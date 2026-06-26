"""
Claim state machine — 8 states, 9 transitions, 2 ZKP gates.

This module implements the state machine defined in Section 5 of the
ShieldPoint Claims Automation Implementation Plan v2.0. Each transition
is guarded by an explicit condition that must be satisfied; if the guard
fails, the transition is rejected with :class:`GuardConditionFailedError`.

States (8)
----------
- ``CLAIM_RECEIVED``        — IntakeAgent has parsed the claim.
- ``VALIDATING``            — ValidatorAgent is cross-referencing the policy DB.
- ``ZKP_POLICY_PROOF``      — ZKP prover is generating a Policy Validity Proof.
- ``CLASSIFYING``           — ClassifierAgent is assigning severity / fraud risk.
- ``ZKP_COMPLIANCE_PROOF``  — ZKP prover is generating a Compliance Proof.
- ``ESCALATING``            — EscalationAgent has routed to a human adjuster.
- ``APPROVED``              — Decision recorded; awaiting payout.
- ``PAID_OUT``              — PayoutAgent has executed the ACH transfer.

Transitions (9)
---------------
1. ``CLAIM_RECEIVED``        → ``VALIDATING``             (always)
2. ``VALIDATING``            → ``ZKP_POLICY_PROOF``       (all required fields present)
3. ``ZKP_POLICY_PROOF``      → ``CLASSIFYING``            (proof verified + confidence ≥ 0.85)
4. ``ZKP_POLICY_PROOF``      → ``ESCALATING``             (proof invalid or confidence < 0.85)
5. ``CLASSIFYING``           → ``ZKP_COMPLIANCE_PROOF``   (severity classified + fraud score computed)
6. ``ZKP_COMPLIANCE_PROOF``  → ``APPROVED``               (compliance proved + low risk + confidence ≥ 0.85)
7. ``ZKP_COMPLIANCE_PROOF``  → ``ESCALATING``             (compliance uncertain or high risk)
8. ``ESCALATING``            → ``APPROVED``               (human adjuster explicit approval)
9. ``APPROVED``              → ``PAID_OUT``               (payment authorization + bank details verified)

The ``claim_update_status`` tool delegates to this state machine for guard
validation before persisting the new status to the claims repository.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class InvalidStateTransitionError(ValueError):
    """Raised when a transition is not defined in the state machine."""

    def __init__(self, from_state: str, to_state: str, *, allowed: list[str]) -> None:
        super().__init__(
            f"Invalid transition: {from_state} → {to_state}. "
            f"Allowed from {from_state}: {allowed}"
        )
        self.from_state = from_state
        self.to_state = to_state
        self.allowed = allowed


class GuardConditionFailedError(ValueError):
    """Raised when a transition's guard condition is not satisfied.

    Carries a structured ``details`` dict so the agent can feed the reason
    back to the LLM (or to the human adjuster in the escalation queue).
    """

    def __init__(
        self, message: str, *, from_state: str, to_state: str, details: dict[str, Any]
    ) -> None:
        super().__init__(message)
        self.from_state = from_state
        self.to_state = to_state
        self.details = details


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------
class ClaimState(str, enum.Enum):
    """The 8 discrete states a claim can occupy."""

    CLAIM_RECEIVED = "claim_received"
    VALIDATING = "validating"
    ZKP_POLICY_PROOF = "zkp_policy_proof"
    CLASSIFYING = "classifying"
    ZKP_COMPLIANCE_PROOF = "zkp_compliance_proof"
    ESCALATING = "escalating"
    APPROVED = "approved"
    PAID_OUT = "paid_out"

    # ---- Legacy / status-quo aliases (existing repo used these) ---- #
    # The plan's 8 states are the canonical set, but the existing demo
    # claims in the repo use older status strings. We accept them as
    # aliases of CLAIM_RECEIVED so the state machine doesn't reject
    # pre-existing demo data.
    SUBMITTED = "submitted"  # alias for CLAIM_RECEIVED
    UNDER_INVESTIGATION = "under_investigation"  # alias for ESCALATING
    DENIED = "denied"  # terminal state (escalation outcome)

    @classmethod
    def normalize(cls, status: "str | ClaimState") -> "ClaimState":
        """Map a raw status string (which may be a legacy alias) to the
        canonical 8-state enum value.

        Accepts either a :class:`ClaimState` instance (returned unchanged
        after canonicalization) or a string (matched against enum values
        and aliases).
        """
        if isinstance(status, ClaimState):
            return status
        if not isinstance(status, str):
            raise TypeError(
                f"Claim status must be a str or ClaimState, got {type(status).__name__}"
            )
        s = status.strip().lower()
        # Direct enum value match
        for member in cls:
            if member.value == s:
                return member
        # Aliases
        if s in {"submitted", "intake", "received", "new"}:
            return cls.CLAIM_RECEIVED
        if s in {"under_investigation", "investigating", "manual_review"}:
            return cls.ESCALATING
        if s in {"denied", "rejected", "closed_denied"}:
            return cls.DENIED
        raise ValueError(f"Unknown claim status: {status!r}")

    def canonical(self) -> "ClaimState":
        """Return the canonical (non-alias) state for this member."""
        if self is ClaimState.SUBMITTED:
            return ClaimState.CLAIM_RECEIVED
        if self is ClaimState.UNDER_INVESTIGATION:
            return ClaimState.ESCALATING
        return self


# ---------------------------------------------------------------------------
# Guard signature
# ---------------------------------------------------------------------------
GuardFn = Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str, dict[str, Any]]]
"""Guard function signature.

Takes ``(claim, context)`` and returns ``(ok, reason, details)``:
- ``ok``       — True if the guard is satisfied.
- ``reason``   — human-readable explanation (passed to the LLM / adjuster).
- ``details``  — structured dict for Langfuse span metadata.
"""


# ---------------------------------------------------------------------------
# Transition dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Transition:
    """A single defined transition in the state machine."""

    from_state: ClaimState
    to_state: ClaimState
    name: str
    description: str
    guard: Optional[GuardFn] = None
    failure_route: Optional[ClaimState] = None  # where to go if guard fails


# ---------------------------------------------------------------------------
# Built-in guard functions
# ---------------------------------------------------------------------------
def _guard_validating_to_zkp_policy_proof(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """All required fields present, no data format errors."""
    required = ["claim_id", "policy_id", "claimant", "amount", "date_of_loss"]
    # Use ``in`` rather than truthiness — amount=0 is a valid (if weird) value.
    missing = [f for f in required if f not in claim or claim.get(f) is None]
    if missing:
        return (
            False,
            f"Missing required fields: {missing}. Returning claim to CLAIM_RECEIVED.",
            {"missing_fields": missing},
        )
    try:
        amount = float(claim.get("amount", 0))
        if amount <= 0:
            return (False, "Amount must be > 0.", {"amount": amount})
    except (TypeError, ValueError):
        return (False, "Amount is not a valid number.", {"amount": claim.get("amount")})
    return (True, "All required fields present.", {})


def _guard_zkp_to_classifying(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Proof verified + confidence >= 0.85."""
    proof_verified = context.get("proof_verified", False)
    confidence = float(context.get("confidence", 0.0))
    if not proof_verified:
        return (False, "ZKP policy proof failed verification.", {"proof_verified": False})
    if confidence < 0.85:
        return (
            False,
            f"Confidence {confidence:.2f} below 0.85 threshold; escalating.",
            {"confidence": confidence, "threshold": 0.85},
        )
    return (True, "Proof verified and confidence sufficient.", {"confidence": confidence})


def _guard_classifying_to_zkp_compliance(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Severity classified + fraud risk score computed."""
    severity = context.get("severity")
    fraud_score = context.get("fraud_risk_score")
    if severity is None:
        return (False, "Severity not yet classified.", {})
    if fraud_score is None:
        return (False, "Fraud risk score not yet computed.", {})
    return (
        True,
        f"Severity={severity}, fraud_score={fraud_score}.",
        {"severity": severity, "fraud_risk_score": fraud_score},
    )


def _guard_zkp_compliance_to_approved(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Compliance proved + low-risk classification + confidence >= 0.85."""
    compliance_proved = context.get("compliance_proved", False)
    risk = context.get("risk_class", "high")
    confidence = float(context.get("confidence", 0.0))
    if not compliance_proved:
        return (False, "Compliance proof failed.", {"compliance_proved": False})
    if risk != "low":
        return (False, f"Risk class {risk!r} is not 'low'; escalating.", {"risk_class": risk})
    if confidence < 0.85:
        return (False, f"Confidence {confidence:.2f} below 0.85.", {"confidence": confidence})
    return (True, "Compliance proved, low risk, confidence sufficient.", {})


def _guard_escalating_to_approved(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Human adjuster explicit approval."""
    approved = context.get("human_approval", False)
    adjuster_id = context.get("adjuster_id")
    if not approved:
        return (False, "Human approval not yet recorded.", {"human_approval": False})
    if not adjuster_id:
        return (False, "Adjuster ID missing on approval.", {})
    return (True, f"Approved by adjuster {adjuster_id}.", {"adjuster_id": adjuster_id})


def _guard_approved_to_paid_out(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Payment authorization + bank details verified."""
    payment_authorized = context.get("payment_authorized", False)
    bank_verified = context.get("bank_details_verified", False)
    if not payment_authorized:
        return (False, "Payment not yet authorized.", {})
    if not bank_verified:
        return (False, "Bank details not yet verified.", {})
    return (True, "Payment authorized and bank verified.", {})


def _always_ok(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    return (True, "Automatic transition.", {})


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
@dataclass
class ClaimStateMachine:
    """The 8-state, 9-transition claim state machine.

    The transition table is built once at construction; guards are looked up
    by ``(from_state, to_state)``. To attempt a transition, call
    :meth:`check_transition` (returns the guard result without persisting)
    or :meth:`transition` (validates + returns the new state, leaving
    persistence to the caller).
    """

    transitions: list[Transition] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.transitions:
            self.transitions = self._default_transitions()
        # Index for fast lookup
        self._index: dict[tuple[ClaimState, ClaimState], Transition] = {}
        for t in self.transitions:
            self._index[(t.from_state.canonical(), t.to_state.canonical())] = t

    @staticmethod
    def _default_transitions() -> list[Transition]:
        return [
            Transition(
                from_state=ClaimState.CLAIM_RECEIVED,
                to_state=ClaimState.VALIDATING,
                name="intake_to_validating",
                description="Always automatic trigger.",
                guard=_always_ok,
            ),
            Transition(
                from_state=ClaimState.VALIDATING,
                to_state=ClaimState.ZKP_POLICY_PROOF,
                name="validating_to_zkp_policy",
                description="All required fields present, no data format errors.",
                guard=_guard_validating_to_zkp_policy_proof,
                failure_route=ClaimState.CLAIM_RECEIVED,
            ),
            Transition(
                from_state=ClaimState.ZKP_POLICY_PROOF,
                to_state=ClaimState.CLASSIFYING,
                name="zkp_policy_to_classifying",
                description="Proof verified + confidence >= 0.85.",
                guard=_guard_zkp_to_classifying,
                failure_route=ClaimState.ESCALATING,
            ),
            Transition(
                from_state=ClaimState.ZKP_POLICY_PROOF,
                to_state=ClaimState.ESCALATING,
                name="zkp_policy_to_escalating",
                description="Proof invalid or confidence < 0.85.",
                guard=_always_ok,
            ),
            Transition(
                from_state=ClaimState.CLASSIFYING,
                to_state=ClaimState.ZKP_COMPLIANCE_PROOF,
                name="classifying_to_zkp_compliance",
                description="Severity classified + fraud risk score computed.",
                guard=_guard_classifying_to_zkp_compliance,
                failure_route=ClaimState.ESCALATING,
            ),
            Transition(
                from_state=ClaimState.ZKP_COMPLIANCE_PROOF,
                to_state=ClaimState.APPROVED,
                name="zkp_compliance_to_approved",
                description="Compliance proved + low risk + confidence >= 0.85.",
                guard=_guard_zkp_compliance_to_approved,
                failure_route=ClaimState.ESCALATING,
            ),
            Transition(
                from_state=ClaimState.ZKP_COMPLIANCE_PROOF,
                to_state=ClaimState.ESCALATING,
                name="zkp_compliance_to_escalating",
                description="Compliance uncertain or high risk.",
                guard=_always_ok,
            ),
            Transition(
                from_state=ClaimState.ESCALATING,
                to_state=ClaimState.APPROVED,
                name="escalating_to_approved",
                description="Human adjuster explicit approval.",
                guard=_guard_escalating_to_approved,
            ),
            Transition(
                from_state=ClaimState.APPROVED,
                to_state=ClaimState.PAID_OUT,
                name="approved_to_paid_out",
                description="Payment authorization + bank details verified.",
                guard=_guard_approved_to_paid_out,
            ),
        ]

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #
    def allowed_targets(self, from_state: str | ClaimState) -> list[ClaimState]:
        """Return the list of states reachable from ``from_state``."""
        canonical = ClaimState.normalize(from_state).canonical()
        return [
            t.to_state.canonical()
            for t in self.transitions
            if t.from_state.canonical() == canonical
        ]

    def get_transition(
        self, from_state: str | ClaimState, to_state: str | ClaimState
    ) -> Transition:
        """Return the :class:`Transition` between two states.

        Raises :class:`InvalidStateTransitionError` if no such transition
        is defined.
        """
        f = ClaimState.normalize(from_state).canonical()
        t = ClaimState.normalize(to_state).canonical()
        transition = self._index.get((f, t))
        if transition is None:
            raise InvalidStateTransitionError(
                from_state=_state_str(from_state),
                to_state=_state_str(to_state),
                allowed=[s.value for s in self.allowed_targets(f)],
            )
        return transition

    def check_transition(
        self,
        from_state: str | ClaimState,
        to_state: str | ClaimState,
        *,
        claim: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Evaluate a transition's guard without persisting anything.

        Returns ``(ok, reason, details)``. If the transition is undefined,
        raises :class:`InvalidStateTransitionError`.
        """
        transition = self.get_transition(from_state, to_state)
        if transition.guard is None:
            return (True, "No guard defined.", {})
        return transition.guard(claim or {}, context or {})

    def transition(
        self,
        from_state: str | ClaimState,
        to_state: str | ClaimState,
        *,
        claim: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> ClaimState:
        """Validate the transition and return the new canonical state.

        Raises:
            - :class:`InvalidStateTransitionError` — transition not defined.
            - :class:`GuardConditionFailedError`   — guard returned False.

        Does NOT persist the new state; the caller (typically the
        ``claim_update_status`` tool) is responsible for writing it to the
        claims repository.
        """
        transition = self.get_transition(from_state, to_state)
        if transition.guard is not None:
            ok, reason, details = transition.guard(claim or {}, context or {})
            if not ok:
                raise GuardConditionFailedError(
                    f"Guard for {transition.name} failed: {reason}",
                    from_state=_state_str(from_state),
                    to_state=_state_str(to_state),
                    details={
                        "transition": transition.name,
                        "reason": reason,
                        **details,
                    },
                )
        return ClaimState.normalize(to_state).canonical()


def _state_str(s: str | ClaimState) -> str:
    """Return the canonical value string for a state input."""
    if isinstance(s, ClaimState):
        return s.canonical().value
    return s
