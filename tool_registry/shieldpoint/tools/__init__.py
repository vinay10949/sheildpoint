"""
Built-in ShieldPoint tool implementations (SP-201).

Each tool is a plain Python function that takes kwargs validated against a
JSON Schema (declared in :data:`TOOL_SCHEMAS`) and returns a dict. The
:func:`build_default_registry` helper wires all six tools into a
:class:`ToolRegistry` with the appropriate repositories injected.

Tools
-----
- ``claim_lookup``         — retrieve a claim by ID from the claims DB
- ``policy_validate``      — check policy status / coverage / limits / dates
- ``payment_authorize``    — initiate ACH payment with validation + dedup
- ``zkp_prove_policy``     — generate a Policy Validity Proof (Circom/SnarkJS stub)
- ``zkp_verify_proof``     — verify a ZKP proof (Groth16 verifier stub, ~10 ms)
- ``claim_update_status``  — transition a claim through the 8-state machine
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime
from typing import Any, Optional

from ..db import (
    ClaimsRepository,
    InMemoryClaimsRepository,
    InMemoryPaymentLedgerRepository,
    InMemoryPolicyRepository,
    PaymentLedgerRepository,
    PolicyRepository,
)
from ..state_machine import (
    ClaimState,
    ClaimStateMachine,
    GuardConditionFailedError,
    InvalidStateTransitionError,
)
from ..tool_registry import ToolRegistry
from ..zkp import prove_policy_validity, verify_policy_validity_proof

logger = logging.getLogger("shieldpoint.tools")


# ---------------------------------------------------------------------------
# JSON Schemas — exposed for both registration and OpenAI export
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "claim_lookup": {
        "type": "object",
        "properties": {
            "claim_id": {
                "type": "string",
                "description": "The claim ID to look up (e.g. 'CLM-2026-0001').",
                "pattern": r"^CLM-\d{4}-\d{3,}$",
            },
        },
        "required": ["claim_id"],
        "additionalProperties": False,
    },
    "policy_validate": {
        "type": "object",
        "properties": {
            "policy_id": {
                "type": "string",
                "description": "The policy ID to validate (e.g. 'HO-2024-001').",
            },
            "as_of_date": {
                "type": "string",
                "description": "ISO date (YYYY-MM-DD) to check the policy against. Defaults to today.",
            },
        },
        "required": ["policy_id"],
        "additionalProperties": False,
    },
    "payment_authorize": {
        "type": "object",
        "properties": {
            "claim_id": {
                "type": "string",
                "description": "The claim ID the payment is for.",
            },
            "amount": {
                "type": "number",
                "description": "Payment amount in USD. Must be > 0.",
                "exclusiveMinimum": 0,
            },
            "payee": {
                "type": "string",
                "description": "Name of the payee.",
                "minLength": 1,
            },
            "policy_id": {
                "type": "string",
                "description": "Associated policy ID (optional).",
                "default": "",
            },
            "idempotency_key": {
                "type": "string",
                "description": (
                    "Caller-supplied dedup key. If a payment with this key "
                    "already exists, the call returns the existing record "
                    "instead of initiating a new ACH transfer."
                ),
            },
            "ach_reference": {
                "type": "string",
                "description": "Optional ACH trace reference from the bank.",
                "default": "",
            },
        },
        "required": ["claim_id", "amount", "payee"],
        "additionalProperties": False,
    },
    "zkp_prove_policy": {
        "type": "object",
        "properties": {
            "claim_id": {"type": "string", "description": "The claim ID."},
            "policy_id": {"type": "string", "description": "The policy ID."},
            "claim_amount": {
                "type": "number",
                "description": "The claim amount being asserted as within coverage.",
            },
            "coverage_limit": {
                "type": "number",
                "description": "The policy coverage limit.",
            },
            "peril_covered": {
                "type": "boolean",
                "description": "Whether the loss peril is covered by the policy.",
            },
            "policy_active": {
                "type": "boolean",
                "description": "Whether the policy is active as of the claim date.",
            },
        },
        "required": [
            "claim_id",
            "policy_id",
            "claim_amount",
            "coverage_limit",
            "peril_covered",
            "policy_active",
        ],
        "additionalProperties": False,
    },
    "zkp_verify_proof": {
        "type": "object",
        "properties": {
            "proof": {
                "type": "string",
                "description": "The ZKP proof string returned by zkp_prove_policy.",
            },
            "public_signals": {
                "type": "object",
                "description": "The public signals dict returned by zkp_prove_policy.",
                "additionalProperties": True,
            },
            "verification_key": {
                "type": "string",
                "description": "Verification key ID. Defaults to 'vk:policy_validity.v1'.",
                "default": "vk:policy_validity.v1",
            },
        },
        "required": ["proof", "public_signals"],
        "additionalProperties": False,
    },
    "claim_update_status": {
        "type": "object",
        "properties": {
            "claim_id": {
                "type": "string",
                "description": "The claim ID to transition.",
            },
            "new_status": {
                "type": "string",
                "description": (
                    "Target state. One of: claim_received, validating, "
                    "zkp_policy_proof, classifying, zkp_compliance_proof, "
                    "escalating, approved, paid_out."
                ),
                "enum": [
                    "claim_received",
                    "validating",
                    "zkp_policy_proof",
                    "classifying",
                    "zkp_compliance_proof",
                    "escalating",
                    "approved",
                    "paid_out",
                ],
            },
            "context": {
                "type": "object",
                "description": (
                    "Guard context. Required keys vary by transition — e.g. "
                    "ZKP_POLICY_PROOF → CLASSIFYING requires "
                    "{proof_verified, confidence}. "
                    "ESCALATING → APPROVED requires {human_approval, adjuster_id}."
                ),
                "additionalProperties": True,
            },
            "updated_by": {
                "type": "string",
                "description": "Identity of the user/agent performing the transition.",
                "default": "system",
            },
        },
        "required": ["claim_id", "new_status"],
        "additionalProperties": False,
    },
}


TOOL_DESCRIPTIONS: dict[str, str] = {
    "claim_lookup": (
        "Retrieve a claim by its ID from the PostgreSQL claims database. "
        "Returns the full claim record including status, documents, and "
        "assigned adjuster."
    ),
    "policy_validate": (
        "Validate a policy by ID. Checks policy status (active/lapsed/cancelled), "
        "coverage type, effective/expiration dates, and perils covered/excluded. "
        "Returns the full policy record with a validation summary."
    ),
    "payment_authorize": (
        "Initiate an ACH payment for an approved claim. Validates amount > 0, "
        "detects duplicate payments via idempotency_key, and records the "
        "authorization in the payment ledger. Returns the payment record."
    ),
    "zkp_prove_policy": (
        "Generate a Zero-Knowledge Proof (Groth16 stub) that demonstrates "
        "policy validity — claim_amount ≤ coverage_limit, peril is covered, "
        "policy is active — without revealing the underlying values. "
        "Production impl (SP-202) shells out to Circom/SnarkJS."
    ),
    "zkp_verify_proof": (
        "Verify a ZKP proof using the Groth16 verifier (~10 ms constant time). "
        "Returns {verified, verifier, latency_ms, reason}. "
        "Production impl (SP-202) shells out to SnarkJS."
    ),
    "claim_update_status": (
        "Transition a claim through the 8-state machine. Each transition is "
        "guarded by an explicit condition (see ClaimStateMachine). If the "
        "guard fails, returns a structured error with the failing condition. "
        "On success, persists the new status to the claims database."
    ),
}


# ---------------------------------------------------------------------------
# 1. claim_lookup
# ---------------------------------------------------------------------------
def claim_lookup(
    claim_id: str,
    *,
    claims_repo: ClaimsRepository,
) -> dict[str, Any]:
    """Look up a claim by ID.

    Returns the full claim record. If the claim is not found, returns
    ``{"error": ..., "claim_id": claim_id}`` (does NOT raise — the agent
    needs to be able to feed this back to the LLM).
    """
    claim = claims_repo.get_claim(claim_id)
    if claim is None:
        logger.warning("claim_lookup: claim_id=%s not found", claim_id)
        return {
            "error": f"Claim '{claim_id}' not found",
            "claim_id": claim_id,
            "found": False,
        }
    logger.info("claim_lookup: claim_id=%s status=%s", claim_id, claim.get("status"))
    return {**claim, "found": True}


# ---------------------------------------------------------------------------
# 2. policy_validate
# ---------------------------------------------------------------------------
def policy_validate(
    policy_id: str,
    *,
    policy_repo: PolicyRepository,
    as_of_date: Optional[str] = None,
) -> dict[str, Any]:
    """Validate a policy by ID.

    Checks:
    - Policy exists
    - Status is ``active``
    - ``as_of_date`` (default: today) falls within
      ``[effective_date, expiration_date]``
    - Perils covered / excluded lists are non-empty and disjoint

    Returns the policy record augmented with a ``validation`` summary
    containing ``is_active``, ``is_in_force``, ``coverage_ok``, and a list
    of ``issues``.
    """
    policy = policy_repo.get_policy(policy_id)
    if policy is None:
        logger.warning("policy_validate: policy_id=%s not found", policy_id)
        return {
            "error": f"Policy '{policy_id}' not found",
            "policy_id": policy_id,
            "found": False,
        }

    # Parse the as-of date
    if as_of_date:
        try:
            ref_date = date.fromisoformat(as_of_date)
        except ValueError:
            return {
                "error": f"Invalid as_of_date: {as_of_date!r} (expected YYYY-MM-DD).",
                "policy_id": policy_id,
                "found": True,
            }
    else:
        ref_date = date.today()

    # Parse effective / expiration dates
    try:
        eff = date.fromisoformat(str(policy.get("effective_date")))
        exp = date.fromisoformat(str(policy.get("expiration_date")))
    except (KeyError, ValueError, TypeError):
        return {
            "error": "Policy record is missing or has invalid effective/expiration dates.",
            "policy_id": policy_id,
            "found": True,
        }

    is_active = policy.get("status") == "active"
    is_in_force = eff <= ref_date <= exp

    covered = set(policy.get("perils_covered") or [])
    excluded = set(policy.get("perils_excluded") or [])
    overlap = covered & excluded
    coverage_ok = not overlap and len(covered) > 0

    issues: list[str] = []
    if not is_active:
        issues.append(f"Policy status is '{policy.get('status')}', not 'active'.")
    if not is_in_force:
        issues.append(
            f"Policy not in force as of {ref_date.isoformat()} "
            f"(effective {eff.isoformat()}, expires {exp.isoformat()})."
        )
    if overlap:
        issues.append(f"Perils appear in both covered and excluded: {sorted(overlap)}.")
    if not covered:
        issues.append("Policy has no covered perils.")

    validation = {
        "is_active": is_active,
        "is_in_force": is_in_force,
        "coverage_ok": coverage_ok,
        "as_of_date": ref_date.isoformat(),
        "issues": issues,
        "valid": is_active and is_in_force and coverage_ok,
    }

    logger.info(
        "policy_validate: policy_id=%s valid=%s issues=%d",
        policy_id, validation["valid"], len(issues),
    )
    return {**policy, "found": True, "validation": validation}


# ---------------------------------------------------------------------------
# 3. payment_authorize
# ---------------------------------------------------------------------------
def payment_authorize(
    claim_id: str,
    amount: float,
    payee: str,
    *,
    payment_repo: PaymentLedgerRepository,
    claims_repo: Optional[ClaimsRepository] = None,
    policy_id: str = "",
    idempotency_key: Optional[str] = None,
    ach_reference: str = "",
) -> dict[str, Any]:
    """Authorize an ACH payment for an approved claim.

    Validates:
    - ``amount > 0``          (rejects zero / negative payments)
    - ``payee`` non-empty
    - Claim exists (if ``claims_repo`` provided)
    - No duplicate payment for the same ``claim_id`` (or same ``idempotency_key``)

    On duplicate detection, returns the existing payment record with
    ``status == "duplicate_detected"`` instead of initiating a new ACH
    transfer. This is the key control that eliminates the $340K/yr in
    invalid payouts called out in the ShieldPoint plan.

    Returns the payment record (existing or newly inserted).
    """
    # ---- Amount / payee validation ---------------------------------- #
    if amount <= 0:
        return {
            "error": f"Amount must be > 0; got {amount!r}.",
            "claim_id": claim_id,
            "amount": amount,
            "status": "rejected",
        }
    if not payee or not payee.strip():
        return {
            "error": "payee must be a non-empty string.",
            "claim_id": claim_id,
            "status": "rejected",
        }

    # ---- Claim existence check (optional) --------------------------- #
    if claims_repo is not None:
        claim = claims_repo.get_claim(claim_id)
        if claim is None:
            return {
                "error": f"Claim '{claim_id}' not found; cannot authorize payment.",
                "claim_id": claim_id,
                "status": "rejected",
            }

    # ---- Duplicate detection ---------------------------------------- #
    # Two flavours of duplicate:
    # (a) Caller-supplied idempotency_key already in the ledger.
    # (b) An existing authorized payment for the same claim_id with the
    #     same amount + payee (catches the case where the caller forgot
    #     to pass an idempotency_key).
    dedup_key = idempotency_key or f"{claim_id}:{amount:.2f}:{payee}"
    existing = payment_repo.find_by_idempotency_key(dedup_key)
    if existing is not None:
        logger.warning(
            "payment_authorize: duplicate detected claim_id=%s key=%s",
            claim_id, dedup_key,
        )
        return {
            **existing,
            "status": "duplicate_detected",
            "duplicate_of": existing["payment_id"],
            "message": (
                "Duplicate payment detected: a payment with the same idempotency "
                "key already exists. Returning the original record instead of "
                "initiating a new ACH transfer."
            ),
        }

    for prior in payment_repo.find_by_claim(claim_id):
        if (
            prior.get("status") in {"authorized", "settled"}
            and abs(float(prior.get("amount", 0)) - float(amount)) < 0.01
            and prior.get("payee") == payee
        ):
            logger.warning(
                "payment_authorize: duplicate detected (claim+amount+payee) claim_id=%s",
                claim_id,
            )
            return {
                **prior,
                "status": "duplicate_detected",
                "duplicate_of": prior["payment_id"],
                "message": (
                    "Duplicate payment detected: an identical authorized payment "
                    "already exists for this claim (same amount and payee). "
                    "Returning the original record."
                ),
            }

    # ---- Insert new authorization ----------------------------------- #
    payment_id = f"PMT-{claim_id}-{uuid.uuid4().hex[:8].upper()}"
    record = {
        "payment_id": payment_id,
        "claim_id": claim_id,
        "policy_id": policy_id,
        "amount": float(amount),
        "payee": payee,
        "ach_reference": ach_reference or f"ACH-{uuid.uuid4().hex[:10].upper()}",
        "status": "authorized",
        "idempotency_key": dedup_key,
        "created_at": time.time(),
    }
    inserted = payment_repo.insert(record)
    logger.info(
        "payment_authorize: payment_id=%s claim_id=%s amount=%.2f payee=%s",
        payment_id, claim_id, amount, payee,
    )
    return inserted


# ---------------------------------------------------------------------------
# 4. zkp_prove_policy
# ---------------------------------------------------------------------------
def zkp_prove_policy(
    claim_id: str,
    policy_id: str,
    claim_amount: float,
    coverage_limit: float,
    peril_covered: bool,
    policy_active: bool,
) -> dict[str, Any]:
    """Generate a Policy Validity Proof (Circom/SnarkJS stub)."""
    return prove_policy_validity(
        claim_id=claim_id,
        policy_id=policy_id,
        claim_amount=claim_amount,
        coverage_limit=coverage_limit,
        peril_covered=peril_covered,
        policy_active=policy_active,
    )


# ---------------------------------------------------------------------------
# 5. zkp_verify_proof
# ---------------------------------------------------------------------------
def zkp_verify_proof(
    proof: str,
    public_signals: dict[str, Any],
    verification_key: str = "vk:policy_validity.v1",
) -> dict[str, Any]:
    """Verify a ZKP proof (Groth16 verifier stub, ~10 ms constant time)."""
    return verify_policy_validity_proof(
        proof=proof,
        public_signals=public_signals,
        verification_key=verification_key,
    )


# ---------------------------------------------------------------------------
# 6. claim_update_status
# ---------------------------------------------------------------------------
def claim_update_status(
    claim_id: str,
    new_status: str,
    *,
    claims_repo: ClaimsRepository,
    state_machine: Optional[ClaimStateMachine] = None,
    context: Optional[dict[str, Any]] = None,
    updated_by: str = "system",
) -> dict[str, Any]:
    """Transition a claim through the 8-state machine.

    Steps:
    1. Look up the claim. If not found, return an error dict.
    2. Normalize the current status to a canonical :class:`ClaimState`.
    3. Validate the transition exists and its guard condition is satisfied
       (using ``context`` for guard inputs like ``proof_verified``,
       ``confidence``, ``human_approval``, etc.).
    4. Persist the new status via ``claims_repo.update_status``.

    On guard failure, returns a structured error with the failing condition
    (does NOT raise — the agent needs to feed this back to the LLM).
    """
    sm = state_machine or ClaimStateMachine()
    ctx = context or {}

    claim = claims_repo.get_claim(claim_id)
    if claim is None:
        return {
            "error": f"Claim '{claim_id}' not found.",
            "claim_id": claim_id,
            "status": "rejected",
        }

    current_status = str(claim.get("status", ""))
    try:
        current_state = ClaimState.normalize(current_status).canonical()
    except ValueError as exc:
        return {
            "error": f"Claim has unknown status {current_status!r}: {exc}",
            "claim_id": claim_id,
            "current_status": current_status,
            "status": "rejected",
        }

    try:
        new_state = sm.transition(
            from_state=current_state,
            to_state=new_status,
            claim=claim,
            context=ctx,
        )
    except InvalidStateTransitionError as exc:
        return {
            "error": str(exc),
            "claim_id": claim_id,
            "current_status": current_state.value,
            "requested_status": new_status,
            "allowed_targets": list(exc.allowed),
            "status": "invalid_transition",
        }
    except GuardConditionFailedError as exc:
        return {
            "error": str(exc),
            "claim_id": claim_id,
            "current_status": current_state.value,
            "requested_status": new_status,
            "guard_details": exc.details,
            "status": "guard_failed",
        }

    # ---- Persist ---------------------------------------------------- #
    updated = claims_repo.update_status(
        claim_id, new_state.value, updated_by=updated_by
    )
    if updated is None:
        return {
            "error": "Claim disappeared between lookup and update.",
            "claim_id": claim_id,
            "status": "rejected",
        }

    logger.info(
        "claim_update_status: claim_id=%s %s → %s by %s",
        claim_id, current_state.value, new_state.value, updated_by,
    )
    return {
        "claim_id": claim_id,
        "previous_status": current_state.value,
        "new_status": new_state.value,
        "updated_by": updated_by,
        "updated_claim": updated,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Default registry builder
# ---------------------------------------------------------------------------
def build_default_registry(
    *,
    claims_repo: Optional[ClaimsRepository] = None,
    policy_repo: Optional[PolicyRepository] = None,
    payment_repo: Optional[PaymentLedgerRepository] = None,
    state_machine: Optional[ClaimStateMachine] = None,
    span_recorder: Any = None,
) -> ToolRegistry:
    """Build a :class:`ToolRegistry` pre-loaded with the six SP-201 tools.

    Repositories default to in-memory implementations seeded with the
    ShieldPoint demo dataset. In production, pass PostgreSQL-backed
    implementations of the same protocols.

    The returned registry has Langfuse span capture wired up via
    ``span_recorder`` (defaults to :class:`NullSpanRecorder`, which keeps
    an in-memory audit log of every tool call).
    """
    from ..langfuse_span import NullSpanRecorder

    registry = ToolRegistry(span_recorder=span_recorder or NullSpanRecorder())

    claims = claims_repo or InMemoryClaimsRepository()
    policies = policy_repo or InMemoryPolicyRepository()
    payments = payment_repo or InMemoryPaymentLedgerRepository()
    sm = state_machine or ClaimStateMachine()

    # ---- claim_lookup ----------------------------------------------- #
    def _claim_lookup(claim_id: str) -> dict[str, Any]:
        return claim_lookup(claim_id, claims_repo=claims)

    _claim_lookup.__doc__ = TOOL_DESCRIPTIONS["claim_lookup"]
    registry.register_tool(
        _claim_lookup,
        name="claim_lookup",
        description=TOOL_DESCRIPTIONS["claim_lookup"],
        schema=TOOL_SCHEMAS["claim_lookup"],
    )

    # ---- policy_validate -------------------------------------------- #
    def _policy_validate(
        policy_id: str, as_of_date: Optional[str] = None
    ) -> dict[str, Any]:
        return policy_validate(
            policy_id, policy_repo=policies, as_of_date=as_of_date
        )

    _policy_validate.__doc__ = TOOL_DESCRIPTIONS["policy_validate"]
    # The as_of_date parameter is optional — we need a schema that allows it.
    # The base TOOL_SCHEMAS["policy_validate"] already declares it as
    # optional (not in "required"), so we can reuse it directly.
    registry.register_tool(
        _policy_validate,
        name="policy_validate",
        description=TOOL_DESCRIPTIONS["policy_validate"],
        schema=TOOL_SCHEMAS["policy_validate"],
    )

    # ---- payment_authorize ------------------------------------------ #
    def _payment_authorize(
        claim_id: str,
        amount: float,
        payee: str,
        policy_id: str = "",
        idempotency_key: Optional[str] = None,
        ach_reference: str = "",
    ) -> dict[str, Any]:
        return payment_authorize(
            claim_id=claim_id,
            amount=amount,
            payee=payee,
            payment_repo=payments,
            claims_repo=claims,
            policy_id=policy_id,
            idempotency_key=idempotency_key,
            ach_reference=ach_reference,
        )

    _payment_authorize.__doc__ = TOOL_DESCRIPTIONS["payment_authorize"]
    registry.register_tool(
        _payment_authorize,
        name="payment_authorize",
        description=TOOL_DESCRIPTIONS["payment_authorize"],
        schema=TOOL_SCHEMAS["payment_authorize"],
    )

    # ---- zkp_prove_policy ------------------------------------------- #
    def _zkp_prove_policy(
        claim_id: str,
        policy_id: str,
        claim_amount: float,
        coverage_limit: float,
        peril_covered: bool,
        policy_active: bool,
    ) -> dict[str, Any]:
        return zkp_prove_policy(
            claim_id=claim_id,
            policy_id=policy_id,
            claim_amount=claim_amount,
            coverage_limit=coverage_limit,
            peril_covered=peril_covered,
            policy_active=policy_active,
        )

    _zkp_prove_policy.__doc__ = TOOL_DESCRIPTIONS["zkp_prove_policy"]
    registry.register_tool(
        _zkp_prove_policy,
        name="zkp_prove_policy",
        description=TOOL_DESCRIPTIONS["zkp_prove_policy"],
        schema=TOOL_SCHEMAS["zkp_prove_policy"],
    )

    # ---- zkp_verify_proof ------------------------------------------- #
    def _zkp_verify_proof(
        proof: str,
        public_signals: dict[str, Any],
        verification_key: str = "vk:policy_validity.v1",
    ) -> dict[str, Any]:
        return zkp_verify_proof(
            proof=proof,
            public_signals=public_signals,
            verification_key=verification_key,
        )

    _zkp_verify_proof.__doc__ = TOOL_DESCRIPTIONS["zkp_verify_proof"]
    registry.register_tool(
        _zkp_verify_proof,
        name="zkp_verify_proof",
        description=TOOL_DESCRIPTIONS["zkp_verify_proof"],
        schema=TOOL_SCHEMAS["zkp_verify_proof"],
    )

    # ---- claim_update_status ---------------------------------------- #
    def _claim_update_status(
        claim_id: str,
        new_status: str,
        context: Optional[dict[str, Any]] = None,
        updated_by: str = "system",
    ) -> dict[str, Any]:
        return claim_update_status(
            claim_id=claim_id,
            new_status=new_status,
            claims_repo=claims,
            state_machine=sm,
            context=context,
            updated_by=updated_by,
        )

    _claim_update_status.__doc__ = TOOL_DESCRIPTIONS["claim_update_status"]
    registry.register_tool(
        _claim_update_status,
        name="claim_update_status",
        description=TOOL_DESCRIPTIONS["claim_update_status"],
        schema=TOOL_SCHEMAS["claim_update_status"],
    )

    return registry
