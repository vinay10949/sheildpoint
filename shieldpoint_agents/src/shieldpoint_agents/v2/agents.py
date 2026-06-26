"""
ShieldPoint V2 Agents — five state-owning agents
=================================================

- :class:`IntakeAgent`        owns ``CLAIM_RECEIVED``
- :class:`ValidatorAgent`     owns ``VALIDATING``
- :class:`ClassifierAgent`    owns ``CLASSIFYING``
- :class:`EscalationAgent`    owns ``ESCALATING`` (HITL)
- :class:`PayoutAgent`        owns ``APPROVED`` → ``PAID_OUT``

Each agent is a deterministic Python class with the signature::

    agent.run(claim: dict, context: dict) -> dict  # updated context

The state machine engine calls ``agent.run()`` to produce a new context
dict, then attempts the transition via ``StateMachineEngine.transition()``
with that context. Guard failures raise
:class:`GuardConditionFailedError` which the orchestrator catches and
routes to ``ESCALATING``.

LLM client injection
--------------------
:class:`ClassifierAgent` accepts an optional ``llm_client`` parameter. In
production this is a real ``openai.OpenAI`` client pointed at LM Studio
(``LM_STUDIO_BASE_URL=http://localhost:1234/v1``). In tests it is a
:class:`FakeLLMClient` that returns deterministic completions.

Langfuse integration
--------------------
All agents emit spans via the shared ``langfuse_wrapper.py``. Each span
carries ``claim_id``, ``agent_id``, and the structured output. When the
Langfuse SDK is not installed, spans no-op.
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

# ---- State machine engine import (sibling package) -----------------------
_SM_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "state_machine_engine", "src",
)
if os.path.isdir(_SM_PATH) and _SM_PATH not in sys.path:
    sys.path.insert(0, _SM_PATH)

# ---- Compliance prover import (sibling package) ---------------------------
_COMP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "zkp_circuit",
)
if os.path.isdir(_COMP_PATH) and _COMP_PATH not in sys.path:
    sys.path.insert(0, _COMP_PATH)

from state_machine_engine import (  # noqa: E402
    GuardConditionFailedError,
    State,
    StateMachineEngine,
)
from compliance import (  # noqa: E402
    ComplianceProver,
    build_record_from_context,
)

# ---- Payout subsystem import (sibling package) ----------------------------
try:
    from .payout import (  # noqa: E402
        ACHProvider,
        ACHResult,
        StubACHProvider,
        BankVerificationService,
        PaymentLedger,
        PaymentRecord,
        InMemoryPaymentLedger,
        ReceiptGenerator,
        ReceiptResult,
        NotificationService,
        NotificationResult,
        StubNotificationService,
        AuditRecordAssembler,
        AuditRecord,
        check_duplicate,
        compute_payment_breakdown,
    )
    _PAYOUT_AVAILABLE = True
except ImportError:
    _PAYOUT_AVAILABLE = False
    ACHProvider = None  # type: ignore
    ACHResult = None  # type: ignore
    StubACHProvider = None  # type: ignore
    BankVerificationService = None  # type: ignore
    PaymentLedger = None  # type: ignore
    PaymentRecord = None  # type: ignore
    InMemoryPaymentLedger = None  # type: ignore
    ReceiptGenerator = None  # type: ignore
    ReceiptResult = None  # type: ignore
    NotificationService = None  # type: ignore
    NotificationResult = None  # type: ignore
    StubNotificationService = None  # type: ignore
    AuditRecordAssembler = None  # type: ignore
    AuditRecord = None  # type: ignore
    check_duplicate = None  # type: ignore
    compute_payment_breakdown = None  # type: ignore

# ---- Fraud detection subsystem import (sibling package) -------------------
try:
    _FRAUD_PATH = os.path.join(_COMP_PATH, "fraud_detection")
    if os.path.isdir(_FRAUD_PATH) and _FRAUD_PATH not in sys.path:
        sys.path.insert(0, _FRAUD_PATH)
    from fraud_detection import (  # noqa: E402
        FraudDetectionClient,
        FraudDetectionResult,
        CommitmentService,
        SharedMerkleTree,
        NonMembershipProver,
    )
    _FRAUD_AVAILABLE = True
except ImportError:
    _FRAUD_AVAILABLE = False
    FraudDetectionClient = None  # type: ignore
    FraudDetectionResult = None  # type: ignore
    CommitmentService = None  # type: ignore
    SharedMerkleTree = None  # type: ignore
    NonMembershipProver = None  # type: ignore

# ---- Langfuse tracer shim (optional) --------------------------------------
try:
    _lf_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..",
        "agent_framework", "observability",
    )
    if os.path.isdir(_lf_path) and _lf_path not in sys.path:
        sys.path.insert(0, _lf_path)
    from langfuse_wrapper import get_tracer  # type: ignore
    _LANGFUSE_AVAILABLE = True
except Exception:
    _LANGFUSE_AVAILABLE = False
    get_tracer = None  # type: ignore

logger = logging.getLogger("shieldpoint_agents.v2.agents")


# ===========================================================================
# Langfuse shim
# ===========================================================================
class _NullSpan:
    def __enter__(self) -> "_NullSpan": return self
    def __exit__(self, *_a: Any) -> None: ...
    def update(self, *a: Any, **kw: Any) -> None: ...


class _NullTracer:
    def start_as_current_span(self, *a: Any, **kw: Any) -> _NullSpan:
        return _NullSpan()
    def update_current_trace(self, *a: Any, **kw: Any) -> None: ...
    def flush(self) -> None: ...
    def shutdown(self) -> None: ...
    def get_current_trace_id(self) -> Optional[str]: return None


def _tracer() -> Any:
    if not _LANGFUSE_AVAILABLE:
        return _NullTracer()
    try:
        return get_tracer()  # type: ignore[misc]
    except Exception:
        return _NullTracer()


TRACER = _tracer()


# ===========================================================================
# LLM client protocol + fake implementation
# ===========================================================================
@runtime_checkable
class LLMClient(Protocol):
    def chat_completion(self, *, model: str, system: str,
                        user: str, temperature: float = 0.1,
                        max_tokens: int = 1024) -> str: ...


class FakeLLMClient:
    """Deterministic LLM client for tests.

    Calls a user-supplied ``responder(prompt) -> str`` function. If no
    responder is given, returns a default JSON classification that
    classifies every claim as ``low`` severity with fraud score 0.1.
    """

    def __init__(
        self,
        responder: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._responder = responder or (lambda _p: json.dumps({
            "severity": "low",
            "claim_type": "property_damage",
            "fraud_risk_score": 0.1,
            "reasoning": "Default fake-classification for testing.",
            "ambiguous": False,
        }))
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, *, model: str, system: str, user: str,
                        temperature: float = 0.1,
                        max_tokens: int = 1024) -> str:
        self.calls.append({
            "model": model, "system": system, "user": user,
            "temperature": temperature, "max_tokens": max_tokens,
        })
        return self._responder(user)


# ===========================================================================
# IntakeAgent — owns CLAIM_RECEIVED
# ===========================================================================
class IntakeAgent:
    """Parses incoming claim data and validates format completeness.

    The IntakeAgent owns the ``CLAIM_RECEIVED`` state. It accepts a raw
    claim payload (dict from the API gateway / email poller / fax OCR),
    validates that all required fields are present and well-formed,
    assigns a claim_id if one is not present, and returns the cleaned
    claim plus a context dict that the ValidatorAgent will consume.

    Required fields (from the policy validity circuit's needs):
    - ``policy_id``            — non-empty string
    - ``claimant``             — non-empty string
    - ``amount``               — positive number
    - ``date_of_loss``         — YYYY-MM-DD string
    - ``description``          — non-empty string

    Optional fields (preserved if present):
    - ``claim_id``, ``documents``, ``claim_type``, ``jurisdiction``,
      ``adjuster_id``, ``metadata``

    On any format error, the agent returns ``context['format_errors']``
    as a list of strings. The state machine's
    ``CLAIM_RECEIVED → VALIDATING`` guard is unconditional (``_always_ok``),
    so a malformed claim still transitions to VALIDATING — but the
    ValidatorAgent's guard will fail and route the claim back to
    CLAIM_RECEIVED for re-intake. This keeps the state machine
    deterministic while allowing the IntakeAgent to do best-effort
    parsing.
    """

    REQUIRED_FIELDS: list[str] = [
        "policy_id", "claimant", "amount", "date_of_loss", "description",
    ]

    def __init__(self, *, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client

    def run(self, claim: dict[str, Any],
            context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        ctx = dict(context or {})
        with TRACER.start_as_current_span(
            name="agent.IntakeAgent.run",
            input={"claim_id": claim.get("claim_id")},
            metadata={"agent_id": "IntakeAgent",
                       "claim_id": claim.get("claim_id")},
        ) as span:
            cleaned, errors = self._parse_and_validate(claim)
            ctx["claim"] = cleaned
            ctx["format_errors"] = errors
            ctx["intake_complete"] = (len(errors) == 0)
            ctx["intake_timestamp"] = time.time()
            span.update(output={
                "intake_complete": ctx["intake_complete"],
                "format_errors": errors,
            })
        return ctx

    # ------------------------------------------------------------------ #
    def _parse_and_validate(
        self, raw: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        cleaned = dict(raw)
        errors: list[str] = []
        # Assign claim_id if missing
        if not cleaned.get("claim_id"):
            cleaned["claim_id"] = f"CLM-{uuid.uuid4().hex[:12].upper()}"
        # Normalize types
        try:
            if "amount" in cleaned and cleaned["amount"] is not None:
                cleaned["amount"] = float(cleaned["amount"])
        except (TypeError, ValueError):
            errors.append(f"amount is not a number: {cleaned.get('amount')!r}")
            cleaned["amount"] = None
        # Required-field check
        for f in self.REQUIRED_FIELDS:
            v = cleaned.get(f)
            if v is None or (isinstance(v, str) and not v.strip()):
                errors.append(f"missing or empty required field: {f}")
        # Date format check
        dol = cleaned.get("date_of_loss")
        if dol and not _is_iso_date(str(dol)):
            errors.append(f"date_of_loss not in YYYY-MM-DD format: {dol!r}")
        # Normalize documents list
        if "documents" not in cleaned or cleaned["documents"] is None:
            cleaned["documents"] = []
        elif not isinstance(cleaned["documents"], list):
            errors.append("documents must be a list")
            cleaned["documents"] = []
        return cleaned, errors


# ===========================================================================
# ValidatorAgent — owns VALIDATING
# ===========================================================================
class ValidatorAgent:
    """Cross-references the claim against the policy DB and 3 additional
    data silos (billing, underwriting, document management).

    The ValidatorAgent owns the ``VALIDATING`` state. It calls each of
    the four silos, aggregates their findings into ``context['silo_records']``
    and ``context['discrepancies']``, and prepares the inputs that the
    ZKP Policy Validity Prover will need (policy_commitment inputs:
    policy_id, salt, coverage_limit, deductible, peril_type, etc.).

    The state machine's ``VALIDATING → ZKP_POLICY_PROOF`` guard checks
    ``context['discrepancies']`` — if non-empty, the guard fails and the
    claim is routed back to ``CLAIM_RECEIVED`` (default failure_route)
    or to ``ESCALATING`` (if ``context['failure_route'] == 'ESCALATING'``).
    """

    def __init__(self, silo_store: Any) -> None:
        """``silo_store`` is an :class:`InMemorySiloStore` (or any object
        exposing ``validate(claim) -> list[SiloRecord]``)."""
        self.silo_store = silo_store

    def run(self, claim: dict[str, Any],
            context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        ctx = dict(context or {})
        claim = ctx.get("claim") or claim
        with TRACER.start_as_current_span(
            name="agent.ValidatorAgent.run",
            input={"claim_id": claim.get("claim_id")},
            metadata={"agent_id": "ValidatorAgent",
                       "claim_id": claim.get("claim_id")},
        ) as span:
            records = self.silo_store.validate(claim)
            discrepancies = [
                {"silo": r.silo_name,
                 "code": r.discrepancy_code,
                 "message": r.discrepancy}
                for r in records if r.discrepancy
            ]
            ctx["silo_records"] = [r.to_dict() for r in records]
            ctx["discrepancies"] = discrepancies
            ctx["validation_complete"] = (len(discrepancies) == 0)
            ctx["validation_timestamp"] = time.time()

            # Prepare inputs for the ZKP Policy Validity Prover.
            # We pull these from the policy_administration silo record
            # (the first one in the list).
            policy_record = records[0].record if records and records[0].found else {}
            ctx["zkp_policy_inputs"] = self._prepare_zkp_inputs(claim, policy_record)
            span.update(output={
                "discrepancies_count": len(discrepancies),
                "validation_complete": ctx["validation_complete"],
                "silo_codes": [r.discrepancy_code for r in records
                                if r.discrepancy_code],
            })
        return ctx

    # ------------------------------------------------------------------ #
    def _prepare_zkp_inputs(
        self, claim: dict[str, Any], policy: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the inputs that the ZKP Policy Validity Prover needs."""
        return {
            "claim_id": claim.get("claim_id"),
            "policy_id": policy.get("policy_id") or claim.get("policy_id"),
            "claim_amount": float(claim.get("amount", 0) or 0),
            "coverage_limit": float(policy.get("limit", 0) or 0),
            "deductible": float(policy.get("deductible", 0) or 0),
            "effective_date": policy.get("effective_date", ""),
            "expiration_date": policy.get("expiration_date", ""),
            "date_of_loss": claim.get("date_of_loss", ""),
            "policy_active": policy.get("status") == "active",
            "peril_type": _infer_peril_type(claim, policy),
            "perils_covered": policy.get("perils_covered", []),
            "perils_excluded": policy.get("perils_excluded", []),
            "jurisdiction": policy.get("jurisdiction", "CA"),
        }


# ===========================================================================
# ClassifierAgent — owns CLASSIFYING
# ===========================================================================
# Severity thresholds (configurable via env vars or constructor kwargs)
DEFAULT_SEVERITY_THRESHOLDS: dict[str, float] = {
    "low":    0.0,        # amount <= 1_000
    "medium": 1_000.0,    # 1_000 < amount <= 10_000
    "high":   10_000.0,   # amount > 10_000
}

# Fraud score thresholds
DEFAULT_FRAUD_RISK_THRESHOLDS: dict[str, float] = {
    "low":    0.30,   # < 0.30 = low risk
    "medium": 0.60,   # 0.30 - 0.60 = medium
    "high":   0.60,   # > 0.60 = high (route to ESCALATING)
}

# Confidence threshold for "is this classification good enough?"
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.85

# Ambiguity window — if fraud score is within ±0.05 of a threshold, mark
# ambiguous so the guard routes to ESCALATING.
DEFAULT_AMBIGUITY_WINDOW: float = 0.05


class ClassifierAgent:
    """Classifies each claim along three dimensions:

    1. **Severity** — low / medium / high based on amount and damage type.
    2. **Claim type** — property_damage, auto, liability, medical, etc.
    3. **Fraud risk score** — float in [0.0, 1.0].

    The fraud risk score is computed by a hybrid approach:

    - **Statistical baseline** — features from claim history (prior
      claims count, time since last claim, amount relative to policy
      limit) are scored against historical patterns.
    - **LLM assessment** — Qwen3.6 receives the claim data, claimant
      history, and statistical patterns and emits a fraud risk score
      with explicit reasoning. The LLM output is parsed as JSON and
      cross-checked against the statistical baseline.
    - **Anomaly detection** — flags claims where amount > 2× the
      claimant's historical average, or where the peril is unusual for
      the jurisdiction.

    Calibration target: <= 8% false positive rate (vs. 70% historically).
    The calibration is achieved by:

    - Setting the high-risk threshold at 0.60 (not 0.30 as in the
      legacy system), so only clearly anomalous claims are flagged.
    - Requiring the LLM's confidence in the fraud classification to be
      >= 0.85; otherwise the claim is routed to ESCALATING rather than
      auto-classified as fraud.
    - Using the LLM's reasoning as the primary signal — the statistical
      baseline only acts as a tiebreaker for borderline cases.
    """

    SYSTEM_PROMPT = """\
You are the ClassifierAgent in the ShieldPoint claims automation system.

Your job: classify the claim along three dimensions and return a JSON object.

Dimensions:
1. severity: "low" | "medium" | "high"
   - low:    claim amount <= $1,000 OR minor cosmetic damage
   - medium: $1,000 < amount <= $10,000 OR moderate structural damage
   - high:   amount > $10,000 OR total loss / severe structural damage

2. claim_type: one of:
   - "property_damage", "auto", "liability", "medical", "water_damage",
     "theft", "vandalism", "fire", "wind", "hail"

3. fraud_risk_score: float in [0.0, 1.0]
   - 0.0 = clearly legitimate
   - 0.3 = some risk indicators but plausibly legitimate
   - 0.6 = significant fraud indicators (multiple prior claims, amount
           unusually high for the claimant's history, etc.)
   - 1.0 = clear fraud (impossible scenario, contradictory evidence)

Fraud indicators to weight (in order of importance):
- Material misrepresentation in underwriting file (weight: 0.4)
- 3+ prior claims in the last 12 months (weight: 0.2)
- Claim amount > 2x claimant's historical average (weight: 0.15)
- Recent policy inception (< 30 days before loss) (weight: 0.1)
- Injury claim with no prior medical records (weight: 0.1)
- Inconsistent statements in claim description (weight: 0.05)

CALIBRATION TARGET: false positive rate <= 8% (vs. 70% historically).
This means: only flag a claim as high-risk (fraud_risk_score > 0.6) if
you are CONFIDENT. When in doubt, score 0.4-0.5 (medium) and let the
human adjuster review via ESCALATING — do NOT auto-flag borderline cases.

Return JSON of this exact shape:
{
  "severity": "low" | "medium" | "high",
  "claim_type": "<one of the values above>",
  "fraud_risk_score": <float in [0.0, 1.0]>,
  "confidence": <float in [0.0, 1.0]>,
  "reasoning": "<2-3 sentences explaining your assessment>",
  "fraud_indicators": ["<indicator 1>", "<indicator 2>", ...],
  "ambiguous": true | false,
  "ambiguity_reason": "<only if ambiguous=true>"
}

If you cannot confidently classify, set "ambiguous": true and explain
in "ambiguity_reason" what information is missing or contradictory.
"""

    def __init__(
        self,
        *,
        llm_client: Optional[LLMClient] = None,
        model: str = "qwen3.6-35b-a3b",
        severity_thresholds: Optional[dict[str, float]] = None,
        fraud_risk_thresholds: Optional[dict[str, float]] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        ambiguity_window: float = DEFAULT_AMBIGUITY_WINDOW,
        fraud_detection_client: Optional[Any] = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.severity_thresholds = severity_thresholds or DEFAULT_SEVERITY_THRESHOLDS
        self.fraud_risk_thresholds = fraud_risk_thresholds or DEFAULT_FRAUD_RISK_THRESHOLDS
        self.confidence_threshold = confidence_threshold
        self.ambiguity_window = ambiguity_window
        # SP-503: Optional fraud detection client for cross-party duplicate checking
        self.fraud_detection_client = fraud_detection_client

    def run(self, claim: dict[str, Any],
            context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        ctx = dict(context or {})
        claim = ctx.get("claim") or claim
        with TRACER.start_as_current_span(
            name="agent.ClassifierAgent.run",
            input={"claim_id": claim.get("claim_id")},
            metadata={"agent_id": "ClassifierAgent",
                       "claim_id": claim.get("claim_id")},
        ) as span:
            # ---- SP-503: Cross-party fraud detection check ----
            # Query the fraud detection network during the CLASSIFYING state.
            # If the claim's commitment already exists in the shared Merkle
            # tree (i.e., filed with another insurer), flag it as a duplicate
            # and route to ESCALATING with a fraud flag.
            if self.fraud_detection_client is not None:
                try:
                    fraud_result = self._check_fraud_detection(claim, ctx)
                    ctx["fraud_detection_result"] = fraud_result
                    if not fraud_result.get("is_unique", True):
                        # Duplicate detected — set the fraud flag
                        ctx["fraud_flag"] = True
                        ctx["fraud_flag_reason"] = (
                            f"Cross-party duplicate detected: commitment "
                            f"{fraud_result.get('commitment_value')} found in "
                            f"shared Merkle tree. Previously filed by insurer="
                            f"{fraud_result.get('duplicate_insurer')}."
                        )
                        ctx["escalation_reason"] = (
                            "Cross-party fraud detection: duplicate claim "
                            "filed with another insurer."
                        )
                        # The compliance guard will fail on this claim and
                        # route to ESCALATING via the standard flow.
                        ctx["ambiguous"] = True
                        ctx["ambiguity_reason"] = ctx["fraud_flag_reason"]
                except Exception as e:
                    logger.warning("Fraud detection check failed: %s", e)
                    ctx["fraud_detection_error"] = str(e)

            # Build the user prompt with claim + history + stats
            user_prompt = self._build_prompt(claim, ctx)
            # Statistical baseline (always computed — used as a sanity check
            # on the LLM output)
            stat_baseline = self._statistical_baseline(claim, ctx)
            ctx["statistical_fraud_baseline"] = stat_baseline

            # LLM call (or fallback to statistical baseline if no client)
            if self.llm_client is not None:
                try:
                    raw = self.llm_client.chat_completion(
                        model=self.model,
                        system=self.SYSTEM_PROMPT,
                        user=user_prompt,
                        temperature=0.1,
                        max_tokens=512,
                    )
                    llm_result = self._parse_llm_output(raw)
                except Exception as e:
                    logger.warning("LLM call failed: %s; falling back to stats", e)
                    llm_result = self._fallback_classification(claim, stat_baseline)
            else:
                llm_result = self._fallback_classification(claim, stat_baseline)

            # Merge LLM result with statistical baseline
            merged = self._merge_with_baseline(llm_result, stat_baseline)
            # Apply ambiguity detection
            merged = self._detect_ambiguity(merged, claim, ctx)

            ctx["severity"] = merged["severity"]
            ctx["claim_type"] = merged["claim_type"]
            ctx["fraud_risk_score"] = merged["fraud_risk_score"]
            ctx["classification_confidence"] = merged["confidence"]
            ctx["classification_reasoning"] = merged["reasoning"]
            ctx["fraud_indicators"] = merged.get("fraud_indicators", [])
            # Preserve the fraud flag from the cross-party check (don't let
            # the LLM's ambiguity detection override it)
            fraud_flag_was_set = ctx.get("fraud_flag", False)
            ctx["ambiguous"] = merged.get("ambiguous", False) or fraud_flag_was_set
            if fraud_flag_was_set and not merged.get("ambiguity_reason"):
                ctx["ambiguity_reason"] = ctx.get("fraud_flag_reason")
            else:
                ctx["ambiguity_reason"] = merged.get("ambiguity_reason")
            ctx["risk_class"] = self._risk_class(merged["fraud_risk_score"])
            ctx["classification_complete"] = True
            ctx["classification_timestamp"] = time.time()

            span.update(output={
                "severity": ctx["severity"],
                "claim_type": ctx["claim_type"],
                "fraud_risk_score": ctx["fraud_risk_score"],
                "risk_class": ctx["risk_class"],
                "ambiguous": ctx["ambiguous"],
                "fraud_flag": ctx.get("fraud_flag", False),
                "reasoning": ctx["classification_reasoning"],
            })
        return ctx

    def _check_fraud_detection(
        self, claim: dict[str, Any], ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """Query the cross-party fraud detection network (SP-503).

        Generates a commitment for the claim and checks it against the
        shared Merkle tree maintained by the coordination layer. If the
        commitment already exists (filed by another insurer), the claim
        is flagged as a potential duplicate.
        """
        if self.fraud_detection_client is None:
            return {"is_unique": True, "skipped": True}

        # Extract fields needed for the commitment
        claimant_id = self._compute_claimant_id(claim, ctx)
        date_of_loss = claim.get("date_of_loss", "")
        location = claim.get("incident_location", claim.get("location", ""))
        peril_type = self._infer_peril_code(claim)
        amount = float(claim.get("amount", 0) or 0)

        result = self.fraud_detection_client.check_claim_uniqueness(
            claim_id=claim.get("claim_id", ""),
            claimant_id=claimant_id,
            date_of_loss=date_of_loss,
            location=location,
            peril_type=peril_type,
            amount=amount,
        )
        # Convert the dataclass to a dict for context storage
        return {
            "is_unique": result.is_unique,
            "commitment_value": result.commitment_value,
            "merkle_root": result.merkle_root,
            "duplicate_insurer": result.duplicate_insurer,
            "checked_at": result.checked_at,
            "proof": {
                "verified": result.proof.verified,
                "proof_type": result.proof.proof_type,
                "statement": result.proof.statement,
                "latency_ms": result.proof.latency_ms,
            },
        }

    def _compute_claimant_id(
        self, claim: dict[str, Any], ctx: dict[str, Any]
    ) -> int:
        """Compute a numeric claimant ID for the commitment.

        In production, this is a Poseidon hash of the claimant's SSN +
        DOB (or a database PK). In tests, we derive a deterministic
        integer from the claimant's name and policy_id.
        """
        claimant = claim.get("claimant", "") or ""
        policy_id = claim.get("policy_id", "") or ""
        import hashlib as _hashlib
        h = _hashlib.sha256(f"{claimant}|{policy_id}".encode()).digest()
        return int.from_bytes(h[:16], "big")

    def _infer_peril_code(self, claim: dict[str, Any]) -> int:
        """Infer the numeric peril code from the claim."""
        text = (
            str(claim.get("claim_type", "")) + " " +
            str(claim.get("description", ""))
        ).lower()
        for peril_name, code in _PERIL_TYPE_MAP.items():
            if peril_name in text:
                return code
        return 1  # default to wind

    # ------------------------------------------------------------------ #
    def _build_prompt(self, claim: dict[str, Any],
                      ctx: dict[str, Any]) -> str:
        """Build the user prompt for the LLM."""
        history = ctx.get("claimant_history", {}) or {}
        stat = self._statistical_baseline(claim, ctx)
        return json.dumps({
            "claim": {
                "claim_id": claim.get("claim_id"),
                "policy_id": claim.get("policy_id"),
                "claimant": claim.get("claimant"),
                "amount": claim.get("amount"),
                "date_of_loss": claim.get("date_of_loss"),
                "description": claim.get("description"),
                "claim_type_hint": claim.get("claim_type"),
            },
            "claimant_history": {
                "prior_claims_count": history.get("prior_claims_count", 0),
                "avg_prior_claim_amount": history.get("avg_prior_claim_amount", 0),
                "days_since_last_claim": history.get("days_since_last_claim"),
                "policy_inception_days_ago": history.get(
                    "policy_inception_days_ago"),
            },
            "statistical_baseline": stat,
            "instructions": (
                "Use the statistical baseline as a sanity check. If your "
                "assessment differs from the baseline by more than 0.2 in "
                "fraud_risk_score, explain why in the reasoning field."
            ),
        }, indent=2)

    def _statistical_baseline(self, claim: dict[str, Any],
                              ctx: dict[str, Any]) -> dict[str, Any]:
        """Compute a deterministic statistical fraud-risk baseline.

        Combines:
        - Material misrepresentation flag (from underwriting silo, if
          present in context).
        - Prior-claims count vs. historical norm.
        - Amount vs. claimant's historical average.
        - Days since policy inception.
        """
        score = 0.0
        indicators: list[str] = []
        history = ctx.get("claimant_history", {}) or {}

        # Misrepresentation (highest weight)
        silo_records = ctx.get("silo_records", []) or []
        for r in silo_records:
            if r.get("discrepancy_code") == "material_misrepresentation":
                score += 0.4
                indicators.append("material_misrepresentation")

        # Prior claims frequency
        prior = history.get("prior_claims_count", 0) or 0
        if prior >= 3:
            score += 0.2
            indicators.append(f"high_prior_claims_count={prior}")
        elif prior >= 2:
            score += 0.1
            indicators.append(f"moderate_prior_claims_count={prior}")

        # Amount vs. historical average
        avg = float(history.get("avg_prior_claim_amount", 0) or 0)
        amount = float(claim.get("amount", 0) or 0)
        if avg > 0 and amount > 2 * avg:
            score += 0.15
            indicators.append(f"amount_2x_historical_avg ({amount} > 2*{avg})")
        elif avg > 0 and amount > 1.5 * avg:
            score += 0.05
            indicators.append("amount_1.5x_historical_avg")

        # Recent policy inception
        days = history.get("policy_inception_days_ago")
        if days is not None and 0 <= days < 30:
            score += 0.1
            indicators.append(f"recent_policy_inception ({days}d)")

        # Claim amount > 50% of policy limit (suspicious for first claim)
        silo_records = ctx.get("silo_records", []) or []
        policy_rec = next((r for r in silo_records
                           if r.get("silo_name") == "policy_administration"
                           and r.get("found")), None)
        if policy_rec and amount > 0:
            limit = float(policy_rec.get("record", {}).get("limit", 0) or 0)
            if limit > 0 and amount > 0.5 * limit:
                score += 0.05
                indicators.append("amount_over_50pct_of_limit")

        # Clamp
        score = min(score, 1.0)
        return {
            "fraud_risk_score": round(score, 3),
            "indicators": indicators,
            "severity_hint": self._severity_from_amount(amount),
        }

    def _severity_from_amount(self, amount: float) -> str:
        if amount <= self.severity_thresholds["medium"]:
            return "low"
        if amount <= self.severity_thresholds["high"]:
            return "medium"
        return "high"

    def _parse_llm_output(self, raw: str) -> dict[str, Any]:
        """Parse the LLM's JSON output, tolerating markdown fences."""
        s = raw.strip()
        # Strip markdown ``` fences
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
            s = re.sub(r"\n?```$", "", s)
            s = s.strip()
        # Find first { ... } block
        if not s.startswith("{"):
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                s = m.group(0)
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM output as JSON: %s\nraw=%r", e, raw)
            return self._fallback_classification({}, {"fraud_risk_score": 0.3,
                                                       "indicators": [],
                                                       "severity_hint": "low"})
        # Validate
        try:
            score = float(obj.get("fraud_risk_score", 0.0))
            score = max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            score = 0.3
        severity = str(obj.get("severity", "low")).lower()
        if severity not in {"low", "medium", "high"}:
            severity = "low"
        return {
            "severity": severity,
            "claim_type": str(obj.get("claim_type", "property_damage")).lower(),
            "fraud_risk_score": score,
            "confidence": float(obj.get("confidence", 0.5)),
            "reasoning": str(obj.get("reasoning", "")),
            "fraud_indicators": list(obj.get("fraud_indicators", []) or []),
            "ambiguous": bool(obj.get("ambiguous", False)),
            "ambiguity_reason": obj.get("ambiguity_reason"),
        }

    def _fallback_classification(self, claim: dict[str, Any],
                                  stat: dict[str, Any]) -> dict[str, Any]:
        """When no LLM client is available (or LLM fails), fall back to
        the statistical baseline. This is also the default path used in
        unit tests."""
        amount = float(claim.get("amount", 0) or 0)
        return {
            "severity": stat.get("severity_hint") or self._severity_from_amount(amount),
            "claim_type": (claim.get("claim_type") or "property_damage").lower(),
            "fraud_risk_score": float(stat.get("fraud_risk_score", 0.1)),
            "confidence": 0.7,  # statistical only — slightly below LLM threshold
            "reasoning": (
                "Statistical-baseline classification (LLM unavailable). "
                f"Indicators: {stat.get('indicators', [])}"
            ),
            "fraud_indicators": stat.get("indicators", []),
            "ambiguous": False,
            "ambiguity_reason": None,
        }

    def _merge_with_baseline(self, llm_result: dict[str, Any],
                              stat: dict[str, Any]) -> dict[str, Any]:
        """If the LLM and statistical baseline diverge by > 0.2 in fraud
        score, mark the result as ambiguous so it routes to ESCALATING."""
        merged = dict(llm_result)
        llm_score = float(llm_result.get("fraud_risk_score", 0.0))
        stat_score = float(stat.get("fraud_risk_score", 0.0))
        if abs(llm_score - stat_score) > 0.2:
            merged["ambiguous"] = True
            merged["ambiguity_reason"] = (
                f"LLM score {llm_score:.2f} diverges from statistical "
                f"baseline {stat_score:.2f} by >0.20."
            )
        # Always include the statistical indicators in the merged output
        merged.setdefault("fraud_indicators", [])
        merged["fraud_indicators"] = list(merged["fraud_indicators"]) + [
            i for i in stat.get("indicators", []) if i not in merged["fraud_indicators"]
        ]
        return merged

    def _detect_ambiguity(self, merged: dict[str, Any],
                           claim: dict[str, Any],
                           ctx: dict[str, Any]) -> dict[str, Any]:
        """Add explicit ambiguity detection for borderline cases."""
        if merged.get("ambiguous"):
            return merged
        score = float(merged.get("fraud_risk_score", 0.0))
        # Within ±window of the high-risk threshold
        high_thresh = self.fraud_risk_thresholds["high"]
        if abs(score - high_thresh) <= self.ambiguity_window:
            merged["ambiguous"] = True
            merged["ambiguity_reason"] = (
                f"Fraud score {score:.2f} within ±{self.ambiguity_window} "
                f"of high-risk threshold {high_thresh}."
            )
        # Confidence below threshold
        conf = float(merged.get("confidence", 0.0))
        if conf < self.confidence_threshold and not merged.get("ambiguous"):
            merged["ambiguous"] = True
            merged["ambiguity_reason"] = (
                f"Classification confidence {conf:.2f} below "
                f"{self.confidence_threshold} threshold."
            )
        return merged

    def _risk_class(self, score: float) -> str:
        if score < self.fraud_risk_thresholds["low"]:
            return "low"
        if score <= self.fraud_risk_thresholds["high"]:
            return "medium"
        return "high"


# ===========================================================================
# EscalationAgent — owns ESCALATING (HITL)
# ===========================================================================
@dataclass
class AdjusterDecision:
    """One human adjuster decision on an escalated claim."""
    claim_id: str
    adjuster_id: str
    action: str  # approve | deny | request_more_info | reclassify
    rationale: str
    timestamp: float = field(default_factory=time.time)
    new_classification: Optional[dict[str, Any]] = None
    extra_info_requested: Optional[list[str]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "adjuster_id": self.adjuster_id,
            "action": self.action,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "new_classification": self.new_classification,
            "extra_info_requested": self.extra_info_requested,
        }


class EscalationAgent:
    """Owns the ``ESCALATING`` state and integrates the
    Human-in-the-Loop (HITL) workflow.

    When a claim is routed to ``ESCALATING`` (due to ZKP proof failure,
    high fraud risk score, classification ambiguity, or compliance
    uncertainty), the EscalationAgent:

    1. Generates a structured case summary for the human adjuster,
       including the automated analysis, escalation reason, ZKP proof
       details (if applicable), and recommended next steps.
    2. Queues the case in the adjuster workbench (in-memory queue in
       tests; production wires this to the adjuster web UI).
    3. Accepts the adjuster's decision (approve / deny / request-more-info
       / reclassify) and logs it to Langfuse + the state log.
    4. Translates the decision into a context dict that the state
       machine's ``ESCALATING → APPROVED`` guard will accept (for
       "approve") or that denies the claim (for "deny").
    """

    def __init__(self) -> None:
        # In-memory queue of pending escalations (claim_id -> summary).
        # Production replaces this with a row in the ``escalation_queue``
        # table polled by the adjuster web UI.
        self._queue: dict[str, dict[str, Any]] = {}
        # In-memory decision log (claim_id -> list of decisions, newest last)
        self._decisions: dict[str, list[AdjusterDecision]] = {}

    def run(self, claim: dict[str, Any],
            context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Generate the case summary and queue the claim for review."""
        ctx = dict(context or {})
        claim = ctx.get("claim") or claim
        with TRACER.start_as_current_span(
            name="agent.EscalationAgent.run",
            input={"claim_id": claim.get("claim_id")},
            metadata={"agent_id": "EscalationAgent",
                       "claim_id": claim.get("claim_id")},
        ) as span:
            summary = self._build_case_summary(claim, ctx)
            ctx["case_summary"] = summary
            ctx["escalation_queued_at"] = time.time()
            ctx["escalation_status"] = "pending_review"
            self._queue[claim.get("claim_id", "")] = summary
            span.update(output={
                "escalation_reason": summary["escalation_reason"],
                "queued_at": ctx["escalation_queued_at"],
            })
        return ctx

    # ------------------------------------------------------------------ #
    def get_pending_cases(self) -> list[dict[str, Any]]:
        """Return all pending escalation cases (for the adjuster UI)."""
        return list(self._queue.values())

    def get_case(self, claim_id: str) -> Optional[dict[str, Any]]:
        return self._queue.get(claim_id)

    def submit_decision(self, decision: AdjusterDecision) -> dict[str, Any]:
        """Record a human adjuster's decision.

        Returns an updated context dict that the orchestrator can feed
        to the state machine's next transition. If the action is
        ``approve``, the context sets ``human_approval=True`` and
        ``adjuster_id`` so the ``ESCALATING → APPROVED`` guard passes.
        If the action is ``deny``, the context sets ``denied=True``
        with the rationale. If ``request_more_info``, the context
        sets ``paused_for_more_info=True`` with the list of requested
        items. If ``reclassify``, the context updates the
        classification fields with the adjuster's new values.
        """
        with TRACER.start_as_current_span(
            name="agent.EscalationAgent.submit_decision",
            input={"claim_id": decision.claim_id,
                    "action": decision.action},
            metadata={
                "agent_id": "EscalationAgent",
                "claim_id": decision.claim_id,
                "adjuster_id": decision.adjuster_id,
                "action": decision.action,
                "rationale": decision.rationale,
            },
        ) as span:
            # Persist decision
            self._decisions.setdefault(decision.claim_id, []).append(decision)
            # Remove from pending queue
            self._queue.pop(decision.claim_id, None)

            ctx: dict[str, Any] = {
                "adjuster_id": decision.adjuster_id,
                "adjuster_rationale": decision.rationale,
                "adjuster_action": decision.action,
                "adjuster_decision_timestamp": decision.timestamp,
            }
            if decision.action == "approve":
                ctx["human_approval"] = True
            elif decision.action == "deny":
                ctx["human_approval"] = False
                ctx["denied"] = True
                ctx["denial_reason"] = decision.rationale
            elif decision.action == "request_more_info":
                ctx["human_approval"] = False
                ctx["paused_for_more_info"] = True
                ctx["extra_info_requested"] = decision.extra_info_requested or []
            elif decision.action == "reclassify":
                ctx["human_approval"] = True  # reclassify = approve with new class
                if decision.new_classification:
                    ctx.update(decision.new_classification)
            else:
                raise ValueError(f"Unknown adjuster action: {decision.action!r}")
            span.update(output={"human_approval": ctx.get("human_approval", False),
                                "action": decision.action})
        return ctx

    def get_decisions(self, claim_id: str) -> list[AdjusterDecision]:
        return list(self._decisions.get(claim_id, []))

    # ------------------------------------------------------------------ #
    def _build_case_summary(self, claim: dict[str, Any],
                             ctx: dict[str, Any]) -> dict[str, Any]:
        """Build the structured case summary for the human adjuster.

        SP-503: When the escalation is due to a cross-party fraud flag,
        the case summary includes a specialised fraud investigation
        section with the ZKP non-membership proof failure details.
        """
        escalation_reason = (
            ctx.get("escalation_reason")
            or self._infer_escalation_reason(ctx)
        )
        summary = {
            "claim_id": claim.get("claim_id"),
            "policy_id": claim.get("policy_id"),
            "claimant": claim.get("claimant"),
            "amount": claim.get("amount"),
            "date_of_loss": claim.get("date_of_loss"),
            "description": claim.get("description"),
            "escalation_reason": escalation_reason,
            "automated_analysis": {
                "severity": ctx.get("severity"),
                "claim_type": ctx.get("claim_type"),
                "fraud_risk_score": ctx.get("fraud_risk_score"),
                "risk_class": ctx.get("risk_class"),
                "classification_reasoning": ctx.get("classification_reasoning"),
                "classification_confidence": ctx.get("classification_confidence"),
            },
            "zkp_proof_details": {
                "policy_proof_verified": ctx.get("policy_proof_verified"),
                "policy_proof_statement": ctx.get("policy_proof_statement"),
                "compliance_proof_verified": ctx.get("compliance_proof_verified"),
                "compliance_proof_statement": ctx.get("compliance_proof_statement"),
                "compliance_jurisdiction": ctx.get("compliance_jurisdiction"),
            },
            "validator_findings": {
                "discrepancies": ctx.get("discrepancies", []),
                "silo_records": ctx.get("silo_records", []),
            },
            "recommended_next_steps": self._recommend_next_steps(escalation_reason, ctx),
            "available_actions": [
                "approve", "deny", "request_more_info", "reclassify",
            ],
            "queued_at": time.time(),
        }

        # ---- SP-503: Add fraud investigation section if fraud-flagged ----
        if ctx.get("fraud_flag") or ctx.get("fraud_detection_result"):
            fd = ctx.get("fraud_detection_result") or {}
            summary["fraud_investigation"] = {
                "fraud_flag": ctx.get("fraud_flag", False),
                "fraud_flag_reason": ctx.get("fraud_flag_reason", ""),
                "is_duplicate": not fd.get("is_unique", True),
                "duplicate_insurer": fd.get("duplicate_insurer"),
                "commitment_value": fd.get("commitment_value"),
                "merkle_root": fd.get("merkle_root"),
                "zkp_proof": (fd.get("proof") or {}),
                "investigation_steps": [
                    "Verify the claimant's identity and confirm whether they "
                    "filed a claim with another insurer for the same incident.",
                    "Request documentation from the claimant proving the loss "
                    "occurred (police report, photos, repair estimates).",
                    "Contact the other insurer (if known) to share investigation "
                    "findings — only the commitment hash, never raw claim data.",
                    "If duplicate filing is confirmed: deny the claim and refer "
                    "to the SIU (Special Investigations Unit).",
                    "If the match was coincidental (same commitment by chance): "
                    "reclassify the claim and proceed through the normal pipeline.",
                ],
                "recommended_action_if_duplicate": "deny",
                "recommended_action_if_coincidental": "reclassify",
            }

        return summary

    def _infer_escalation_reason(self, ctx: dict[str, Any]) -> str:
        """If the orchestrator didn't set an explicit escalation reason,
        infer one from the context."""
        # SP-503: Check fraud flag first (highest priority)
        if ctx.get("fraud_flag"):
            return (
                f"Cross-party fraud detection: {ctx.get('fraud_flag_reason', 'duplicate claim detected')}"
            )
        if ctx.get("compliance_proved") is False:
            return "ZKP compliance proof failed verification."
        if ctx.get("proof_verified") is False:
            return "ZKP policy proof failed verification."
        if ctx.get("ambiguous"):
            return (
                f"Classification ambiguous: {ctx.get('ambiguity_reason')}"
            )
        if ctx.get("risk_class") == "high":
            return (
                f"High fraud risk score: {ctx.get('fraud_risk_score'):.2f}"
            )
        if ctx.get("discrepancies"):
            return (
                f"Data silo discrepancies: "
                f"{[d.get('code') for d in ctx['discrepancies']]}"
            )
        if ctx.get("classification_confidence", 1.0) < self._conf_threshold(ctx):
            return (
                f"Classification confidence {ctx.get('classification_confidence'):.2f} "
                "below threshold."
            )
        return "Escalated by guard condition (unspecified reason)."

    def _conf_threshold(self, ctx: dict[str, Any]) -> float:
        return float(ctx.get("confidence_threshold",
                              DEFAULT_CONFIDENCE_THRESHOLD))

    def _recommend_next_steps(self, reason: str,
                              ctx: dict[str, Any]) -> list[str]:
        """Heuristic recommendations based on the escalation reason."""
        steps: list[str] = []
        if "compliance" in reason.lower():
            steps.append("Review the compliance proof failure details and "
                         "determine if the claim can be brought into "
                         "compliance with additional documentation.")
        if "policy proof" in reason.lower():
            steps.append("Verify the policy is in force and the peril is "
                         "covered. If the policy is lapsed, deny the claim.")
        if "ambiguous" in reason.lower():
            steps.append("Request additional information from the claimant "
                         "to resolve the classification ambiguity.")
        if "fraud" in reason.lower() or "cross-party" in reason.lower() or "duplicate" in reason.lower():
            # SP-503: Cross-party fraud investigation steps
            steps.append("Review the cross-party fraud detection ZKP proof "
                         "failure details — the claim's commitment matches "
                         "an existing entry in the shared Merkle tree.")
            steps.append("Contact the claimant to confirm whether they filed "
                         "a claim with another insurer for the same incident.")
            steps.append("Request supporting documentation (police report, "
                         "photos, repair estimates) to verify the loss.")
            steps.append("If duplicate filing is CONFIRMED: deny the claim "
                         "and refer to the SIU (Special Investigations Unit).")
            steps.append("If the match was COINCIDENTAL (same commitment by "
                         "chance): reclassify the claim and proceed through "
                         "the normal pipeline.")
        elif "fraud" in reason.lower():
            steps.append("Review the claimant's prior claim history and "
                         "the underwriting file for material misrepresentation.")
            steps.append("If fraud is confirmed, deny and refer to SIU.")
        if ctx.get("discrepancies"):
            steps.append("Resolve data silo discrepancies: "
                         + ", ".join(d.get("code", "") for d in ctx["discrepancies"]))
        if not steps:
            steps.append("Review the case and decide: approve, deny, "
                         "request more info, or reclassify.")
        return steps


# ===========================================================================
# PayoutAgent — owns APPROVED → PAID_OUT
# ===========================================================================
class PayoutAgent:
    """Owns the ``APPROVED → PAID_OUT`` transition (SP-405).

    The PayoutAgent executes the final payment for approved claims:
    1. **Duplicate payment detection** — queries the payment ledger for
       any existing payment with the same claim_id. If found, the payment
       is skipped and the existing record is returned (idempotency).
    2. **Bank details verification** — calls the bank verification service
       to confirm the payee's bank account and routing number are valid.
     3. **ACH payment initiation** — calls the ACH provider to initiate
       the electronic funds transfer.
    4. **Payment record creation** — inserts an immutable payment record
       into the ledger with the full breakdown (gross, deductible, co-pay,
       net).
    5. **PDF receipt generation** — generates a professional PDF receipt
       with the full payment breakdown and audit trail.
    6. **Claimant notification** — sends an email to the claimant with
       the payment confirmation and receipt attachment.
    7. **Audit record assembly** — assembles the complete audit record
       (all agent traces + ZKP proof references + payment breakdown) and
       stores it in Langfuse + PostgreSQL.

    Guard condition for APPROVED → PAID_OUT:
    - Payment authorization must be present (``payment_authorized=True``)
    - Bank details must be verified (``bank_details_verified=True``)

    Backward compatibility
    ----------------------
    The agent accepts the legacy ``payment_ledger`` parameter (a dict-based
    ledger) for existing tests. When the new payout subsystem is available
    and no explicit components are provided, default stub instances are
    created (suitable for tests and local development).
    """

    def __init__(
        self,
        payment_ledger: Optional[Any] = None,
        *,
        ach_provider: Optional[Any] = None,
        bank_verification: Optional[Any] = None,
        receipt_generator: Optional[Any] = None,
        notification_service: Optional[Any] = None,
        audit_assembler: Optional[Any] = None,
    ) -> None:
        # Use the enhanced ledger if available; fall back to the legacy
        # in-memory ledger for backward compatibility with existing tests.
        if payment_ledger is not None:
            self.payment_ledger = payment_ledger
        elif _PAYOUT_AVAILABLE:
            self.payment_ledger = InMemoryPaymentLedger()
        else:
            self.payment_ledger = _InMemoryLedger()

        # Inject the new subsystem components (or create defaults)
        self.ach_provider = ach_provider or (
            StubACHProvider() if _PAYOUT_AVAILABLE else None
        )
        self.bank_verification = bank_verification or (
            BankVerificationService() if _PAYOUT_AVAILABLE else None
        )
        self.receipt_generator = receipt_generator or (
            ReceiptGenerator() if _PAYOUT_AVAILABLE else None
        )
        self.notification_service = notification_service or (
            StubNotificationService() if _PAYOUT_AVAILABLE else None
        )
        self.audit_assembler = audit_assembler or (
            AuditRecordAssembler() if _PAYOUT_AVAILABLE else None
        )

    def run(self, claim: dict[str, Any],
            context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        ctx = dict(context or {})
        claim = ctx.get("claim") or claim
        with TRACER.start_as_current_span(
            name="agent.PayoutAgent.run",
            input={"claim_id": claim.get("claim_id")},
            metadata={"agent_id": "PayoutAgent",
                       "claim_id": claim.get("claim_id")},
        ) as span:
            # Idempotency key prevents double-payment on retry
            claim_id = claim.get("claim_id", "")
            idem_key = f"payout-{claim_id}"

            # ---- Step 1: Duplicate payment detection ----
            existing = self._check_duplicate(claim_id, idem_key)
            if existing:
                ctx["payment_authorized"] = True
                ctx["bank_details_verified"] = True
                ctx["payment_record"] = self._record_to_dict(existing)
                ctx["duplicate_payment_prevented"] = True
                ctx["payout_timestamp"] = time.time()
                span.update(output={
                    "payment_authorized": True,
                    "bank_details_verified": True,
                    "duplicate_payment_prevented": True,
                    "ach_reference": (ctx.get("payment_record") or {}).get("ach_reference"),
                })
                return ctx

            # ---- Step 2: Bank details verification ----
            bank_verified = self._verify_bank_details(claim, ctx)

            # ---- Step 3: ACH payment initiation ----
            payment_authorized = False
            ach_result = None
            if bank_verified and self.ach_provider is not None:
                breakdown = self._compute_breakdown(claim, ctx)
                ach_result = self.ach_provider.initiate_payment(
                    amount=breakdown["net"],
                    payee_name=claim.get("claimant", ""),
                    bank_account=claim.get("bank_account", "000123456789"),
                    bank_routing=claim.get("bank_routing", "021000021"),
                    idempotency_key=idem_key,
                )
                payment_authorized = ach_result.success
                ctx["ach_result"] = {
                    "success": ach_result.success,
                    "ach_reference": ach_result.ach_reference,
                    "amount": ach_result.amount,
                    "status": ach_result.status,
                    "settlement_date": ach_result.settlement_date,
                }
            elif bank_verified:
                # Legacy fallback (no ACH provider injected)
                payment_authorized = self._authorize_payment(claim, ctx)

            # ---- Step 4: Create payment record ----
            if payment_authorized and bank_verified:
                breakdown = self._compute_breakdown(claim, ctx)
                ach_ref = (
                    ach_result.ach_reference if ach_result
                    else f"ACH-{uuid.uuid4().hex[:10].upper()}"
                )
                record_dict = {
                    "payment_id": f"PMT-{uuid.uuid4().hex[:12].upper()}",
                    "claim_id": claim_id,
                    "policy_id": claim.get("policy_id"),
                    "payee": claim.get("claimant"),
                    "gross_amount": breakdown["gross"],
                    "deductible_applied": breakdown["deductible"],
                    "copay_amount": breakdown["copay"],
                    "net_payable": breakdown["net"],
                    "amount": breakdown["net"],  # backward compat
                    "ach_reference": ach_ref,
                    "status": "settled" if ach_result is None else ach_result.status,
                    "settlement_date": ach_result.settlement_date if ach_result else None,
                    "idempotency_key": idem_key,
                }
                inserted = self._insert_record(record_dict)
                ctx["payment_record"] = inserted

                # ---- Step 5: Generate PDF receipt ----
                if self.receipt_generator is not None:
                    receipt_result = self.receipt_generator.generate(
                        payment_record=inserted,
                        claim=claim,
                        audit_trail=self._build_audit_trail_summary(ctx),
                        zkp_proofs=self._collect_zkp_proof_refs(ctx),
                    )
                    ctx["receipt"] = {
                        "success": receipt_result.success,
                        "file_path": receipt_result.file_path,
                        "file_format": receipt_result.file_format,
                        "receipt_id": receipt_result.receipt_id,
                    }

                # ---- Step 6: Send claimant notification ----
                if self.notification_service is not None:
                    claimant_email = claim.get("email", "")
                    notif_result = self.notification_service.send_payment_confirmation(
                        recipient_email=claimant_email or "claimant@example.com",
                        recipient_name=claim.get("claimant", "Claimant"),
                        claim_id=claim_id,
                        payment_record=inserted,
                        receipt_path=ctx.get("receipt", {}).get("file_path") if ctx.get("receipt") else None,
                    )
                    ctx["notification"] = {
                        "success": notif_result.success,
                        "recipient": notif_result.recipient,
                        "message_id": notif_result.message_id,
                    }

                # ---- Step 7: Assemble audit record ----
                # Set payout_timestamp BEFORE assembling so the audit record
                # includes the PayoutAgent trace.
                ctx["payout_timestamp"] = time.time()
                if self.audit_assembler is not None:
                    audit_record = self.audit_assembler.assemble(
                        claim=claim,
                        context=ctx,
                        payment_record=inserted,
                    )
                    ctx["audit_record"] = audit_record.to_dict()

            ctx["payment_authorized"] = payment_authorized
            ctx["bank_details_verified"] = bank_verified
            if "payout_timestamp" not in ctx:
                ctx["payout_timestamp"] = time.time()
            span.update(output={
                "payment_authorized": ctx["payment_authorized"],
                "bank_details_verified": ctx["bank_details_verified"],
                "ach_reference": (ctx.get("payment_record") or {}).get("ach_reference"),
                "receipt_generated": "receipt" in ctx,
                "notification_sent": "notification" in ctx,
                "audit_record_assembled": "audit_record" in ctx,
            })
        return ctx

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _check_duplicate(self, claim_id: str, idem_key: str) -> Optional[Any]:
        """Check the ledger for an existing payment (duplicate detection)."""
        # Try the new ledger interface first
        if hasattr(self.payment_ledger, "find_by_idempotency_key"):
            return self.payment_ledger.find_by_idempotency_key(idem_key)
        return None

    def _verify_bank_details(self, claim: dict[str, Any],
                             ctx: dict[str, Any]) -> bool:
        """Verify payee bank details via the bank verification service."""
        if self.bank_verification is not None:
            valid, _ = self.bank_verification.verify(
                bank_account=claim.get("bank_account", "000123456789"),
                bank_routing=claim.get("bank_routing", "021000021"),
                payee_name=claim.get("claimant", ""),
            )
            return valid
        # Legacy fallback
        return bool(claim.get("bank_details_verified", True))

    def _authorize_payment(self, claim: dict[str, Any],
                           ctx: dict[str, Any]) -> bool:
        """Legacy authorization fallback (no ACH provider)."""
        return bool(claim.get("payment_authorized", True))

    def _compute_breakdown(self, claim: dict[str, Any],
                           ctx: dict[str, Any]) -> dict[str, float]:
        """Compute the payment breakdown (gross, deductible, copay, net)."""
        gross = float(claim.get("amount", 0) or 0)
        deductible = float(ctx.get("deductible", claim.get("deductible", 0)) or 0)
        copay_pct = float(ctx.get("copay_pct", claim.get("copay_pct", 0)) or 0)
        if _PAYOUT_AVAILABLE and compute_payment_breakdown is not None:
            return compute_payment_breakdown(
                gross_amount=gross,
                deductible=deductible,
                copay_pct=copay_pct,
            )
        # Fallback
        net = max(0, gross - deductible)
        net = net - (net * copay_pct)
        return {
            "gross": round(gross, 2),
            "deductible": round(deductible, 2),
            "copay": round(gross * copay_pct, 2),
            "net": round(net, 2),
        }

    def _insert_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Insert a payment record, handling both new and legacy ledger interfaces."""
        if _PAYOUT_AVAILABLE and isinstance(self.payment_ledger, InMemoryPaymentLedger):
            # Use the new PaymentRecord dataclass
            pr = PaymentRecord(
                payment_id=record["payment_id"],
                claim_id=record["claim_id"],
                policy_id=record.get("policy_id", ""),
                payee=record.get("payee", ""),
                gross_amount=record.get("gross_amount", record.get("amount", 0)),
                deductible_applied=record.get("deductible_applied", 0),
                copay_amount=record.get("copay_amount", 0),
                net_payable=record.get("net_payable", record.get("amount", 0)),
                ach_reference=record.get("ach_reference", ""),
                status=record.get("status", "settled"),
                idempotency_key=record["idempotency_key"],
                settlement_date=record.get("settlement_date"),
            )
            inserted = self.payment_ledger.insert(pr)
            return inserted.to_dict()
        # Legacy dict-based ledger
        return self.payment_ledger.insert(record)

    def _record_to_dict(self, record: Any) -> dict[str, Any]:
        """Convert a payment record (dataclass or dict) to a dict."""
        if hasattr(record, "to_dict"):
            return record.to_dict()
        return dict(record) if isinstance(record, dict) else {}

    def _build_audit_trail_summary(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Build a summary of the audit trail for the receipt."""
        traces = []
        if ctx.get("intake_timestamp") or ctx.get("claim"):
            traces.append({
                "agent": "IntakeAgent", "state": "CLAIM_RECEIVED",
                "timestamp": time.strftime("%Y-%m-%d", time.gmtime(ctx.get("intake_timestamp", 0))),
                "outcome": "success",
            })
        if ctx.get("classification_complete"):
            traces.append({
                "agent": "ClassifierAgent", "state": "CLASSIFYING",
                "timestamp": time.strftime("%Y-%m-%d", time.gmtime(ctx.get("classification_timestamp", 0))),
                "outcome": "success",
            })
        if ctx.get("compliance_proof_verified") is not None:
            traces.append({
                "agent": "ZKPProver-ComplianceGate", "state": "ZKP_COMPLIANCE_PROOF",
                "timestamp": time.strftime("%Y-%m-%d", time.gmtime(ctx.get("compliance_proof_timestamp", 0))),
                "outcome": "success" if ctx.get("compliance_proof_verified") else "failed",
            })
        traces.append({
            "agent": "PayoutAgent", "state": "PAID_OUT",
            "timestamp": time.strftime("%Y-%m-%d", time.gmtime(time.time())),
            "outcome": "success",
        })
        return {"agent_traces": traces}

    def _collect_zkp_proof_refs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Collect ZKP proof references for the receipt."""
        refs = {}
        if ctx.get("policy_proof_verified") is not None:
            refs["Policy Validity"] = {
                "verified": ctx.get("policy_proof_verified"),
                "reference": ctx.get("policy_proof_ref", "N/A"),
            }
        if ctx.get("compliance_proof_verified") is not None:
            refs["Compliance Verification"] = {
                "verified": ctx.get("compliance_proof_verified"),
                "reference": ctx.get("compliance_root", "N/A"),
            }
        if ctx.get("fraud_detection_result"):
            fd = ctx["fraud_detection_result"]
            proof = fd.get("proof") or {}
            refs["Fraud Detection (Non-Membership)"] = {
                "verified": proof.get("verified"),
                "reference": fd.get("commitment_value", "N/A"),
            }
        return refs


class _InMemoryLedger:
    """Minimal in-memory payment ledger with idempotency support.

    Kept for backward compatibility with existing tests that pass a
    dict-based ledger. New code should use :class:`InMemoryPaymentLedger`
    from the payout subsystem instead.
    """
    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def find_by_claim(self, claim_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._records if r["claim_id"] == claim_id]

    def find_by_idempotency_key(self, key: str) -> Optional[dict[str, Any]]:
        for r in self._records:
            if r.get("idempotency_key") == key:
                return dict(r)
        return None

    def insert(self, record: dict[str, Any]) -> dict[str, Any]:
        record = dict(record)
        if "created_at" not in record:
            record["created_at"] = time.time()
        self._records.append(record)
        return dict(record)

    def all_records(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._records]


# ===========================================================================
# Helpers
# ===========================================================================
def _is_iso_date(s: str) -> bool:
    """YYYY-MM-DD format check."""
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return False
    try:
        time.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


_PERIL_TYPE_MAP: dict[str, int] = {
    "wind": 1, "hail": 2, "fire": 3, "theft": 4, "vandalism": 5,
    "lightning": 6, "collision": 7, "comprehensive": 8, "flood": 9,
    "earthquake": 10, "wear_and_tear": 12, "mold": 13,
    "intentional_damage": 14, "water_damage": 15,
}


def _infer_peril_type(claim: dict[str, Any],
                      policy: Optional[dict[str, Any]]) -> int:
    """Infer the numeric peril code from the claim description / claim type."""
    text = (
        str(claim.get("claim_type", "")) + " " +
        str(claim.get("description", ""))
    ).lower()
    for peril_name, code in _PERIL_TYPE_MAP.items():
        if peril_name in text:
            return code
    return 1  # default to wind


# ===========================================================================
# Orchestrator — wires all five agents + state machine together
# ===========================================================================
class ClaimOrchestrator:
    """End-to-end orchestrator that drives a claim through all states.

    Used by the integration tests to process 200 / 100 / 500 / 20 claim
    batches. Production uses the same class but with real Langfuse,
    Postgres, and LM Studio wiring.
    """

    def __init__(
        self,
        *,
        engine: Optional[StateMachineEngine] = None,
        intake: Optional[IntakeAgent] = None,
        validator: Optional[ValidatorAgent] = None,
        classifier: Optional[ClassifierAgent] = None,
        escalation: Optional[EscalationAgent] = None,
        payout: Optional[PayoutAgent] = None,
        zkp_policy_prover: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        zkp_compliance_prover: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        compliance_prover: Optional[ComplianceProver] = None,
    ) -> None:
        self.engine = engine or StateMachineEngine()
        self.intake = intake or IntakeAgent()
        self.validator = validator or ValidatorAgent(_DefaultSiloStore())
        self.classifier = classifier or ClassifierAgent()
        self.escalation = escalation or EscalationAgent()
        self.payout = payout or PayoutAgent()
        self.zkp_policy_prover = zkp_policy_prover or _stub_policy_prover
        # Use the real ComplianceProver (auto-falls back to stub if
        # circuit artifacts aren't compiled).
        self.compliance_prover = compliance_prover or ComplianceProver()
        self.zkp_compliance_prover = (
            zkp_compliance_prover or self._default_compliance_prover
        )

    def process(self, raw_claim: dict[str, Any],
                *, adjuster_decisions: Optional[dict[str, AdjusterDecision]] = None
                ) -> tuple[State, dict[str, Any]]:
        """Drive a claim end-to-end. Returns (final_state, final_context).

        If the claim routes to ESCALATING and ``adjuster_decisions`` is
        provided, the orchestrator submits the matching decision and
        continues. If no decision is provided for an escalated claim,
        the orchestrator stops at ESCALATING.
        """
        ctx: dict[str, Any] = {}
        claim_id = raw_claim.get("claim_id") or f"CLM-{uuid.uuid4().hex[:12].upper()}"
        raw_claim = dict(raw_claim)
        raw_claim["claim_id"] = claim_id

        # Initialize claim in state machine
        self.engine.initialize_claim(claim_id)
        state = State.CLAIM_RECEIVED

        # ---- IntakeAgent: CLAIM_RECEIVED ----
        ctx = self.intake.run(raw_claim, ctx)
        try:
            state = self.engine.transition(
                claim_id, State.CLAIM_RECEIVED, State.VALIDATING,
                claim=ctx["claim"], context=ctx,
            )
        except GuardConditionFailedError:
            return State.CLAIM_RECEIVED, ctx

        # ---- ValidatorAgent: VALIDATING ----
        ctx = self.validator.run(ctx["claim"], ctx)
        try:
            state = self.engine.transition(
                claim_id, State.VALIDATING, State.ZKP_POLICY_PROOF,
                claim=ctx["claim"], context=ctx,
            )
        except GuardConditionFailedError:
            # Discrepancies — fail back to CLAIM_RECEIVED (re-intake).
            # In production this would notify the claimant.
            return State.CLAIM_RECEIVED, ctx

        # ---- ZKP Policy Proof: ZKP_POLICY_PROOF ----
        proof_result = self.zkp_policy_prover(ctx.get("zkp_policy_inputs", {}))
        # Guard `_guard_zkp_policy_to_classifying` reads `proof_verified`
        # and `confidence` from the context — set both, plus the
        # prefixed aliases for the case summary.
        ctx["proof_verified"] = bool(proof_result.get("verified"))
        ctx["policy_proof_verified"] = bool(proof_result.get("verified"))
        ctx["policy_proof_statement"] = proof_result.get("statement")
        ctx["policy_proof_confidence"] = float(proof_result.get("confidence", 0.9))
        ctx["confidence"] = float(proof_result.get("confidence", 0.9))
        try:
            state = self.engine.transition(
                claim_id, State.ZKP_POLICY_PROOF, State.CLASSIFYING,
                claim=ctx["claim"], context=ctx,
            )
        except GuardConditionFailedError:
            # Route to ESCALATING explicitly
            state = self.engine.transition(
                claim_id, State.ZKP_POLICY_PROOF, State.ESCALATING,
                claim=ctx["claim"], context=ctx,
            )
            return self._handle_escalation(claim_id, ctx, adjuster_decisions)

        # ---- ClassifierAgent: CLASSIFYING ----
        ctx = self.classifier.run(ctx["claim"], ctx)
        try:
            state = self.engine.transition(
                claim_id, State.CLASSIFYING, State.ZKP_COMPLIANCE_PROOF,
                claim=ctx["claim"], context=ctx,
            )
        except GuardConditionFailedError:
            state = self.engine.transition(
                claim_id, State.CLASSIFYING, State.ESCALATING,
                claim=ctx["claim"], context=ctx,
            )
            return self._handle_escalation(claim_id, ctx, adjuster_decisions)

        # ---- ZKP Compliance Proof: ZKP_COMPLIANCE_PROOF ----
        comp_result = self.zkp_compliance_prover(ctx)
        # Guard `_guard_zkp_compliance_to_approved` reads
        # `compliance_proved`, `risk_class`, and `confidence` from context.
        ctx["compliance_proved"] = bool(comp_result.get("verified"))
        ctx["compliance_proof_verified"] = bool(comp_result.get("verified"))
        ctx["compliance_proof_statement"] = comp_result.get("statement")
        ctx["compliance_jurisdiction"] = comp_result.get("jurisdiction")
        ctx["confidence"] = float(comp_result.get("confidence", 0.9))
        try:
            state = self.engine.transition(
                claim_id, State.ZKP_COMPLIANCE_PROOF, State.APPROVED,
                claim=ctx["claim"], context=ctx,
            )
        except GuardConditionFailedError:
            state = self.engine.transition(
                claim_id, State.ZKP_COMPLIANCE_PROOF, State.ESCALATING,
                claim=ctx["claim"], context=ctx,
            )
            return self._handle_escalation(claim_id, ctx, adjuster_decisions)

        # ---- PayoutAgent: APPROVED → PAID_OUT ----
        ctx = self.payout.run(ctx["claim"], ctx)
        try:
            state = self.engine.transition(
                claim_id, State.APPROVED, State.PAID_OUT,
                claim=ctx["claim"], context=ctx,
            )
        except GuardConditionFailedError:
            return State.APPROVED, ctx

        return state, ctx

    def _default_compliance_prover(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Use the real ComplianceProver to generate a compliance proof.

        Builds a :class:`ComplianceClaimRecord` from the orchestrator
        context, calls ``ComplianceProver.prove()``, and also runs the
        traditional compliance checker in parallel (per the acceptance
        criteria). Returns a dict the orchestrator can feed into the
        state machine guard via ``ctx['compliance_proved']``.
        """
        # Settlement defaults to full claim amount (ratio 1.0) since
        # the claim is being processed for full payment at this stage.
        record = build_record_from_context(ctx, approved=1, settlement_ratio=1.0)
        result = self.compliance_prover.prove(record)
        # Run traditional check in parallel for the 12-month parallel run
        trad = self.compliance_prover.traditional_check(record)
        # If the ZKP and traditional paths disagree, log a warning
        # (production: raise an alert to the compliance team).
        if result.verified != trad.get("compliant"):
            logger.warning(
                "Compliance ZKP/traditional divergence: zkp=%s trad=%s "
                "claim_id=%s",
                result.verified, trad.get("compliant"),
                (ctx.get("claim") or {}).get("claim_id"),
            )
        return {
            "verified": result.verified,
            "statement": result.statement,
            "jurisdiction": result.jurisdiction,
            "claim_type": result.claim_type,
            "confidence": 0.95 if result.verified else 0.3,
            "proof_type": result.proof_type,
            "checks": result.checks,
            "traditional_compliant": trad.get("compliant"),
            "traditional_checks": trad.get("checks"),
            "compliance_root": result.compliance_root,
            "proof": result.proof,
            "public_signals": result.public_signals,
        }

    def _handle_escalation(
        self,
        claim_id: str,
        ctx: dict[str, Any],
        adjuster_decisions: Optional[dict[str, AdjusterDecision]],
    ) -> tuple[State, dict[str, Any]]:
        ctx = self.escalation.run(ctx.get("claim", {}), ctx)
        if adjuster_decisions and claim_id in adjuster_decisions:
            decision = adjuster_decisions[claim_id]
            decision_ctx = self.escalation.submit_decision(decision)
            ctx.update(decision_ctx)
            if decision.action == "approve" or decision.action == "reclassify":
                try:
                    state = self.engine.transition(
                        claim_id, State.ESCALATING, State.APPROVED,
                        claim=ctx.get("claim", {}), context=ctx,
                    )
                    # Continue to payout
                    ctx = self.payout.run(ctx.get("claim", {}), ctx)
                    try:
                        state = self.engine.transition(
                            claim_id, State.APPROVED, State.PAID_OUT,
                            claim=ctx.get("claim", {}), context=ctx,
                        )
                    except GuardConditionFailedError:
                        return State.APPROVED, ctx
                    return state, ctx
                except GuardConditionFailedError:
                    return State.ESCALATING, ctx
            elif decision.action == "deny":
                return State.ESCALATING, ctx  # terminal — claim denied
            elif decision.action == "request_more_info":
                return State.ESCALATING, ctx  # paused
        return State.ESCALATING, ctx


# ===========================================================================
# Stub ZKP provers (used when the real Circom circuits aren't compiled)
# ===========================================================================
def _stub_policy_prover(inputs: dict[str, Any]) -> dict[str, Any]:
    """Deterministic stub: verifies if claim_amount <= coverage_limit,
    peril covered, policy active. The real implementation lives in
    ``zkp_circuit/zkp_prover.py``."""
    claim_amount = float(inputs.get("claim_amount", 0) or 0)
    coverage_limit = float(inputs.get("coverage_limit", 0) or 0)
    policy_active = bool(inputs.get("policy_active"))
    perils = list(inputs.get("perils_covered", []) or [])
    peril_type = inputs.get("peril_type", 1)
    # Map peril_type code back to name (best effort)
    peril_name = next((k for k, v in _PERIL_TYPE_MAP.items() if v == peril_type), "wind")
    peril_covered = peril_name in perils
    verified = (claim_amount <= coverage_limit) and policy_active and peril_covered
    return {
        "verified": verified,
        "statement": (
            f"Policy validity proof {'VERIFIED' if verified else 'FAILED'}: "
            f"amount={claim_amount} ≤ limit={coverage_limit}; "
            f"peril_covered={peril_covered}; policy_active={policy_active}."
        ),
        "confidence": 0.95 if verified else 0.3,
        "proof_type": "stub",
    }


def _stub_compliance_prover(ctx: dict[str, Any]) -> dict[str, Any]:
    """Deterministic stub: verifies that the claim processing record
    satisfies the regulatory constraints of the claim's jurisdiction.
    The real implementation lives in
    ``zkp_circuit/compliance/compliance_prover.py``."""
    # Pull the jurisdiction from the zkp_policy_inputs in ctx
    inputs = ctx.get("zkp_policy_inputs", {}) or {}
    jurisdiction = inputs.get("jurisdiction", "CA")
    risk_class = ctx.get("risk_class", "low")
    fraud_score = float(ctx.get("fraud_risk_score", 0.0) or 0.0)
    # The stub "verifies" compliance if the risk is low/medium and the
    # fraud score is below the high threshold.
    verified = risk_class in {"low", "medium"} and fraud_score < 0.60
    return {
        "verified": verified,
        "statement": (
            f"Compliance proof {'VERIFIED' if verified else 'FAILED'} "
            f"for jurisdiction={jurisdiction}; risk_class={risk_class}; "
            f"fraud_score={fraud_score:.2f}."
        ),
        "jurisdiction": jurisdiction,
        "confidence": 0.95 if verified else 0.3,
        "proof_type": "stub",
    }


# ===========================================================================
# Default silo store for the orchestrator
# ===========================================================================
def _DefaultSiloStore():
    """Lazy import to avoid circular dependency."""
    from .silos import InMemorySiloStore
    return InMemorySiloStore()
