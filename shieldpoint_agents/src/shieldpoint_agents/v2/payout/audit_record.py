"""
SP-405 — Comprehensive Audit Record Assembly
=============================================

Assembles a complete audit record for each paid-out claim, including:

1. **Full agent trace** — every agent that processed the claim, in order,
   with timestamps, decisions, and outcomes.
2. **ZKP proof references** — all ZKP proofs generated and verified
   during processing (policy validity, compliance, fraud-detection
   non-membership).
3. **Payment breakdown** — gross amount, deductible, co-pay, net payable.
4. **State machine transitions** — every state transition with guard
   results and timestamps.
5. **Adjuster decisions** — any human-in-the-loop decisions (if the
   claim was escalated).

The audit record is:

- Stored in **Langfuse** as a structured trace (the top-level trace
  contains a span per agent + a final "audit_record" span with the
  full assembled record).
- Stored in **PostgreSQL** in the ``claim_audit_records`` table as an
  immutable JSON document (append-only; never updated or deleted).

This dual storage provides both real-time observability (Langfuse)
and long-term immutable audit (PostgreSQL) — the regulatory
requirement for insurance claim records.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("shieldpoint.payout.audit")


@dataclass(frozen=True)
class AgentTrace:
    """A single agent's processing step in the audit trail.

    Attributes
    ----------
    agent : str
        Agent name (e.g. "IntakeAgent", "ClassifierAgent").
    state : str
        State machine state owned by this agent (e.g. "CLASSIFYING").
    timestamp : float
        When the agent processed this claim.
    trace_id : str, optional
        Langfuse trace ID (for cross-referencing).
    span_id : str, optional
        Langfuse span ID.
    input_summary : dict
        Summary of the agent's input (no PII — only structural fields).
    output_summary : dict
        Summary of the agent's output (decision, scores, etc.).
    outcome : str
        "success", "escalated", "failed", or "skipped".
    duration_ms : float
        Processing time in milliseconds.
    """

    agent: str
    state: str
    timestamp: float
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"
    duration_ms: float = 0.0


@dataclass(frozen=True)
class ZKPProofRef:
    """A reference to a ZKP proof in the audit trail.

    Attributes
    ----------
    proof_type : str
        "policy_validity", "compliance_verification", "cross_agent_claim_limit",
        "non_membership", or "fraud_detection".
    verified : bool
        Whether the proof verified successfully.
    statement : str
        Human-readable proof statement.
    proof_ref : str
        Reference ID (commitment hash or proof ID) for cross-referencing.
    public_signals : dict
        Public signals of the proof (safe to share).
    timestamp : float
        When the proof was generated/verified.
    """

    proof_type: str
    verified: bool
    statement: str
    proof_ref: str
    public_signals: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class StateTransition:
    """A single state machine transition in the audit trail.

    Attributes
    ----------
    from_state : str
    to_state : str
    transition_name : str
    agent : str
    guard_ok : bool
    guard_reason : str
    timestamp : float
    trace_id : str, optional
    """

    from_state: Optional[str]
    to_state: str
    transition_name: str
    agent: str
    guard_ok: bool
    guard_reason: str
    timestamp: float
    trace_id: Optional[str] = None


@dataclass(frozen=True)
class AuditRecord:
    """The complete audit record for a claim.

    This is the immutable record stored in Langfuse + PostgreSQL.

    Attributes
    ----------
    audit_id : str
        Unique audit record ID.
    claim_id : str
    policy_id : str
    claimant : str
    final_state : str
        Final state machine state (e.g. "PAID_OUT" or "ESCALATING").
    created_at : float
    agent_traces : list[AgentTrace]
        Every agent that processed the claim, in order.
    zkp_proof_refs : list[ZKPProofRef]
        Every ZKP proof generated/verified during processing.
    state_transitions : list[StateTransition]
        Every state machine transition.
    payment_record : dict, optional
        The final payment record (if the claim was paid out).
    adjuster_decisions : list[dict]
        Any human adjuster decisions (if escalated).
    fraud_detection : dict, optional
        Fraud detection check result (if the fraud network was queried).
    content_hash : str
        SHA-256 hash of the full record (for integrity verification).
    """

    audit_id: str
    claim_id: str
    policy_id: str
    claimant: str
    final_state: str
    created_at: float
    agent_traces: list[AgentTrace] = field(default_factory=list)
    zkp_proof_refs: list[ZKPProofRef] = field(default_factory=list)
    state_transitions: list[StateTransition] = field(default_factory=list)
    payment_record: Optional[dict[str, Any]] = None
    adjuster_decisions: list[dict[str, Any]] = field(default_factory=list)
    fraud_detection: Optional[dict[str, Any]] = None
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "claim_id": self.claim_id,
            "policy_id": self.policy_id,
            "claimant": self.claimant,
            "final_state": self.final_state,
            "created_at": self.created_at,
            "agent_traces": [
                {
                    "agent": t.agent, "state": t.state,
                    "timestamp": t.timestamp, "trace_id": t.trace_id,
                    "span_id": t.span_id, "input_summary": t.input_summary,
                    "output_summary": t.output_summary, "outcome": t.outcome,
                    "duration_ms": t.duration_ms,
                }
                for t in self.agent_traces
            ],
            "zkp_proof_refs": [
                {
                    "proof_type": p.proof_type, "verified": p.verified,
                    "statement": p.statement, "proof_ref": p.proof_ref,
                    "public_signals": p.public_signals, "timestamp": p.timestamp,
                }
                for p in self.zkp_proof_refs
            ],
            "state_transitions": [
                {
                    "from_state": t.from_state, "to_state": t.to_state,
                    "transition_name": t.transition_name, "agent": t.agent,
                    "guard_ok": t.guard_ok, "guard_reason": t.guard_reason,
                    "timestamp": t.timestamp, "trace_id": t.trace_id,
                }
                for t in self.state_transitions
            ],
            "payment_record": self.payment_record,
            "adjuster_decisions": self.adjuster_decisions,
            "fraud_detection": self.fraud_detection,
            "content_hash": self.content_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


class AuditRecordAssembler:
    """Assembles audit records from claim processing context.

    Called by the PayoutAgent after a successful payout (or by the
    EscalationAgent for denied claims) to create the final immutable
    audit record.

    The assembler pulls data from:

    1. The claim context dict (which accumulates data as each agent runs).
    2. The state machine engine's transition log (persisted to
       PostgreSQL or SQLite).
    3. The Langfuse trace (via the langfuse_wrapper — if available).
    4. The payment ledger (for the final payment record).
    """

    def __init__(self) -> None:
        pass

    def assemble(
        self,
        *,
        claim: dict[str, Any],
        context: dict[str, Any],
        payment_record: Optional[dict[str, Any]] = None,
        state_log: Optional[list[dict[str, Any]]] = None,
    ) -> AuditRecord:
        """Assemble the complete audit record.

        Parameters
        ----------
        claim : dict
            The original claim data.
        context : dict
            The accumulated context from all agents (includes agent
            outputs, ZKP proof results, fraud detection results, etc.).
        payment_record : dict, optional
            The final payment record (if the claim was paid out).
        state_log : list[dict], optional
            The state machine's transition log (from the engine).
        """
        audit_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
        created_at = time.time()

        # Build agent traces from the context
        agent_traces = self._build_agent_traces(context)

        # Build ZKP proof references from the context
        zkp_proof_refs = self._build_zkp_proof_refs(context)

        # Build state transitions from the state log
        state_transitions = self._build_state_transitions(state_log or [])

        # Extract adjuster decisions (if any)
        adjuster_decisions = self._extract_adjuster_decisions(context)

        # Extract fraud detection result (if any)
        fraud_detection = context.get("fraud_detection_result")
        if fraud_detection is not None:
            fraud_detection = {
                "is_unique": fraud_detection.get("is_unique"),
                "commitment_value": fraud_detection.get("commitment_value"),
                "merkle_root": fraud_detection.get("merkle_root"),
                "duplicate_insurer": fraud_detection.get("duplicate_insurer"),
                "proof_type": (fraud_detection.get("proof") or {}).get("proof_type"),
            }

        # Determine final state
        final_state = context.get("final_state", "PAID_OUT")

        record = AuditRecord(
            audit_id=audit_id,
            claim_id=claim.get("claim_id", "UNKNOWN"),
            policy_id=claim.get("policy_id", "UNKNOWN"),
            claimant=claim.get("claimant", "UNKNOWN"),
            final_state=final_state,
            created_at=created_at,
            agent_traces=agent_traces,
            zkp_proof_refs=zkp_proof_refs,
            state_transitions=state_transitions,
            payment_record=payment_record,
            adjuster_decisions=adjuster_decisions,
            fraud_detection=fraud_detection,
        )

        # Compute content hash for integrity
        content_hash = self._compute_hash(record)
        # Return a new record with the hash set
        return AuditRecord(
            audit_id=record.audit_id,
            claim_id=record.claim_id,
            policy_id=record.policy_id,
            claimant=record.claimant,
            final_state=record.final_state,
            created_at=record.created_at,
            agent_traces=record.agent_traces,
            zkp_proof_refs=record.zkp_proof_refs,
            state_transitions=record.state_transitions,
            payment_record=record.payment_record,
            adjuster_decisions=record.adjuster_decisions,
            fraud_detection=record.fraud_detection,
            content_hash=content_hash,
        )

    def _build_agent_traces(
        self, context: dict[str, Any]
    ) -> list[AgentTrace]:
        """Build the ordered list of agent traces from the context."""
        traces: list[AgentTrace] = []

        # IntakeAgent
        if context.get("intake_timestamp") or context.get("claim"):
            traces.append(AgentTrace(
                agent="IntakeAgent",
                state="CLAIM_RECEIVED",
                timestamp=context.get("intake_timestamp", 0),
                trace_id=context.get("trace_id"),
                output_summary={
                    "claim_id": context.get("claim", {}).get("claim_id"),
                    "validation_passed": context.get("intake_valid", True),
                },
                outcome="success",
                duration_ms=context.get("intake_duration_ms", 0),
            ))

        # ValidatorAgent
        if context.get("validation_timestamp") or context.get("silo_records"):
            traces.append(AgentTrace(
                agent="ValidatorAgent",
                state="VALIDATING",
                timestamp=context.get("validation_timestamp", 0),
                trace_id=context.get("trace_id"),
                input_summary={
                    "claim_id": context.get("claim", {}).get("claim_id"),
                },
                output_summary={
                    "discrepancies": context.get("discrepancies", []),
                    "silo_records_count": len(context.get("silo_records", [])),
                },
                outcome="success" if not context.get("discrepancies") else "escalated",
                duration_ms=context.get("validation_duration_ms", 0),
            ))

        # ZKP Policy Prover
        if context.get("policy_proof_verified") is not None:
            traces.append(AgentTrace(
                agent="ZKPProver-PolicyGate",
                state="ZKP_POLICY_PROOF",
                timestamp=context.get("policy_proof_timestamp", 0),
                output_summary={
                    "verified": context.get("policy_proof_verified"),
                    "statement": context.get("policy_proof_statement"),
                },
                outcome="success" if context.get("policy_proof_verified") else "escalated",
            ))

        # ClassifierAgent
        if context.get("classification_complete"):
            traces.append(AgentTrace(
                agent="ClassifierAgent",
                state="CLASSIFYING",
                timestamp=context.get("classification_timestamp", 0),
                trace_id=context.get("trace_id"),
                output_summary={
                    "severity": context.get("severity"),
                    "claim_type": context.get("claim_type"),
                    "fraud_risk_score": context.get("fraud_risk_score"),
                    "risk_class": context.get("risk_class"),
                    "ambiguous": context.get("ambiguous"),
                },
                outcome="success" if not context.get("ambiguous") else "escalated",
                duration_ms=context.get("classification_duration_ms", 0),
            ))

        # Fraud Detection (if checked)
        if context.get("fraud_detection_result"):
            fd = context["fraud_detection_result"]
            traces.append(AgentTrace(
                agent="FraudDetectionNetwork",
                state="CLASSIFYING",
                timestamp=fd.get("checked_at", 0),
                output_summary={
                    "is_unique": fd.get("is_unique"),
                    "duplicate_insurer": fd.get("duplicate_insurer"),
                },
                outcome="success" if fd.get("is_unique") else "escalated",
            ))

        # ZKP Compliance Prover
        if context.get("compliance_proof_verified") is not None:
            traces.append(AgentTrace(
                agent="ZKPProver-ComplianceGate",
                state="ZKP_COMPLIANCE_PROOF",
                timestamp=context.get("compliance_proof_timestamp", 0),
                output_summary={
                    "verified": context.get("compliance_proof_verified"),
                    "jurisdiction": context.get("compliance_jurisdiction"),
                    "statement": context.get("compliance_proof_statement"),
                },
                outcome="success" if context.get("compliance_proof_verified") else "escalated",
            ))

        # EscalationAgent (if escalated)
        if context.get("escalation_queued_at"):
            traces.append(AgentTrace(
                agent="EscalationAgent",
                state="ESCALATING",
                timestamp=context.get("escalation_queued_at", 0),
                output_summary={
                    "escalation_reason": context.get("escalation_reason"),
                    "case_summary": "generated",
                },
                outcome="escalated",
            ))

        # PayoutAgent
        if context.get("payout_timestamp"):
            traces.append(AgentTrace(
                agent="PayoutAgent",
                state="PAID_OUT",
                timestamp=context.get("payout_timestamp", 0),
                output_summary={
                    "payment_authorized": context.get("payment_authorized"),
                    "bank_details_verified": context.get("bank_details_verified"),
                    "ach_reference": (context.get("payment_record") or {}).get("ach_reference"),
                    "duplicate_payment_prevented": context.get("duplicate_payment_prevented", False),
                },
                outcome="success" if context.get("payment_authorized") else "failed",
            ))

        return traces

    def _build_zkp_proof_refs(
        self, context: dict[str, Any]
    ) -> list[ZKPProofRef]:
        """Build the list of ZKP proof references from the context."""
        refs: list[ZKPProofRef] = []

        # Policy validity proof
        if context.get("policy_proof_verified") is not None:
            refs.append(ZKPProofRef(
                proof_type="policy_validity",
                verified=context.get("policy_proof_verified", False),
                statement=context.get("policy_proof_statement", ""),
                proof_ref=context.get("policy_proof_ref", ""),
                public_signals={
                    "policy_commitment": context.get("policy_commitment"),
                },
                timestamp=context.get("policy_proof_timestamp", 0),
            ))

        # Cross-agent claim limit proof
        if context.get("zkp_proof"):
            proof = context["zkp_proof"]
            refs.append(ZKPProofRef(
                proof_type="cross_agent_claim_limit",
                verified=proof.get("verified", False),
                statement="Claim amount within policy coverage limit.",
                proof_ref=proof.get("policy_commitment", ""),
                public_signals={
                    "claim_amount": proof.get("claim_amount"),
                    "policy_commitment": proof.get("policy_commitment"),
                },
            ))

        # Compliance proof
        if context.get("compliance_proof_verified") is not None:
            refs.append(ZKPProofRef(
                proof_type="compliance_verification",
                verified=context.get("compliance_proof_verified", False),
                statement=context.get("compliance_proof_statement", ""),
                proof_ref=context.get("compliance_root", ""),
                public_signals={
                    "jurisdiction": context.get("compliance_jurisdiction"),
                    "compliance_root": context.get("compliance_root"),
                },
                timestamp=context.get("compliance_proof_timestamp", 0),
            ))

        # Fraud detection non-membership proof
        if context.get("fraud_detection_result"):
            fd = context["fraud_detection_result"]
            proof = fd.get("proof") or {}
            refs.append(ZKPProofRef(
                proof_type="non_membership",
                verified=proof.get("verified", False),
                statement=proof.get("statement", ""),
                proof_ref=fd.get("commitment_value", ""),
                public_signals={
                    "merkle_root": fd.get("merkle_root"),
                    "new_commitment": fd.get("commitment_value"),
                },
                timestamp=fd.get("checked_at", 0),
            ))

        return refs

    def _build_state_transitions(
        self, state_log: list[dict[str, Any]]
    ) -> list[StateTransition]:
        """Build state transitions from the engine's log."""
        transitions: list[StateTransition] = []
        for entry in state_log:
            transitions.append(StateTransition(
                from_state=entry.get("from_state"),
                to_state=entry.get("to_state", ""),
                transition_name=entry.get("transition_name", ""),
                agent=entry.get("agent", ""),
                guard_ok=entry.get("guard_ok", True),
                guard_reason=entry.get("guard_reason", ""),
                timestamp=entry.get("timestamp", 0),
                trace_id=entry.get("trace_id"),
            ))
        return transitions

    def _extract_adjuster_decisions(
        self, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract adjuster decisions from the context."""
        decisions: list[dict[str, Any]] = []
        if context.get("adjuster_action"):
            decisions.append({
                "adjuster_id": context.get("adjuster_id"),
                "action": context.get("adjuster_action"),
                "rationale": context.get("adjuster_rationale"),
                "timestamp": context.get("adjuster_decision_timestamp"),
            })
        return decisions

    def _compute_hash(self, record: AuditRecord) -> str:
        """Compute a SHA-256 hash of the record (excluding the hash itself)."""
        data = record.to_dict()
        data.pop("content_hash", None)
        # Sort keys for deterministic hashing
        content = json.dumps(data, sort_keys=True, default=str).encode()
        return hashlib.sha256(content).hexdigest()
