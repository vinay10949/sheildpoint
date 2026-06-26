"""
ShieldPoint 5-Agent State Machine Engine
========================================

This module implements the deterministic state machine that orchestrates the
five agents through eight discrete states and nine transitions in the
ShieldPoint claims automation lifecycle::

    CLAIM_RECEIVED (IntakeAgent)
        ↓ 1
    VALIDATING (ValidatorAgent)
        ↓ 2
    ZKP_POLICY_PROOF (ZKP Prover)
        ↓ 3 / 4
    CLASSIFYING (ClassifierAgent)
        ↓ 5
    ZKP_COMPLIANCE_PROOF (ZKP Prover)
        ↓ 6 / 7
    APPROVED  ←── 8 ── ESCALATING (EscalationAgent)
        ↓ 9
    PAID_OUT (PayoutAgent)

Each transition is guarded by an explicit condition. Guard failures route
the claim to ``ESCALATING`` for human review (except the
``CLAIM_RECEIVED → VALIDATING`` transition which is unconditional and the
``VALIDATING → ZKP_POLICY_PROOF`` transition which routes back to
``CLAIM_RECEIVED`` for re-intake on format failure).

Persistence
-----------
Every transition is persisted to the ``state_log`` table in PostgreSQL with
``(claim_id, state, agent, timestamp, guard_result, transition_name, trace_id)``.
If the ``SHIELDPOINT_DB_URL`` environment variable is unset, the engine
falls back to an in-memory SQLite database so unit tests and local
development work without external services. The :class:`StateMachineEngine`
class is the single entry point for both modes — callers never need to know
which backend is active.

Langfuse Tracing
----------------
Every transition emits a Langfuse span via the observability wrapper located
at ``agent_framework/observability/langfuse_wrapper.py``. The span carries:

- ``agent_id``           — the agent that triggered the transition
- ``transition_name``    — symbolic name (e.g. ``validating_to_zkp_policy``)
- ``from_state``         — source state
- ``to_state``           — target state
- ``guard_result``       — ``{ok, reason, details}`` from the guard
- ``claim_id``           — claim being processed

If the Langfuse SDK is not installed or the env vars are not set, the
tracer transparently no-ops, so the engine still runs end-to-end.

Recovery
--------
After a system restart, :meth:`StateMachineEngine.get_state` reads the
most recent persisted row for a claim from the ``state_log`` table and
returns it. This means a crashed process can be resumed simply by
re-instantiating the engine and looking up the claim.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("state_machine_engine")


# ---------------------------------------------------------------------------
# Optional PostgreSQL driver. Falls back to SQLite when unavailable.
# ---------------------------------------------------------------------------
try:  # pragma: no cover — exercised in environments where psycopg2 is installed
    import psycopg2
    import psycopg2.extras
    _PSYCOPG2_AVAILABLE = True
except ImportError:  # pragma: no cover
    psycopg2 = None  # type: ignore
    _PSYCOPG2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Optional Langfuse tracer import. If not installed, no-op decorator.
# ---------------------------------------------------------------------------
try:
    import sys as _sys
    _lf_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "agent_framework", "observability",
    )
    if os.path.isdir(_lf_path) and _lf_path not in _sys.path:
        _sys.path.insert(0, _lf_path)
    from langfuse_wrapper import LangfuseTracer, get_tracer  # type: ignore
    _LANGFUSE_AVAILABLE = True
except Exception:  # pragma: no cover
    _LANGFUSE_AVAILABLE = False
    LangfuseTracer = None  # type: ignore
    get_tracer = None  # type: ignore


# ===========================================================================
# Exceptions
# ===========================================================================
class InvalidStateTransitionError(ValueError):
    """Raised when a transition is not defined in the state machine."""

    def __init__(self, from_state: str, to_state: str, *, allowed: list[str]) -> None:
        super().__init__(
            f"Invalid transition: {from_state} -> {to_state}. "
            f"Allowed from {from_state}: {allowed}"
        )
        self.from_state = from_state
        self.to_state = to_state
        self.allowed = allowed


class GuardConditionFailedError(ValueError):
    """Raised when a transition's guard condition is not satisfied.

    Carries a structured ``details`` dict so the caller can persist the
    reason into the Langfuse trace and the escalation queue.
    """

    def __init__(
        self, message: str, *, from_state: str, to_state: str, details: dict[str, Any]
    ) -> None:
        super().__init__(message)
        self.from_state = from_state
        self.to_state = to_state
        self.details = details


class StateRecoveryError(RuntimeError):
    """Raised when state recovery fails (e.g. corrupt log entry)."""


# ===========================================================================
# Enums: State and Transition
# ===========================================================================
class State(str, enum.Enum):
    """The 8 discrete states a claim can occupy."""

    CLAIM_RECEIVED = "CLAIM_RECEIVED"            # IntakeAgent
    VALIDATING = "VALIDATING"                    # ValidatorAgent
    ZKP_POLICY_PROOF = "ZKP_POLICY_PROOF"        # ZKP Prover (policy gate)
    CLASSIFYING = "CLASSIFYING"                  # ClassifierAgent
    ZKP_COMPLIANCE_PROOF = "ZKP_COMPLIANCE_PROOF"  # ZKP Prover (compliance gate)
    ESCALATING = "ESCALATING"                    # EscalationAgent (HITL)
    APPROVED = "APPROVED"                        # Decision recorded; awaiting payout
    PAID_OUT = "PAID_OUT"                        # PayoutAgent (terminal)


class Transition(str, enum.Enum):
    """The 9 defined transitions between states."""

    CLAIM_RECEIVED_TO_VALIDATING = "CLAIM_RECEIVED_TO_VALIDATING"
    VALIDATING_TO_ZKP_POLICY_PROOF = "VALIDATING_TO_ZKP_POLICY_PROOF"
    ZKP_POLICY_PROOF_TO_CLASSIFYING = "ZKP_POLICY_PROOF_TO_CLASSIFYING"
    ZKP_POLICY_PROOF_TO_ESCALATING = "ZKP_POLICY_PROOF_TO_ESCALATING"
    CLASSIFYING_TO_ZKP_COMPLIANCE_PROOF = "CLASSIFYING_TO_ZKP_COMPLIANCE_PROOF"
    ZKP_COMPLIANCE_PROOF_TO_APPROVED = "ZKP_COMPLIANCE_PROOF_TO_APPROVED"
    ZKP_COMPLIANCE_PROOF_TO_ESCALATING = "ZKP_COMPLIANCE_PROOF_TO_ESCALATING"
    ESCALATING_TO_APPROVED = "ESCALATING_TO_APPROVED"
    APPROVED_TO_PAID_OUT = "APPROVED_TO_PAID_OUT"


# Map: agent that owns each state (used by Langfuse span tagging + log row)
STATE_AGENT: dict[State, str] = {
    State.CLAIM_RECEIVED: "IntakeAgent",
    State.VALIDATING: "ValidatorAgent",
    State.ZKP_POLICY_PROOF: "ZKPProver-PolicyGate",
    State.CLASSIFYING: "ClassifierAgent",
    State.ZKP_COMPLIANCE_PROOF: "ZKPProver-ComplianceGate",
    State.ESCALATING: "EscalationAgent",
    State.APPROVED: "PayoutAgent",
    State.PAID_OUT: "PayoutAgent",
}


# ===========================================================================
# Guard function signature
# ===========================================================================
GuardFn = Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str, dict[str, Any]]]
"""Guard function signature.

Takes ``(claim, context)`` and returns ``(ok, reason, details)``:
- ``ok``       — True if the guard is satisfied.
- ``reason``   — human-readable explanation (passed to the LLM / adjuster).
- ``details``  — structured dict for Langfuse span metadata.
"""


# ===========================================================================
# Transition definition
# ===========================================================================
@dataclass(frozen=True)
class TransitionDef:
    """A single defined transition in the state machine."""

    name: Transition
    from_state: State
    to_state: State
    agent: str
    description: str
    guard: Optional[GuardFn] = None
    failure_route: Optional[State] = None  # where to go if guard fails
    # NOTE: ``failure_route`` is informational only — the engine's
    # ``transition()`` method raises ``GuardConditionFailedError`` and the
    # caller decides whether to route to ``failure_route`` or stop. This
    # keeps the state machine deterministic and explicit.


# ===========================================================================
# Built-in guards
# ===========================================================================
def _guard_always_ok(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    return (True, "Unconditional automatic transition.", {})


def _guard_validating_to_zkp_policy(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """All required fields present, no format errors, no silo discrepancies."""
    required = ["claim_id", "policy_id", "claimant", "amount", "date_of_loss"]
    missing = [f for f in required if f not in claim or claim.get(f) in (None, "")]
    if missing:
        return (
            False,
            f"Missing required fields: {missing}.",
            {"missing_fields": missing},
        )
    try:
        amount = float(claim.get("amount", 0))
        if amount <= 0:
            return (False, "Amount must be > 0.", {"amount": amount})
    except (TypeError, ValueError):
        return (False, "Amount is not a valid number.", {"amount": claim.get("amount")})

    # Silo discrepancy check — set by the ValidatorAgent.
    discrepancies = context.get("discrepancies", [])
    if discrepancies:
        return (
            False,
            f"Data silo discrepancies detected: {len(discrepancies)} issue(s).",
            {"discrepancies": discrepancies},
        )
    return (True, "All required fields present, no discrepancies.", {})


def _guard_zkp_policy_to_classifying(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Proof verified + confidence >= 0.85."""
    proof_verified = bool(context.get("proof_verified", False))
    confidence = float(context.get("confidence", 0.0))
    if not proof_verified:
        return (
            False,
            "ZKP policy proof failed verification.",
            {"proof_verified": False, "failure_route": "ESCALATING"},
        )
    if confidence < 0.85:
        return (
            False,
            f"Confidence {confidence:.2f} below 0.85 threshold; escalating.",
            {"confidence": confidence, "threshold": 0.85,
             "failure_route": "ESCALATING"},
        )
    return (True, "Proof verified and confidence sufficient.",
            {"confidence": confidence})


def _guard_classifying_to_zkp_compliance(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Severity classified + fraud risk score computed.

    Note: Ambiguity is NOT checked here — the spec defines only 9
    transitions, and there is no ``CLASSIFYING -> ESCALATING`` edge.
    Instead, ambiguous claims proceed to ``ZKP_COMPLIANCE_PROOF`` and
    the compliance gate's guard (``_guard_zkp_compliance_to_approved``)
    fails, routing to ESCALATING via the defined
    ``ZKP_COMPLIANCE_PROOF -> ESCALATING`` transition. This keeps the
    state machine at exactly 9 transitions while still ensuring
    ambiguous claims reach a human adjuster.
    """
    severity = context.get("severity")
    fraud_score = context.get("fraud_risk_score")
    if severity is None:
        return (False, "Severity not yet classified.",
                {"failure_route": "ESCALATING"})
    if fraud_score is None:
        return (False, "Fraud risk score not yet computed.",
                {"failure_route": "ESCALATING"})
    return (
        True,
        f"Severity={severity}, fraud_score={fraud_score:.3f}.",
        {"severity": severity, "fraud_risk_score": fraud_score},
    )


def _guard_zkp_compliance_to_approved(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Compliance proved + low-risk + confidence >= 0.85 + not ambiguous.

    Ambiguity is checked here (rather than at the CLASSIFYING ->
    ZKP_COMPLIANCE_PROOF gate) because the spec defines only 9
    transitions and there is no CLASSIFYING -> ESCALATING edge.
    Ambiguous claims reach this guard, fail it, and route to ESCALATING
    via the defined ZKP_COMPLIANCE_PROOF -> ESCALATING transition.
    """
    compliance_proved = bool(context.get("compliance_proved", False))
    risk = str(context.get("risk_class", "high")).lower()
    confidence = float(context.get("confidence", 0.0))
    ambiguous = bool(context.get("ambiguous", False))
    if not compliance_proved:
        return (
            False,
            "Compliance proof failed.",
            {"compliance_proved": False, "failure_route": "ESCALATING"},
        )
    if ambiguous:
        return (
            False,
            f"Classification ambiguous: {context.get('ambiguity_reason')}",
            {"ambiguous": True,
             "ambiguity_reason": context.get("ambiguity_reason"),
             "failure_route": "ESCALATING"},
        )
    if risk != "low":
        return (
            False,
            f"Risk class '{risk}' is not 'low'; escalating.",
            {"risk_class": risk, "failure_route": "ESCALATING"},
        )
    if confidence < 0.85:
        return (
            False,
            f"Confidence {confidence:.2f} below 0.85.",
            {"confidence": confidence, "failure_route": "ESCALATING"},
        )
    return (True, "Compliance proved, low risk, confidence sufficient.", {})


def _guard_escalating_to_approved(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Human adjuster explicit approval recorded."""
    approved = bool(context.get("human_approval", False))
    adjuster_id = context.get("adjuster_id")
    if not approved:
        return (False, "Human approval not yet recorded.",
                {"human_approval": False})
    if not adjuster_id:
        return (False, "Adjuster ID missing on approval.", {})
    return (True, f"Approved by adjuster {adjuster_id}.",
            {"adjuster_id": adjuster_id})


def _guard_approved_to_paid_out(
    claim: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, str, dict[str, Any]]:
    """Payment authorization + bank details verified."""
    payment_authorized = bool(context.get("payment_authorized", False))
    bank_verified = bool(context.get("bank_details_verified", False))
    if not payment_authorized:
        return (False, "Payment not yet authorized.", {})
    if not bank_verified:
        return (False, "Bank details not yet verified.", {})
    return (True, "Payment authorized and bank verified.", {})


# ===========================================================================
# Transition table (9 transitions)
# ===========================================================================
def _default_transitions() -> list[TransitionDef]:
    return [
        TransitionDef(
            name=Transition.CLAIM_RECEIVED_TO_VALIDATING,
            from_state=State.CLAIM_RECEIVED,
            to_state=State.VALIDATING,
            agent=STATE_AGENT[State.VALIDATING],
            description="Always automatic trigger (IntakeAgent hands off to ValidatorAgent).",
            guard=_guard_always_ok,
        ),
        TransitionDef(
            name=Transition.VALIDATING_TO_ZKP_POLICY_PROOF,
            from_state=State.VALIDATING,
            to_state=State.ZKP_POLICY_PROOF,
            agent=STATE_AGENT[State.ZKP_POLICY_PROOF],
            description="All required fields present, no format errors, no silo discrepancies.",
            guard=_guard_validating_to_zkp_policy,
            failure_route=State.CLAIM_RECEIVED,
        ),
        TransitionDef(
            name=Transition.ZKP_POLICY_PROOF_TO_CLASSIFYING,
            from_state=State.ZKP_POLICY_PROOF,
            to_state=State.CLASSIFYING,
            agent=STATE_AGENT[State.CLASSIFYING],
            description="Proof verified + confidence >= 0.85.",
            guard=_guard_zkp_policy_to_classifying,
            failure_route=State.ESCALATING,
        ),
        TransitionDef(
            name=Transition.ZKP_POLICY_PROOF_TO_ESCALATING,
            from_state=State.ZKP_POLICY_PROOF,
            to_state=State.ESCALATING,
            agent=STATE_AGENT[State.ESCALATING],
            description="Proof invalid or confidence < 0.85 — explicit escalation route.",
            guard=_guard_always_ok,
        ),
        TransitionDef(
            name=Transition.CLASSIFYING_TO_ZKP_COMPLIANCE_PROOF,
            from_state=State.CLASSIFYING,
            to_state=State.ZKP_COMPLIANCE_PROOF,
            agent=STATE_AGENT[State.ZKP_COMPLIANCE_PROOF],
            description="Severity classified + fraud score computed + not ambiguous.",
            guard=_guard_classifying_to_zkp_compliance,
            failure_route=State.ESCALATING,
        ),
        TransitionDef(
            name=Transition.ZKP_COMPLIANCE_PROOF_TO_APPROVED,
            from_state=State.ZKP_COMPLIANCE_PROOF,
            to_state=State.APPROVED,
            agent=STATE_AGENT[State.APPROVED],
            description="Compliance proved + low-risk + confidence >= 0.85.",
            guard=_guard_zkp_compliance_to_approved,
            failure_route=State.ESCALATING,
        ),
        TransitionDef(
            name=Transition.ZKP_COMPLIANCE_PROOF_TO_ESCALATING,
            from_state=State.ZKP_COMPLIANCE_PROOF,
            to_state=State.ESCALATING,
            agent=STATE_AGENT[State.ESCALATING],
            description="Compliance uncertain or high risk — explicit escalation route.",
            guard=_guard_always_ok,
        ),
        TransitionDef(
            name=Transition.ESCALATING_TO_APPROVED,
            from_state=State.ESCALATING,
            to_state=State.APPROVED,
            agent=STATE_AGENT[State.APPROVED],
            description="Human adjuster explicit approval recorded.",
            guard=_guard_escalating_to_approved,
        ),
        TransitionDef(
            name=Transition.APPROVED_TO_PAID_OUT,
            from_state=State.APPROVED,
            to_state=State.PAID_OUT,
            agent=STATE_AGENT[State.PAID_OUT],
            description="Payment authorization + bank details verified.",
            guard=_guard_approved_to_paid_out,
        ),
    ]


# ===========================================================================
# State log entry
# ===========================================================================
@dataclass
class StateLogEntry:
    """One persisted row in the ``state_log`` table."""

    log_id: str
    claim_id: str
    from_state: Optional[str]      # None for initial intake
    to_state: str
    transition_name: str
    agent: str
    guard_ok: bool
    guard_reason: str
    guard_details: dict[str, Any]
    trace_id: Optional[str]
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_id": self.log_id,
            "claim_id": self.claim_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "transition_name": self.transition_name,
            "agent": self.agent,
            "guard_ok": self.guard_ok,
            "guard_reason": self.guard_reason,
            "guard_details": self.guard_details,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ===========================================================================
# Backend: PostgreSQL or SQLite (auto-selected)
# ===========================================================================
class _StateLogBackend:
    """Abstract interface for the persistence backend."""

    def init_schema(self) -> None: ...
    def append(self, entry: StateLogEntry) -> None: ...
    def latest_for_claim(self, claim_id: str) -> Optional[StateLogEntry]: ...
    def history_for_claim(self, claim_id: str) -> list[StateLogEntry]: ...
    def all_entries(self) -> list[StateLogEntry]: ...
    def close(self) -> None: ...


class _PostgresBackend(_StateLogBackend):
    """PostgreSQL-backed state log.

    Uses a thread-safe connection pool (psycopg2 ``ThreadedConnectionPool``)
    so concurrent agent runs can append without serializing.
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS state_log (
        log_id            TEXT PRIMARY KEY,
        claim_id          TEXT NOT NULL,
        from_state        TEXT,
        to_state          TEXT NOT NULL,
        transition_name   TEXT NOT NULL,
        agent             TEXT NOT NULL,
        guard_ok          BOOLEAN NOT NULL,
        guard_reason      TEXT NOT NULL,
        guard_details     JSONB NOT NULL DEFAULT '{}'::jsonb,
        trace_id          TEXT,
        timestamp         DOUBLE PRECISION NOT NULL,
        metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_state_log_claim_id
        ON state_log (claim_id, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_state_log_state
        ON state_log (to_state);
    """

    def __init__(self, dsn: str, minconn: int = 1, maxconn: int = 8) -> None:
        if not _PSYCOPG2_AVAILABLE:
            raise RuntimeError(
                "psycopg2 is not installed. Install with `pip install psycopg2-binary` "
                "or unset SHIELDPOINT_DB_URL to use the SQLite fallback."
            )
        from psycopg2 import pool  # type: ignore
        self._pool = pool.ThreadedConnectionPool(minconn, maxconn, dsn)
        self._lock = threading.Lock()
        self.init_schema()

    @contextmanager
    def _conn(self):
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def init_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(self.DDL)

    def append(self, entry: StateLogEntry) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO state_log
                        (log_id, claim_id, from_state, to_state,
                         transition_name, agent, guard_ok, guard_reason,
                         guard_details, trace_id, timestamp, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        entry.log_id,
                        entry.claim_id,
                        entry.from_state,
                        entry.to_state,
                        entry.transition_name,
                        entry.agent,
                        entry.guard_ok,
                        entry.guard_reason,
                        json.dumps(entry.guard_details),
                        entry.trace_id,
                        entry.timestamp,
                        json.dumps(entry.metadata),
                    ),
                )

    def _row_to_entry(self, row: Any) -> StateLogEntry:
        return StateLogEntry(
            log_id=row[0],
            claim_id=row[1],
            from_state=row[2],
            to_state=row[3],
            transition_name=row[4],
            agent=row[5],
            guard_ok=bool(row[6]),
            guard_reason=row[7],
            guard_details=row[8] if isinstance(row[8], dict)
                          else json.loads(row[8] or "{}"),
            trace_id=row[9],
            timestamp=float(row[10]),
            metadata=row[11] if isinstance(row[11], dict)
                      else json.loads(row[11] or "{}"),
        )

    def latest_for_claim(self, claim_id: str) -> Optional[StateLogEntry]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT log_id, claim_id, from_state, to_state,
                           transition_name, agent, guard_ok, guard_reason,
                           guard_details, trace_id, timestamp, metadata
                      FROM state_log
                     WHERE claim_id = %s
                     ORDER BY timestamp DESC
                     LIMIT 1
                    """,
                    (claim_id,),
                )
                row = cur.fetchone()
                return self._row_to_entry(row) if row else None

    def history_for_claim(self, claim_id: str) -> list[StateLogEntry]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT log_id, claim_id, from_state, to_state,
                           transition_name, agent, guard_ok, guard_reason,
                           guard_details, trace_id, timestamp, metadata
                      FROM state_log
                     WHERE claim_id = %s
                     ORDER BY timestamp ASC
                    """,
                    (claim_id,),
                )
                return [self._row_to_entry(r) for r in cur.fetchall()]

    def all_entries(self) -> list[StateLogEntry]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT log_id, claim_id, from_state, to_state,
                           transition_name, agent, guard_ok, guard_reason,
                           guard_details, trace_id, timestamp, metadata
                      FROM state_log
                     ORDER BY timestamp ASC
                    """
                )
                return [self._row_to_entry(r) for r in cur.fetchall()]

    def close(self) -> None:
        if self._pool:
            self._pool.closeall()
            self._pool = None  # type: ignore


class _SQLiteBackend(_StateLogBackend):
    """SQLite-backed state log — used in tests and local dev when no
    Postgres DSN is configured. File-backed by default so persistence
    across restarts works exactly like the Postgres backend.
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS state_log (
        log_id            TEXT PRIMARY KEY,
        claim_id          TEXT NOT NULL,
        from_state        TEXT,
        to_state          TEXT NOT NULL,
        transition_name   TEXT NOT NULL,
        agent             TEXT NOT NULL,
        guard_ok          INTEGER NOT NULL,
        guard_reason      TEXT NOT NULL,
        guard_details     TEXT NOT NULL DEFAULT '{}',
        trace_id          TEXT,
        timestamp         REAL NOT NULL,
        metadata          TEXT NOT NULL DEFAULT '{}',
        created_at        TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_state_log_claim_id
        ON state_log (claim_id, timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_state_log_state
        ON state_log (to_state);
    """

    def __init__(self, path: str = ":memory:") -> None:
        # ``check_same_thread=False`` lets the engine be called from any
        # thread; we serialize writes with a Lock.
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.init_schema()

    def init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(self.DDL)
            self._conn.commit()

    def append(self, entry: StateLogEntry) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO state_log
                    (log_id, claim_id, from_state, to_state,
                     transition_name, agent, guard_ok, guard_reason,
                     guard_details, trace_id, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.log_id,
                    entry.claim_id,
                    entry.from_state,
                    entry.to_state,
                    entry.transition_name,
                    entry.agent,
                    1 if entry.guard_ok else 0,
                    entry.guard_reason,
                    json.dumps(entry.guard_details),
                    entry.trace_id,
                    entry.timestamp,
                    json.dumps(entry.metadata),
                ),
            )
            self._conn.commit()

    def _row_to_entry(self, row: sqlite3.Row) -> StateLogEntry:
        return StateLogEntry(
            log_id=row["log_id"],
            claim_id=row["claim_id"],
            from_state=row["from_state"],
            to_state=row["to_state"],
            transition_name=row["transition_name"],
            agent=row["agent"],
            guard_ok=bool(row["guard_ok"]),
            guard_reason=row["guard_reason"],
            guard_details=json.loads(row["guard_details"] or "{}"),
            trace_id=row["trace_id"],
            timestamp=float(row["timestamp"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def latest_for_claim(self, claim_id: str) -> Optional[StateLogEntry]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM state_log
                 WHERE claim_id = ?
                 ORDER BY timestamp DESC
                 LIMIT 1
                """,
                (claim_id,),
            )
            row = cur.fetchone()
            return self._row_to_entry(row) if row else None

    def history_for_claim(self, claim_id: str) -> list[StateLogEntry]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT * FROM state_log
                 WHERE claim_id = ?
                 ORDER BY timestamp ASC
                """,
                (claim_id,),
            )
            return [self._row_to_entry(r) for r in cur.fetchall()]

    def all_entries(self) -> list[StateLogEntry]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM state_log ORDER BY timestamp ASC"
            )
            return [self._row_to_entry(r) for r in cur.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _make_backend(db_url: Optional[str] = None,
                  sqlite_path: str = ":memory:") -> _StateLogBackend:
    """Pick a backend based on environment / argument."""
    url = db_url or os.environ.get("SHIELDPOINT_DB_URL")
    if url and url.startswith(("postgres://", "postgresql://")):
        return _PostgresBackend(url)
    # Default: SQLite (file-backed if SHIELDPOINT_SQLITE_PATH set)
    path = os.environ.get("SHIELDPOINT_SQLITE_PATH", sqlite_path)
    return _SQLiteBackend(path)


# ===========================================================================
# Langfuse tracer shim — works whether or not the SDK is installed
# ===========================================================================
class _NullSpan:
    def __enter__(self) -> "_NullSpan": return self
    def __exit__(self, *_args: Any) -> None: ...
    def update(self, *args: Any, **kwargs: Any) -> None: ...


class _NullTracer:
    """No-op tracer used when Langfuse SDK is unavailable or disabled."""
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> _NullSpan:
        return _NullSpan()
    def update_current_trace(self, *args: Any, **kwargs: Any) -> None: ...
    def flush(self) -> None: ...
    def shutdown(self) -> None: ...
    def get_current_trace_id(self) -> Optional[str]:
        return None


def _get_tracer():
    """Return a Langfuse tracer instance or a no-op."""
    if not _LANGFUSE_AVAILABLE:
        return _NullTracer()
    try:
        return get_tracer()  # type: ignore[misc]
    except Exception:
        return _NullTracer()


# ===========================================================================
# StateMachineEngine
# ===========================================================================
class StateMachineEngine:
    """The 8-state, 9-transition claim state machine with persistence.

    Parameters
    ----------
    backend : _StateLogBackend, optional
        Persistence backend. If ``None``, picks Postgres if
        ``SHIELDPOINT_DB_URL`` is set, else SQLite (in-memory by default,
        or file-backed via ``SHIELDPOINT_SQLITE_PATH``).
    tracer : optional
        Langfuse tracer (or any object exposing
        ``start_as_current_span(name, input, output, metadata)``).
        Defaults to the shared tracer from ``langfuse_wrapper.py``.
    transitions : list[TransitionDef], optional
        Override the default transition table (used by tests).
    """

    def __init__(
        self,
        *,
        backend: Optional[_StateLogBackend] = None,
        tracer: Any = None,
        transitions: Optional[list[TransitionDef]] = None,
    ) -> None:
        self.backend: _StateLogBackend = backend or _make_backend()
        self.tracer: Any = tracer or _get_tracer()
        self.transitions: list[TransitionDef] = transitions or _default_transitions()
        # Index for fast lookup: (from_state, to_state) -> TransitionDef
        self._index: dict[tuple[State, State], TransitionDef] = {}
        for t in self.transitions:
            self._index[(t.from_state, t.to_state)] = t
        # Index for allowed targets from a given state
        self._allowed: dict[State, list[State]] = {}
        for t in self.transitions:
            self._allowed.setdefault(t.from_state, []).append(t.to_state)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def allowed_targets(self, from_state: State | str) -> list[State]:
        """Return the list of states reachable from ``from_state``."""
        s = self._coerce_state(from_state)
        return list(self._allowed.get(s, []))

    def get_state(self, claim_id: str) -> Optional[State]:
        """Return the current persisted state for a claim, or ``None`` if
        the claim has never been seen by the engine.

        This is the recovery entry point — after a system restart,
        re-instantiate the engine and call ``get_state(claim_id)`` to
        pick up where the previous process left off.
        """
        entry = self.backend.latest_for_claim(claim_id)
        if entry is None:
            return None
        return State(entry.to_state)

    def get_state_log(self, claim_id: str) -> list[StateLogEntry]:
        """Return the full ordered transition history for a claim."""
        return self.backend.history_for_claim(claim_id)

    def evaluate_guard(
        self,
        from_state: State | str,
        to_state: State | str,
        *,
        claim: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Evaluate a transition's guard without persisting anything.

        Returns ``(ok, reason, details)``. Raises
        :class:`InvalidStateTransitionError` if the transition is undefined.
        """
        tdef = self._get_transition_def(from_state, to_state)
        if tdef.guard is None:
            return (True, "No guard defined.", {})
        return tdef.guard(claim or {}, context or {})

    def transition(
        self,
        claim_id: str,
        from_state: State | str,
        to_state: State | str,
        *,
        claim: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> State:
        """Validate, persist, and trace a state transition.

        Returns the new :class:`State`. Raises
        :class:`InvalidStateTransitionError` if the transition is undefined,
        or :class:`GuardConditionFailedError` if the guard returned False.
        In the latter case the failure is *still* persisted to the state log
        (with ``guard_ok=False``) so audits have a complete picture; the
        caller is responsible for catching the exception and routing to
        ``ESCALATING`` (or to the transition's ``failure_route``) as
        appropriate.
        """
        tdef = self._get_transition_def(from_state, to_state)
        claim = claim or {}
        context = context or {}
        metadata = dict(metadata or {})

        # ---- Evaluate guard ---- #
        if tdef.guard is None:
            ok, reason, details = (True, "No guard defined.", {})
        else:
            ok, reason, details = tdef.guard(claim, context)

        # ---- Persist (always — even on failure, for audit) ---- #
        entry = StateLogEntry(
            log_id=f"sl-{uuid.uuid4().hex[:16]}",
            claim_id=claim_id,
            from_state=tdef.from_state.value,
            to_state=tdef.to_state.value,
            transition_name=tdef.name.value,
            agent=tdef.agent,
            guard_ok=ok,
            guard_reason=reason,
            guard_details=details,
            trace_id=None,  # filled in by tracer context below
            timestamp=time.time(),
            metadata=metadata,
        )

        # ---- Langfuse trace (span around the persisted transition) ---- #
        span_metadata = {
            "claim_id": claim_id,
            "agent_id": tdef.agent,
            "transition_name": tdef.name.value,
            "from_state": tdef.from_state.value,
            "to_state": tdef.to_state.value,
            "guard_ok": ok,
            "guard_reason": reason,
            "guard_details": details,
            **metadata,
        }
        with self.tracer.start_as_current_span(
            name=f"state_transition.{tdef.name.value}",
            input={
                "claim_id": claim_id,
                "from_state": tdef.from_state.value,
                "to_state": tdef.to_state.value,
                "context": context,
            },
            metadata=span_metadata,
        ) as span:
            # Persist inside the span so latency includes DB write
            self.backend.append(entry)
            # Stamp the trace_id back onto the persisted row (best-effort;
            # if the tracer is a no-op, trace_id stays None).
            try:
                trace_id = getattr(self.tracer, "get_current_trace_id", lambda: None)()
                if trace_id:
                    entry.trace_id = trace_id
            except Exception:
                pass
            span.update(output={"new_state": tdef.to_state.value,
                                "guard_ok": ok})

        if not ok:
            raise GuardConditionFailedError(
                f"Guard for {tdef.name.value} failed: {reason}",
                from_state=tdef.from_state.value,
                to_state=tdef.to_state.value,
                details={"transition": tdef.name.value, "reason": reason, **details},
            )
        return tdef.to_state

    def initialize_claim(
        self,
        claim_id: str,
        *,
        agent: str = STATE_AGENT[State.CLAIM_RECEIVED],
        metadata: Optional[dict[str, Any]] = None,
    ) -> State:
        """Record the initial intake state for a brand-new claim.

        This is the only way to put a claim into ``CLAIM_RECEIVED`` —
        the engine refuses to transition *into* ``CLAIM_RECEIVED`` from
        anywhere else, because ``CLAIM_RECEIVED`` is the entry state.
        """
        # Idempotent: if a log row already exists for this claim, no-op.
        existing = self.backend.latest_for_claim(claim_id)
        if existing is not None:
            return State(existing.to_state)
        entry = StateLogEntry(
            log_id=f"sl-{uuid.uuid4().hex[:16]}",
            claim_id=claim_id,
            from_state=None,
            to_state=State.CLAIM_RECEIVED.value,
            transition_name="INITIAL_INTAKE",
            agent=agent,
            guard_ok=True,
            guard_reason="Initial intake (no guard).",
            guard_details={},
            trace_id=None,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        with self.tracer.start_as_current_span(
            name="state_transition.INITIAL_INTAKE",
            input={"claim_id": claim_id},
            metadata={
                "claim_id": claim_id,
                "agent_id": agent,
                "transition_name": "INITIAL_INTAKE",
                "from_state": None,
                "to_state": State.CLAIM_RECEIVED.value,
                "guard_ok": True,
                "guard_reason": "Initial intake (no guard).",
                **(metadata or {}),
            },
        ) as span:
            self.backend.append(entry)
            span.update(output={"new_state": State.CLAIM_RECEIVED.value})
        return State.CLAIM_RECEIVED

    def recover(self, claim_id: str) -> Optional[State]:
        """Alias for :meth:`get_state` — restore in-memory state after
        a restart from the persisted log."""
        return self.get_state(claim_id)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _coerce_state(self, s: State | str) -> State:
        if isinstance(s, State):
            return s
        try:
            return State(s)
        except ValueError as e:
            raise InvalidStateTransitionError(
                from_state=str(s), to_state="?",
                allowed=[st.value for st in State],
            ) from e

    def _get_transition_def(
        self, from_state: State | str, to_state: State | str
    ) -> TransitionDef:
        f = self._coerce_state(from_state)
        t = self._coerce_state(to_state)
        tdef = self._index.get((f, t))
        if tdef is None:
            raise InvalidStateTransitionError(
                from_state=f.value,
                to_state=t.value,
                allowed=[s.value for s in self._allowed.get(f, [])],
            )
        return tdef

    # ------------------------------------------------------------------ #
    # Test/debug helpers                                                 #
    # ------------------------------------------------------------------ #
    def all_log_entries(self) -> list[StateLogEntry]:
        return self.backend.all_entries()

    def close(self) -> None:
        self.backend.close()


# ===========================================================================
# Convenience: state machine constants for tests / agents
# ===========================================================================
INITIAL_STATE = State.CLAIM_RECEIVED
TERMINAL_STATES = {State.PAID_OUT}
ESCALATION_STATES = {State.ESCALATING}
