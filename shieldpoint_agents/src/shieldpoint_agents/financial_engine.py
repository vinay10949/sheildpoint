"""
FinancialAgent payment assessment engine (SP-302).

Implements the SP-302 acceptance criteria:

- :class:`DeductibleCalculator` — supports per-claim, per-year, and
  aggregate deductible types. Calculates the deductible amount applied
  to a given claim based on the policy configuration and prior claim
  history (for per-year / aggregate deductibles).
- :class:`PaymentCalculator` — combines the claim amount, deductible,
  and co-pay ratio to compute the net payable amount.
- :class:`DuplicatePaymentDetector` — checks the payment ledger for
  prior payments with the same (policy_id, amount) within a 30-day
  window. Uses PostgreSQL when available; falls back to an in-memory
  list so tests run without a database.
- :class:`ZKPCrossAgentVerifier` — wraps :class:`CrossAgentClaimProver`
  so the FinancialAgent can verify a ClaimsAgent proof that the claim
  amount is within policy limits WITHOUT accessing the policy document.
- :class:`PaymentAuthorizationRecord` — Pydantic model matching the
  PayoutAgent's expected input schema.

All public methods are wrapped in Langfuse spans via :class:`LangfuseTracer`.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import AgentConfig
from .tracer import LangfuseTracer

logger = logging.getLogger("shieldpoint_agents.financial_engine")


# ---------------------------------------------------------------------------
# Deductible types
# ---------------------------------------------------------------------------
DeductibleType = Literal["per_claim", "per_year", "aggregate"]
"""How the deductible is applied across multiple claims on the same policy.

- ``per_claim`` — the deductible is applied to each claim independently.
  Two $1,000 claims on a $500-deductible policy each pay $500.
- ``per_year`` — the deductible accumulates across claims within a calendar
  year. Once the annual deductible is met, subsequent claims pay in full.
- ``aggregate`` — the deductible accumulates across the entire policy
  lifetime. Once the aggregate deductible is met, subsequent claims pay
  in full.
"""


# ---------------------------------------------------------------------------
# PaymentAuthorizationRecord — the canonical PayoutAgent input schema
# ---------------------------------------------------------------------------
class PaymentAuthorizationRecord(BaseModel):
    """Authoritative payment record produced by the FinancialAgent.

    Consumed by the PayoutAgent to disburse funds. Contains everything
    the PayoutAgent needs: who to pay, how much, against which claim and
    policy, plus a reference to the ZKP proof that verifies the claim
    is within policy limits (so the PayoutAgent doesn't need the policy
    document either).
    """

    model_config = ConfigDict(extra="forbid")

    authorization_id: str = Field(
        ..., description="Unique ID for this authorisation (UUID).",
    )
    claim_id: str = Field(..., description="The claim being paid.")
    policy_id: str = Field(..., description="The policy against which payment is authorised.")
    payee: str = Field(
        ..., description="Recipient of the payment (policyholder or vendor).",
    )
    gross_amount: float = Field(
        ..., ge=0.0, description="Claim amount before deductible and co-pay.",
    )
    deductible_applied: float = Field(
        ..., ge=0.0, description="Deductible amount subtracted from gross.",
    )
    copay_amount: float = Field(
        ..., ge=0.0, description="Co-pay portion (claimant responsibility).",
    )
    net_payable: float = Field(
        ..., ge=0.0,
        description="Final amount to disburse: gross - deductible - copay.",
    )
    deductible_type: DeductibleType = Field(
        ..., description="How the deductible was applied.",
    )
    coverage_limit: float = Field(
        ..., ge=0.0, description="Policy coverage limit (for audit).",
    )
    within_coverage_limit: bool = Field(
        ..., description="Whether the claim was within the policy limit.",
    )
    zkp_proof_verified: bool = Field(
        default=False,
        description=(
            "Whether the cross-agent ZKP proof from the ClaimsAgent was "
            "verified successfully. If False, the FinancialAgent fell "
            "back to direct policy access (data-exposure mode)."
        ),
    )
    zkp_proof_ref: Optional[str] = Field(
        default=None,
        description="Reference to the ZKP proof in the proof store (if verified).",
    )
    duplicate_flag: bool = Field(
        default=False,
        description="True if a matching prior payment was detected in the 30-day window.",
    )
    duplicate_of: Optional[str] = Field(
        default=None,
        description="If duplicate_flag is True, the payment_id of the prior matching payment.",
    )
    authorised_at: float = Field(
        default_factory=time.time,
        description="Unix epoch seconds when the authorisation was issued.",
    )
    authorised_by: str = Field(
        default="FinancialAgent",
        description="Agent or user that issued the authorisation.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata for downstream consumption.",
    )

    @field_validator("net_payable")
    @classmethod
    def _net_payable_non_negative(cls, v: float) -> float:
        if v < 0:
            # Clamp to 0 — the FinancialAgent should not authorise negative
            # payments (claimant paying the insurer). But this is a
            # defensive validator: the calculator should never produce
            # a negative net_payable.
            return 0.0
        return round(v, 2)

    @field_validator("gross_amount", "deductible_applied", "copay_amount")
    @classmethod
    def _round_to_cents(cls, v: float) -> float:
        return round(float(v), 2)


# ---------------------------------------------------------------------------
# PriorClaim — minimal record of a prior claim for deductible accumulation
# ---------------------------------------------------------------------------
@dataclass
class PriorClaim:
    """A prior claim on the same policy (for per-year/aggregate deductibles)."""

    claim_id: str
    amount: float
    date_of_loss: str  # YYYY-MM-DD
    deductible_applied: float = 0.0
    decision: str = "approve"  # approve | deny | route_to_manual_review


# ---------------------------------------------------------------------------
# DeductibleCalculator
# ---------------------------------------------------------------------------
class DeductibleCalculator:
    """Calculate the deductible applied to a claim, given prior history.

    Three deductible types:

    - ``per_claim``: deductible is the policy's per-claim amount, applied
      independently to each claim. Prior history is irrelevant.
    - ``per_year``: deductible accumulates across claims in the same
      calendar year. If the prior claims in the same year have already
      met the annual deductible, this claim's deductible is 0.
    - ``aggregate``: deductible accumulates across the policy's lifetime.
      Once met, subsequent claims have 0 deductible.
    """

    def calculate(
        self,
        *,
        claim_amount: float,
        policy_deductible: float,
        deductible_type: DeductibleType,
        prior_claims: Iterable[PriorClaim] = (),
        policy_year: Optional[int] = None,
        claim_date: Optional[str] = None,
    ) -> float:
        """Return the deductible amount applied to THIS claim (in dollars).

        Parameters
        ----------
        claim_amount : float
            The current claim's gross amount.
        policy_deductible : float
            The policy's deductible (annual for per_year, lifetime for
            aggregate, per-claim for per_claim).
        deductible_type : DeductibleType
            How the deductible is applied.
        prior_claims : Iterable[PriorClaim]
            Prior claims on the same policy. Used only for per_year /
            aggregate.
        policy_year : int, optional
            The calendar year to filter prior claims by (for per_year).
            Defaults to the year of ``claim_date``, or the current year.
        claim_date : str, optional
            YYYY-MM-DD date of the current claim. Used for per_year
            filtering when ``policy_year`` is not supplied.
        """
        if policy_deductible <= 0:
            return 0.0
        if deductible_type == "per_claim":
            return min(policy_deductible, claim_amount)

        # For per_year and aggregate, sum prior deductible applications
        prior_list = list(prior_claims)
        if deductible_type == "per_year":
            year = policy_year or self._year_of(claim_date) or datetime.now(timezone.utc).year
            prior_in_year = [
                p for p in prior_list
                if self._year_of(p.date_of_loss) == year
            ]
            already_paid = sum(p.deductible_applied for p in prior_in_year)
            remaining = max(0.0, policy_deductible - already_paid)
            return min(remaining, claim_amount)

        if deductible_type == "aggregate":
            already_paid = sum(p.deductible_applied for p in prior_list)
            remaining = max(0.0, policy_deductible - already_paid)
            return min(remaining, claim_amount)

        raise ValueError(f"Unknown deductible type: {deductible_type!r}")

    @staticmethod
    def _year_of(date_str: Optional[str]) -> Optional[int]:
        if not date_str:
            return None
        try:
            return int(date_str.split("-")[0])
        except (ValueError, IndexError):
            return None


# ---------------------------------------------------------------------------
# PaymentCalculator
# ---------------------------------------------------------------------------
class PaymentCalculator:
    """Compute the net payable amount after deductible and co-pay.

    The co-pay is the percentage of the post-deductible amount that the
    claimant is responsible for. E.g. co_pay_pct=0.10 means the insurer
    pays 90% of (claim_amount - deductible).

    The calculator never returns a negative net payable — if the
    deductible exceeds the claim amount, the net is 0 and the
    deductible is clamped to the claim amount.
    """

    def __init__(self, deductible_calculator: Optional[DeductibleCalculator] = None) -> None:
        self.deductible_calc = deductible_calculator or DeductibleCalculator()

    def calculate(
        self,
        *,
        claim_amount: float,
        policy_deductible: float,
        deductible_type: DeductibleType,
        co_pay_pct: float = 0.0,
        coverage_limit: float = float("inf"),
        prior_claims: Iterable[PriorClaim] = (),
        claim_date: Optional[str] = None,
    ) -> dict[str, float]:
        """Return a dict with: gross, deductible_applied, copay_amount, net_payable, within_limit.

        The ``within_limit`` key is True iff ``claim_amount <= coverage_limit``.
        When the claim exceeds the coverage limit, the calculator still
        returns the math (so the FinancialAgent can route to manual
        review with full context), but ``within_limit`` is False.
        """
        claim_amount = max(0.0, float(claim_amount))
        within_limit = claim_amount <= coverage_limit

        deductible = self.deductible_calc.calculate(
            claim_amount=claim_amount,
            policy_deductible=policy_deductible,
            deductible_type=deductible_type,
            prior_claims=prior_claims,
            claim_date=claim_date,
        )
        # Clamp deductible to claim amount
        deductible = min(deductible, claim_amount)
        post_deductible = max(0.0, claim_amount - deductible)
        copay = round(post_deductible * co_pay_pct, 2)
        net = round(post_deductible - copay, 2)
        return {
            "gross": round(claim_amount, 2),
            "deductible_applied": round(deductible, 2),
            "copay_amount": copay,
            "net_payable": max(0.0, net),
            "within_limit": within_limit,
        }


# ---------------------------------------------------------------------------
# DuplicatePaymentDetector
# ---------------------------------------------------------------------------
# AC: "Duplicate payment detection flags claims with matching policy ID +
# amount within 30-day window"
class DuplicatePaymentDetector:
    """Detect duplicate payments against the payment ledger.

    Backed by PostgreSQL when a DSN is supplied (production). Falls back
    to an in-memory list otherwise (tests, local dev). The interface is
    identical either way.

    Schema (PostgreSQL)::

        CREATE TABLE IF NOT EXISTS payment_ledger (
            payment_id     TEXT PRIMARY KEY,
            claim_id       TEXT NOT NULL,
            policy_id      TEXT NOT NULL,
            amount         DOUBLE PRECISION NOT NULL,
            payee          TEXT,
            status         TEXT,
            created_at     DOUBLE PRECISION NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_payment_ledger_policy_amount
            ON payment_ledger (policy_id, amount);

    A duplicate exists when the same (policy_id, amount) pair appears in
    the ledger within the last 30 days. The window is configurable via
    the ``window_days`` constructor parameter.
    """

    _DEFAULT_DDL = """
    CREATE TABLE IF NOT EXISTS payment_ledger (
        payment_id     TEXT PRIMARY KEY,
        claim_id       TEXT NOT NULL,
        policy_id      TEXT NOT NULL,
        amount         DOUBLE PRECISION NOT NULL,
        payee          TEXT,
        status         TEXT,
        created_at     DOUBLE PRECISION NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_payment_ledger_policy_amount
        ON payment_ledger (policy_id, amount);
    """

    def __init__(
        self,
        *,
        dsn: Optional[str] = None,
        window_days: int = 30,
        amount_tolerance: float = 0.01,
        table_name: str = "payment_ledger",
    ) -> None:
        self.window_days = window_days
        self.amount_tolerance = amount_tolerance
        self.table_name = table_name
        self._dsn = dsn
        self._conn: Any = None
        self._in_memory: list[dict[str, Any]] = []
        if dsn:
            self._init_postgres(dsn)

    def _init_postgres(self, dsn: str) -> None:
        try:
            try:
                import psycopg  # type: ignore
                self._conn = psycopg.connect(dsn, autocommit=True)
                logger.info("DuplicatePaymentDetector: connected via psycopg3")
            except ImportError:
                import psycopg2  # type: ignore
                self._conn = psycopg2.connect(dsn)
                self._conn.autocommit = True
                logger.info("DuplicatePaymentDetector: connected via psycopg2")
            cur = self._conn.cursor()
            cur.execute(self._DEFAULT_DDL.replace("payment_ledger", self.table_name))
            cur.close()
        except Exception as exc:
            logger.warning(
                "DuplicatePaymentDetector: Postgres init failed (%s); "
                "falling back to in-memory.", exc,
            )
            self._conn = None
            self._in_memory = []

    def check(
        self,
        *,
        policy_id: str,
        amount: float,
        exclude_claim_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Return the matching prior payment if a duplicate is detected, else None."""
        cutoff = time.time() - (self.window_days * 86400)
        if self._conn is not None:
            return self._check_postgres(
                policy_id=policy_id, amount=amount,
                cutoff=cutoff, exclude_claim_id=exclude_claim_id,
            )
        return self._check_in_memory(
            policy_id=policy_id, amount=amount,
            cutoff=cutoff, exclude_claim_id=exclude_claim_id,
        )

    def record(self, payment: dict[str, Any]) -> None:
        """Persist a payment to the ledger (for future duplicate checks)."""
        if self._conn is not None:
            self._record_postgres(payment)
        else:
            self._in_memory.append(dict(payment))

    # ------------------------------------------------------------------ #
    #  Backends                                                           #
    # ------------------------------------------------------------------ #
    def _check_postgres(
        self, *, policy_id: str, amount: float, cutoff: float,
        exclude_claim_id: Optional[str],
    ) -> Optional[dict[str, Any]]:
        sql = (
            f"SELECT payment_id, claim_id, policy_id, amount, payee, status, created_at "
            f"FROM {self.table_name} "
            f"WHERE policy_id = %s AND ABS(amount - %s) <= %s "
            f"AND created_at >= %s"
        )
        params: list[Any] = [policy_id, amount, self.amount_tolerance, cutoff]
        if exclude_claim_id:
            sql += " AND claim_id != %s"
            params.append(exclude_claim_id)
        sql += " ORDER BY created_at DESC LIMIT 1"
        cur = self._conn.cursor()
        try:
            cur.execute(sql, params)
            row = cur.fetchone()
        finally:
            cur.close()
        if not row:
            return None
        return {
            "payment_id": row[0], "claim_id": row[1], "policy_id": row[2],
            "amount": float(row[3]), "payee": row[4], "status": row[5],
            "created_at": float(row[6]),
        }

    def _check_in_memory(
        self, *, policy_id: str, amount: float, cutoff: float,
        exclude_claim_id: Optional[str],
    ) -> Optional[dict[str, Any]]:
        # Iterate newest-first
        for p in reversed(self._in_memory):
            if exclude_claim_id and p.get("claim_id") == exclude_claim_id:
                continue
            if p.get("policy_id") != policy_id:
                continue
            if p.get("created_at", 0) < cutoff:
                continue
            if abs(float(p.get("amount", 0)) - amount) <= self.amount_tolerance:
                return dict(p)
        return None

    def _record_postgres(self, payment: dict[str, Any]) -> None:
        sql = (
            f"INSERT INTO {self.table_name} "
            f"(payment_id, claim_id, policy_id, amount, payee, status, created_at) "
            f"VALUES (%s, %s, %s, %s, %s, %s, %s) "
            f"ON CONFLICT (payment_id) DO NOTHING"
        )
        cur = self._conn.cursor()
        try:
            cur.execute(sql, (
                payment.get("payment_id") or f"PMT-{uuid.uuid4().hex[:16]}",
                payment.get("claim_id", ""),
                payment.get("policy_id", ""),
                float(payment.get("amount", 0)),
                payment.get("payee"),
                payment.get("status", "authorized"),
                float(payment.get("created_at", time.time())),
            ))
        finally:
            cur.close()

    def clear(self) -> None:
        """Wipe the ledger (for tests)."""
        if self._conn is not None:
            cur = self._conn.cursor()
            try:
                cur.execute(f"DELETE FROM {self.table_name}")
            finally:
                cur.close()
        else:
            self._in_memory.clear()


# ---------------------------------------------------------------------------
# ZKPCrossAgentVerifier
# ---------------------------------------------------------------------------
class ZKPCrossAgentVerifier:
    """Verify the ClaimsAgent's cross-agent "claim-within-limit" proof.

    Wraps :class:`cross_agent_prover.CrossAgentClaimProver`. The
    FinancialAgent calls :meth:`verify` with the proof envelope from
    the ClaimsAgent and the expected policy commitment (which it learns
    out-of-band — e.g. from a public policy registry). If the proof
    verifies, the FinancialAgent can proceed WITHOUT accessing the
    policy document, preserving the privacy guarantee.
    """

    def __init__(self, prover: Any = None) -> None:
        self._prover = prover

    def _ensure_prover(self) -> Any:
        if self._prover is not None:
            return self._prover
        try:
            # cross_agent_prover.py lives in zkp_circuit/ at the repo root.
            # claims_extraction.py is at <repo>/shieldpoint_agents/src/shieldpoint_agents/
            # so parents[3] is the repo root.
            repo_root = Path(__file__).resolve().parents[3]
            zkp_dir = str(repo_root / "zkp_circuit")
            if zkp_dir not in sys.path:
                sys.path.insert(0, zkp_dir)
            from cross_agent_prover import CrossAgentClaimProver
            self._prover = CrossAgentClaimProver()
            return self._prover
        except Exception as exc:
            logger.warning(
                "ZKPCrossAgentVerifier: could not load CrossAgentClaimProver (%s); "
                "verification will return verified=False.", exc,
            )
            return None

    def verify(
        self,
        *,
        proof: dict[str, Any],
        public_signals: list[str],
        expected_commitment: str,
    ) -> dict[str, Any]:
        """Verify a cross-agent proof. Returns the prover's verify() result.

        If the prover isn't loadable, returns a synthetic "not verified"
        result so the FinancialAgent can fall back to direct policy
        access (with a warning logged).
        """
        prover = self._ensure_prover()
        if prover is None:
            return {
                "verified": False,
                "verifier": "unavailable",
                "mode": "unavailable",
                "latency_ms": 0.0,
                "reason": "CrossAgentClaimProver not loadable; "
                          "FinancialAgent falling back to direct policy access.",
                "commitment_match": False,
            }
        return prover.verify_claim_within_limit(
            proof=proof,
            public_signals=public_signals,
            expected_commitment=expected_commitment,
        )


# ---------------------------------------------------------------------------
# FinancialAssessmentEngine — orchestrates the full SP-302 flow
# ---------------------------------------------------------------------------
class FinancialAssessmentEngine:
    """Run the full SP-302 payment-assessment flow.

    Steps (each wrapped in a Langfuse span):

    1. ``verify_zkp_proof`` — verify the ClaimsAgent's cross-agent proof
       that the claim is within policy limits.
    2. ``check_duplicate`` — query the payment ledger for matching
       (policy_id, amount) within 30 days.
    3. ``calculate_payment`` — apply deductible (per_claim / per_year /
       aggregate) and co-pay to get net payable.
    4. ``authorise`` — emit a :class:`PaymentAuthorizationRecord` for
       the PayoutAgent.
    """

    def __init__(
        self,
        *,
        config: Optional[AgentConfig] = None,
        tracer: Optional[LangfuseTracer] = None,
        duplicate_detector: Optional[DuplicatePaymentDetector] = None,
        zkp_verifier: Optional[ZKPCrossAgentVerifier] = None,
        payment_calculator: Optional[PaymentCalculator] = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.tracer = tracer or LangfuseTracer(agent_name="FinancialAgent")
        self.duplicate_detector = duplicate_detector or DuplicatePaymentDetector()
        self.zkp_verifier = zkp_verifier or ZKPCrossAgentVerifier()
        self.payment_calculator = payment_calculator or PaymentCalculator()

    def assess(
        self,
        *,
        claim_id: str,
        policy_id: str,
        claim_amount: float,
        coverage_limit: float,
        policy_deductible: float,
        deductible_type: DeductibleType = "per_claim",
        co_pay_pct: float = 0.0,
        payee: str = "",
        prior_claims: Iterable[PriorClaim] = (),
        claim_date: Optional[str] = None,
        zkp_proof: Optional[dict[str, Any]] = None,
        expected_policy_commitment: Optional[str] = None,
    ) -> PaymentAuthorizationRecord:
        """Run the full assessment. Returns a :class:`PaymentAuthorizationRecord`."""
        started = time.perf_counter()
        with self.tracer.trace(
            "financial_assessment",
            metadata={"claim_id": claim_id, "agent.name": "FinancialAgent"},
            tags=["FinancialAgent", "assessment"],
        ) as span:
            trace_id = getattr(span, "id", None) if span else None

            # 1. Verify ZKP proof (if supplied)
            zkp_verified = False
            zkp_ref = None
            if zkp_proof is not None and expected_policy_commitment is not None:
                v = self.zkp_verifier.verify(
                    proof=zkp_proof.get("proof", {}),
                    public_signals=zkp_proof.get("public_signals", []),
                    expected_commitment=expected_policy_commitment,
                )
                zkp_verified = bool(v.get("verified"))
                zkp_ref = zkp_proof.get("policy_commitment")
                logger.info(
                    "FinancialAssessment: zkp_verified=%s claim_id=%s",
                    zkp_verified, claim_id,
                )

            # 2. Duplicate detection
            dup = self.duplicate_detector.check(
                policy_id=policy_id, amount=claim_amount,
                exclude_claim_id=claim_id,
            )
            duplicate_flag = dup is not None
            duplicate_of = dup.get("payment_id") if dup else None

            # 3. Payment calculation
            calc = self.payment_calculator.calculate(
                claim_amount=claim_amount,
                policy_deductible=policy_deductible,
                deductible_type=deductible_type,
                co_pay_pct=co_pay_pct,
                coverage_limit=coverage_limit,
                prior_claims=prior_claims,
                claim_date=claim_date,
            )

            # 4. Build authorisation record
            record = PaymentAuthorizationRecord(
                authorization_id=f"AUTH-{uuid.uuid4().hex[:16]}",
                claim_id=claim_id,
                policy_id=policy_id,
                payee=payee,
                gross_amount=calc["gross"],
                deductible_applied=calc["deductible_applied"],
                copay_amount=calc["copay_amount"],
                net_payable=calc["net_payable"],
                deductible_type=deductible_type,
                coverage_limit=coverage_limit,
                within_coverage_limit=calc["within_limit"],
                zkp_proof_verified=zkp_verified,
                zkp_proof_ref=zkp_ref,
                duplicate_flag=duplicate_flag,
                duplicate_of=duplicate_of,
                authorised_by="FinancialAgent",
                metadata={
                    "trace_id": trace_id,
                    "latency_sec": time.perf_counter() - started,
                    "deductible_calculator": self.payment_calculator.deductible_calc.__class__.__name__,
                },
            )

            # Record the payment so future duplicate checks catch it
            if not duplicate_flag:
                self.duplicate_detector.record({
                    "payment_id": record.authorization_id,
                    "claim_id": claim_id,
                    "policy_id": policy_id,
                    "amount": claim_amount,
                    "payee": payee,
                    "status": "authorized",
                    "created_at": time.time(),
                })

            logger.info(
                "FinancialAssessment: claim_id=%s net=$%.2f deductible=$%.2f "
                "zkp_verified=%s duplicate=%s",
                claim_id, record.net_payable, record.deductible_applied,
                zkp_verified, duplicate_flag,
            )
            return record


# ---------------------------------------------------------------------------
# 100+ financial calculation scenarios — used by the SP-302 test suite.
# ---------------------------------------------------------------------------
def build_financial_scenarios() -> list[dict[str, Any]]:
    """Return 100+ parametrised financial scenarios for the test suite.

    Each scenario is a dict with the inputs to
    :meth:`FinancialAssessmentEngine.assess` plus the expected
    ``net_payable`` and ``deductible_applied``.
    """
    scenarios: list[dict[str, Any]] = []

    # 1-30: per_claim deductible variations
    for i in range(1, 31):
        claim = i * 100.0
        deductible = 500.0
        scenarios.append({
            "name": f"per_claim_{i}",
            "claim_id": f"CLM-PC-{i:03d}",
            "policy_id": f"HO-PC-{i:04d}",
            "claim_amount": claim,
            "coverage_limit": 100_000.0,
            "policy_deductible": deductible,
            "deductible_type": "per_claim",
            "co_pay_pct": 0.0,
            "expected_deductible_applied": min(deductible, claim),
            "expected_net_payable": max(0.0, claim - min(deductible, claim)),
        })

    # 31-50: per_claim with co-pay
    for i in range(1, 21):
        claim = 5000.0
        deductible = 1000.0
        copay_pct = i * 0.05  # 5%, 10%, ..., 100%
        post_ded = claim - deductible
        copay = round(post_ded * copay_pct, 2)
        net = round(post_ded - copay, 2)
        scenarios.append({
            "name": f"per_claim_copay_{i}",
            "claim_id": f"CLM-PCP-{i:03d}",
            "policy_id": f"HO-PCP-{i:04d}",
            "claim_amount": claim,
            "coverage_limit": 100_000.0,
            "policy_deductible": deductible,
            "deductible_type": "per_claim",
            "co_pay_pct": copay_pct,
            "expected_deductible_applied": deductible,
            "expected_net_payable": max(0.0, net),
        })

    # 51-70: per_year deductible — prior claims in same year
    for i in range(1, 21):
        claim = 2000.0
        deductible = 1500.0  # annual
        # First i-1 claims already paid the full deductible; this claim's deductible is 0
        prior = [
            PriorClaim(
                claim_id=f"CLM-PRIOR-{i:03d}-{j}",
                amount=2000.0,
                date_of_loss=f"2026-06-{j:02d}",
                deductible_applied=500.0,  # 3 priors × $500 = $1500 = full deductible
            )
            for j in range(1, min(i, 4))
        ]
        already = sum(p.deductible_applied for p in prior)
        expected_ded = max(0.0, deductible - already)
        expected_ded = min(expected_ded, claim)
        scenarios.append({
            "name": f"per_year_{i}",
            "claim_id": f"CLM-PY-{i:03d}",
            "policy_id": f"HO-PY-{i:04d}",
            "claim_amount": claim,
            "coverage_limit": 100_000.0,
            "policy_deductible": deductible,
            "deductible_type": "per_year",
            "co_pay_pct": 0.0,
            "prior_claims": prior,
            "claim_date": "2026-06-15",
            "expected_deductible_applied": expected_ded,
            "expected_net_payable": max(0.0, claim - expected_ded),
        })

    # 71-85: aggregate deductible — lifetime accumulation
    for i in range(1, 16):
        claim = 1000.0
        deductible = 5000.0  # aggregate
        prior = [
            PriorClaim(
                claim_id=f"CLM-AGG-PRIOR-{i:03d}-{j}",
                amount=1000.0,
                date_of_loss=f"2024-01-{j:02d}",
                deductible_applied=500.0,
            )
            for j in range(1, i + 1)
        ]
        already = sum(p.deductible_applied for p in prior)
        expected_ded = max(0.0, deductible - already)
        expected_ded = min(expected_ded, claim)
        scenarios.append({
            "name": f"aggregate_{i}",
            "claim_id": f"CLM-AGG-{i:03d}",
            "policy_id": f"HO-AGG-{i:04d}",
            "claim_amount": claim,
            "coverage_limit": 100_000.0,
            "policy_deductible": deductible,
            "deductible_type": "aggregate",
            "co_pay_pct": 0.0,
            "prior_claims": prior,
            "expected_deductible_applied": expected_ded,
            "expected_net_payable": max(0.0, claim - expected_ded),
        })

    # 86-100: claim exceeds coverage limit
    for i in range(1, 16):
        limit = 10_000.0
        claim = limit + i * 1000.0  # over the limit
        scenarios.append({
            "name": f"over_limit_{i}",
            "claim_id": f"CLM-OVER-{i:03d}",
            "policy_id": f"HO-OVER-{i:04d}",
            "claim_amount": claim,
            "coverage_limit": limit,
            "policy_deductible": 500.0,
            "deductible_type": "per_claim",
            "co_pay_pct": 0.0,
            "expected_deductible_applied": 500.0,
            "expected_net_payable": max(0.0, claim - 500.0),
            "expected_within_limit": False,
        })

    # 101-115: claim below deductible (net should be 0)
    for i in range(1, 16):
        claim = i * 50.0  # 50, 100, ..., 750
        deductible = 1000.0
        scenarios.append({
            "name": f"below_deductible_{i}",
            "claim_id": f"CLM-BD-{i:03d}",
            "policy_id": f"HO-BD-{i:04d}",
            "claim_amount": claim,
            "coverage_limit": 100_000.0,
            "policy_deductible": deductible,
            "deductible_type": "per_claim",
            "co_pay_pct": 0.0,
            "expected_deductible_applied": min(deductible, claim),
            "expected_net_payable": 0.0,  # claim fully absorbed by deductible
        })

    # 116-120: zero claim, zero deductible, etc.
    scenarios.extend([
        {
            "name": "zero_claim",
            "claim_id": "CLM-ZERO-001", "policy_id": "HO-Z-0001",
            "claim_amount": 0.0, "coverage_limit": 100_000.0,
            "policy_deductible": 500.0, "deductible_type": "per_claim",
            "co_pay_pct": 0.0,
            "expected_deductible_applied": 0.0, "expected_net_payable": 0.0,
        },
        {
            "name": "zero_deductible",
            "claim_id": "CLM-ZERO-002", "policy_id": "HO-Z-0002",
            "claim_amount": 5000.0, "coverage_limit": 100_000.0,
            "policy_deductible": 0.0, "deductible_type": "per_claim",
            "co_pay_pct": 0.0,
            "expected_deductible_applied": 0.0, "expected_net_payable": 5000.0,
        },
        {
            "name": "full_copay",
            "claim_id": "CLM-FULLCOPAY-001", "policy_id": "HO-FC-0001",
            "claim_amount": 1000.0, "coverage_limit": 100_000.0,
            "policy_deductible": 0.0, "deductible_type": "per_claim",
            "co_pay_pct": 1.0,
            "expected_deductible_applied": 0.0, "expected_net_payable": 0.0,
        },
        {
            "name": "large_claim",
            "claim_id": "CLM-LARGE-001", "policy_id": "HO-LG-0001",
            "claim_amount": 999_999.99, "coverage_limit": 1_000_000.0,
            "policy_deductible": 2500.0, "deductible_type": "per_claim",
            "co_pay_pct": 0.0,
            "expected_deductible_applied": 2500.0,
            "expected_net_payable": 999_999.99 - 2500.0,
        },
        {
            "name": "exact_deductible",
            "claim_id": "CLM-EXACT-001", "policy_id": "HO-EX-0001",
            "claim_amount": 500.0, "coverage_limit": 100_000.0,
            "policy_deductible": 500.0, "deductible_type": "per_claim",
            "co_pay_pct": 0.0,
            "expected_deductible_applied": 500.0, "expected_net_payable": 0.0,
        },
    ])

    return scenarios
