"""
SP-405 — Enhanced Payment Ledger with Duplicate Detection
==========================================================

The payment ledger tracks all ACH payments made by the PayoutAgent.
It supports:

- **Idempotency**: every payment is keyed by an idempotency key
  (``payout-{claim_id}``) so retries don't cause double-payment.
- **Duplicate detection**: before initiating a new payment, the ledger
  is queried for any existing payment with the same claim_id or
  idempotency key. If found, the payment is skipped and the existing
  record is returned.
- **Audit trail**: every payment record includes the full breakdown
  (gross, deductible, co-pay, net) and the ACH reference for
  reconciliation.

In production, the ledger is backed by PostgreSQL (the
``payment_ledger`` table). In tests, the :class:`InMemoryPaymentLedger`
provides the same interface without external dependencies.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger("shieldpoint.payout.ledger")


@dataclass(frozen=True)
class PaymentRecord:
    """Immutable record of a single ACH payment.

    Attributes
    ----------
    payment_id : str
        Internal payment ID (``PMT-XXXXXXXXXXXX``).
    claim_id : str
        The claim this payment settles.
    policy_id : str
        The policy under which the claim was filed.
    payee : str
        Name of the payment recipient.
    gross_amount : float
        Original claim amount (before deductible / co-pay).
    deductible_applied : float
        Deductible amount subtracted from the gross.
    copay_amount : float
        Co-pay amount subtracted from the gross.
    net_payable : float
        Actual amount paid via ACH (gross - deductible - copay).
    ach_reference : str
        ACH transaction reference from the banking provider.
    status : str
        "initiated", "settled", "failed", "duplicate_prevented".
    idempotency_key : str
        Key used for deduplication (``payout-{claim_id}``).
    created_at : float
        Unix timestamp of payment creation.
    settlement_date : str, optional
        Expected ACH settlement date (ISO format).
    metadata : dict
        Additional metadata (ZKP proof refs, agent traces, etc.).
    """

    payment_id: str
    claim_id: str
    policy_id: str
    payee: str
    gross_amount: float
    deductible_applied: float
    copay_amount: float
    net_payable: float
    ach_reference: str
    status: str
    idempotency_key: str
    created_at: float = field(default_factory=time.time)
    settlement_date: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "payment_id": self.payment_id,
            "claim_id": self.claim_id,
            "policy_id": self.policy_id,
            "payee": self.payee,
            "gross_amount": self.gross_amount,
            "deductible_applied": self.deductible_applied,
            "copay_amount": self.copay_amount,
            "net_payable": self.net_payable,
            "ach_reference": self.ach_reference,
            "status": self.status,
            "idempotency_key": self.idempotency_key,
            "created_at": self.created_at,
            "settlement_date": self.settlement_date,
            "metadata": self.metadata,
        }


@runtime_checkable
class PaymentLedger(Protocol):
    """Protocol for payment ledger backends."""

    def find_by_claim(self, claim_id: str) -> list[PaymentRecord]: ...
    def find_by_idempotency_key(self, key: str) -> Optional[PaymentRecord]: ...
    def insert(self, record: PaymentRecord) -> PaymentRecord: ...
    def all_records(self) -> list[PaymentRecord]: ...


class InMemoryPaymentLedger:
    """In-memory payment ledger with idempotency support.

    Used for tests and local development. Production replaces this with
    a PostgreSQL-backed implementation (same interface).
    """

    def __init__(self) -> None:
        self._records: list[PaymentRecord] = []
        self._by_idempotency: dict[str, PaymentRecord] = {}
        self._by_claim: dict[str, list[PaymentRecord]] = {}

    def find_by_claim(self, claim_id: str) -> list[PaymentRecord]:
        return list(self._by_claim.get(claim_id, []))

    def find_by_idempotency_key(self, key: str) -> Optional[PaymentRecord]:
        return self._by_idempotency.get(key)

    def insert(self, record: PaymentRecord) -> PaymentRecord:
        # Check idempotency
        existing = self._by_idempotency.get(record.idempotency_key)
        if existing is not None:
            logger.info(
                "Duplicate payment prevented for claim %s "
                "(idempotency key=%s)", record.claim_id, record.idempotency_key
            )
            return existing
        self._records.append(record)
        self._by_idempotency[record.idempotency_key] = record
        self._by_claim.setdefault(record.claim_id, []).append(record)
        return record

    def all_records(self) -> list[PaymentRecord]:
        return list(self._records)


def check_duplicate(
    ledger: PaymentLedger,
    claim_id: str,
) -> Optional[PaymentRecord]:
    """Check the ledger for an existing payment for ``claim_id``.

    Returns the existing :class:`PaymentRecord` if a payment has already
    been made, or None if this is a new payment.

    This is the duplicate-detection function called by the PayoutAgent
    before initiating any ACH transfer. It implements the AC:
    "Duplicate payment detection prevents payment on already-paid claims."
    """
    # Check by idempotency key first (fastest)
    idem_key = f"payout-{claim_id}"
    existing = ledger.find_by_idempotency_key(idem_key)
    if existing is not None:
        return existing
    # Also check by claim_id (in case the key format changed)
    records = ledger.find_by_claim(claim_id)
    if records:
        # Return the most recent successful payment
        successful = [r for r in records if r.status in {"initiated", "settled"}]
        if successful:
            return successful[-1]
    return None


def compute_payment_breakdown(
    *,
    gross_amount: float,
    deductible: float,
    copay_pct: float = 0.0,
    copay_cap: Optional[float] = None,
) -> dict[str, float]:
    """Compute the payment breakdown (gross, deductible, co-pay, net).

    Parameters
    ----------
    gross_amount : float
        The original claim amount.
    deductible : float
        The policy deductible (applied first).
    copay_pct : float
        Co-pay percentage (0.0 to 1.0), applied AFTER deductible.
    copay_cap : float, optional
        Maximum co-pay amount (caps the co-pay).

    Returns
    -------
    dict
        {"gross": ..., "deductible": ..., "copay": ..., "net": ...}
    """
    if gross_amount < 0:
        raise ValueError("Gross amount must be >= 0")
    if deductible < 0:
        raise ValueError("Deductible must be >= 0")
    if not (0 <= copay_pct <= 1):
        raise ValueError("Co-pay percentage must be in [0, 1]")

    after_deductible = max(0, gross_amount - deductible)
    copay = after_deductible * copay_pct
    if copay_cap is not None:
        copay = min(copay, copay_cap)
    net = after_deductible - copay
    return {
        "gross": round(gross_amount, 2),
        "deductible": round(deductible, 2),
        "copay": round(copay, 2),
        "net": round(net, 2),
    }
