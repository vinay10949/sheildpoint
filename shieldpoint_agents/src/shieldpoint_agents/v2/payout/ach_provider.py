"""
SP-405 — ACH Payment Initiation Provider
=========================================

Initiates ACH (Automated Clearing House) transfers for approved claim
payouts. In production, this calls a real banking API (e.g. Plaid, Stripe,
or a direct bank API). In tests, the :class:`StubACHProvider` returns
deterministic success responses.

The provider is injected into the :class:`PayoutAgent` via the
``ach_provider`` parameter, following the same dependency-injection
pattern as the LLM client in :class:`ClassifierAgent`.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger("shieldpoint.payout.ach")


@dataclass(frozen=True)
class ACHResult:
    """Result of an ACH payment initiation.

    Attributes
    ----------
    success : bool
        True if the ACH transfer was successfully initiated.
    ach_reference : str
        The ACH transaction reference number (for tracking).
    amount : float
        The amount transferred (USD).
    status : str
        "initiated", "settled", "failed", or "pending".
    error : str, optional
        Error message if the ACH initiation failed.
    initiated_at : float
        Unix timestamp of initiation.
    settlement_date : str, optional
        Expected settlement date (ISO format).
    raw_response : dict
        Raw response from the banking API (for audit).
    """

    success: bool
    ach_reference: str
    amount: float
    status: str
    error: Optional[str] = None
    initiated_at: float = field(default_factory=time.time)
    settlement_date: Optional[str] = None
    raw_response: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ACHProvider(Protocol):
    """Protocol for ACH payment providers.

    Implementations:
    - :class:`StubACHProvider` — deterministic stub for tests
    - Production: ``PlaidACHProvider``, ``StripeACHProvider``, etc.
    """

    def initiate_payment(
        self,
        *,
        amount: float,
        payee_name: str,
        bank_account: str,
        bank_routing: str,
        idempotency_key: str,
        **kwargs: Any,
    ) -> ACHResult: ...


class StubACHProvider:
    """Deterministic stub ACH provider for tests and local development.

    Always succeeds unless:
    - The amount is <= 0
    - The bank_account or bank_routing is empty
    - The idempotency_key has already been used (returns the cached result)

    This stub lets the full payout pipeline run end-to-end in CI without
    a real bank connection.
    """

    def __init__(self) -> None:
        self._idempotency_cache: dict[str, ACHResult] = {}
        self.call_log: list[dict[str, Any]] = []

    def initiate_payment(
        self,
        *,
        amount: float,
        payee_name: str,
        bank_account: str,
        bank_routing: str,
        idempotency_key: str,
        **kwargs: Any,
    ) -> ACHResult:
        self.call_log.append({
            "amount": amount,
            "payee_name": payee_name,
            "bank_account": bank_account,
            "idempotency_key": idempotency_key,
            "timestamp": time.time(),
        })

        # Idempotency: return cached result if we've seen this key
        if idempotency_key in self._idempotency_cache:
            logger.info("ACH idempotency hit for key=%s", idempotency_key)
            return self._idempotency_cache[idempotency_key]

        # Validate inputs
        if amount <= 0:
            result = ACHResult(
                success=False, ach_reference="", amount=amount,
                status="failed", error="Amount must be > 0",
            )
            self._idempotency_cache[idempotency_key] = result
            return result

        if not bank_account or not bank_routing:
            result = ACHResult(
                success=False, ach_reference="", amount=amount,
                status="failed", error="Missing bank account or routing number",
            )
            self._idempotency_cache[idempotency_key] = result
            return result

        # Generate ACH reference
        ach_ref = f"ACH-{uuid.uuid4().hex[:10].upper()}"

        # Compute expected settlement date (2 business days from now)
        from datetime import datetime, timedelta
        settlement = (datetime.utcnow() + timedelta(days=2)).strftime("%Y-%m-%d")

        result = ACHResult(
            success=True,
            ach_reference=ach_ref,
            amount=amount,
            status="initiated",
            settlement_date=settlement,
            raw_response={
                "provider": "stub",
                "transaction_id": ach_ref,
                "amount": amount,
                "currency": "USD",
                "network": "ACH",
                "sec_code": "PPD",  # Prearranged Payment and Deposit
            },
        )
        self._idempotency_cache[idempotency_key] = result
        return result


class BankVerificationService:
    """Verifies payee bank details before ACH initiation.

    In production, this calls a bank verification API (e.g. Plaid's
    ``/accounts/balance/get`` or Microbilt's bank verification service).
    In tests, the stub returns True unless explicitly overridden.
    """

    def __init__(self, *, always_valid: bool = True) -> None:
        self.always_valid = always_valid
        self.call_log: list[dict[str, Any]] = []

    def verify(
        self,
        *,
        bank_account: str,
        bank_routing: str,
        payee_name: str,
    ) -> tuple[bool, str]:
        """Verify bank details.

        Returns
        -------
        tuple of (is_valid, message)
            is_valid is True if the bank details pass verification.
            message describes the verification result.
        """
        self.call_log.append({
            "bank_account": bank_account,
            "bank_routing": bank_routing,
            "payee_name": payee_name,
            "timestamp": time.time(),
        })

        if self.always_valid:
            return (True, "Bank details verified (stub).")

        # Basic format validation
        if not bank_account or len(bank_account) < 4:
            return (False, "Bank account number too short.")
        if not bank_routing or len(bank_routing) != 9:
            return (False, "Routing number must be 9 digits.")
        if not payee_name:
            return (False, "Payee name required.")

        return (True, "Bank details verified.")
