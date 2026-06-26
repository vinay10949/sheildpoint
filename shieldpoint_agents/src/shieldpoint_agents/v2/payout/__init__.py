"""
ShieldPoint Payout Subsystem (SP-405)
======================================

This package implements the full payout pipeline for approved claims:

- :mod:`ach_provider`      — ACH payment initiation (with stub for tests)
- :mod:`ledger`            — Enhanced payment ledger with duplicate detection
- :mod:`receipt_generator` — PDF receipt generation with payment breakdown
- :mod:`notification`      — Claimant email notification
- :mod:`audit_record`      — Comprehensive audit record assembly

The enhanced :class:`PayoutAgent` (in :mod:`agents`) wires these
together to execute the APPROVED → PAID_OUT transition.
"""

from .ach_provider import ACHProvider, ACHResult, StubACHProvider, BankVerificationService
from .ledger import (
    PaymentLedger, PaymentRecord, InMemoryPaymentLedger,
    check_duplicate, compute_payment_breakdown,
)
from .receipt_generator import ReceiptGenerator, ReceiptResult
from .notification import NotificationService, NotificationResult, StubNotificationService
from .audit_record import AuditRecordAssembler, AuditRecord

__all__ = [
    "ACHProvider",
    "ACHResult",
    "StubACHProvider",
    "BankVerificationService",
    "PaymentLedger",
    "PaymentRecord",
    "InMemoryPaymentLedger",
    "check_duplicate",
    "compute_payment_breakdown",
    "ReceiptGenerator",
    "ReceiptResult",
    "NotificationService",
    "NotificationResult",
    "StubNotificationService",
    "AuditRecordAssembler",
    "AuditRecord",
]
