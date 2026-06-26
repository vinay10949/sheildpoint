"""
ShieldPoint Claim Intake Automation (SP-203)
============================================

Automated claim intake pipeline that replaces the manual 2.3-day process
for claims arriving via web portal, email (IMAP polling), and fax
(digitised PDF). Uses Tesseract OCR to extract structured data from
fax/email attachments, validates the extracted data against required
fields, and formats the claim into the standard JSON schema expected by
the IntakeAgent.

Public API
----------

- :class:`IntakeConfig` — runtime configuration (env-driven).
- :func:`intake_web_claim`, :func:`intake_email_claim`,
  :func:`intake_fax_claim` — pipeline entry points.
- :class:`EmailPoller` — background IMAP poller.
- :class:`StandardClaim` — the canonical claim schema handed to the
  IntakeAgent.
- :class:`IntakeResult` — the envelope returned by every intake call.
- :func:`reset_stores`, :func:`store_stats` — store management (mostly
  for tests).

Latency targets
---------------

- Web claims: < 30 seconds end-to-end (no OCR).
- Email/fax claims: < 2 minutes end-to-end (includes OCR + extraction).

The 30-second web-claim target is verified by the load test
(``scripts/run_load_test.py``), which submits 100 concurrent claims and
measures P99 latency.
"""

from __future__ import annotations

from .config import IntakeConfig
from .email_poller import EmailPoller, PollResult
from .extractor import ExtractionResult, extract
from .ocr import OCRResult, OCRError, is_tesseract_available, ocr_attachment, ocr_pdf
from .pipeline import intake_email_claim, intake_fax_claim, intake_web_claim
from .schemas import (
    ClaimStatus,
    ClaimType,
    EmailClaimSubmission,
    FaxClaimSubmission,
    FieldError,
    IntakeResult,
    IntakeSource,
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    ReviewItem,
    StandardClaim,
    WebClaimSubmission,
    new_claim_id,
    new_request_id,
)
from .store import (
    get_accepted,
    get_review,
    list_accepted,
    list_review,
    next_review,
    put_accepted,
    put_review,
    reset_stores,
    resolve_review,
    store_stats,
)
from .validator import ValidationResult, validate

__all__ = [
    "ClaimStatus",
    "ClaimType",
    "EmailClaimSubmission",
    "EmailPoller",
    "ExtractionResult",
    "FaxClaimSubmission",
    "FieldError",
    "IntakeConfig",
    "IntakeResult",
    "IntakeSource",
    "OCRResult",
    "OCRError",
    "OPTIONAL_FIELDS",
    "PollResult",
    "REQUIRED_FIELDS",
    "ReviewItem",
    "StandardClaim",
    "ValidationResult",
    "WebClaimSubmission",
    "extract",
    "get_accepted",
    "get_review",
    "intake_email_claim",
    "intake_fax_claim",
    "intake_web_claim",
    "is_tesseract_available",
    "list_accepted",
    "list_review",
    "new_claim_id",
    "new_request_id",
    "next_review",
    "ocr_attachment",
    "ocr_pdf",
    "put_accepted",
    "put_review",
    "reset_stores",
    "resolve_review",
    "store_stats",
    "validate",
]

__version__ = "0.1.0"
