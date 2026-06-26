"""
Pydantic models for the claim intake pipeline (SP-203).

Three logical groups of models:

1. **Inbound** — the request shapes accepted from each intake channel:

   - :class:`WebClaimSubmission` — POSTed by the web portal as JSON.
   - :class:`EmailClaimSubmission` — synthesised by the IMAP poller after
     extracting attachments.
   - :class:`FaxClaimSubmission` — synthesised by the fax ingestion path
     (raw PDF bytes).

2. **Standard claim JSON** — :class:`StandardClaim` is the canonical schema
   handed off to the IntakeAgent downstream. This is the "contract" between
   intake and the rest of the agent framework.

3. **Result / review** — :class:`IntakeResult` is what the API returns.
   :class:`ReviewItem` is what the manual-review queue stores.

The required-field set is exactly the five named in the SP-203 acceptance
criteria: ``policyholder_name``, ``policy_id``, ``claim_type``,
``date_of_loss``, ``damage_description``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class IntakeSource(str, Enum):
    """Where the claim entered the pipeline."""

    WEB = "web"
    EMAIL = "email"
    FAX = "fax"


class ClaimStatus(str, Enum):
    """Lifecycle status of an intake claim.

    - ``accepted``    — passed validation, ready for the IntakeAgent.
    - ``in_review``   — invalid/incomplete; sitting in the manual review queue.
    - ``rejected``    — too many missing fields; persisted for audit only.
    """

    ACCEPTED = "accepted"
    IN_REVIEW = "in_review"
    REJECTED = "rejected"


class ClaimType(str, Enum):
    """Categorisation of the claim.

    The intake pipeline is permissive: anything the extractor produces is
    accepted, but the value must round-trip through one of these enum members.
    Unknown strings are normalised to ``OTHER`` so downstream tools never
    crash on a novel peril type.
    """

    HOMEOWNERS = "homeowners"
    AUTO = "auto"
    PROPERTY = "property"
    LIABILITY = "liability"
    HEALTH = "health"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Standard claim JSON — the schema handed to the IntakeAgent
# ---------------------------------------------------------------------------
# The five required fields named in the SP-203 AC. Anything optional is
# listed in ``OPTIONAL_FIELDS`` below.
REQUIRED_FIELDS: tuple[str, ...] = (
    "policyholder_name",
    "policy_id",
    "claim_type",
    "date_of_loss",
    "damage_description",
)

OPTIONAL_FIELDS: tuple[str, ...] = (
    "amount_claimed",
    "incident_location",
    "adjuster_id",
    "phone",
    "email",
    "policy_effective_date",
    "policy_expiration_date",
)


class StandardClaim(BaseModel):
    """The canonical claim schema handed off to the IntakeAgent.

    All five required fields must be present and non-empty. Optional fields
    default to ``None`` when the extractor cannot populate them.
    """

    model_config = ConfigDict(extra="forbid")

    # ---- Required ---------------------------------------------------------
    policyholder_name: str = Field(
        ..., min_length=1, description="Full name of the policyholder."
    )
    policy_id: str = Field(
        ..., min_length=1, description="Policy identifier (e.g. HO-2024-001)."
    )
    claim_type: ClaimType = Field(
        ..., description="Categorisation of the claim."
    )
    date_of_loss: str = Field(
        ...,
        min_length=1,
        description=(
            "ISO-8601 date (YYYY-MM-DD) when the loss occurred. Kept as a "
            "string so the schema is JSON-serialisable without a custom encoder."
        ),
    )
    damage_description: str = Field(
        ..., min_length=1, description="Free-text description of the damage."
    )

    # ---- Optional ---------------------------------------------------------
    amount_claimed: float | None = Field(
        default=None, ge=0, description="Claimed amount in USD, if known."
    )
    incident_location: str | None = Field(
        default=None, description="Where the loss occurred."
    )
    adjuster_id: str | None = Field(
        default=None, description="Assigned adjuster, if any."
    )
    phone: str | None = Field(
        default=None, description="Claimant contact phone."
    )
    email: str | None = Field(
        default=None, description="Claimant contact email."
    )
    policy_effective_date: str | None = Field(
        default=None, description="ISO-8601 date the policy started."
    )
    policy_expiration_date: str | None = Field(
        default=None, description="ISO-8601 date the policy expires."
    )

    @field_validator("date_of_loss")
    @classmethod
    def _validate_iso_date(cls, v: str) -> str:
        """Ensure the date is a real calendar date in YYYY-MM-DD form."""
        v = v.strip()
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                f"date_of_loss must be YYYY-MM-DD, got '{v}'"
            ) from exc
        return v


# ---------------------------------------------------------------------------
# Inbound submissions — one per channel
# ---------------------------------------------------------------------------
class _BaseSubmission(BaseModel):
    """Shared metadata for every inbound submission."""

    model_config = ConfigDict(extra="forbid")

    source: IntakeSource
    #: Caller-supplied correlation ID. If absent, the pipeline assigns one.
    request_id: str | None = None
    #: Email subject (for email-source claims) or fax header line.
    subject: str | None = None


class WebClaimSubmission(_BaseSubmission):
    """Body of ``POST /intake/claims`` from the web portal.

    The portal may submit either:

    - A **pre-structured** claim (all required fields populated by the form
      UI). The pipeline validates and accepts it as-is.
    - A **semi-structured** claim — typically a free-text damage description
      plus a policy ID. The extractor pulls the rest out of the description.

    Both forms are accepted; the validator decides which path to take.
    """

    source: IntakeSource = IntakeSource.WEB

    # Required fields may be empty here — the validator decides what to do.
    policyholder_name: str = ""
    policy_id: str = ""
    claim_type: str = ""
    date_of_loss: str = ""
    damage_description: str = ""

    # Optional fields
    amount_claimed: float | None = None
    incident_location: str | None = None
    adjuster_id: str | None = None
    phone: str | None = None
    email: str | None = None


class EmailClaimSubmission(_BaseSubmission):
    """Synthesised by the IMAP poller after extracting attachments.

    The poller hands us raw email metadata + each attachment's bytes. The
    pipeline runs OCR on the attachment(s) and the extractor pulls structured
    fields out of the resulting text.
    """

    source: IntakeSource = IntakeSource.EMAIL
    sender: str
    received_at: str  # ISO-8601 timestamp from the IMAP envelope
    #: List of (filename, mime_type, bytes) tuples — one per attachment.
    attachments: list[tuple[str, str, bytes]] = Field(default_factory=list)


class FaxClaimSubmission(_BaseSubmission):
    """Synthesised by the fax ingestion path (raw PDF bytes).

    Fax PDFs are digitised by the upstream fax gateway. The pipeline runs
    Tesseract OCR on each page.
    """

    source: IntakeSource = IntakeSource.FAX
    fax_number: str | None = None
    received_at: str  # ISO-8601 timestamp
    #: Raw PDF bytes.
    pdf_bytes: bytes


# ---------------------------------------------------------------------------
# Result + review
# ---------------------------------------------------------------------------
class FieldError(BaseModel):
    """A single field-level validation error."""

    model_config = ConfigDict(extra="forbid")

    field: str
    message: str


class IntakeResult(BaseModel):
    """Top-level envelope returned by the intake API.

    Either ``status == "accepted"`` and ``claim`` is populated, or
    ``status in {"in_review", "rejected"}`` and ``errors`` lists the
    field-level problems.
    """

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    status: ClaimStatus
    source: IntakeSource
    accepted: bool = Field(
        description="Convenience boolean: True iff status == 'accepted'."
    )
    claim: StandardClaim | None = None
    errors: list[FieldError] = Field(default_factory=list)
    #: Pipeline timings in seconds, for the latency ACs.
    latency_sec: float = 0.0
    #: Where in the queue this claim sits, if routed to review.
    review_queue_position: int | None = None
    request_id: str | None = None


class ReviewItem(BaseModel):
    """One entry in the manual-review queue."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    status: ClaimStatus
    source: IntakeSource
    received_at: str  # ISO-8601
    #: The raw submission payload, for the reviewer to see what came in.
    raw_submission: dict[str, Any]
    #: Field-level errors that triggered the review.
    errors: list[FieldError]
    #: Whatever the extractor managed to pull out, even if incomplete.
    partial_claim: dict[str, Any] | None = None
    #: OCR text, if the source was fax/email (helps the reviewer verify).
    ocr_text: str | None = None
    request_id: str | None = None


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------
def new_claim_id() -> str:
    """Generate a new claim ID of the form ``CLM-<YYYY>-<10hex>``."""
    now = datetime.now(timezone.utc)
    return f"CLM-{now.year}-{uuid4().hex[:10].upper()}"


def new_request_id() -> str:
    """Generate a new request correlation ID."""
    return f"REQ-{uuid4().hex[:12].upper()}"
