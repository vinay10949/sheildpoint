"""
End-to-end intake pipeline (SP-203).

Orchestrates: submission → (OCR if needed) → extract → validate → store.

Three entry points, one per source:

- :func:`intake_web_claim`       — synchronous, called from the FastAPI
  handler. Target latency < 30s.
- :func:`intake_email_claim`     — called by the IMAP poller for each
  message with attachments. Target latency < 2 min.
- :func:`intake_fax_claim`       — called by the fax ingestion path
  (POSTed by the fax gateway as raw PDF bytes). Target latency < 2 min.

All three return an :class:`IntakeResult`. The IntakeAgent downstream reads
accepted claims from :func:`store.get_accepted` and rejected/in-review
claims from :func:`store.list_review`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .config import IntakeConfig
from .extractor import extract as extract_fields
from .ocr import OCRError, normalise_ocr_text, ocr_attachment, ocr_pdf
from .schemas import (
    ClaimStatus,
    EmailClaimSubmission,
    FaxClaimSubmission,
    FieldError,
    IntakeResult,
    IntakeSource,
    ReviewItem,
    StandardClaim,
    WebClaimSubmission,
    new_claim_id,
)
from .store import put_accepted, put_review
from .validator import to_standard_claim, validate

logger = logging.getLogger("claim_intake.pipeline")


# ---------------------------------------------------------------------------
# Shared routing logic
# ---------------------------------------------------------------------------
def _route(
    *,
    claim_id: str,
    source: IntakeSource,
    raw_submission: dict[str, Any],
    extracted_fields: dict[str, Any],
    ocr_text: str | None,
    config: IntakeConfig,
    start_time: float,
    request_id: str | None,
) -> IntakeResult:
    """Run validation and route to accepted/review/rejected."""
    vresult = validate(extracted_fields, config=config)
    latency = time.monotonic() - start_time

    if vresult.ok:
        try:
            claim = to_standard_claim(vresult.cleaned)
        except Exception as exc:
            # Should not happen — validate() already filtered — but guard.
            logger.exception(
                "claim_id=%s: post-validation StandardClaim build failed: %s",
                claim_id, exc,
            )
            err = FieldError(
                field="__root__",
                message=f"Internal validation error: {exc}",
            )
            item = ReviewItem(
                claim_id=claim_id,
                status=ClaimStatus.IN_REVIEW,
                source=source,
                received_at=_now_iso(),
                raw_submission=raw_submission,
                errors=[err],
                partial_claim=vresult.cleaned,
                ocr_text=ocr_text,
                request_id=request_id,
            )
            pos = put_review(item)
            return IntakeResult(
                claim_id=claim_id,
                status=ClaimStatus.IN_REVIEW,
                source=source,
                accepted=False,
                errors=[err],
                latency_sec=latency,
                review_queue_position=pos,
                request_id=request_id,
            )
        return put_accepted(
            claim, claim_id, source=source,
            request_id=request_id, latency_sec=latency,
        )

    # Not OK — route to review (or reject if too many missing fields)
    is_rejected = vresult.missing_required_count >= config.review_max_missing_fields
    status = ClaimStatus.REJECTED if is_rejected else ClaimStatus.IN_REVIEW
    item = ReviewItem(
        claim_id=claim_id,
        status=status,
        source=source,
        received_at=_now_iso(),
        raw_submission=raw_submission,
        errors=vresult.errors,
        partial_claim=vresult.cleaned,
        ocr_text=ocr_text,
        request_id=request_id,
    )
    pos = put_review(item)
    return IntakeResult(
        claim_id=claim_id,
        status=status,
        source=source,
        accepted=False,
        errors=vresult.errors,
        latency_sec=latency,
        review_queue_position=pos,
        request_id=request_id,
    )


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Web claim intake
# ---------------------------------------------------------------------------
def intake_web_claim(
    submission: WebClaimSubmission, *,
    config: IntakeConfig | None = None,
) -> IntakeResult:
    """Process a claim submitted via the web portal.

    The portal may pre-populate any of the required fields. Whatever is
    missing is left blank — the validator flags it. No OCR is performed
    (the portal already gives us structured data).
    """
    cfg = config or IntakeConfig.from_env()
    start = time.monotonic()
    claim_id = new_claim_id()
    request_id = submission.request_id

    # Pass through whatever the portal submitted.
    passthrough: dict[str, Any] = {}
    for k in (
        "policyholder_name", "policy_id", "claim_type", "date_of_loss",
        "damage_description", "amount_claimed", "incident_location",
        "adjuster_id", "phone", "email",
    ):
        v = getattr(submission, k, None)
        if v is not None and v != "":
            passthrough[k] = v

    # No OCR for web claims — but we still run the extractor's normalisation
    # (e.g. claim_type "Home" → "homeowners") on the damage_description if
    # claim_type is missing.
    if "claim_type" not in passthrough and passthrough.get("damage_description"):
        ex = extract_fields(
            passthrough["damage_description"], config=cfg, passthrough=passthrough,
        )
        passthrough.update({k: v for k, v in ex.fields.items() if v})

    raw_submission = submission.model_dump()
    return _route(
        claim_id=claim_id,
        source=IntakeSource.WEB,
        raw_submission=raw_submission,
        extracted_fields=passthrough,
        ocr_text=None,
        config=cfg,
        start_time=start,
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Email claim intake
# ---------------------------------------------------------------------------
def intake_email_claim(
    submission: EmailClaimSubmission, *,
    config: IntakeConfig | None = None,
) -> IntakeResult:
    """Process a claim received via email with attachments.

    Runs OCR on each attachment (concatenating text), then extracts fields.
    """
    cfg = config or IntakeConfig.from_env()
    start = time.monotonic()
    claim_id = new_claim_id()
    request_id = submission.request_id

    all_text_parts: list[str] = []
    ocr_error: str | None = None

    for filename, mime_type, data in submission.attachments:
        try:
            res = ocr_attachment(filename, mime_type, data, config=cfg)
            all_text_parts.append(res.text)
        except OCRError as exc:
            logger.warning("claim_id=%s: OCR failed on %s: %s",
                           claim_id, filename, exc)
            ocr_error = str(exc)

    ocr_text = normalise_ocr_text("\n\n".join(all_text_parts))
    if not ocr_text and ocr_error:
        # All attachments failed OCR — route to review with the error.
        latency = time.monotonic() - start
        err = FieldError(
            field="__attachments__",
            message=f"OCR failed on all attachments: {ocr_error}",
        )
        item = ReviewItem(
            claim_id=claim_id,
            status=ClaimStatus.IN_REVIEW,
            source=IntakeSource.EMAIL,
            received_at=_now_iso(),
            raw_submission={
                "sender": submission.sender,
                "received_at": submission.received_at,
                "subject": submission.subject,
                "attachment_count": len(submission.attachments),
                "attachment_names": [a[0] for a in submission.attachments],
            },
            errors=[err],
            partial_claim=None,
            ocr_text=None,
            request_id=request_id,
        )
        pos = put_review(item)
        return IntakeResult(
            claim_id=claim_id,
            status=ClaimStatus.IN_REVIEW,
            source=IntakeSource.EMAIL,
            accepted=False,
            errors=[err],
            latency_sec=latency,
            review_queue_position=pos,
            request_id=request_id,
        )

    ex = extract_fields(ocr_text, config=cfg, passthrough=None)

    raw_submission = submission.model_dump()
    # Pydantic can't JSON-serialise bytes by default; strip them.
    raw_submission["attachments"] = [
        {"filename": a[0], "mime_type": a[1], "size_bytes": len(a[2])}
        for a in submission.attachments
    ]
    return _route(
        claim_id=claim_id,
        source=IntakeSource.EMAIL,
        raw_submission=raw_submission,
        extracted_fields=ex.fields,
        ocr_text=ocr_text,
        config=cfg,
        start_time=start,
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Fax claim intake
# ---------------------------------------------------------------------------
def intake_fax_claim(
    submission: FaxClaimSubmission, *,
    config: IntakeConfig | None = None,
) -> IntakeResult:
    """Process a claim received as a digitised fax PDF."""
    cfg = config or IntakeConfig.from_env()
    start = time.monotonic()
    claim_id = new_claim_id()
    request_id = submission.request_id

    try:
        ocr_result = ocr_pdf(submission.pdf_bytes, config=cfg)
        ocr_text = normalise_ocr_text(ocr_result.text)
    except OCRError as exc:
        logger.warning("claim_id=%s: fax OCR failed: %s", claim_id, exc)
        latency = time.monotonic() - start
        err = FieldError(
            field="__pdf__",
            message=f"Fax OCR failed: {exc}",
        )
        item = ReviewItem(
            claim_id=claim_id,
            status=ClaimStatus.IN_REVIEW,
            source=IntakeSource.FAX,
            received_at=_now_iso(),
            raw_submission={
                "fax_number": submission.fax_number,
                "received_at": submission.received_at,
                "pdf_size_bytes": len(submission.pdf_bytes),
            },
            errors=[err],
            partial_claim=None,
            ocr_text=None,
            request_id=request_id,
        )
        pos = put_review(item)
        return IntakeResult(
            claim_id=claim_id,
            status=ClaimStatus.IN_REVIEW,
            source=IntakeSource.FAX,
            accepted=False,
            errors=[err],
            latency_sec=latency,
            review_queue_position=pos,
            request_id=request_id,
        )

    ex = extract_fields(ocr_text, config=cfg, passthrough=None)

    raw_submission = submission.model_dump()
    raw_submission["pdf_size_bytes"] = len(submission.pdf_bytes)
    raw_submission.pop("pdf_bytes", None)  # don't persist raw bytes in review queue

    return _route(
        claim_id=claim_id,
        source=IntakeSource.FAX,
        raw_submission=raw_submission,
        extracted_fields=ex.fields,
        ocr_text=ocr_text,
        config=cfg,
        start_time=start,
        request_id=request_id,
    )
