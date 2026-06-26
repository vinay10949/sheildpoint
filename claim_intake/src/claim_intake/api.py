"""
FastAPI server for the Claim Intake Automation service (SP-203).

Endpoints
---------

- ``GET  /health``                       — liveness + store stats
- ``POST /intake/claims``                — web portal submission (JSON body)
- ``POST /intake/claims/fax``            — fax PDF ingestion (multipart upload)
- ``POST /intake/claims/email``          — manual email-trigger ingestion
  (for testing without a live IMAP server)
- ``GET  /intake/claims/{claim_id}``     — retrieve an accepted claim
- ``GET  /review/queue``                 — list the manual review queue
- ``GET  /review/queue/{claim_id}``      — fetch a single review item
- ``POST /review/queue/{claim_id}/resolve`` — promote or reject a review item
- ``POST /review/queue/next``            — pop the oldest review item (FIFO)
- ``GET  /email/poller/status``          — poller state (running, last poll)
- ``POST /email/poller/start``           — start the background poller
- ``POST /email/poller/stop``            — stop the background poller
- ``POST /email/poller/poll-now``        — trigger a single poll cycle (sync)

The server also starts the email poller automatically on startup if
``INTAKE_IMAP_ENABLED=true`` is set.
"""

from __future__ import annotations

import base64
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import IntakeConfig
from .email_poller import EmailPoller
from .pipeline import intake_email_claim, intake_fax_claim, intake_web_claim
from .schemas import (
    ClaimStatus,
    EmailClaimSubmission,
    FaxClaimSubmission,
    IntakeResult,
    IntakeSource,
    StandardClaim,
    WebClaimSubmission,
    new_request_id,
)
from .store import (
    get_accepted,
    get_review,
    list_accepted,
    list_review,
    next_review,
    resolve_review,
    store_stats,
)

logger = logging.getLogger("claim_intake.api")


# ---------------------------------------------------------------------------
# App + state
# ---------------------------------------------------------------------------
#: Module-level poller (started on app startup if IMAP is enabled).
_poller: EmailPoller | None = None


def _get_poller() -> EmailPoller:
    global _poller
    if _poller is None:
        _poller = EmailPoller(config=IntakeConfig.from_env())
    return _poller


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the email poller lifecycle alongside the FastAPI app."""
    cfg = IntakeConfig.from_env()
    if cfg.imap_enabled:
        try:
            _get_poller().start()
            logger.info("Email poller auto-started (INTAKE_IMAP_ENABLED=true)")
        except Exception as exc:
            logger.error("Email poller failed to start: %s", exc)
    yield
    # Shutdown
    global _poller
    if _poller is not None:
        _poller.stop(timeout=2.0)
        _poller = None


app = FastAPI(
    title="ShieldPoint Claim Intake",
    version="0.1.0",
    description=(
        "Automated claim intake pipeline: web portal, email (IMAP), and "
        "fax (PDF) sources. OCR-based extraction with field validation "
        "and manual review queue."
    ),
    lifespan=lifespan,
)

# CORS for the web portal — tighten in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe — also returns store stats and poller state."""
    cfg = IntakeConfig.from_env()
    poller = _poller
    return {
        "status": "ok",
        "intake_api_version": "0.1.0",
        "imap_enabled": cfg.imap_enabled,
        "imap_host": cfg.imap_host or None,
        "ocr_tesseract_available": _tesseract_available_cached(),
        "llm_enabled": cfg.llm_enabled,
        "store": store_stats(),
        "poller": {
            "running": poller.is_running if poller else False,
            "poll_count": poller.poll_count if poller else 0,
            "processed_uid_count": poller.processed_uid_count if poller else 0,
            "last_poll": (
                {
                    "messages_found": poller.last_poll.messages_found,
                    "messages_processed": poller.last_poll.messages_processed,
                    "errors": poller.last_poll.errors,
                    "elapsed_sec": poller.last_poll.elapsed_sec,
                    "first_error": poller.last_poll.first_error,
                }
                if poller and poller.last_poll else None
            ),
        },
    }


def _tesseract_available_cached() -> bool:
    from .ocr import is_tesseract_available
    return is_tesseract_available()


# ---------------------------------------------------------------------------
# Web portal intake
# ---------------------------------------------------------------------------
class WebClaimRequest(BaseModel):
    """Request body for ``POST /intake/claims``.

    All fields are optional at the API boundary — the validator decides
    whether the result is acceptable, in review, or rejected. This lets
    the portal submit partially-filled forms and let the intake pipeline
    fill in the gaps via the damage description.
    """

    policyholder_name: str = ""
    policy_id: str = ""
    claim_type: str = ""
    date_of_loss: str = ""
    damage_description: str = ""
    amount_claimed: float | None = None
    incident_location: str | None = None
    adjuster_id: str | None = None
    phone: str | None = None
    email: str | None = None
    request_id: str | None = None


@app.post("/intake/claims", response_model_exclude_none=True)
def submit_web_claim(req: WebClaimRequest) -> dict[str, Any]:
    """Accept a claim from the web portal and return the intake result."""
    submission = WebClaimSubmission(
        source=IntakeSource.WEB,
        request_id=req.request_id or new_request_id(),
        policyholder_name=req.policyholder_name,
        policy_id=req.policy_id,
        claim_type=req.claim_type,
        date_of_loss=req.date_of_loss,
        damage_description=req.damage_description,
        amount_claimed=req.amount_claimed,
        incident_location=req.incident_location,
        adjuster_id=req.adjuster_id,
        phone=req.phone,
        email=req.email,
    )
    result = intake_web_claim(submission)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Fax intake (multipart upload)
# ---------------------------------------------------------------------------
@app.post("/intake/claims/fax", response_model_exclude_none=True)
async def submit_fax_claim(
    file: UploadFile = File(...),
    fax_number: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Accept a digitised fax PDF and run OCR-based intake."""
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF upload")
    submission = FaxClaimSubmission(
        source=IntakeSource.FAX,
        request_id=request_id or new_request_id(),
        fax_number=fax_number,
        received_at=datetime.now(timezone.utc).isoformat(),
        pdf_bytes=pdf_bytes,
    )
    result = intake_fax_claim(submission)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Email intake (manual trigger — for testing without a live IMAP server)
# ---------------------------------------------------------------------------
class ManualEmailRequest(BaseModel):
    """Request body for ``POST /intake/claims/email``.

    Each attachment is a (filename, mime_type, base64_data) tuple. This
    endpoint exists so the load test and integration tests can exercise
    the email-intake path without standing up an IMAP server.
    """

    sender: str
    received_at: str
    subject: str | None = None
    request_id: str | None = None
    attachments: list[dict[str, str]] = []  # [{filename, mime_type, data_b64}]


@app.post("/intake/claims/email", response_model_exclude_none=True)
def submit_email_claim(req: ManualEmailRequest) -> dict[str, Any]:
    """Manually trigger email-claim intake (no IMAP required)."""
    attachments: list[tuple[str, str, bytes]] = []
    for att in req.attachments:
        try:
            data = base64.b64decode(att["data_b64"])
        except (KeyError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid attachment: {exc}",
            ) from exc
        attachments.append((
            att.get("filename", "unnamed"),
            att.get("mime_type", "application/octet-stream"),
            data,
        ))
    submission = EmailClaimSubmission(
        source=IntakeSource.EMAIL,
        request_id=req.request_id or new_request_id(),
        sender=req.sender,
        received_at=req.received_at,
        subject=req.subject,
        attachments=attachments,
    )
    result = intake_email_claim(submission)
    return result.model_dump()


# ---------------------------------------------------------------------------
# Claim retrieval
# ---------------------------------------------------------------------------
@app.get("/intake/claims/{claim_id}")
def get_claim(claim_id: str) -> dict[str, Any]:
    """Retrieve an accepted claim by ID."""
    rec = get_accepted(claim_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found")
    return rec


@app.get("/intake/claims")
def list_claims(limit: int = 100) -> dict[str, Any]:
    """List the most recent N accepted claims."""
    return {"claims": list_accepted(limit=limit), **store_stats()}


# ---------------------------------------------------------------------------
# Manual review queue
# ---------------------------------------------------------------------------
@app.get("/review/queue")
def list_review_queue(limit: int = 100, status: str | None = None) -> dict[str, Any]:
    """List the manual review queue."""
    filter_status: ClaimStatus | None = None
    if status:
        try:
            filter_status = ClaimStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{status}' (valid: {[s.value for s in ClaimStatus]})",
            )
    items = list_review(limit=limit, status=filter_status)
    return {
        "queue_depth": len(items),
        "items": [it.model_dump() for it in items],
    }


@app.get("/review/queue/{claim_id}")
def get_review_item(claim_id: str) -> dict[str, Any]:
    """Fetch a single review item."""
    item = get_review(claim_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Review item '{claim_id}' not found")
    return item.model_dump()


class ResolveReviewRequest(BaseModel):
    """Body for ``POST /review/queue/{claim_id}/resolve``.

    If ``decision == "accept"``, ``claim`` must contain the corrected fields
    needed to build a :class:`StandardClaim`. If ``decision == "reject"``,
    ``claim`` is ignored and the item is marked rejected.
    """

    decision: str  # "accept" | "reject"
    claim: dict[str, Any] | None = None
    request_id: str | None = None


@app.post("/review/queue/{claim_id}/resolve", response_model_exclude_none=True)
def resolve_review_item(claim_id: str, req: ResolveReviewRequest) -> dict[str, Any]:
    """Promote a review item to accepted, or mark it rejected."""
    item = get_review(claim_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Review item '{claim_id}' not found")

    if req.decision == "accept":
        if not req.claim:
            raise HTTPException(
                status_code=400, detail="claim is required when decision='accept'"
            )
        try:
            claim = StandardClaim(**req.claim)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid claim fields: {exc}"
            ) from exc
        result = resolve_review(
            claim_id, accepted_claim=claim,
            source=item.source, request_id=req.request_id,
        )
        return result.model_dump()
    elif req.decision == "reject":
        result = resolve_review(
            claim_id, accepted_claim=None,
            source=item.source, request_id=req.request_id,
        )
        return result.model_dump()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"decision must be 'accept' or 'reject', got '{req.decision}'",
        )


@app.post("/review/queue/next", response_model_exclude_none=True)
def pop_next_review() -> dict[str, Any]:
    """Pop the oldest review item from the queue (FIFO).

    Returns 404 if the queue is empty.
    """
    item = next_review()
    if item is None:
        raise HTTPException(status_code=404, detail="Review queue is empty")
    return item.model_dump()


# ---------------------------------------------------------------------------
# Email poller control
# ---------------------------------------------------------------------------
@app.get("/email/poller/status")
def poller_status() -> dict[str, Any]:
    """Return the current state of the email poller."""
    poller = _get_poller()
    return {
        "running": poller.is_running,
        "poll_count": poller.poll_count,
        "processed_uid_count": poller.processed_uid_count,
        "last_poll": (
            {
                "messages_found": poller.last_poll.messages_found,
                "messages_processed": poller.last_poll.messages_processed,
                "errors": poller.last_poll.errors,
                "elapsed_sec": poller.last_poll.elapsed_sec,
                "first_error": poller.last_poll.first_error,
            }
            if poller.last_poll else None
        ),
    }


@app.post("/email/poller/start")
def poller_start() -> dict[str, Any]:
    """Start the background email poller."""
    poller = _get_poller()
    if poller.is_running:
        return {"status": "already_running"}
    poller.start()
    return {"status": "started"}


@app.post("/email/poller/stop")
def poller_stop() -> dict[str, Any]:
    """Stop the background email poller."""
    poller = _get_poller()
    if not poller.is_running:
        return {"status": "already_stopped"}
    poller.stop(timeout=2.0)
    return {"status": "stopped"}


@app.post("/email/poller/poll-now")
def poller_poll_now() -> dict[str, Any]:
    """Trigger a single poll cycle synchronously and return the result."""
    poller = _get_poller()
    result = poller.poll_once()
    return {
        "messages_found": result.messages_found,
        "messages_processed": result.messages_processed,
        "errors": result.errors,
        "elapsed_sec": result.elapsed_sec,
        "first_error": result.first_error,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    cfg = IntakeConfig.from_env()
    uvicorn.run(
        app,
        host=cfg.api_host,
        port=cfg.api_port,
        log_level="info",
    )
