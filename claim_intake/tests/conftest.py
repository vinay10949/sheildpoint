"""
Shared pytest fixtures for claim_intake tests.

Pattern follows the existing ``shieldpoint_agents/tests/conftest.py``:
- Add src/ to sys.path so tests work before `pip install -e .`.
- Provide a `clean_stores` autouse fixture for store isolation.
- Provide reusable sample submissions + a synthetic-fax PDF factory.
"""

from __future__ import annotations

import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

# Make `claim_intake` importable when running from the package dir.
PKG_SRC = Path(__file__).resolve().parent.parent / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))


# ---------------------------------------------------------------------------
# Store isolation — reset before every test
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_stores():
    """Reset the in-memory stores before each test."""
    from claim_intake.store import reset_stores
    reset_stores()
    yield
    reset_stores()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def test_config() -> Any:
    """An IntakeConfig with IMAP + LLM disabled (deterministic)."""
    from claim_intake.config import IntakeConfig
    return IntakeConfig(
        imap_enabled=False,
        llm_base_url="",  # regex-only extraction
    )


# ---------------------------------------------------------------------------
# Web portal submission fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def valid_web_submission() -> Any:
    """A web submission with all required fields — should be accepted."""
    from claim_intake.schemas import WebClaimSubmission
    return WebClaimSubmission(
        policyholder_name="Alice Homeowner",
        policy_id="HO-2024-001",
        claim_type="homeowners",
        date_of_loss="2026-03-14",
        damage_description="Wind damage to roof shingles during storm.",
    )


@pytest.fixture
def incomplete_web_submission() -> Any:
    """A web submission missing several required fields — should go to review."""
    from claim_intake.schemas import WebClaimSubmission
    return WebClaimSubmission(
        policyholder_name="Bob Driver",
        # policy_id missing
        claim_type="auto",
        # date_of_loss missing
        damage_description="Rear-end collision on highway.",
    )


@pytest.fixture
def empty_web_submission() -> Any:
    """A web submission with NO required fields — should be rejected
    (all 5 required missing >= review_max_missing_fields threshold)."""
    from claim_intake.schemas import WebClaimSubmission
    return WebClaimSubmission()  # all fields default to empty


# ---------------------------------------------------------------------------
# Synthetic fax PDF factory
# ---------------------------------------------------------------------------
@pytest.fixture
def make_fax_pdf() -> Any:
    """Factory: build a synthetic fax-style PDF with known text content.

    Returns the PDF as bytes. Uses reportlab to lay out a simple
    claim-form template. The OCR test verifies that Tesseract can extract
    >95% of the typed characters.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    def _build(
        *,
        policyholder_name: str = "Alice Homeowner",
        policy_id: str = "HO-2024-001",
        claim_type: str = "homeowners",
        date_of_loss: str = "2026-03-14",
        damage_description: str = (
            "Wind damage to roof shingles during severe thunderstorm on March 14, 2026. "
            "Approximately 30% of shingles blown off the south-facing roof slope. "
            "Water intrusion noted in upstairs bedroom."
        ),
        amount: str = "$1,250.00",
        include_subject: bool = True,
    ) -> bytes:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        width, height = letter

        y = height - 72  # 1 inch margin
        if include_subject:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(72, y, "SHIELDPOINT CLAIM INTAKE FORM")
            y -= 30
            c.setFont("Helvetica", 11)
            c.drawString(72, y, f"Fax received: 2026-03-15 08:42 EST")
            y -= 36

        c.setFont("Helvetica", 11)
        c.drawString(72, y, f"Policyholder Name: {policyholder_name}")
        y -= 18
        c.drawString(72, y, f"Policy ID: {policy_id}")
        y -= 18
        c.drawString(72, y, f"Claim Type: {claim_type}")
        y -= 18
        c.drawString(72, y, f"Date of Loss: {date_of_loss}")
        y -= 18
        c.drawString(72, y, f"Amount Claimed: {amount}")
        y -= 30

        c.drawString(72, y, "Damage Description:")
        y -= 18
        # Wrap the description text manually.
        from reportlab.lib.utils import simpleSplit
        wrapped = simpleSplit(
            damage_description, "Helvetica", 11, width - 144
        )
        for line in wrapped:
            c.drawString(72, y, line)
            y -= 14

        c.showPage()
        c.save()
        return buf.getvalue()

    return _build


# ---------------------------------------------------------------------------
# Email submission fixture (uses make_fax_pdf to build an attachment)
# ---------------------------------------------------------------------------
@pytest.fixture
def make_email_submission(make_fax_pdf) -> Any:
    """Factory: build an EmailClaimSubmission with one PDF attachment."""
    from claim_intake.schemas import EmailClaimSubmission

    def _build(**fax_kwargs) -> Any:
        pdf_bytes = make_fax_pdf(**fax_kwargs)
        return EmailClaimSubmission(
            sender="adjuster@shieldpoint.example",
            received_at=datetime.now(timezone.utc).isoformat(),
            subject="New claim submission — Alice Homeowner",
            attachments=[("claim_form.pdf", "application/pdf", pdf_bytes)],
        )

    return _build


# ---------------------------------------------------------------------------
# Fax submission fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def make_fax_submission(make_fax_pdf) -> Any:
    """Factory: build a FaxClaimSubmission."""
    from claim_intake.schemas import FaxClaimSubmission

    def _build(**fax_kwargs) -> Any:
        pdf_bytes = make_fax_pdf(**fax_kwargs)
        return FaxClaimSubmission(
            fax_number="+1-555-0100",
            received_at=datetime.now(timezone.utc).isoformat(),
            pdf_bytes=pdf_bytes,
        )

    return _build
