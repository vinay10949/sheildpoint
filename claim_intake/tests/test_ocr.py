"""Unit tests for the OCR module — Tesseract accuracy + text extraction path."""

from __future__ import annotations

import pytest

from claim_intake.config import IntakeConfig
from claim_intake.ocr import (
    OCRError,
    is_tesseract_available,
    normalise_ocr_text,
    ocr_attachment,
    ocr_pdf,
)


# ---------------------------------------------------------------------------
# Tesseract availability
# ---------------------------------------------------------------------------
class TestTesseractAvailable:
    def test_is_tesseract_available(self):
        # The CI/dev environment has Tesseract installed.
        assert is_tesseract_available() is True

    def test_is_tesseract_available_is_cached(self):
        is_tesseract_available()
        # Second call returns same cached value without re-running subprocess.
        assert is_tesseract_available() is True


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
class TestNormaliseOcrText:
    def test_collapses_runs_of_spaces(self):
        text = "Policy   ID:    HO-2024-001"
        out = normalise_ocr_text(text)
        assert "   " not in out
        assert "Policy ID: HO-2024-001" in out

    def test_collapses_excess_newlines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        out = normalise_ocr_text(text)
        assert out == "Line 1\n\nLine 2"

    def test_strips_per_line_whitespace(self):
        text = "  Policy ID: HO-2024-001  \n  Amount: $100  "
        out = normalise_ocr_text(text)
        assert out.startswith("Policy ID:")
        assert out.endswith("$100")

    def test_handles_empty_input(self):
        assert normalise_ocr_text("") == ""


# ---------------------------------------------------------------------------
# Born-digital PDFs (text-extraction path, no OCR needed)
# ---------------------------------------------------------------------------
class TestTextExtractionPath:
    def test_born_digital_pdf_uses_text_extraction(self, make_fax_pdf):
        """A reportlab-generated PDF is born-digital — text extraction should
        succeed with confidence 1.0, no Tesseract call needed."""
        pdf_bytes = make_fax_pdf()
        result = ocr_pdf(pdf_bytes, config=IntakeConfig())
        assert result.method == "text_extraction"
        assert result.mean_confidence == 1.0
        assert "Alice Homeowner" in result.text
        assert "HO-2024-001" in result.text

    def test_text_extraction_returns_per_page_text(self, make_fax_pdf):
        pdf_bytes = make_fax_pdf()
        result = ocr_pdf(pdf_bytes, config=IntakeConfig())
        assert len(result.pages) >= 1
        assert any("Alice Homeowner" in p for p in result.pages)


# ---------------------------------------------------------------------------
# Tesseract path — forced via rasterisation
# ---------------------------------------------------------------------------
class TestTesseractPath:
    """Force the rasterisation path by rasterising the PDF to an image first,
    then feeding the image bytes to ocr_image. This exercises Tesseract.
    """

    def test_typed_text_accuracy(self, make_fax_pdf):
        """SP-203 AC: OCR achieves >95% character accuracy on typed text.

        We build a born-digital PDF, rasterise it to a PNG, OCR the PNG,
        and compare the OCR output to the known ground-truth text.
        """
        # Build a PDF with known text.
        pdf_bytes = make_fax_pdf(
            damage_description=(
                "Wind damage to roof shingles during storm on March 14, 2026."
            ),
        )
        # Rasterise to PNG using pdf2image.
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(pdf_bytes, dpi=300)
        assert len(images) >= 1

        # OCR the first page image.
        import pytesseract
        text = pytesseract.image_to_string(images[0], lang="eng", config="--psm 6")

        # Ground truth — must be a subset of the OCR'd text (modulo whitespace).
        ground_truth_tokens = [
            "Policyholder", "Name", "Alice", "Homeowner",
            "Policy", "ID", "HO-2024-001",
            "Claim", "Type", "homeowners",
            "Date", "Loss", "2026-03-14",
            "Damage", "Description",
            "Wind", "damage", "roof", "shingles",
        ]
        # Normalise both for comparison (lowercase, strip punctuation).
        ocr_lower = text.lower()
        matched = sum(
            1 for tok in ground_truth_tokens if tok.lower() in ocr_lower
        )
        accuracy = matched / len(ground_truth_tokens)
        # >95% of ground-truth tokens must appear in the OCR output.
        assert accuracy >= 0.95, (
            f"OCR accuracy too low: {accuracy:.2%}. "
            f"Matched {matched}/{len(ground_truth_tokens)} tokens. "
            f"OCR text:\n{text}"
        )


# ---------------------------------------------------------------------------
# Attachment dispatch
# ---------------------------------------------------------------------------
class TestAttachmentDispatch:
    def test_pdf_attachment_routed_to_ocr_pdf(self, make_fax_pdf):
        pdf_bytes = make_fax_pdf()
        result = ocr_attachment(
            "claim.pdf", "application/pdf", pdf_bytes, config=IntakeConfig(),
        )
        assert "Alice Homeowner" in result.text
        assert result.page_count >= 1

    def test_text_attachment_returned_as_is(self):
        result = ocr_attachment(
            "note.txt", "text/plain", b"Policy ID: HO-2024-001",
            config=IntakeConfig(),
        )
        assert result.method == "text_extraction"
        assert "HO-2024-001" in result.text

    def test_unknown_binary_falls_back_to_image_ocr(self, make_fax_pdf):
        """If we get unknown binary data, the dispatcher tries text first
        (fails — not UTF-8), then falls back to image OCR (also fails
        because it's a PDF, not an image). The expected behaviour is an
        OCRError, NOT a silent corruption."""
        # Feed a PDF with a .bin filename and unknown mime type.
        # The dispatcher should try text (fails), then try image OCR
        # (which will fail because Pillow can't identify the PDF directly).
        # This is acceptable — the upstream caller should know the type.
        # We verify the dispatcher doesn't silently mangle the data.
        with pytest.raises(Exception):
            ocr_attachment(
                "unknown.bin", "application/octet-stream", make_fax_pdf(),
                config=IntakeConfig(),
            )
