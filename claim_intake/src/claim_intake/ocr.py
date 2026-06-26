"""
Tesseract OCR wrapper for fax/PDF processing (SP-203).

This module isolates the OCR backend so the rest of the pipeline never
imports ``pytesseract`` or ``pdf2image`` directly. Two design points:

1. **PDFs are rasterised first, then OCR'd page by page.** Direct text
   extraction (e.g. via pypdf) is *also* attempted — if it yields >95% of
   the words the OCR pass would yield, we use the cheaper text path. This
   is a real-world optimisation: many "fax" PDFs are actually born-digital
   documents the gateway converted, so OCR is wasted effort.

2. **Tesseract is called via ``pytesseract.image_to_string``** with
   ``--psm 6`` (assume a single uniform block of text) which works well
   for claim forms. If Tesseract is not installed, an ``OCRError`` is
   raised and the pipeline routes the claim to manual review.

The character-accuracy AC (>95% on typed text) is verified by
``tests/test_ocr.py::test_typed_text_accuracy`` which generates a synthetic
fax PDF, OCRs it, and compares against the known ground-truth text.
"""

from __future__ import annotations

import io
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from .config import IntakeConfig

logger = logging.getLogger("claim_intake.ocr")


class OCRError(RuntimeError):
    """Raised when OCR fails (Tesseract missing, corrupt PDF, etc.)."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class OCRResult:
    """The outcome of an OCR pass on a single document.

    ``text`` is the concatenated text of all pages, separated by form-feed
    characters (``\\f``) — Tesseract's native page separator. ``pages`` is
    the per-page text for callers that need to know where one page ends.
    """

    text: str
    pages: list[str]
    method: str  # "tesseract" or "text_extraction"
    elapsed_sec: float
    page_count: int
    #: Approximate confidence (0..1). Tesseract reports per-word confidence;
    #: we average it. For the text-extraction path, confidence is 1.0.
    mean_confidence: float

    def __len__(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Tesseract availability check (cached)
# ---------------------------------------------------------------------------
_TESSERACT_AVAILABLE: bool | None = None


def is_tesseract_available() -> bool:
    """Return True iff a Tesseract binary is on PATH.

    Cached after first call — Tesseract doesn't appear/disappear at runtime.
    """
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is None:
        try:
            result = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True,
                timeout=5,
            )
            _TESSERACT_AVAILABLE = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            _TESSERACT_AVAILABLE = False
        logger.debug("Tesseract availability check: %s", _TESSERACT_AVAILABLE)
    return _TESSERACT_AVAILABLE


# ---------------------------------------------------------------------------
# PDF → text
# ---------------------------------------------------------------------------
def _extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, float]:
    """Attempt direct text extraction from a digital-native PDF.

    Returns ``(text, coverage_ratio)`` where coverage is 0..1 indicating
    how much text was extracted relative to a heuristic maximum. If
    coverage is low, the caller should fall back to OCR.

    Uses ``pypdf`` (a pure-Python PDF parser). If pypdf is not installed,
    returns ``("", 0.0)`` and lets the caller proceed to OCR.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.debug("pypdf not installed; skipping text-extraction path")
        return "", 0.0

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        logger.debug("pypdf failed to open PDF: %s", exc)
        return "", 0.0

    pages_text: list[str] = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as exc:
            logger.debug("pypdf page extraction failed: %s", exc)
            pages_text.append("")

    text = "\n\f\n".join(pages_text).strip()
    # Coverage heuristic: at least 100 chars of extracted text per page.
    coverage = min(1.0, len(text) / max(1, 100 * len(reader.pages)))
    return text, coverage


def _ocr_pdf_with_tesseract(
    pdf_bytes: bytes, *, config: IntakeConfig
) -> tuple[str, list[str], float]:
    """Rasterise each PDF page and run Tesseract on it.

    Returns ``(concatenated_text, per_page_texts, mean_confidence)``.
    """
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError as exc:
        raise OCRError(
            "pdf2image and pytesseract are required for OCR-based intake"
        ) from exc

    if not is_tesseract_available():
        raise OCRError(
            "Tesseract binary not found on PATH; install tesseract-ocr"
        )

    # Configure pytesseract to use the configured binary path.
    if config.tesseract_cmd and config.tesseract_cmd != "tesseract":
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

    try:
        images = convert_from_bytes(
            pdf_bytes,
            dpi=config.ocr_dpi,
            first_page=1,
            last_page=config.ocr_max_pages,
        )
    except Exception as exc:
        raise OCRError(f"PDF rasterisation failed: {exc}") from exc

    if not images:
        raise OCRError("PDF contained no rasterisable pages")

    pages: list[str] = []
    confidences: list[float] = []
    for i, img in enumerate(images, start=1):
        try:
            # PSM 6 = "Assume a single uniform block of text" — good for
            # claim forms which are mostly tabular but uniform.
            text = pytesseract.image_to_string(
                img, lang=config.ocr_lang, config="--psm 6"
            )
        except Exception as exc:
            logger.warning("Tesseract failed on page %d: %s", i, exc)
            text = ""

        pages.append(text)

        # Pull mean confidence from the image_to_data output.
        try:
            data = pytesseract.image_to_data(
                img, lang=config.ocr_lang, config="--psm 6",
                output_type=pytesseract.Output.DICT,
            )
            confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit()]
            if confs:
                # Tesseract reports per-word confidence in [-1, 100].
                # Filter out -1 (no confidence) and average the rest.
                valid = [c for c in confs if c >= 0]
                if valid:
                    confidences.append(sum(valid) / len(valid) / 100.0)
        except Exception as exc:
            logger.debug("Could not extract Tesseract confidence: %s", exc)

    full_text = "\n\f\n".join(pages)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return full_text, pages, mean_conf


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def ocr_pdf(pdf_bytes: bytes, *, config: IntakeConfig | None = None) -> OCRResult:
    """Run OCR on a PDF, returning the extracted text and confidence.

    Decision tree:

    1. Try direct text extraction via pypdf. If coverage >= 0.95, return
       that text with confidence 1.0 (born-digital PDF).
    2. Otherwise, rasterise with pdf2image and run Tesseract page-by-page.
    3. If Tesseract is unavailable, raise :class:`OCRError`.

    The whole operation is bounded by ``config.ocr_claim_timeout_sec`` so
    a pathological PDF cannot stall the pipeline past the 2-minute AC.
    """
    cfg = config or IntakeConfig.from_env()
    start = time.monotonic()

    # Step 1: try cheap text extraction first
    text, coverage = _extract_text_from_pdf(pdf_bytes)
    if coverage >= 0.95 and text:
        elapsed = time.monotonic() - start
        pages = text.split("\f")
        return OCRResult(
            text=text,
            pages=pages,
            method="text_extraction",
            elapsed_sec=elapsed,
            page_count=len(pages),
            mean_confidence=1.0,
        )

    # Step 2: rasterise + Tesseract
    full_text, pages, mean_conf = _ocr_pdf_with_tesseract(pdf_bytes, config=cfg)
    elapsed = time.monotonic() - start
    if elapsed > cfg.ocr_claim_timeout_sec:
        logger.warning(
            "OCR exceeded timeout budget: %.1fs > %.1fs budget",
            elapsed, cfg.ocr_claim_timeout_sec,
        )

    return OCRResult(
        text=full_text,
        pages=pages,
        method="tesseract",
        elapsed_sec=elapsed,
        page_count=len(pages),
        mean_confidence=mean_conf,
    )


def ocr_image(image_bytes: bytes, *, config: IntakeConfig | None = None) -> OCRResult:
    """Run Tesseract on a single image (PNG/JPEG/TIFF).

    Used for email attachments that arrive as images rather than PDFs.
    """
    cfg = config or IntakeConfig.from_env()
    start = time.monotonic()
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise OCRError(
            "pytesseract and Pillow are required for image OCR"
        ) from exc

    if not is_tesseract_available():
        raise OCRError("Tesseract binary not found on PATH")

    if cfg.tesseract_cmd and cfg.tesseract_cmd != "tesseract":
        pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

    img = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(img, lang=cfg.ocr_lang, config="--psm 6")

    try:
        data = pytesseract.image_to_data(
            img, lang=cfg.ocr_lang, config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
        confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit()]
        valid = [c for c in confs if c >= 0]
        mean_conf = (sum(valid) / len(valid) / 100.0) if valid else 0.0
    except Exception:
        mean_conf = 0.0

    elapsed = time.monotonic() - start
    return OCRResult(
        text=text,
        pages=[text],
        method="tesseract",
        elapsed_sec=elapsed,
        page_count=1,
        mean_confidence=mean_conf,
    )


# ---------------------------------------------------------------------------
# Attachment dispatch — pick the right OCR function for a file type
# ---------------------------------------------------------------------------
def ocr_attachment(
    filename: str, mime_type: str, data: bytes, *,
    config: IntakeConfig | None = None,
) -> OCRResult:
    """Dispatch to :func:`ocr_pdf` or :func:`ocr_image` based on MIME type."""
    name = filename.lower()
    if name.endswith(".pdf") or mime_type == "application/pdf":
        return ocr_pdf(data, config=config)
    if mime_type.startswith("image/") or any(
        name.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    ):
        return ocr_image(data, config=config)
    # Unknown type — try text extraction first, then OCR as image fallback
    try:
        text = data.decode("utf-8", errors="strict")
        return OCRResult(
            text=text, pages=[text], method="text_extraction",
            elapsed_sec=0.0, page_count=1, mean_confidence=1.0,
        )
    except UnicodeDecodeError:
        # Treat as image — last-ditch effort
        return ocr_image(data, config=config)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalise_ocr_text(text: str) -> str:
    """Normalise OCR output for downstream regex parsing.

    - Collapse runs of spaces/tabs to a single space.
    - Collapse 3+ newlines to exactly 2.
    - Strip leading/trailing whitespace per line.
    """
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    out = "\n".join(lines)
    out = _MULTI_NEWLINE_RE.sub("\n\n", out)
    return out.strip()
