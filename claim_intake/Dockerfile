# =============================================================================
# ShieldPoint Claim Intake Service — Dockerfile (SP-203)
# -----------------------------------------------------------------------------
# Builds a production-ready image for the claim intake API + IMAP poller +
# Tesseract OCR pipeline.
#
# System dependencies:
#   - tesseract-ocr + tesseract-ocr-eng  — OCR engine
#   - poppler-utils                       — pdftoppm for PDF rasterisation
#
# Python dependencies: see pyproject.toml.
# =============================================================================

FROM python:3.12-slim AS base

# ---- System dependencies -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python app ----------------------------------------------------------
WORKDIR /app

# Copy the package and install it.
COPY pyproject.toml requirements.txt README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY samples/ ./samples/

RUN pip install --no-cache-dir -e .

# ---- Runtime -------------------------------------------------------------
EXPOSE 8001

# Health check via the /health endpoint.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD curl -fsS http://localhost:8001/health || exit 1

# Run the FastAPI app via uvicorn.
CMD ["python", "-m", "uvicorn", "claim_intake.api:app", \
     "--host", "0.0.0.0", "--port", "8001", \
     "--log-level", "info"]

# ---- Labels --------------------------------------------------------------
LABEL shieldpoint.project="claims-automation" \
      shieldpoint.layer="intake" \
      shieldpoint.service="claim-intake" \
      shieldpoint.ticket="SP-203"
