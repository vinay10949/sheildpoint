# =============================================================================
# ShieldPoint Agent Framework — Dockerfile
# -----------------------------------------------------------------------------
# Ticket: SHLD-13 — "Dockerfile builds agent container with all dependencies,
# connects to LM Studio and Langfuse."
#
# Builds a slim Python 3.11 image with the `shieldpoint_agents` package
# installed (editable, so the legacy `agent_framework/` directory at the repo
# root stays importable). The container connects to:
#
#   - LM Studio    at http://host.docker.internal:1234/v1
#   - Langfuse     at http://langfuse:3000
#
# Both are configured via env vars in docker-compose.agent.yml.
# =============================================================================

FROM python:3.11-slim AS base

# ---- System deps ----------------------------------------------------------
# - curl  : used by the HEALTHCHECK
# - build-essential + gcc : needed to compile jsonschema/pydantic C extensions
#   on fresh images (no-op if wheels are available).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Install the package --------------------------------------------------
# Copy the entire repo so the editable install of `shieldpoint_agents` can
# also pick up the sibling `agent_framework/` directory via the bootstrap
# in src/shieldpoint_agents/_bootstrap.py.
COPY . /app

# Install runtime deps from pyproject.toml + the [api] extras (FastAPI/uvicorn)
# so the container can serve the optional /health and /run endpoints. Dev
# deps (pytest etc.) are NOT installed — those belong to the test image.
RUN pip install --no-cache-dir \
        "/app/shieldpoint_agents[api]"

# ---- Runtime user ---------------------------------------------------------
# Non-root user for the agent process. UID/GID 1000 are conventional for the
# first non-root user on most Linux distros.
RUN useradd --uid 1000 --create-home --shell /bin/bash shieldpoint
USER shieldpoint

# ---- Environment ----------------------------------------------------------
# These are defaults for local dev. In production, override via
# docker-compose.agent.yml or Kubernetes env vars.
ENV LM_STUDIO_BASE_URL=http://host.docker.internal:1234/v1 \
    LM_STUDIO_API_KEY=lm-studio \
    QWEN_MODEL_ID=qwen3.6-35b-a3b-q4_k_m \
    LLM_TIMEOUT_SEC=10 \
    MAX_REACT_ITERATIONS=5 \
    LANGFUSE_HOST=http://langfuse:3000 \
    LANGFUSE_ENABLED=true \
    SHIELDPOINT_ENV=production \
    SHIELDPOINT_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose the FastAPI demo server port (only used when running `api` entrypoint).
EXPOSE 8000

# ---- Healthcheck ----------------------------------------------------------
# The default entrypoint starts a tiny FastAPI server exposing /health. If
# you override the entrypoint (e.g. to run a worker), replace this check
# with a process-level probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# ---- Default entrypoint: serve the demo API -------------------------------
# Override with `--entrypoint python -m shieldpoint_agents.example` to run
# the CLI demo instead.
CMD ["python", "-m", "shieldpoint_agents.api"]
