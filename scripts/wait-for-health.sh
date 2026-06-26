#!/usr/bin/env bash
# =============================================================================
# wait-for-health.sh — poll LM Studio desktop /v1/models until 200 or timeout
# -----------------------------------------------------------------------------
# Used by start.sh to gate on LM Studio desktop being ready (model loaded,
# local server started). Polls the HOST URL (http://localhost:1234) since this
# script runs on the Mac, not inside a container.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# shellcheck disable=SC1091
source ./scripts/lib/common.sh

if [[ -f .env ]]; then
    # shellcheck disable=SC1090
    set -a; source .env; set +a
fi

HOST_URL="${LM_STUDIO_BASE_URL_HOST:-http://localhost:1234/v1}"
URL="${HOST_URL}/models"

MAX_WAIT="${LM_STUDIO_HEALTH_TIMEOUT:-120}"
POLL=3
START=$(date +%s)

log "Polling ${URL} (max wait ${MAX_WAIT}s, poll ${POLL}s)..."

while true; do
    if curl -fsS --max-time 5 "${URL}" >/dev/null 2>&1; then
        ELAPSED=$(( $(date +%s) - START ))
        ok "LM Studio healthy after ${ELAPSED}s."
        exit 0
    fi
    NOW=$(date +%s)
    if (( NOW - START > MAX_WAIT )); then
        die "Timed out after ${MAX_WAIT}s. Is LM Studio desktop local server started?"
    fi
    printf '.' >&2
    sleep "${POLL}"
done
