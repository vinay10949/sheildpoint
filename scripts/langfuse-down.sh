#!/usr/bin/env bash
# =============================================================================
# langfuse-down.sh — tear down the ShieldPoint Langfuse stack
# -----------------------------------------------------------------------------
# Uses the `langfuse` profile in docker-compose.yml. Reads env vars from the
# single .env file (no separate .env.langfuse anymore).
#
# By default, keeps persistent volumes (traces preserved across restarts).
# Use --volumes to wipe all trace data (DESTRUCTIVE — confirms first).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source ./scripts/lib/common.sh

if [[ "${1:-}" == "--volumes" || "${1:-}" == "-v" ]]; then
    # Destructive: wipe all trace data
    read -r -p "This will DELETE all Langfuse traces, observations, and config. Type 'WIPE' to confirm: " CONFIRM
    if [[ "${CONFIRM}" != "WIPE" ]]; then
        die "Aborted — volumes preserved."
    fi
    warn "Tearing down stack and DELETING volumes ..."
    "${DOCKER_COMPOSE[@]}" --profile langfuse down -v
    ok "Stack down + volumes deleted."
else
    log "Tearing down Langfuse containers (volumes preserved) ..."
    "${DOCKER_COMPOSE[@]}" --profile langfuse down
    ok "Stack down. Traces preserved in Docker volumes."
    log "To wipe all trace data: $0 --volumes"
fi
