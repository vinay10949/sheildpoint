#!/usr/bin/env bash
# =============================================================================
# stop.sh — tear down ShieldPoint agent framework containers
# -----------------------------------------------------------------------------
# LM Studio desktop on the Mac is NOT affected — it keeps running. To stop
# LM Studio itself, use the app's "Stop Server" button in the Local Server tab.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck disable=SC1091
source ./scripts/lib/common.sh

log "Stopping ShieldPoint agent framework containers..."
"${DOCKER_COMPOSE[@]}" down --remove-orphans

ok "Agent framework stopped. LM Studio desktop on Mac is still running."
warn "To stop LM Studio itself: open the app → Local Server tab → click 'Stop Server'."
