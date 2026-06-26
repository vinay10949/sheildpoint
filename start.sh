#!/usr/bin/env bash
# =============================================================================
# start.sh — bring up ShieldPoint agent framework + verify LM Studio desktop
# -----------------------------------------------------------------------------
# Architecture: LM Studio runs NATIVELY on the Mac. This script:
#   1. Verifies LM Studio desktop is running with Qwen loaded
#   2. Brings up the agent framework containers (currently just smoke-probe;
#      agent-api / postgres / redis will be added in Sprint 2)
#   3. Verifies a container can reach LM Studio via host.docker.internal
#   4. Prints ready banner with connection info for both host and container
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck disable=SC1091
source ./scripts/lib/common.sh

if [[ ! -f .env ]]; then
    die "Missing .env — copy .env.example and edit values first:\n  cp .env.example .env && \$EDITOR .env"
fi

# shellcheck disable=SC1090
set -a; source .env; set +a

# ----------------------------------------------------------------------------
# 1. Verify LM Studio desktop is up (host-side check)
# ----------------------------------------------------------------------------
log "Step 1/3 — Verifying LM Studio desktop on Mac..."
./scripts/verify-lm-studio-desktop.sh

# ----------------------------------------------------------------------------
# 2. Bring up agent framework containers
# ----------------------------------------------------------------------------
log "Step 2/3 — Bringing up agent framework containers..."
# Currently only the smoke-probe sidecar is defined (under the 'smoke' profile).
# When Sprint 2 adds agent-api / postgres / redis, they'll start here unconditionally.
"${DOCKER_COMPOSE[@]}" --profile smoke up -d smoke-probe

# ----------------------------------------------------------------------------
# 3. Verify container → host.docker.internal → LM Studio reachability
# ----------------------------------------------------------------------------
log "Step 3/3 — Verifying container can reach LM Studio via host.docker.internal..."

# Install curl in the alpine sidecar and probe LM Studio
docker exec shieldpoint-smoke-probe sh -c \
    'apk add --no-cache curl >/dev/null 2>&1 && curl -fsS --max-time 5 http://host.docker.internal:1234/v1/models' \
    >/dev/null 2>&1 \
    && ok "Container reachability verified (smoke-probe → http://host.docker.internal:1234)" \
    || die "Container cannot reach LM Studio. Check:\n  - LM Studio server is running on Mac (port 1234)\n  - Docker Desktop is up to date (host.docker.internal requires Docker Desktop 18.03+)\n  - LM Studio Settings → Server → 'Enable CORS' is checked (allows cross-origin from Docker)"

# ----------------------------------------------------------------------------
# Ready banner
# ----------------------------------------------------------------------------
cat <<EOF

$(ok "ShieldPoint agent framework is UP")

  LM Studio (Mac native):
    Host URL:       $(echo "${LM_STUDIO_BASE_URL_HOST:-http://localhost:1234/v1}" | sed 's|/v1$||')
    Loaded model:   ${QWEN_MODEL_ID:-qwen3.6-35b-a3b-q4_k_m}

  Agent framework (Docker):
    Network:        shieldpoint-net (172.28.0.0/16)
    LM Studio URL:  ${LM_STUDIO_BASE_URL:-http://host.docker.internal:1234/v1}
                    (use this URL inside agent containers)

Next:
  make smoke       # 5-stage acceptance test (health / inference / throughput / VRAM / network)
  make status      # live status of LM Studio + containers
  make down        # tear down agent framework (LM Studio desktop keeps running)

EOF
