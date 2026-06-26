#!/usr/bin/env bash
# =============================================================================
# status.sh — live status of LM Studio (Mac) + agent framework containers
# -----------------------------------------------------------------------------
# Reports:
#   - LM Studio desktop process state
#   - LM Studio HTTP health + loaded model
#   - Agent framework containers
#   - Throughput sample (3-run avg, 50 tokens)
# Exits 0 if healthy, 1 otherwise.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck disable=SC1091
source ./scripts/lib/common.sh

if [[ -f .env ]]; then
    # shellcheck disable=SC1090
    set -a; source .env; set +a
fi

HOST_URL="${LM_STUDIO_BASE_URL_HOST:-http://localhost:1234/v1}"
MODEL="${QWEN_MODEL_ID:-qwen3.6-35b-a3b-q4_k_m}"

# ----------------------------------------------------------------------------
# LM Studio desktop process
# ----------------------------------------------------------------------------
echo "=== LM Studio desktop process ==="
if pgrep -fi "LM Studio" >/dev/null 2>&1; then
    ok "LM Studio desktop app is running"
else
    warn "LM Studio desktop process not found (or pgrep can't see it)"
fi

# ----------------------------------------------------------------------------
# LM Studio HTTP health
# ----------------------------------------------------------------------------
echo
echo "=== LM Studio health (GET ${HOST_URL}/models) ==="
if HEALTH=$(curl -fsS --max-time 5 "${HOST_URL}/models" 2>/dev/null); then
    LOADED=$(echo "${HEALTH}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(', '.join(m.get('id','?') for m in d.get('data',[])))" 2>/dev/null || echo "<parse error>")
    ok "200 OK — loaded model(s): ${LOADED}"
else
    die "LM Studio unreachable at ${HOST_URL}/models — is the local server started?"
fi

# ----------------------------------------------------------------------------
# Agent framework containers
# ----------------------------------------------------------------------------
echo
echo "=== Agent framework containers ==="
"${DOCKER_COMPOSE[@]}" ps 2>/dev/null || warn "No agent framework containers running"

# ----------------------------------------------------------------------------
# Container reachability (if smoke-probe is up)
# ----------------------------------------------------------------------------
echo
echo "=== Container → LM Studio reachability ==="
if docker ps --format '{{.Names}}' | grep -q '^shieldpoint-smoke-probe$'; then
    if docker exec shieldpoint-smoke-probe sh -c \
        'curl -fsS --max-time 5 http://host.docker.internal:1234/v1/models' \
        >/dev/null 2>&1; then
        ok "smoke-probe → http://host.docker.internal:1234 → 200 OK"
    else
        fail "smoke-probe cannot reach LM Studio"
    fi
else
    warn "smoke-probe not running — start with: make dev"
fi

# ----------------------------------------------------------------------------
# Throughput sample
# ----------------------------------------------------------------------------
echo
echo "=== Throughput sample (50 tokens, 3 runs) ==="
python3 ./scripts/inference-bench.py \
    --base-url "${HOST_URL}" \
    --model "${MODEL}" \
    --prompt "Summarize the key coverages in a standard homeowners insurance policy." \
    --max-tokens 50 \
    --runs 3 \
    --quiet 2>/dev/null \
    || warn "throughput benchmark failed"

echo
ok "Status: healthy"
