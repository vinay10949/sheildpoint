#!/usr/bin/env bash
# =============================================================================
# vram-check.sh — print LM Studio memory usage hint (Mac native)
# -----------------------------------------------------------------------------
# On Mac, VRAM is shared unified memory — there's no nvidia-smi equivalent.
# LM Studio desktop exposes memory usage in the Local Server tab UI.
#
# This script:
#   1. Prints the target VRAM ceiling from .env
#   2. Attempts to query LM Studio's metrics endpoint (if available)
#   3. Falls back to a UI hint if the endpoint is not exposed
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
MAX_GB="${SMOKE_VRAM_MAX_GB:-10}"

echo "Target VRAM ceiling: ${MAX_GB} GB (per SP-100 AC)"
echo

# LM Studio may expose a /v1/metrics or similar endpoint — try a few.
echo "Attempting to query LM Studio memory metrics..."
for endpoint in "/v1/metrics" "/metrics" "/v1/system" "/system"; do
    RESP=$(curl -fsS --max-time 3 "${HOST_URL}${endpoint}" 2>/dev/null || echo "")
    if [[ -n "${RESP}" ]] && echo "${RESP}" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
        echo "  ${endpoint}: ${RESP}"
        echo
        echo "Parsed memory fields:"
        echo "${RESP}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
def walk(obj, prefix=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(t in k.lower() for t in ['mem', 'vram', 'gpu', 'usage']):
                print(f'  {prefix}{k}: {v}')
            walk(v, prefix + k + '.')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, prefix + f'[{i}].')
walk(d)
" 2>/dev/null || echo "  <no memory-related fields found>"
        exit 0
    fi
done

warn "LM Studio does not expose a memory metrics endpoint via HTTP."
echo
echo "Manual check — open LM Studio desktop → Local Server tab → look for:"
echo "  - 'Memory' gauge (should show <= ${MAX_GB} GB)"
echo "  - 'Context' usage indicator"
echo "  - Active model name in dropdown"
echo
echo "Or check macOS unified memory pressure:"
echo "  Activity Monitor → Memory tab → look for 'LM Studio' process"
echo "  Or:  ps -p \$(pgrep -fi 'LM Studio') -o rss,vsz"
