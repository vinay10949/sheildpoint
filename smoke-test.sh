#!/usr/bin/env bash
# =============================================================================
# smoke-test.sh — 5-stage acceptance test for ShieldPoint LM Studio (SP-100)
# -----------------------------------------------------------------------------
# Architecture: LM Studio runs NATIVELY on Mac. Docker containers reach it via
# host.docker.internal. This smoke test verifies:
#
#   1. HEALTH           GET /v1/models on Mac returns 200 + Qwen model id
#   2. INFERENCE        POST /v1/chat/completions 50-token prompt < 2s
#   3. THROUGHPUT       200-token generation, >=80 tok/s accept / <50 alert
#   4. VRAM             LM Studio API reports VRAM usage (or LM Studio UI hint)
#   5. CONTAINER-TO-HOST Alpine sidecar on shieldpoint-net reaches
#                       http://host.docker.internal:1234/v1/models
#
# Exit code: 0 if all PASS, 1 if any FAIL, 2 if any WARN-only.
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
CONTAINER_URL="http://host.docker.internal:1234/v1"
MODEL="${QWEN_MODEL_ID:-qwen3.6-35b-a3b-q4_k_m}"

PASS=0; WARN=0; FAIL=0
results=()
record() { results+=("$1"); }
case_fn() {
    case "$1" in
        PASS) PASS=$((PASS+1)); record "$(ok "PASS") $2" ;;
        WARN) WARN=$((WARN+1)); record "$(warn "WARN") $2" ;;
        FAIL) FAIL=$((FAIL+1)); record "$(fail "FAIL") $2" ;;
    esac
}

# ============================================================================
# Stage 1 — HEALTH (Mac host URL)
# ============================================================================
echo "=== Stage 1/5: HEALTH (GET ${HOST_URL}/models) ==="
if HEALTH=$(curl -fsS --max-time 10 "${HOST_URL}/models" 2>/dev/null); then
    LOADED=$(echo "${HEALTH}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "<parse error>")
    if [[ "${LOADED}" == "${MODEL}" ]]; then
        case_fn PASS "loaded model matches '${MODEL}'"
    else
        case_fn WARN "loaded model '${LOADED}' != expected '${MODEL}' — check LM Studio desktop Local Server model dropdown"
    fi
else
    case_fn FAIL "GET ${HOST_URL}/models did not return 200 — is LM Studio local server started?"
fi

# ============================================================================
# Stage 2 — INFERENCE (latency)
# ============================================================================
echo
echo "=== Stage 2/5: INFERENCE (50-token chat completion, <2s target) ==="
PAYLOAD=$(cat <<EOF
{"model":"${MODEL}","messages":[{"role":"user","content":"Reply with one short sentence about insurance."}],"max_tokens":50,"temperature":0.1,"stream":false}
EOF
)
START_NS=$(python3 -c "import time; print(int(time.time_ns()))")
RESP=$(curl -fsS --max-time 30 -X POST "${HOST_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${LM_STUDIO_API_KEY:-lm-studio}" \
    -d "${PAYLOAD}" 2>/dev/null || echo "")
END_NS=$(python3 -c "import time; print(int(time.time_ns()))")
LATENCY_MS=$(( (END_NS - START_NS) / 1000000 ))

if [[ -z "${RESP}" ]] || ! echo "${RESP}" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    case_fn FAIL "POST /v1/chat/completions did not return valid JSON"
else
    CONTENT=$(echo "${RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:60])" 2>/dev/null || echo "<parse error>")
    MAX_SEC="${SMOKE_INFERENCE_LATENCY_MAX_SEC:-2}"
    if (( LATENCY_MS <= MAX_SEC * 1000 )); then
        case_fn PASS "50-token completion in ${LATENCY_MS}ms (<=${MAX_SEC}s) — content: '${CONTENT}...'"
    else
        case_fn WARN "50-token completion took ${LATENCY_MS}ms (> ${MAX_SEC}s) — Mac may be slower than A100 target. Adjust SMOKE_INFERENCE_LATENCY_MAX_SEC in .env if this is expected."
    fi
fi

# ============================================================================
# Stage 3 — THROUGHPUT (200-token generation)
# ============================================================================
echo
echo "=== Stage 3/5: THROUGHPUT (200-token generation) ==="
TPS=$(python3 ./scripts/inference-bench.py \
    --base-url "${HOST_URL}" \
    --model "${MODEL}" \
    --prompt "List 10 common reasons auto insurance claims are denied, with a one-sentence explanation for each." \
    --max-tokens 200 \
    --runs 1 \
    --print-tps-only 2>/dev/null || echo "0")

ACCEPT_TPS="${SMOKE_THROUGHPUT_ACCEPT_TPS:-80}"
ALERT_TPS="${SMOKE_THROUGHPUT_ALERT_TPS:-50}"
if (( $(echo "${TPS} >= ${ACCEPT_TPS}" | bc -l 2>/dev/null || echo 0) )); then
    case_fn PASS "throughput ${TPS} tok/s (>= ${ACCEPT_TPS} accept threshold)"
elif (( $(echo "${TPS} >= ${ALERT_TPS}" | bc -l 2>/dev/null || echo 0) )); then
    case_fn WARN "throughput ${TPS} tok/s (between alert ${ALERT_TPS} and accept ${ACCEPT_TPS} — Mac may not hit A100 targets)"
else
    case_fn WARN "throughput ${TPS} tok/s (< alert threshold ${ALERT_TPS}) — Mac hardware may not meet A100 spec. Adjust SMOKE_THROUGHPUT_ACCEPT_TPS in .env to your hardware baseline."
fi

# ============================================================================
# Stage 4 — VRAM (LM Studio API / OS check — no nvidia-smi on Mac)
# ============================================================================
echo
echo "=== Stage 4/5: VRAM (LM Studio desktop UI hint) ==="
echo "  On Mac, VRAM is shared unified memory — there's no nvidia-smi equivalent."
echo "  Check LM Studio desktop → Local Server tab → 'Memory' gauge."
echo "  Target: <= ${SMOKE_VRAM_MAX_GB:-10} GB (per SP-100 AC)"
echo
# We can't auto-measure VRAM on Mac. Default to a manual-check WARN, but allow
# the operator to acknowledge it (after verifying the LM Studio Memory gauge)
# by setting SMOKE_VRAM_ACK=1 in .env — that turns it into a PASS for clean exit.
if [[ "${SMOKE_VRAM_ACK:-0}" == "1" ]]; then
    case_fn PASS "VRAM acknowledged manually (SMOKE_VRAM_ACK=1) — operator verified <= ${SMOKE_VRAM_MAX_GB:-10} GB in LM Studio UI"
else
    case_fn WARN "VRAM auto-check not available on Mac — verify manually in LM Studio UI (target: <= ${SMOKE_VRAM_MAX_GB:-10} GB). Set SMOKE_VRAM_ACK=1 in .env once verified."
fi

# ============================================================================
# Stage 5 — CONTAINER-TO-HOST (network reachability AC — the key one)
# ============================================================================
echo
echo "=== Stage 5/5: CONTAINER-TO-HOST (alpine → http://host.docker.internal:1234) ==="
# Ensure smoke-probe is up
"${DOCKER_COMPOSE[@]}" --profile smoke up -d --force-recreate smoke-probe >/dev/null 2>&1
sleep 2

# Install curl in alpine sidecar and probe LM Studio via host.docker.internal
PROBE_OUT=$(docker exec shieldpoint-smoke-probe sh -c \
    'apk add --no-cache curl >/dev/null 2>&1 && curl -fsS --max-time 5 '"${CONTAINER_URL}"'/models' \
    2>/dev/null || echo "")

if [[ -n "${PROBE_OUT}" ]] && echo "${PROBE_OUT}" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['data'][0]['id']" 2>/dev/null; then
    PROBE_MODEL=$(echo "${PROBE_OUT}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null)
    case_fn PASS "alpine sidecar on shieldpoint-net reached ${CONTAINER_URL}/models — got model '${PROBE_MODEL}'. Docker network AC met."
else
    case_fn FAIL "alpine sidecar could not reach ${CONTAINER_URL}/models. Common causes:\n    - LM Studio server not started (open desktop app → Local Server → Start Server)\n    - Docker Desktop too old (need 18.03+ for host.docker.internal)\n    - LM Studio server bound to wrong interface (Settings → Server → bind to 0.0.0.0 or 127.0.0.1, NOT 0.0.0.0:1234 only)"
fi

# ============================================================================
# Summary
# ============================================================================
echo
echo "==================================================================="
echo "  SMOKE TEST SUMMARY"
echo "==================================================================="
for r in "${results[@]}"; do echo "  ${r}"; done
echo "==================================================================="
echo "  PASS: ${PASS}   WARN: ${WARN}   FAIL: ${FAIL}"
echo "==================================================================="

if (( FAIL > 0 )); then
    die "Smoke test FAILED — see above. Do NOT proceed to agent development."
elif (( WARN > 0 )); then
    warn "Smoke test PASSED with warnings — review before declaring SP-100 done."
    exit 2
else
    ok "All smoke tests PASSED. SP-100 acceptance criteria met."
    exit 0
fi
