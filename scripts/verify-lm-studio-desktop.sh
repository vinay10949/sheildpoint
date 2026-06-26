#!/usr/bin/env bash
# =============================================================================
# verify-lm-studio-desktop.sh — confirm LM Studio desktop is running on Mac
# -----------------------------------------------------------------------------
# Checks:
#   1. LM Studio desktop app is running (pgrep "LM Studio")
#   2. Local server is started on http://localhost:1234 (GET /v1/models → 200)
#   3. Qwen3.6 35B A3B model is loaded (model id in response matches .env)
#   4. Inference works (50-token chat completion)
#
# If any check fails, prints actionable instructions for the LM Studio GUI.
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
EXPECTED_MODEL="${QWEN_MODEL_ID:-qwen3.6-35b-a3b-q4_k_m}"

echo "==================================================================="
echo "  Verifying LM Studio desktop (Mac local)"
echo "==================================================================="
echo

# ----------------------------------------------------------------------------
# 1. Process check
# ----------------------------------------------------------------------------
echo "=== 1/4 — LM Studio desktop process ==="
if pgrep -fi "LM Studio" >/dev/null 2>&1; then
    ok "LM Studio desktop app is running"
else
    warn "LM Studio desktop process not found via pgrep"
    warn "(if you're on Linux or the binary name differs, this check may be a false negative)"
    warn "Continuing — the HTTP check below is authoritative."
fi

# ----------------------------------------------------------------------------
# 2. HTTP health
# ----------------------------------------------------------------------------
echo
echo "=== 2/4 — Local server on ${HOST_URL} ==="
if HEALTH=$(curl -fsS --max-time 5 "${HOST_URL}/models" 2>/dev/null); then
    ok "GET /v1/models → 200 OK"
else
    fail "GET /v1/models failed at ${HOST_URL}/models"
    cat >&2 <<EOF

  Action required in LM Studio desktop:

  1. Open LM Studio app
  2. Click the "Local Server" icon (right-side toolbar, '<->' symbol)
  3. Click "Start Server"
  4. Confirm it shows "Server running on http://localhost:1234"
  5. Re-run: ./scripts/verify-lm-studio-desktop.sh

EOF
    exit 1
fi

# ----------------------------------------------------------------------------
# 3. Model loaded
# ----------------------------------------------------------------------------
echo
echo "=== 3/4 — Qwen model loaded ==="
LOADED_MODELS=$(echo "${HEALTH}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for m in d.get('data', []):
    print('  - ' + m.get('id', '<unknown>'))
" 2>/dev/null || echo "  <parse error>")

echo "Models currently loaded in LM Studio:"
echo "${LOADED_MODELS}"

if echo "${HEALTH}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
ids = [m.get('id', '') for m in d.get('data', [])]
sys.exit(0 if any('${EXPECTED_MODEL}' in i for i in ids) else 1)
" 2>/dev/null; then
    ok "Qwen model '${EXPECTED_MODEL}' is loaded"
else
    fail "Qwen model '${EXPECTED_MODEL}' NOT loaded in LM Studio"
    cat >&2 <<EOF

  Action required in LM Studio desktop:

  1. Open LM Studio app
  2. Click "My Models" (folder icon in left toolbar)
  3. Find "Qwen3.6 35B A3B GGUF Q4_K_M" (or whatever you downloaded)
  4. Click the model to load it (wait for "Model loaded" status)
  5. Switch to "Local Server" tab
  6. From the model dropdown at the top, select the loaded Qwen model
  7. Click "Start Server"
  8. Re-run: ./scripts/verify-lm-studio-desktop.sh

EOF
    exit 1
fi

# ----------------------------------------------------------------------------
# 4. Inference test
# ----------------------------------------------------------------------------
echo
echo "=== 4/4 — Inference test (50 tokens) ==="
PAYLOAD=$(cat <<EOF
{"model":"${EXPECTED_MODEL}","messages":[{"role":"user","content":"Reply with one short sentence about insurance."}],"max_tokens":50,"temperature":0.1,"stream":false}
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
    fail "Inference test failed"
    exit 1
fi

CONTENT=$(echo "${RESP}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['choices'][0]['message']['content'][:80])
" 2>/dev/null || echo "<parse error>")

MAX_SEC="${SMOKE_INFERENCE_LATENCY_MAX_SEC:-2}"
if (( LATENCY_MS <= MAX_SEC * 1000 )); then
    ok "Inference OK in ${LATENCY_MS}ms (<=${MAX_SEC}s target)"
else
    warn "Inference took ${LATENCY_MS}ms (> ${MAX_SEC}s) — Mac may be slower than A100 target"
fi
echo "  Response: '${CONTENT}...'"

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo
echo "==================================================================="
ok "LM Studio desktop is ready"
echo "==================================================================="
echo "  Host URL (Mac terminal / browser):  ${HOST_URL}"
echo "  Container URL (agent framework):    http://host.docker.internal:1234/v1"
echo "  Loaded model:                       ${EXPECTED_MODEL}"
echo "==================================================================="
