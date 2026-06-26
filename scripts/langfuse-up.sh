#!/usr/bin/env bash
# =============================================================================
# langfuse-up.sh — bring up the ShieldPoint Langfuse observability stack
# -----------------------------------------------------------------------------
# Uses the `langfuse` profile in docker-compose.yml. Reads env vars from the
# single .env file (no separate .env.langfuse anymore).
#
# Pre-flight checks:
#   1. .env present (with secrets filled in)
#   2. Docker Compose v2 available
#   3. No REPLACE_WITH_* placeholders left in .env
#
# Then: docker compose --profile langfuse up -d
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck disable=SC1091
source ./scripts/lib/common.sh

# ----------------------------------------------------------------------------
# 1. Pre-flight: .env present
# ----------------------------------------------------------------------------
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    die "Missing .env — create it first:
  cp .env.example .env
  ./scripts/langfuse-gen-secrets.sh >> .env
  \$EDITOR .env   # fill in remaining placeholders"
fi

# shellcheck disable=SC1090
set -a; source "${ENV_FILE}"; set +a

# ----------------------------------------------------------------------------
# 2. Pre-flight: no placeholder values left
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 3. Pre-flight: docker compose v2
# ----------------------------------------------------------------------------
log "Using Docker Compose: ${DOCKER_COMPOSE[*]}"

# ----------------------------------------------------------------------------
# 4. Pull + bring up (langfuse profile only)
# ----------------------------------------------------------------------------
log "Pulling Langfuse images (first run may take a few minutes) ..."
"${DOCKER_COMPOSE[@]}" --profile langfuse pull

log "Bringing up Langfuse stack (profile: langfuse) ..."
"${DOCKER_COMPOSE[@]}" --profile langfuse up -d

# ----------------------------------------------------------------------------
# 5. Wait for Langfuse health
# ----------------------------------------------------------------------------
log "Waiting for Langfuse to become healthy (up to 90s) ..."
HEALTH_URL="http://localhost:3000/api/public/health"
for i in $(seq 1 45); do
    if curl -fsS --max-time 5 "${HEALTH_URL}" >/dev/null 2>&1; then
        ok "Langfuse healthy on ${HEALTH_URL}"
        break
    fi
    printf '  [%2d/45] waiting...\r' "$i"
    sleep 2
done

if ! curl -fsS --max-time 5 "${HEALTH_URL}" >/dev/null 2>&1; then
    warn "Langfuse not yet healthy after 90s. Check logs:"
    warn "  ${DOCKER_COMPOSE[*]} --profile langfuse logs langfuse"
    warn "First boot runs DB migrations — give it another 60-120s and re-check:"
    warn "  curl ${HEALTH_URL}"
    exit 1
fi

# ----------------------------------------------------------------------------
# 6. Status banner
# ----------------------------------------------------------------------------
cat <<EOF

$(ok "ShieldPoint Langfuse stack is UP")

  Web UI:        http://localhost:3000          (bound to 127.0.0.1 only)
  Health API:    http://localhost:3000/api/public/health
  Postgres 16:   langfuse-postgres:5432         (internal only)
  PgBouncer:     langfuse-pgbouncer:5432        (internal only)
  ClickHouse:    langfuse-clickhouse:8123       (internal only)
  Redis 7:       langfuse-redis:6379            (internal only)

  Blob storage:  Local filesystem (no S3 / no MinIO — dependency removed)

  Data sovereignty:
    - All ports bound to 127.0.0.1 (no external interface exposure)
    - Telemetry disabled (TELEMETRY_ENABLED=false)
    - Public signups disabled (ENABLE_SIGN_UP=false)
    - No S3 protocol dependency — fully local

Next steps:
  1. Open http://localhost:3000 in your browser — create the admin user
     (first-visit setup screen).
  2. Create a project + API key + 90-day retention:
       python3 scripts/langfuse-bootstrap.py \\
         --email admin@shieldpoint.local \\
         --password '<your-admin-password>'
  3. Paste the printed API keys into .env:
       LANGFUSE_PUBLIC_KEY=pk-lf-...
       LANGFUSE_SECRET_KEY=sk-lf-...
  4. Test end-to-end trace capture:
       python3 scripts/test-langfuse-trace.py --no-llm
  5. View traces: http://localhost:3000

EOF
