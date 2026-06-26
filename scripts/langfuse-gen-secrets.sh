#!/usr/bin/env bash
# =============================================================================
# langfuse-gen-secrets.sh — generate cryptographically strong secrets for the
# ShieldPoint Langfuse stack. Output is appended to .env (or stdout if
# redirected).
#
# Usage:
#   ./scripts/langfuse-gen-secrets.sh                # prints to stdout
#   ./scripts/langfuse-gen-secrets.sh >> .env        # appends to single .env
# =============================================================================
set -euo pipefail

NEXTAUTH_SECRET=$(openssl rand -base64 32)
SALT=$(openssl rand -base64 32)
ENCRYPTION_KEY=$(openssl rand -base64 32)
DB_PASSWORD=$(openssl rand -hex 24)
CLICKHOUSE_PASSWORD=$(openssl rand -hex 24)

cat <<EOF
# ---- Auto-generated secrets (langfuse-gen-secrets.sh @ $(date -u +"%Y-%m-%dT%H:%M:%SZ")) ----
LANGFUSE_NEXTAUTH_SECRET=${NEXTAUTH_SECRET}
LANGFUSE_SALT=${SALT}
LANGFUSE_ENCRYPTION_KEY=${ENCRYPTION_KEY}
LANGFUSE_DB_PASSWORD=${DB_PASSWORD}
CLICKHOUSE_PASSWORD=${CLICKHOUSE_PASSWORD}
EOF

cat >&2 <<EOF

[gen-secrets] Generated 5 secrets.
Next:
  1. Paste these into .env (replacing the REPLACE_WITH_* placeholders)
  2. Bring up the stack:
       make langfuse-up
  3. After first login to http://localhost:3000, create a project and copy
     the public + secret API keys into .env as:
       LANGFUSE_PUBLIC_KEY=pk-...
       LANGFUSE_SECRET_KEY=sk-...
EOF
