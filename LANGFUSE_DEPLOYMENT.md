# ShieldPoint Langfuse Deployment Guide

**Ticket:** SHLD-9 — Deploy self-hosted Langfuse stack via docker-compose
**Phase:** 0 (Foundation) — must be completed in Week 1
**Assignee:** Langfuse Administrator / ML Platform Engineer

---

## 1. Overview

This guide deploys a self-hosted **Langfuse v3** observability stack on the
ShieldPoint internal network. The stack captures LLM traces (prompts,
completions, latency, token counts, tool invocations) from every agent run
without sending any data to langfuse.io cloud — satisfying the SHLD-43 data
sovereignty requirement.

**Key design choices (refactored):**

1. **Single `docker-compose.yml`** — Langfuse services live under the
   `langfuse` profile alongside the existing `smoke` profile (smoke-probe
   sidecar). Future agent services (`agent-api`, `claims-db`, `redis`) will
   live in the same file under no profile (always-on) or their own profile.
2. **Single `.env` file** — All env vars (LM Studio + Langfuse + future
   agents) live in `.env`. No more `.env.langfuse`.
3. **No S3 dependency** — MinIO has been removed entirely. Langfuse uses
   local-filesystem blob storage (`LANGFUSE_S3_MEDIA_UPLOAD_ENABLED=false`).
   For an observability stack that captures text traces, this is sufficient
   and removes the only S3-protocol dependency from the stack.

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  shieldpoint-net  (Docker bridge, 172.28.0.0/16)                 │
│                                                                  │
│   ┌────────────┐                                                 │
│   │  langfuse  │ (v3 — web + worker in one container)            │
│   │  :3000     │                                                 │
│   └─────┬──────┘                                                 │
│         │                                                        │
│         ├─▶ langfuse-pgbouncer :5432 ─▶ langfuse-postgres :5432  │
│         │   (transaction-pool)            (PostgreSQL 16)         │
│         │                                                        │
│         ├─▶ langfuse-redis  :6379  (queue + cache)               │
│         └─▶ langfuse-clickhouse :8123 (observation store)        │
│                                                                  │
│   Blob storage: local filesystem (NO S3, NO MinIO)               │
│                                                                  │
│   ┌────────────┐                                                 │
│   │ agent-api  │ (Sprint 2) — sends traces via Langfuse SDK     │
│   └────────────┘                                                 │
└──────────────────────────────────────────────────────────────────┘
                │
                │  127.0.0.1:3000  (UI bound to localhost only —
                │                  not exposed on external interfaces)
                ▼
         Mac host / browser
```

### Services in `docker-compose.yml`

| Service              | Profile    | Purpose                       | External Binding |
|----------------------|------------|-------------------------------|-------------------|
| `smoke-probe`        | `smoke`    | AC test sidecar               | (none)            |
| `langfuse`           | `langfuse` | Web UI + API + worker         | `127.0.0.1:3000`  |
| `langfuse-postgres`  | `langfuse` | Metadata DB (PostgreSQL 16)   | (none)            |
| `langfuse-pgbouncer` | `langfuse` | Connection pool               | (none)            |
| `langfuse-clickhouse`| `langfuse` | Observation store             | (none)            |
| `langfuse-redis`     | `langfuse` | Queue + cache                 | (none)            |

**No port is bound to `0.0.0.0`** — all external bindings use `127.0.0.1`
so traffic from other machines on the network cannot reach the stack.

---

## 2. Prerequisites

| Requirement            | Version    | Notes                                    |
|------------------------|------------|------------------------------------------|
| Docker Engine          | 24.x+      | `docker --version`                       |
| Docker Compose v2      | 2.20+      | `docker compose version`                 |
| Python                 | 3.10+      | For the SDK wrapper + test scripts       |
| `pip` packages         | langfuse, openai, httpx | See `agent_framework/observability/requirements.txt` |
| Free disk space        | ≥3 GB      | Postgres + ClickHouse volumes (MinIO removed) |
| Free RAM               | ≥3 GB      | Langfuse + ClickHouse + Postgres         |

Verify Docker is up:
```bash
docker info >/dev/null 2>&1 && echo "Docker OK" || echo "Docker not running"
```

---

## 3. Deployment Steps

### Step 1 — Configure secrets (single .env file)

```bash
cd sheildpoint/

# 1a. Copy the env template (single unified file for the whole stack)
cp .env.example .env

# 1b. Generate cryptographically strong secrets and append to .env
./scripts/langfuse-gen-secrets.sh >> .env

# 1c. Open .env and verify all REPLACE_WITH_* placeholders are gone
$EDITOR .env
```

After this step, `.env` should contain values for:
- `LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_SALT`, `LANGFUSE_ENCRYPTION_KEY`
- `LANGFUSE_DB_PASSWORD`, `CLICKHOUSE_PASSWORD`
- `LM_STUDIO_BASE_URL`, `LM_STUDIO_API_KEY`, `QWEN_MODEL_ID` (from the LM Studio section)
- `LANGFUSE_BOOTSTRAP_EMAIL`, `LANGFUSE_BOOTSTRAP_PASSWORD` (admin user you'll create in Step 3)

**Do NOT** set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` yet — those are
generated by the bootstrap script after the Langfuse UI is reachable.

### Step 2 — Bring up the stack

```bash
make langfuse-up
```

This runs `scripts/langfuse-up.sh` which:
1. Pre-flight checks (`.env` exists, no placeholders, docker compose v2)
2. Pulls all Docker images (~1.5 GB on first run, ~2 min — MinIO removed)
3. `docker compose --profile langfuse up -d`
4. Waits up to 90s for `http://localhost:3000/api/public/health` to return 200
   (first boot runs DB migrations — may take 60-120s)

**Expected output:**
```
[OK] ShieldPoint Langfuse stack is UP

  Web UI:        http://localhost:3000
  Health API:    http://localhost:3000/api/public/health
  Postgres 16:   langfuse-postgres:5432         (internal only)
  PgBouncer:     langfuse-pgbouncer:5432        (internal only)
  ClickHouse:    langfuse-clickhouse:8123       (internal only)
  Redis 7:       langfuse-redis:6379            (internal only)

  Blob storage:  Local filesystem (no S3 / no MinIO — dependency removed)
```

### Step 3 — Create the admin user (first UI visit)

Open `http://localhost:3000` in your browser. On first visit, Langfuse shows
a setup screen — create the admin user:

- **Email:** `admin@shieldpoint.local` (or your org email)
- **Password:** choose a strong password, save it in your password manager
- **Name:** `ShieldPoint Admin`

This admin user is stored locally in the Postgres database — no data is sent
to langfuse.io.

**Important:** Update `LANGFUSE_BOOTSTRAP_PASSWORD` in `.env` to match the
password you just chose — the bootstrap script in Step 4 needs it.

### Step 4 — Bootstrap the project + API keys + 90-day retention

```bash
make langfuse-bootstrap
# Uses LANGFUSE_BOOTSTRAP_EMAIL + LANGFUSE_BOOTSTRAP_PASSWORD from .env
```

Or run the script directly with arguments:

```bash
python3 scripts/langfuse-bootstrap.py \
    --email admin@shieldpoint.local \
    --password 'YOUR_ADMIN_PASSWORD' \
    --project-name "ShieldPoint Claims Automation" \
    --retention-days 90
```

This script:
1. Waits for Langfuse health endpoint
2. Logs in with admin credentials
3. Creates the project "ShieldPoint Claims Automation" (idempotent)
4. Creates a project API key pair (`pk-lf-...` + `sk-lf-...`)
5. Sets the trace retention policy to 90 days (SHLD-9 AC)
6. Prints the API keys for you to paste into `.env`

**Sample output:**
```
[bootstrap] ===== API KEYS — copy to .env =====
LANGFUSE_PUBLIC_KEY=pk-lf-abc123def456...
LANGFUSE_SECRET_KEY=sk-lf-xyz789ghi012...
[bootstrap] ============================================
```

### Step 5 — Add API keys to .env

```bash
$EDITOR .env
# Set:
#   LANGFUSE_HOST=http://localhost:3000
#   LANGFUSE_PUBLIC_KEY=pk-lf-...    (from Step 4)
#   LANGFUSE_SECRET_KEY=sk-lf-...    (from Step 4)
```

### Step 6 — Verify end-to-end trace capture

```bash
# With real LM Studio (must be running on port 1234):
make langfuse-test-trace

# Without LM Studio (uses mock LLM response, still sends a real trace):
python3 scripts/test-langfuse-trace.py --no-llm
```

The test:
1. Checks Langfuse health endpoint returns 200
2. Imports the Python SDK + ShieldPoint wrapper
3. Sends a sample LLM call (real or mocked) decorated with `@observe_llm`
4. Polls the Langfuse API for the trace
5. Verifies the trace contains: **prompt, completion, latency, token count**
6. Verifies `LANGFUSE_HOST` is internal (data sovereignty check)

**Expected output:**
```
=== Stage 1/5: Langfuse health endpoint (UI + DB up) ===
[OK]   Langfuse health endpoint returned 200

=== Stage 2/5: Python SDK + ShieldPoint wrapper import ===
[OK]   langfuse SDK + ShieldPoint wrapper loaded; tracer enabled

=== Stage 3/5: Send sample LLM call to LM Studio ===
[OK]   Trace sent. Captured: prompt + completion + 60 tokens

=== Stage 4/5: Verify trace was captured by Langfuse ===
[OK]   Trace captured with: prompt, completion, token count, latency ✓

=== Stage 5/5: Data sovereignty check ===
[OK]   LANGFUSE_HOST is internal + telemetry disabled in compose.

  PASS: 5   WARN: 0   FAIL: 0
```

### Step 7 — View traces in the UI

Open `http://localhost:3000` → log in → **Traces** tab. You should see a
trace named `e2e_test_trace` with one generation `shieldpoint_e2e_test_classification`
containing:
- Input: claim text + system prompt
- Output: LLM classification JSON
- Usage: 42 input tokens, 18 output tokens, 60 total
- Latency: ~500ms (visible in metadata)

---

## 4. Integration with the Agent Framework

The Python wrapper lives at:
```
agent_framework/observability/langfuse_wrapper.py
```

Import in any agent module:

```python
from agent_framework.observability import (
    observe_llm,    # decorator for LLM calls
    observe_tool,   # decorator for tool calls
    trace_context,  # context manager for trace boundaries
    tracer,         # singleton ShieldPointTracer instance
)
```

### Example: ReAct loop with full tracing

```python
from openai import OpenAI
from agent_framework.observability import observe_llm, observe_tool, trace_context

client = OpenAI(
    base_url=os.environ["LM_STUDIO_BASE_URL"],  # http://host.docker.internal:1234/v1
    api_key=os.environ["LM_STUDIO_API_KEY"],
)

@observe_llm(name="react_think")
def think(observation: str) -> dict:
    """Ask Qwen to emit the next Thought/Action/ActionInput."""
    response = client.chat.completions.create(
        model=os.environ["QWEN_MODEL_ID"],
        messages=[
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": observation},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return response  # wrapper extracts content, tokens, latency automatically

@observe_tool(name="claim_lookup")
def claim_lookup(claim_id: str) -> dict:
    return db.claims.find_one(id=claim_id)

@observe_tool(name="policy_validate")
def policy_validate(policy_id: str, claim: dict) -> dict:
    return policy_engine.validate(policy_id, claim)

def run_agent(claim_id: str, user_id: str = "adjuster-42"):
    with trace_context(
        name="react_agent_run",
        user_id=user_id,
        session_id=claim_id,
        metadata={"claim_id": claim_id, "phase": "1", "pattern": "tool-using"},
        tags=["react", "phase-1"],
    ):
        for step in range(MAX_STEPS):
            thought = think(current_observation)
            tool_name = thought["action"]
            tool_input = thought["action_input"]

            if tool_name == "claim_lookup":
                result = claim_lookup(tool_input["claim_id"])
            elif tool_name == "policy_validate":
                result = policy_validate(tool_input["policy_id"], tool_input["claim"])
            # ... other tools
```

Every `think()` call appears as a **GENERATION** on the trace (with model,
input messages, output content, token usage, latency). Every tool call
appears as a **SPAN** (with input args, output, latency, error if any).
Errors are captured automatically with stack traces.

### What gets captured per LLM call

| Field         | Source                              | Example                            |
|---------------|-------------------------------------|------------------------------------|
| name          | `@observe_llm(name=...)`            | `classify_claim`                   |
| input.messages| kwargs["messages"] or args[0]      | `[{"role": "user", "content":...}]`|
| input.model   | kwargs["model"]                    | `qwen3.6-35b-a3b-q4_k_m`           |
| input.params  | other kwargs (temperature, etc.)   | `{"temperature": 0.1}`             |
| output.content| response.choices[0].message.content | `"medium"`                         |
| output.tool_calls | response.choices[0].message.tool_calls | `[{"function": {...}}]`        |
| output.finish_reason | response.choices[0].finish_reason | `"stop"`                       |
| usage_details | response.usage                     | `{"input": 42, "output": 18, "total": 60}` |
| metadata.latency_ms | wrapper-measured wall clock  | `487`                              |
| metadata.function | `fn.__qualname__`               | `classify_claim`                   |
| metadata.error | if exception was raised            | `"ConnectionError: ..."`           |

---

## 5. Daily Operations

| Action                              | Command                            |
|-------------------------------------|------------------------------------|
| Bring up the Langfuse stack         | `make langfuse-up`                 |
| Tear down (volumes preserved)       | `make langfuse-down`               |
| Tear down + DELETE all data         | `make langfuse-down -- --volumes`  |
| Status of containers                | `make langfuse-status`             |
| Tail logs                           | `make langfuse-logs`               |
| Tail specific service logs          | `make langfuse-logs TARGET=postgres` |
| Run e2e trace test                  | `make langfuse-test-trace`         |
| Validate compose syntax             | `make langfuse-lint`               |
| Generate new secrets                | `make langfuse-gen-secrets`        |
| Re-bootstrap project + API keys     | `make langfuse-bootstrap`          |

### Equivalent docker compose commands

```bash
# Bring up Langfuse stack (uses langfuse profile)
docker compose --profile langfuse up -d

# Tear down (keep data)
docker compose --profile langfuse down

# Tear down + DELETE volumes
docker compose --profile langfuse down -v

# Tail logs
docker compose --profile langfuse logs -f langfuse
docker compose --profile langfuse logs -f langfuse-postgres

# Status
docker compose --profile langfuse ps

# Bring up BOTH Langfuse and smoke-probe
docker compose --profile langfuse --profile smoke up -d
```

Docker Compose auto-loads `.env` from the project root — no `--env-file` flag
needed.

---

## 6. Acceptance Criteria Verification

| SHLD-9 AC | How verified |
|-----------|--------------|
| Langfuse web UI accessible on port 3000 within internal network | `curl http://localhost:3000/api/public/health` returns 200; port is bound to `127.0.0.1` only |
| PostgreSQL 16 database running with persistent volume for trace storage | `docker volume inspect sheildpoint_langfuse_pgdata` exists; `docker exec shieldpoint-langfuse-db psql -U langfuse -d langfuse -c '\dt'` shows tables |
| Langfuse Python SDK (langfuse>=2.0) integrated into agent framework skeleton | `agent_framework/observability/langfuse_wrapper.py` exists; `from agent_framework.observability import observe_llm` succeeds |
| First test trace successfully captured with prompt, completion, latency, and token count | `python3 scripts/test-langfuse-trace.py` exits 0; UI shows the trace with all 4 fields |
| Docker Compose stack starts cleanly with `docker compose up -d` | `make langfuse-up` returns 0 |
| No data leaves ShieldPoint network — all endpoints bound to internal interfaces only | All `ports:` mappings use `127.0.0.1:` prefix; `TELEMETRY_ENABLED=false`; `LANGFUSE_HOST` does not contain `langfuse.io` or any external domain; **no S3 dependency** (MinIO removed, blob storage uses local filesystem) |

---

## 7. Troubleshooting

### `make langfuse-up` fails with "dependency cycle detected"

This was fixed in v1.0 of the compose file. If you see it, you have an old
version — `git pull` and re-run.

### Langfuse never becomes healthy (90s timeout)

```bash
# Check container status
docker compose --profile langfuse ps

# Check Langfuse logs (look for migration errors)
docker compose --profile langfuse logs langfuse | tail -50

# Common causes:
# 1. .env still has REPLACE_WITH_* placeholders
grep REPLACE_WITH .env
# 2. Postgres didn't come up — check its logs:
docker compose --profile langfuse logs langfuse-postgres
# 3. ClickHouse OOM — increase Docker Desktop's memory limit to 4 GB+
```

### `make langfuse-bootstrap` fails with "Login failed (HTTP 401)"

You haven't created the admin user in the UI yet. Open
`http://localhost:3000` in your browser, complete the first-visit setup
screen (create admin user), then re-run `make langfuse-bootstrap`.

### `make langfuse-test-trace` Stage 4 fails: trace never appears

The Langfuse ingestion pipeline is async (traces go through Redis → worker
→ ClickHouse). The test retries for 30s. If it still fails:

```bash
# Check that ClickHouse is up
docker exec shieldpoint-langfuse-clickhouse wget -qO- http://localhost:8123/ping

# Check Langfuse worker logs for ingestion errors
docker logs shieldpoint-langfuse 2>&1 | grep -i error | tail -20

# Verify the trace was sent by querying the API directly:
curl -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
    http://localhost:3000/api/public/traces
```

### LM Studio unreachable from the test script

If you're running the test from the Mac host (not inside a container):
```bash
curl http://localhost:1234/v1/models
```
If this fails, LM Studio's local server isn't started. Open the LM Studio
desktop app → Local Server → Start Server.

If you want to test the trace pipeline without LM Studio:
```bash
python3 scripts/test-langfuse-trace.py --no-llm
```
This uses a mocked LLM response — the trace is still sent to Langfuse with
real usage numbers (42/18/60 tokens).

### Postgres connection exhaustion

If you see `FATAL: sorry, too many clients already` in the Langfuse logs,
increase the PgBouncer pool size in `docker-compose.yml`:

```yaml
langfuse-pgbouncer:
  environment:
    - DEFAULT_POOL_SIZE=50    # was 20
    - MAX_CLIENT_CONN=500     # was 200
```

Then:
```bash
make langfuse-down
make langfuse-up
```

### Want to wipe all traces and start over (DESTRUCTIVE)

```bash
make langfuse-down -- --volumes
make langfuse-up
make langfuse-bootstrap   # re-create project + API keys
```

This deletes ALL traces, scores, prompts, datasets. The 90-day retention
policy is re-set automatically by the bootstrap script.

---

## 8. Data Sovereignty Audit (SHLD-43)

To verify no data leaves the ShieldPoint network during a trace:

```bash
# 1. Start a 30s tcpdump on non-loopback interfaces
sudo timeout 30 tcpdump -i any -nn \
    'not (src host 127.0.0.1 and dst host 127.0.0.1)' \
    -c 50 -w /tmp/langfuse-egress.pcap &

# 2. Send a trace (in another terminal)
python3 scripts/test-langfuse-trace.py --no-llm

# 3. Wait for tcpdump to finish, then inspect
sleep 28
sudo kill -INT $(pgrep tcpdump) 2>/dev/null
tcpdump -r /tmp/langfuse-egress.pcap -nn | head -20
```

**Expected:** zero packets captured (no egress traffic). If you see any
packets to non-localhost IPs, that's an egress violation — check:

1. `LANGFUSE_HOST` in `.env` is `http://localhost:3000` (not
   `https://cloud.langfuse.com` or similar).
2. `TELEMETRY_ENABLED=false` and `LANGFUSE_ENABLE_TELEMETRY=false` are
   both set in `docker-compose.yml`.
3. No container has a port bound to `0.0.0.0` (only `127.0.0.1`).
4. **No S3 endpoint configured** — `LANGFUSE_S3_MEDIA_UPLOAD_ENABLED=false`
   is set in `docker-compose.yml`, so Langfuse does not attempt any S3
   API calls (to Amazon S3, MinIO, or any other S3-compatible service).

Verify (3) with:
```bash
docker compose --profile langfuse ps --format json | \
    python3 -c "
import json, sys
for line in sys.stdin:
    s = json.loads(line)
    ports = s.get('Publishers', []) or []
    for p in ports:
        if p.get('URL'):
            print(f\"{s['Service']}: {p['URL']}:{p['PublishedPort']}->{p['TargetPort']}\")
"
```

All lines should show `127.0.0.1:` as the URL prefix.

---

## 9. Migration Path (Phase 5 — Production Hardening)

When ShieldPoint moves from Mac dev to a production VLAN:

1. **Move the stack** to a dedicated observability VM on the internal VLAN.
2. **Change `LANGFUSE_NEXTAUTH_URL`** in `.env` to
   `http://<observability-vm-vlan-ip>:3000` — but ONLY accessible from
   inside the VLAN (firewall rule on the VM).
3. **Change `ports:` mappings** in `docker-compose.yml` from
   `127.0.0.1:3000:3000` to `<vlan-ip>:3000:3000` (or `0.0.0.0:3000:3000`
   if the VM's firewall restricts access).
4. **Add backups**: cron a `pg_dump` of `langfuse-postgres` + a
   `clickhouse-backup` of `langfuse-clickhouse` to a local NAS.
5. **Add monitoring**: Prometheus exporter on Langfuse + alert on
   `langfuse_health_status != 1` or `clickhouse_disk_usage > 80%`.
6. **Rotate secrets** quarterly: re-run `./scripts/langfuse-gen-secrets.sh`
   into a new `.env` and `make langfuse-down && make langfuse-up`.

If media upload (images, PDFs) becomes necessary in production, you can
re-enable S3-compatible storage at that time — either MinIO (self-hosted,
keeps data local) or Amazon S3 (only if data sovereignty allows). For
Phase 0, local filesystem is sufficient.

---

## 10. References

- **Langfuse self-hosting docs:** https://langfuse.com/self-hosting/docker-compose
- **Langfuse Python SDK:** https://python.reference.langfuse.com/
- **Langfuse v3 migration guide:** https://langfuse.com/docs/migration/upgrade-v3
- **Docker Compose profiles:** https://docs.docker.com/compose/profiles/
- **PgBouncer config reference:** https://www.pgbouncer.org/config.html
- **Source tickets:** `ShieldPoint_Claims_Automation_Plan_v2.pdf` §9.2;
  `ShieldPoint_Jira_Tickets.xlsx` SHLD-1, SHLD-9, SHLD-42, SHLD-43
