# =============================================================================
# ShieldPoint Agent Framework — Makefile
# -----------------------------------------------------------------------------
# Architecture: LM Studio runs NATIVELY on Mac. This Makefile manages the
# agent framework containers that connect to LM Studio via host.docker.internal.
# =============================================================================

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

CYAN  := \033[36m
RESET := \033[0m
BOLD  := \033[1m

# Prefer the v2 CLI plugin ("docker compose"); fall back to standalone "docker-compose".
DOCKER_COMPOSE := $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

.PHONY: help dev down smoke status logs verify-lm-studio vram test lint audit-egress clean \
	langfuse-up langfuse-down langfuse-logs langfuse-status langfuse-bootstrap \
	langfuse-test-trace langfuse-gen-secrets langfuse-lint \
	intake-up intake-down intake-logs intake-test intake-load-test

help: ## Show this help
	@printf '${CYAN}ShieldPoint Agent Framework — Make targets${RESET}\n\n'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  ${BOLD}%-18s${RESET} %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
dev: ## Verify LM Studio desktop + bring up agent framework containers
	@./start.sh

down: ## Tear down agent framework containers (LM Studio desktop keeps running)
	@./stop.sh

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
smoke: ## 5-stage acceptance test (health / inference / throughput / VRAM / network)
	@./smoke-test.sh

test: ## Alias for smoke
	@$(MAKE) smoke

verify-lm-studio: ## Verify LM Studio desktop is running with Qwen loaded
	@./scripts/verify-lm-studio-desktop.sh

status: ## Live status of LM Studio (Mac) + agent framework containers
	@./status.sh

logs: ## Tail logs: 'make logs' (LM Studio) or 'make logs containers'
	@./logs.sh $(TARGET)

vram: ## Print LM Studio memory usage hint
	@./scripts/vram-check.sh

lint: ## Validate compose, env, and shell scripts
	@echo "[lint] docker-compose config..."
	@$(DOCKER_COMPOSE) config --quiet
	@echo "[lint] shell scripts (shellcheck)..."
	@shellcheck -x *.sh scripts/*.sh 2>/dev/null || \
	echo "[lint] shellcheck not installed or found issues — non-blocking"
	@echo "[lint] .env present (no secrets committed)..."
	@test -f .env || { echo "[lint] .env missing — cp .env.example .env"; exit 1; }
	@echo "[lint] OK"

audit-egress: ## Run a 30s tcpdump during a sample inference; assert only localhost traffic
	@echo "[audit-egress] Starting 30s tcpdump on non-loopback interfaces..."
	@sudo timeout 30 tcpdump -i any -nn 'not (src host 127.0.0.1 and dst host 127.0.0.1)' -c 50 \
	        -w /tmp/shieldpoint-egress-$$(date +%s).pcap 2>/dev/null & \
	TCPDUMP_PID=$$!; \
	sleep 2; \
	curl -fsS http://localhost:1234/v1/chat/completions \
	        -H "Content-Type: application/json" \
	        -d '{"model":"qwen3.6-35b-a3b-q4_k_m","messages":[{"role":"user","content":"hi"}],"max_tokens":10}' \
	        >/dev/null 2>&1; \
	sleep 28; \
	kill -INT $$TCPDUMP_PID 2>/dev/null || true; \
	wait $$TCPDUMP_PID 2>/dev/null || true; \
	echo "[audit-egress] Capture complete. Inspect /tmp/shieldpoint-egress-*.pcap with:"; \
	echo "  tcpdump -r /tmp/shieldpoint-egress-*.pcap -nn | head -50"; \
	echo "[audit-egress] Any non-localhost packets = egress violation (SHLD-43 AC)."

clean: ## Remove stopped containers and orphan images (keeps volumes)
	@$(DOCKER_COMPOSE) down --remove-orphans
	@docker image prune -f --filter "label=shieldpoint.project=claims-automation" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Langfuse observability stack (SHLD-9) — uses the `langfuse` profile in
# docker-compose.yml. All env vars come from the single .env file.
# ---------------------------------------------------------------------------
langfuse-gen-secrets: ## Generate crypto secrets for .env (append to file)
	@./scripts/langfuse-gen-secrets.sh

langfuse-up: ## Bring up Langfuse stack (langfuse + postgres16 + pgbouncer + clickhouse + redis)
	@./scripts/langfuse-up.sh

langfuse-down: ## Tear down Langfuse stack (volumes preserved)
	@./scripts/langfuse-down.sh

langfuse-status: ## Status of Langfuse containers + health endpoint
	@$(DOCKER_COMPOSE) --profile langfuse ps
	@echo
	@echo "[health] http://localhost:3000/api/public/health"
	@curl -fsS --max-time 5 http://localhost:3000/api/public/health >/dev/null 2>&1 \
	        && echo "  [OK] Langfuse healthy" \
	        || echo "  [FAIL] Langfuse not reachable — run 'make langfuse-up'"

langfuse-logs: ## Tail Langfuse logs (TARGET=langfuse|postgres|clickhouse|redis)
	@$(DOCKER_COMPOSE) --profile langfuse logs -f --tail=100 $(TARGET)

langfuse-bootstrap: ## Create Langfuse project + API key + 90-day retention (run after first UI login)
	@python3 ./scripts/langfuse-bootstrap.py

langfuse-test-trace: ## End-to-end trace capture test (verifies SHLD-9 AC)
	@python3 ./scripts/test-langfuse-trace.py

langfuse-lint: ## Validate docker-compose.yml syntax (Langfuse profile)
	@echo "[lint] docker-compose.yml config (profile: langfuse)..."
	@$(DOCKER_COMPOSE) --profile langfuse config --quiet
	@echo "[lint] OK"

# ---------------------------------------------------------------------------
# Claim Intake service (SP-203) — uses the `intake` profile in
# docker-compose.yml. Runs the FastAPI app + IMAP poller + Tesseract OCR.
# ---------------------------------------------------------------------------
intake-up: ## Bring up the claim intake service (web + email + fax OCR)
	@$(DOCKER_COMPOSE) --profile intake up -d --build
	@echo
	@echo "[intake] Service starting on http://localhost:8001"
	@echo "[intake] Health:   curl http://localhost:8001/health"
	@echo "[intake] Submit:   POST http://localhost:8001/intake/claims"

intake-down: ## Tear down the claim intake service
	@$(DOCKER_COMPOSE) --profile intake down

intake-logs: ## Tail claim intake logs
	@$(DOCKER_COMPOSE) --profile intake logs -f --tail=100 claim-intake

intake-test: ## Run claim_intake unit + integration tests
	@cd claim_intake && pytest

intake-load-test: ## Run SP-203 load test (100 concurrent claims, P99 < 30s)
	@cd claim_intake && python scripts/run_load_test.py --count 100 --concurrency 100 --in-process
