#!/usr/bin/env bash
# =============================================================================
# common.sh — shared bash helpers for ShieldPoint LM Studio scripts
# -----------------------------------------------------------------------------
# Sourced by start.sh / stop.sh / status.sh / smoke-test.sh / lms-bootstrap.sh
# =============================================================================

# --- ANSI color codes (disabled if not a TTY) ------------------------------
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'
    C_RED=$'\033[31m'
    C_GREEN=$'\033[32m'
    C_YELLOW=$'\033[33m'
    C_BLUE=$'\033[34m'
    C_BOLD=$'\033[1m'
else
    C_RESET=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_BOLD=""
fi

log()   { printf '%s[INFO]%s %s\n'  "${C_BLUE}"   "${C_RESET}" "$*"; }
ok()    { printf '%s[OK]%s %s\n'   "${C_GREEN}"  "${C_RESET}" "$*"; }
warn()  { printf '%s[WARN]%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
fail()  { printf '%s[FAIL]%s %s\n' "${C_RED}"    "${C_RESET}" "$*" >&2; }
die()   { fail "$*"; exit 1; }

# Ensure required commands exist before proceeding
require_cmd() {
    local missing=()
    for c in "$@"; do
        command -v "$c" >/dev/null 2>&1 || missing+=("$c")
    done
    if (( ${#missing[@]} > 0 )); then
        die "Missing required commands: ${missing[*]}"
    fi
}

# Resolve the Docker Compose command for this host.
# Prefers the v2 CLI plugin ("docker compose"); falls back to the standalone
# "docker-compose" binary. Sets DOCKER_COMPOSE as an array so callers can do:
#   "${DOCKER_COMPOSE[@]}" up -d ...
detect_docker_compose() {
    if docker compose version >/dev/null 2>&1; then
        DOCKER_COMPOSE=(docker compose)
    elif command -v docker-compose >/dev/null 2>&1; then
        DOCKER_COMPOSE=(docker-compose)
    else
        die "Neither 'docker compose' (v2 plugin) nor 'docker-compose' (standalone) is installed."
    fi
}
# Resolve eagerly so every sourcing script gets ${DOCKER_COMPOSE[@]}.
detect_docker_compose

# Verify we're in the project root (must contain docker-compose.yml)
require_project_root() {
    if [[ ! -f docker-compose.yml ]]; then
        die "Must run from project root (directory containing docker-compose.yml). Current: $(pwd)"
    fi
}
