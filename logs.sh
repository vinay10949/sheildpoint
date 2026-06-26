#!/usr/bin/env bash
# =============================================================================
# logs.sh — tail LM Studio desktop logs (or agent framework container logs)
# -----------------------------------------------------------------------------
# LM Studio desktop on Mac writes logs to:
#   ~/Library/Logs/LM Studio/lm-studio.log
#   (older versions: ~/Library/Application Support/LM Studio/logs/)
#
# Usage:
#   ./logs.sh                 # tail LM Studio desktop log
#   ./logs.sh lm-studio       # same as above (explicit)
#   ./logs.sh containers      # tail agent framework container logs
#   ./logs.sh error           # grep LM Studio log for errors
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# shellcheck disable=SC1091
source ./scripts/lib/common.sh

TARGET="${1:-lm-studio}"

case "${TARGET}" in
    lm-studio|lmstudio|desktop|mac)
        # Find the LM Studio log file
        LOG_FILE=""
        for candidate in \
            "$HOME/Library/Logs/LM Studio/lm-studio.log" \
            "$HOME/Library/Application Support/LM Studio/logs/lm-studio.log" \
            "$HOME/Library/Application Support/LM Studio/lm-studio.log"; do
            if [[ -f "${candidate}" ]]; then
                LOG_FILE="${candidate}"
                break
            fi
        done

        if [[ -z "${LOG_FILE}" ]]; then
            die "LM Studio log file not found in standard Mac locations.\n  Searched:\n    ~/Library/Logs/LM Studio/\n    ~/Library/Application Support/LM Studio/\n  Open LM Studio desktop → Help → Reveal Logs to find it manually."
        fi

        if [[ -n "${2:-}" ]]; then
            log "Tailing ${LOG_FILE} filtered by '$2' (Ctrl+C to stop)..."
            tail -n 200 -f "${LOG_FILE}" | grep --line-buffered -i "$2"
        else
            log "Tailing ${LOG_FILE} (Ctrl+C to stop)..."
            tail -n 200 -f "${LOG_FILE}"
        fi
        ;;

    containers|container|docker|compose)
        log "Tailing agent framework container logs (Ctrl+C to stop)..."
        # Streams logs from all services defined in docker-compose.yml
        "${DOCKER_COMPOSE[@]}" logs -f --tail 200
        ;;

    *)
        # Treat as grep filter against LM Studio desktop log
        log "Tailing LM Studio log filtered by '${TARGET}' (Ctrl+C to stop)..."
        LOG_FILE="$HOME/Library/Logs/LM Studio/lm-studio.log"
        [[ -f "${LOG_FILE}" ]] || LOG_FILE="$HOME/Library/Application Support/LM Studio/logs/lm-studio.log"
        [[ -f "${LOG_FILE}" ]] || die "LM Studio log not found."
        tail -n 200 -f "${LOG_FILE}" | grep --line-buffered -i "${TARGET}"
        ;;
esac
