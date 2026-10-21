#!/usr/bin/env bash
# Stop the server cleanly (graceful shutdown; never kill a multi-GPU vLLM with -9).
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
docker stop -t 120 "${NAME:-ds41-flash}"
