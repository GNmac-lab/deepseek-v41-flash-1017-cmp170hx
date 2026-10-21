#!/usr/bin/env bash
# ./start.sh            start (or restart) the server and wait until it answers
# ./start.sh smoke      one chat, one tool call and one image request against the running server
# ./start.sh logs       follow the server log
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || { echo "no .env: cp .env.example .env and set MODELS_DIR"; exit 1; }
set -a; . ./.env; set +a
PORT=${PORT:-8000}; NAME=${NAME:-ds41-flash}
case "${1:-start}" in
  start|restart)
    docker image inspect "${IMAGE:-ds41-cmp170hx:latest}" > /dev/null 2>&1 || docker/build.sh
    ./serve.sh
    echo "waiting for the engine (first start compiles kernels: ~6-10 min)..."
    for _ in $(seq 1 180); do
      if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then echo "ready: http://127.0.0.1:$PORT/v1 (model deepseek-v4.1-flash)"; exit 0; fi
      docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "the server exited:"; docker logs --tail 40 "$NAME"; exit 1; }
      sleep 10
    done
    echo "not ready after 30 min; see: docker logs $NAME"; exit 1 ;;
  smoke) python3 examples/smoke.py --port "$PORT" ;;
  logs) docker logs -f "$NAME" ;;
  *) echo "usage: $0 [start|restart|smoke|logs]"; exit 1 ;;
esac
