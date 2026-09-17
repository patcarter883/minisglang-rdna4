#!/usr/bin/env bash
# Launch the serving control panel (host-side, CPU-only: it only shells out to docker compose).
set -euo pipefail
cd "$(dirname "$0")"
PY="$(command -v python3)"
exec "$PY" server.py
