#!/usr/bin/env bash
# Launch the GPU serving control panel (host-side; CPU-only; no gpu-lease needed here —
# the panel itself just orchestrates docker/gpu-lease on your behalf).
set -euo pipefail
cd "$(dirname "$0")"
PY="$(command -v python3)"
exec "$PY" server.py
