#!/usr/bin/env bash
# Capture a minisgl-native bs=1 decode profile. This has never been done — every trace in the repo
# is a vLLM run, and minisgl's decode bottleneck is logged as "unprofiled".
#
# NOTE ON READING IT: ROCm kineto emits no device-kernel durations, so the chrome trace gives
# cpu_op / launch COUNTS, not GPU time. That is still the right instrument for an op-count question.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
TRACE="${TRACE:-/engine/tools/_decode_prof.pt.trace.json.gz}"

wait_ready() { for _ in $(seq 1 180); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }

LEASE_NAME=decprof \
MINISGL_PROFILE="$TRACE" MINISGL_PROFILE_SKIP=40 MINISGL_PROFILE_STEPS=50 \
  gpu-lease -n 2 --detach --name decprof -- \
  docker compose -p lease-decprof --profile serve up -d >/dev/null 2>&1

if ! wait_ready; then
  echo "BOOT FAILED"; docker compose -p lease-decprof --profile serve logs --tail 30
  docker compose -p lease-decprof --profile serve down >/dev/null 2>&1; exit 1
fi

# 256 decode steps > SKIP+STEPS(90), so the window lands in steady-state decode
python3 "$REPO/tools/_hostloop_driver.py" 256 "$REPO/tools/_decode_prof_out.txt"
sleep 8
docker compose -p lease-decprof --profile serve down >/dev/null 2>&1
ls -la "$REPO/tools/"_decode_prof*.gz 2>/dev/null
