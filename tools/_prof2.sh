#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
wait_ready(){ for _ in $(seq 1 200); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
LEASE_NAME=prof2 MINISGL_IMAGE=minisgl-rdna4:lean-b000dd0 VLLM_W4A8_MOE_G2FUSE_BYLANE=1 \
  MINISGL_PROFILE=/engine/tools/_prof2.pt.trace.json.gz MINISGL_PROFILE_SKIP=40 MINISGL_PROFILE_STEPS=50 \
  gpu-lease -n 2 --detach --name prof2 -- docker compose -p lease-prof2 --profile serve up -d >/dev/null 2>&1
if ! wait_ready; then echo "BOOT FAILED"; docker compose -p lease-prof2 --profile serve logs --tail 25; \
  docker compose -p lease-prof2 --profile serve down >/dev/null 2>&1; exit 1; fi
python3 "$REPO/tools/_hostloop_driver.py" 256 "$REPO/tools/_prof2_out.txt" 2>&1 | grep tok/s
sleep 8
docker compose -p lease-prof2 --profile serve down >/dev/null 2>&1
ls -la "$REPO/tools/_prof2.pt.trace.json.gz" 2>/dev/null
