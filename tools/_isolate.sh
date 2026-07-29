#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
wait_ready(){ for _ in $(seq 1 150); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
LEASE_NAME=iso MINISGL_IMAGE=minisgl-rdna4:lean-b000dd0 \
  MINISGL_GDN_PROJ_GEMV=1 MINISGL_MINV_DECODE_GEMV=1 VLLM_W4A8_MOE_G2FUSE_BYLANE=1 \
  MINISGL_ROUTER_FUSED=0 MINISGL_EXTRA_ARGS="--no-gdn-radix" \
  gpu-lease -n 2 --detach --name iso -- docker compose -p lease-iso --profile serve up -d >/dev/null 2>&1
if wait_ready; then
  echo "BOOTS OK with baked moe_hip -> the rebuilt moe_hip is the fault"
  python3 "$REPO/tools/_hostloop_driver.py" 256 "$REPO/tools/_iso_out.txt" 2>&1 | grep tok/s
else
  echo "STILL FAILS with baked moe_hip -> fault is elsewhere"
  docker compose -p lease-iso --profile serve logs --tail 8 2>&1 | sed 's/^/  /'
fi
docker compose -p lease-iso --profile serve down >/dev/null 2>&1
