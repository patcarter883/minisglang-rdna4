#!/usr/bin/env bash
# Boot ONCE with the fused router and be patient: log a heartbeat so "still capturing" is
# distinguishable from "wedged". Previous runs were killed at ~1 min, which may have been premature.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; PKG=/home/pat/code/rdna4-hip-kernels-build/moe/torch-ext/moe_hip
LEASE_NAME=rpat MINISGL_IMAGE=minisgl-rdna4:lean-b000dd0 MOE_HIP_PKG="$PKG" \
  MINISGL_GDN_PROJ_GEMV=1 MINISGL_MINV_DECODE_GEMV=1 VLLM_W4A8_MOE_G2FUSE_BYLANE=1 \
  MINISGL_ROUTER_FUSED=1 MINISGL_EXTRA_ARGS="--no-gdn-radix" \
  gpu-lease -n 2 --detach --name rpat -- docker compose -p lease-rpat --profile serve up -d >/dev/null 2>&1
for i in $(seq 1 90); do
  if curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1; then
    echo "READY after $((i*10))s"
    python3 "$REPO/tools/_hostloop_driver.py" 256 "$REPO/tools/_rpat_out.txt" 2>&1 | grep tok/s
    docker compose -p lease-rpat --profile serve down >/dev/null 2>&1; exit 0
  fi
  [ $((i % 6)) -eq 0 ] && echo "  ${i}0s: $(docker logs lease-rpat-serve 2>&1 | tail -1 | tr -d '\r' | tail -c 90)"
  sleep 10
done
echo "STILL NOT READY after 900s -> genuinely wedged"
docker logs lease-rpat-serve 2>&1 | tail -5
docker compose -p lease-rpat --profile serve down >/dev/null 2>&1
