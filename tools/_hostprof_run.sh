#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
PKG=/home/pat/code/rdna4-hip-kernels-mmax/fp8_wmma/torch-ext/fp8_wmma
wait_ready(){ for _ in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
cleanup(){ docker compose -p lease-hp --profile serve down >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
LEASE_NAME=hp MINISGL_IMAGE=minisgl-rdna4:lean-b000dd0 FP8_WMMA_PKG="$PKG" \
  MINISGL_GDN_PROJ_GEMV=1 MINISGL_MINV_DECODE_GEMV=1 MINISGL_HOSTPROF=50 \
  gpu-lease -n 2 --detach --name hp -- docker compose -p lease-hp --profile serve up -d >/dev/null 2>&1
wait_ready || { echo "BOOT FAILED"; docker compose -p lease-hp --profile serve logs --tail 15; exit 1; }
python3 "$REPO/tools/_hostloop_driver.py" 256 "$REPO/tools/_hp_out.txt" 2>&1 | grep tok/s
sleep 3
echo "=== per-stage host breakdown ==="
docker compose -p lease-hp --profile serve logs 2>&1 | grep -iE "hostprof|\[hp\]" | tail -8
