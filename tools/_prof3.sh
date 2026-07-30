#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
PKG=/home/pat/code/rdna4-hip-kernels-mmax/fp8_wmma/torch-ext/fp8_wmma
[ -f "$PKG/_ops.py" ] || { echo "FATAL: $PKG not built"; exit 1; }
wait_ready(){ for _ in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
cleanup(){ docker compose -p lease-p3 --profile serve down >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
LEASE_NAME=p3 MINISGL_IMAGE=minisgl-rdna4:lean FP8_WMMA_PKG="$PKG" MINISGL_GDN_PROJ_GEMV=1 MINISGL_MINV_DECODE_GEMV=1 \
  MINISGL_PROFILE=/engine/tools/_prof4.pt.trace.json.gz MINISGL_PROFILE_SKIP=40 MINISGL_PROFILE_STEPS=50 \
  gpu-lease -n 2 --detach --name p3 -- docker compose -p lease-p3 --profile serve up -d >/dev/null 2>&1
if ! wait_ready; then echo "BOOT FAILED"; docker compose -p lease-p3 --profile serve logs --tail 20; exit 1; fi
python3 "$REPO/tools/_hostloop_driver.py" 256 "$REPO/tools/_prof4_out.txt" 2>&1 | grep tok/s
sleep 8
