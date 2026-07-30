#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; OUT="$REPO/tools/_router_final.txt"
PKG=/home/pat/code/rdna4-hip-kernels-build/moe/torch-ext/moe_hip
[ -f "$PKG/_ops.py" ] || { echo "FATAL: not built"; exit 1; }
: > "$OUT"
run(){ local v="$1" label="$2"
  echo "=== $label (ROUTER_FUSED=$v) ===" | tee -a "$OUT"
  LEASE_NAME=rfin MINISGL_IMAGE=minisgl-rdna4:lean MOE_HIP_PKG="$PKG" \
    MINISGL_GDN_PROJ_GEMV=1 MINISGL_MINV_DECODE_GEMV=1 VLLM_W4A8_MOE_G2FUSE_BYLANE=1 \
    MINISGL_ROUTER_FUSED="$v" MINISGL_EXTRA_ARGS="--no-gdn-radix" \
    gpu-lease -n 2 --detach --name rfin -- docker compose -p lease-rfin --profile serve up -d >/dev/null 2>&1
  for i in $(seq 1 60); do
    if curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1; then
      echo "  ready after $((i*10))s" | tee -a "$OUT"
      python3 "$REPO/tools/_hostloop_driver.py" 256 "$REPO/tools/_rfin_$label.txt" 2>&1 | grep tok/s | sed 's/^/  /' | tee -a "$OUT"
      docker compose -p lease-rfin --profile serve down >/dev/null 2>&1; sleep 4; return 0
    fi
    # poll: distinguish "still capturing" from "spinning"
    if [ $((i % 6)) -eq 0 ]; then
      u=$(rocm-smi --showuse 2>/dev/null | grep -m1 "GPU\[0\]" | grep -oE "[0-9]+$")
      echo "  ${i}0s gpu_use=${u:-?}%" | tee -a "$OUT"
    fi
    sleep 10
  done
  echo "  WEDGED (600s)" | tee -a "$OUT"
  docker compose -p lease-rfin --profile serve down >/dev/null 2>&1; sleep 4; return 1; }
run 0 torch_chain
run 1 fused
echo "=== done ===" | tee -a "$OUT"
