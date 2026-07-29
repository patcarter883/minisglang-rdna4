#!/usr/bin/env bash
# What does raising *_GEMV_MAXM from 2 to 16 cost at CONCURRENCY?
#
# MAXM=16 is what puts ordinary decode and spec-verify on the same kernel (no M crossing), but it
# also routes concurrent batches (M up to max_running) onto a GEMV that is M=1-shaped and loses to
# WMMA from about M=4. bs=1 cannot see that. Measures both legs at bs=1 AND conc=4/8.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; OUT="$REPO/tools/_maxm_ab.txt"
FP8=/home/pat/code/rdna4-hip-kernels-bfgemv/fp8_wmma/torch-ext/fp8_wmma
MOE=/home/pat/code/rdna4-hip-kernels-build/moe/torch-ext/moe_hip
IMG="${MINISGL_IMAGE:-minisgl-rdna4:lean-b000dd0}"
for p in "$FP8" "$MOE"; do
  [ -f "$p/_ops.py" ] || { echo "FATAL: $p not built"; exit 1; }
  pkg=$(basename "$p")
  docker run --rm -v "$p:/pkg/$pkg:ro" --entrypoint bash "$IMG" \
    -lc "PYTHONPATH=/pkg python -c 'import torch, $pkg'" >/dev/null 2>&1 \
    || { echo "FATAL: $p does not import in $IMG (rebuild INSIDE that image)"; exit 1; }
done
: > "$OUT"

run(){ local maxm="$1"
  echo "=== MAXM=$maxm ===" | tee -a "$OUT"
  LEASE_NAME=mab MINISGL_IMAGE="$IMG" FP8_WMMA_PKG="$FP8" MOE_HIP_PKG="$MOE" \
    MINISGL_GDN_PROJ_GEMV=1 MINISGL_MINV_DECODE_GEMV=1 \
    MINISGL_GDN_PROJ_GEMV_MAXM="$maxm" MINISGL_MINV_DECODE_GEMV_MAXM="$maxm" \
    VLLM_W4A8_MOE_G2FUSE_BYLANE=1 MINISGL_ROUTER_FUSED=1 MINISGL_EXTRA_ARGS="--no-gdn-radix" \
    gpu-lease -n 2 --detach --name mab -- \
    docker compose -p lease-mab --profile serve up -d >/dev/null 2>&1
  for i in $(seq 1 60); do
    if curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1; then
      echo "  ready after $((i*10))s" | tee -a "$OUT"
      # VERIFY the knob actually reached the container. compose only forwards variables it names,
      # so `MAXM=2 docker compose up` against a compose that never mentions MAXM silently measures
      # the default — an A/B comparing a config against itself, with no symptom.
      got=$(docker exec lease-mab-serve printenv MINISGL_MINV_DECODE_GEMV_MAXM 2>/dev/null)
      echo "  container MAXM=${got:-<UNSET>}" | tee -a "$OUT"
      [ "$got" = "$maxm" ] || { echo "  FATAL: env did not reach container (wanted $maxm)" | tee -a "$OUT"
        docker compose -p lease-mab --profile serve down >/dev/null 2>&1; sleep 4; return 1; }
      python3 "$REPO/tools/_hostloop_driver.py" 256 "$REPO/tools/_maxm_${maxm}.txt" 2>&1 \
        | grep tok/s | sed 's/^/  bs=1  /' | tee -a "$OUT"
      for c in 4 8; do
        python3 "$REPO/tools/_conc_driver.py" "$c" 128 2>&1 | grep agg | sed 's/^/  /' | tee -a "$OUT"
      done
      docker compose -p lease-mab --profile serve down >/dev/null 2>&1; sleep 4; return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -q lease-mab; then
      echo "  CONTAINER DIED" | tee -a "$OUT"
      docker compose -p lease-mab --profile serve logs --tail 25 2>&1 | tail -25 | tee -a "$OUT"
      docker compose -p lease-mab --profile serve down >/dev/null 2>&1; sleep 4; return 1
    fi
    [ $((i % 6)) -eq 0 ] && echo "  ${i}0s gpu_use=$(rocm-smi --showuse 2>/dev/null | grep -m1 'GPU\[0\]' | grep -oE '[0-9]+$')%" | tee -a "$OUT"
    sleep 10
  done
  echo "  WEDGED (600s)" | tee -a "$OUT"
  docker compose -p lease-mab --profile serve down >/dev/null 2>&1; sleep 4; return 1; }

run 2
run 16
echo "=== done ===" | tee -a "$OUT"
