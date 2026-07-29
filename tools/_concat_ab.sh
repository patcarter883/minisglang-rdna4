#!/usr/bin/env bash
# A/B the fused column projections (layers/fusedcol.py) e2e, sampled, bs=1.
#   A: MINISGL_FUSE_COL_PROJ=0  — every projection dispatched separately (baseline)
#   B: MINISGL_FUSE_COL_PROJ=1  — gdn qkvz+ba as one GEMV, moe gate+shared_gate+gate_up as one
# Both legs carry the bylane/tiling-table fp8_wmma build so ONLY the fusion differs.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; OUT="$REPO/tools/_concat_ab.txt"
FP8=/home/pat/code/rdna4-hip-kernels-bfgemv/fp8_wmma/torch-ext/fp8_wmma
MOE=/home/pat/code/rdna4-hip-kernels-build/moe/torch-ext/moe_hip
IMG="${MINISGL_IMAGE:-minisgl-rdna4:lean-b000dd0}"
# PREFLIGHT. Checking for _ops.py is NOT enough: a package built in a different image links against
# that image's libtorch and dies at load with "libc10.so: cannot open shared object file" — which
# presents as a serve that never reaches the GPU (gpu_use 0% until the 600s timeout), not as an
# obvious build error. So actually IMPORT each shadow-mounted package in the SERVE image.
for p in "$FP8" "$MOE"; do
  [ -f "$p/_ops.py" ] || { echo "FATAL: $p not built (_ops.py missing)"; exit 1; }
  pkg=$(basename "$p")
  docker run --rm -v "$p:/pkg/$pkg:ro" --entrypoint bash "$IMG" \
    -lc "PYTHONPATH=/pkg python -c 'import torch, $pkg'" >/dev/null 2>&1 \
    || { echo "FATAL: $p does not import in $IMG (ABI mismatch — rebuild it INSIDE that image)"; exit 1; }
done
echo "preflight: both packages import in $IMG"
: > "$OUT"

run(){ local v="$1" label="$2"
  echo "=== $label (MINISGL_FUSE_COL_PROJ=$v) ===" | tee -a "$OUT"
  LEASE_NAME=cab MINISGL_IMAGE=minisgl-rdna4:lean-b000dd0 \
    FP8_WMMA_PKG="$FP8" MOE_HIP_PKG="$MOE" \
    MINISGL_GDN_PROJ_GEMV=1 MINISGL_MINV_DECODE_GEMV=1 VLLM_W4A8_MOE_G2FUSE_BYLANE=1 \
    MINISGL_ROUTER_FUSED=1 MINISGL_FUSE_COL_PROJ="$v" MINISGL_EXTRA_ARGS="--no-gdn-radix" \
    gpu-lease -n 2 --detach --name cab -- \
    docker compose -p lease-cab --profile serve up -d >/dev/null 2>&1
  for i in $(seq 1 60); do
    if curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1; then
      echo "  ready after $((i*10))s" | tee -a "$OUT"
      python3 "$REPO/tools/_hostloop_driver.py" 256 "$REPO/tools/_concat_$label.txt" 2>&1 \
        | grep tok/s | sed 's/^/  /' | tee -a "$OUT"
      echo "  --- first 200 chars of output (coherence) ---" | tee -a "$OUT"
      head -c 200 "$REPO/tools/_concat_$label.txt" | sed 's/^/  /' | tee -a "$OUT"; echo | tee -a "$OUT"
      docker compose -p lease-cab --profile serve down >/dev/null 2>&1; sleep 4; return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -q lease-cab; then
      echo "  CONTAINER DIED" | tee -a "$OUT"
      docker compose -p lease-cab --profile serve logs --tail 30 2>&1 | tail -30 | tee -a "$OUT"
      docker compose -p lease-cab --profile serve down >/dev/null 2>&1; sleep 4; return 1
    fi
    if [ $((i % 6)) -eq 0 ]; then
      u=$(rocm-smi --showuse 2>/dev/null | grep -m1 "GPU\[0\]" | grep -oE "[0-9]+$")
      echo "  ${i}0s gpu_use=${u:-?}%" | tee -a "$OUT"
    fi
    sleep 10
  done
  echo "  WEDGED (600s)" | tee -a "$OUT"
  docker compose -p lease-cab --profile serve down >/dev/null 2>&1; sleep 4; return 1; }

run 0 unfused
run 1 fused
echo "=== done ===" | tee -a "$OUT"
