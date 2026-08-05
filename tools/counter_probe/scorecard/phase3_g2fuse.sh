#!/usr/bin/env bash
# phase3_g2fuse.sh — IN-CONTAINER. The two gaps phase2 left.
#
# (a) The PRODUCTION fused MoE decode gemm2 (`moe_gemm2_gather_reduce_core`, grid.y = M). Phase 2
#     measured the UNFUSED gemv_decode_core gemm2, which is not what the engine calls at decode --
#     it is the path the fused kernel replaced. Since the served models are MoE, this is the one
#     that decides the claim.
#
# (b) Dense GEMV at PRODUCTION decode shapes (N,K around 2048), not the synthetic 16384^2.
#     This matters because the "81% of HBM" verdict was taken at a huge synthetic shape and does not
#     transfer: the same kernels at real serve shapes read a fraction of that. Measuring only the
#     synthetic shape is how a percentage-of-roofline claim comes to close off work at shapes it
#     never measured.
set -uo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SELF/pmc_lib.sh"
OUT="${OUT:-/out}/phase3"; mkdir -p "$OUT"

echo "=== phase3: build ==="
hipcc -O3 --offload-arch=gfx1201 -I/kern/fp8_wmma/fp8_wmma_rocm \
      -o /tmp/g2 /probe/scorecard/moe_g2fuse_probe.hip 2>&1 | grep -iE " error" | head -20
hipcc -O3 --offload-arch=gfx1201 -I/kern/fp8_wmma/fp8_wmma_rocm \
      -o /tmp/kp /probe/scorecard/kernel_probes.hip 2>&1 | grep -iE " error" | head -20
[ -x /tmp/g2 ] || { echo "BUILD FAILED (g2)"; exit 1; }
[ -x /tmp/kp ] || { echo "BUILD FAILED (kp)"; exit 1; }

echo "=== phase3: smoke ==="
/tmp/g2 1 32 8 512 2048 128 16 || { echo "SMOKE g2fuse K=512 FAILED"; exit 1; }
/tmp/g2 1 32 8 256 2048 128 16 || { echo "SMOKE g2fuse K=256 FAILED"; exit 1; }

G_WAVE="SQ_WAVES SQ_BUSY_CYCLES"; G_ANY="SQ_WAIT_ANY"; G_INSTW="SQ_WAIT_INST_ANY"
G_WC="SQ_WAVE_CYCLES"; G_INST="SQ_INSTS_VALU SQ_INSTS_SALU"
G_MEM2="GL2C_HIT GL2C_MISS"; G_OCC="OccupancyPercent"; G_BUSY="MemUnitBusy"; G_ISS="WAVE_ISSUE_WAIT"

sweep() {  # sweep <outroot> <tag> <bin> <args...>
  local root="$1" tag="$2" bin="$3"; shift 3
  echo ""; echo "--- sweep $tag :: $bin $* ---"
  local i=0
  for grp in "$G_WAVE" "$G_ANY" "$G_INSTW" "$G_WC" "$G_INST" "$G_MEM2" "$G_OCC" "$G_BUSY" "$G_ISS"; do
    i=$((i+1)); pmc_run "$root/$tag" "g$i" "$grp" -- "$bin" "$@"
  done
}

# (a) fused gemm2, served geometry: E=32 top_k=8 N=2048, K=512 (K-on-lanes) and K=256 (BYLANE).
for m in 1 5 6 30; do
  sweep "$OUT" "g2f_K512_M$m" /tmp/g2 $m 32 8 512 2048 128 16
  sweep "$OUT" "g2f_K256_M$m" /tmp/g2 $m 32 8 256 2048 128 16
done

# (b) dense int4/fp8 GEMV at PRODUCTION decode shapes (Qwen3.6-35B-A3B TP=2 hidden 2048).
sweep "$OUT" dense_prod_2048 /tmp/kp dense 1 2048 2048 128
sweep "$OUT" dense_prod_6144 /tmp/kp dense 1 6144 2048 128

echo "=== phase3 done ==="
