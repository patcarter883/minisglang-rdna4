#!/usr/bin/env bash
# ONE held two-card lease for the whole W4A8 tile measurement batch.
#
# WHY TWO CARDS FOR SINGLE-CARD WORK: the two cards share board power, PSU headroom, PCIe and
# thermals, so a neighbour job on card 1 moves card 0's clocks. `-n 2` makes the whole BOX
# exclusive; the tools then SELECT the 64-CU card BY PROPERTIES (never by ordinal) and time only
# that one. Never two timing runs concurrently -- this script is strictly serial.
#
# ORDER IS LOAD-BEARING: correctness gates run BEFORE any timing. Putting the bit-identity gate
# after the timing loop is exactly how the MoE gemm1 SIGFPE stayed hidden for months.
#
#   gpu-lease -n 2 --timeout 14400 -- bash tools/w4a8_wn2_batch.sh
set -uo pipefail
cd /home/pat/code/minisgl-rdna4-wn2
export TILE_WT=/home/pat/code/minisgl-rdna4-wn2
export TILE_KERNELS=/home/pat/code/rdna4-hip-kernels-wn2
export TILE_EXCLUSIVE=1
R=tools/_fixtures
mkdir -p "$R"

step () {
  echo ""
  echo "############################################################"
  echo "## $1"
  echo "############################################################"
}

# ---------------------------------------------------------------- 1. CORRECTNESS
step "1/4  MoE fused-gemm1 BN fault gate (SIGFPE + silent mis-launch)"
TILE_TOOL=tools/w4a8_moe_bn_fault_repro.py bash tools/w4a8_moe_bn_fault_run.sh \
  --out /engine/$R/moe_bn_fault_gate.txt
echo "exit=$?"

step "2/4  MoE tile verify -- correctness first, M ladder now past 32"
TILE_TOOL=tools/w4a8_moe_tile_verify.py bash tools/w4a8_dense_tile_surface_run.sh \
  --out /engine/$R/moe_tile_verify_bnfix.txt
echo "exit=$?"

# ---------------------------------------------------------------- 2. THE MERGE NUMBER
step "3/4  DENSE tile verify -- chooser vs the shipped 256x128 hard-wire (live)"
TILE_TOOL=tools/w4a8_dense_tile_verify.py bash tools/w4a8_dense_tile_surface_run.sh \
  --out /engine/$R/dense_tile_verify_card0.txt \
  --csv /engine/$R/dense_tile_verify_card0.csv \
  --oracle-csv /engine/$R/dense_tile_surface_card0.csv
echo "exit=$?"

# ---------------------------------------------------------------- 3. THE CONTAMINATED FIXTURE
# Every M>32 cell of the existing MoE surface was measured through the fused gemm1+silu launcher
# while it silently redirected 5 of the 7 swept BN to the BN=64 body. Those rows are not slow or
# noisy, they are WRONG (wrong column coverage, under-reported time). Re-measure on the fixed build.
step "4/4  MoE tile surface RE-MEASURE (the old one is BN-contaminated at M>32)"
TILE_TOOL=tools/w4a8_moe_tile_surface.py bash tools/w4a8_dense_tile_surface_run.sh \
  --out /engine/$R/moe_tile_surface_card0_bnfix.txt \
  --csv /engine/$R/moe_tile_surface_card0_bnfix.csv
echo "exit=$?"

echo ""
echo "BATCH DONE"
