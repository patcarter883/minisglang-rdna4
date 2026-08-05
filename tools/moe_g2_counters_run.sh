#!/usr/bin/env bash
# Occupancy + workgroup count of the fused MoE decode gemm2, BEFORE and AFTER the grid.z split,
# at the SERVED shape. HOST side; takes its own profile_standard window.
#   gpu-lease -n 1 --timeout 3600 -- bash tools/moe_g2_counters_run.sh
# Counters ONLY — profile_standard pins clocks, so no timing may be quoted from this run.
set -uo pipefail
WT="${WT:-/home/pat/code/minisgl-rdna4-g2sk}"
export IMG="${IMG:-rocm/dev-ubuntu-24.04:7.14.0-full}"
export KERN="${KERN:-/home/pat/code/rdna4-hip-kernels-g2sk}"
export OUT="${OUT:-$WT/tools/counter_probe/results/g2split}"
mkdir -p "$OUT"
bash "$WT/tools/counter_probe/scorecard/run_sweep.sh" phase4_g2split
