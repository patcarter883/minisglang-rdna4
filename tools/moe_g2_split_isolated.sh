#!/usr/bin/env bash
# ONE GPU window for the isolated evaluation of the MoE decode gemm2 splits.
#   gpu-lease -n 1 --timeout 5400 -- bash tools/moe_g2_split_isolated.sh
# (1) bit-exactness gate, single-process, with the sk1b control
# (2) slice-count SURFACE of the fused gather-reduce at the served M (derivation vs the optimum)
# (3) slice-count surface of the SCATTER arm at the only two M it serves (1 and 2)
# (4) whole-MoE through the engine dispatch
set -uo pipefail
WT="${WT:-/home/pat/code/minisgl-rdna4-g2sk}"
export G2_MOUNT="-v /home/pat/code/rdna4-hip-kernels-g2sk:/kern:ro" G2_KERN=/kern/fp8_wmma/torch-ext
mkdir -p "$WT/tools/_fixtures"
run() { bash "$WT/tools/moe_g2_probe_run.sh" "$@"; }

echo "############ (1) bit-exactness gate (single process, sk1b control) ############"
run tools/moe_g2_split_parity.py --m 3,4,5,6,8,16,32 2>&1 | tail -50
echo ""
echo "############ (2) fused gather-reduce: slice-count surface ############"
run tools/moe_g2_bench.py --m 3,5,6,8,16,32 --fused-sk 1,2,4,8 --tag fused_surface \
    --json /engine/tools/_fixtures/moe_g2_fused_sk.json 2>&1 | grep -vE "^\[bench\] shape"
echo ""
echo "############ (3) scatter arm: slice-count surface at M=1,2 ############"
run tools/moe_g2_bench.py --m 1,2 --scatter-sk 1,2,4,8 --tag scatter_surface \
    --json /engine/tools/_fixtures/moe_g2_scatter_sk.json 2>&1 | grep -vE "^\[bench\] shape"
echo ""
echo "############ (4) whole-MoE through the engine dispatch ############"
G2_ENV="-e MINISGL_MOE_G2_SPLIT_DEBUG=1" run tools/moe_g2_served_probe.py --m 1,2,5,6 2>&1 | tail -22
