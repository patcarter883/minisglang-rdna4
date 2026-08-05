#!/usr/bin/env bash
# ONE held two-card lease: smoke the BN/WARPS_N attribution probe, then run it in full.
#
#   gpu-lease -n 2 --timeout 7200 -- bash tools/w4a8_dense_bn_diag_run.sh <outdir>
#
# TWO cards, not one, and that is a measurement decision rather than a compute one: the pair share
# board power, PSU headroom, PCIe and case thermals, so a neighbour under load moves these numbers
# without ever touching the card being timed. Holding both makes the BOX exclusive. Only card 0 is
# ever timed, and it is selected inside the tool BY PROPERTIES (64 CU), never by ordinal.
set -uo pipefail
# OUTDIR is a path INSIDE the worktree; the container sees the worktree at /engine, so every path
# handed to the tool must be translated. Writing a host path into the container is the failure that
# costs a lease.
OUTDIR="${1:?usage: w4a8_dense_bn_diag_run.sh <outdir-relative-to-worktree>}"
WT="${TILE_WT:-/home/pat/code/minisgl-rdna4-wn}"
KWT="${TILE_KERNELS:-/home/pat/code/rdna4-hip-kernels-wn}"
OUTDIR="${OUTDIR#$WT/}"
mkdir -p "$WT/$OUTDIR"
HOSTDIR="$WT/$OUTDIR"
OUTDIR="/engine/$OUTDIR"

export TILE_WT="$WT" TILE_KERNELS="$KWT" TILE_EXCLUSIVE=1
export TILE_TOOL=tools/w4a8_dense_bn_diag.py

echo "=== SMOKE (must print a device line and two timings) ==="
bash "$WT"/tools/w4a8_dense_tile_surface_run.sh \
    --exp stage --gs 32 --ms 1 --base-n 8192 \
    --out "$OUTDIR/smoke.txt" --csv "$OUTDIR/smoke.csv"
rc=$?
if [ $rc -ne 0 ] || ! grep -q "^done" "$HOSTDIR/smoke.txt" 2>/dev/null; then
    echo "SMOKE FAILED (rc=$rc) -- not spending the lease on the full sweep" >&2
    exit 1
fi

echo "=== FULL ==="
bash "$WT"/tools/w4a8_dense_tile_surface_run.sh \
    --exp stage,swiz,nsweep,ksweep --swiz 0,1 \
    --out "$OUTDIR/bn_diag.txt" --csv "$OUTDIR/bn_diag.csv"
echo "rc=$?"
