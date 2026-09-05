#!/usr/bin/env bash
# ONE leg of the 48-layer TP=2 boot A/B, with the box recorded around it and sampled through it.
#
# Everything here exists because a previous attempt's numbers were contaminated:
#  * `gpu-lease -n 2` and NOTHING else concurrent. `-n 2` is HOW MANY cards; TP=2 needs both, and
#    holding both is also what keeps a co-tenant off the DDR/NVMe/cores this measurement is bound by.
#    It BLOCKS by default — waiting is the coordination.
#  * The box is recorded BEFORE and AFTER (load average, MemAvailable, ARC, docker ps) because a
#    boot compared across different MemAvailable is not an A/B; the 2026-09-06 baseline started at
#    68.45 GiB and a leg starting at 60 GiB is a different experiment.
#  * `tools/offload/box_sampler.py` runs at 1 Hz on the HOST for the whole leg, so the per-chunk
#    rows' `t_end_wall` can be joined against /proc/vmstat. That join is the only way to tell an
#    op-specific defect from a box-wide reclaim stall.
#  * `timeout` + a named container + a reaping trap: an orphan holding both cards and 48 GiB of
#    pinned RAM poisons every subsequent measurement on this box.
set -euo pipefail

TAG="${TAG:?set TAG, e.g. L48_r2_before}"
OUTDIR="${OUTDIR:?set OUTDIR}"
REPO="${REPO:-/home/pat/code/minisgl-rdna4-ramperf}"
KERN_EXT="${KERN_EXT:-/home/pat/code/rdna4-hip-kernels-e4m3/fp8_wmma/torch-ext}"
RUN_TIMEOUT="${RUN_TIMEOUT:-2700}"
ARGS="${ARGS:---tp 2 --layers 48 --device-gb 8.1 --host-gb 28.0 --cuda-graph-max-bs 2}"
CHUNK_MIB="${CHUNK_MIB:-1372}"
# THE ARENA FLOOR IS A GATE, NOT A MEASUREMENT PARAMETER -- and on THIS box the default 12 GiB is
# unreachable. `MemAvailable` does not count the ZFS ARC, which sits at 14-16 GiB here and IS
# reclaimable, so a nominal floor of F GiB is really a ~(F+15) GiB floor. With the ARC warm from the
# checkpoint, the 48.23 GiB plan + a 12 GiB floor needs 60.23 GiB against a 56.6 GiB reading and
# aborts in `reserve()` before pinning a single page (this blocked round 1 entirely, twice).
# It is lowered for BOTH legs of the A/B and recorded here. It cannot affect either leg's timing:
# it only decides whether `reserve()` proceeds, and the swap tripwire ARMS LATER at a lower floor,
# so it also cannot abort a leg the default would have completed.
FLOOR_GIB="${FLOOR_GIB:-4}"
export CHUNK_MIB FLOOR_GIB

mkdir -p "$OUTDIR"

box() {
  {
    echo "=== box $1 $TAG $(date -Is) chunk=${CHUNK_MIB}MiB floor=${FLOOR_GIB}GiB args='${ARGS}' ==="
    uptime
    free -g
    grep -E 'MemAvailable|SwapFree|MemFree' /proc/meminfo
    awk '/^size /{printf "ARC %.2f GiB\n", $NF/1073741824}' /proc/spl/kstat/zfs/arcstats
    echo "--- non-infra containers (must be none) ---"
    docker ps --format '{{.Names}} {{.Image}}' | grep -Ev 'turnstone|firecrawl|hermes|cloudflared|grafana|prometheus|gpu-exporter|searxng|caddy|postgres|redis|rabbitmq|playwright' || true
    gpu-status || true
  } >> "$OUTDIR/${TAG}.box.txt" 2>&1
}

box BEFORE
python3 "$REPO/tools/offload/box_sampler.py" "$OUTDIR/${TAG}.box_sampler.jsonl" 1.0 &
SAMP=$!
cleanup() { kill "$SAMP" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

set +e
REPO="$REPO" KERN_EXT="$KERN_EXT" OUTDIR="$OUTDIR" TAG="$TAG" RUN_TIMEOUT="$RUN_TIMEOUT" \
  gpu-lease -n 2 -- "$REPO/tools/offload/run_boot_timeline.sh" \
  $ARGS --json "/out/${TAG}.test.json" 2>&1 | tee "$OUTDIR/${TAG}.run.log"
RC=${PIPESTATUS[0]}
set -e

kill "$SAMP" 2>/dev/null || true
box AFTER
echo "exit=$RC" >> "$OUTDIR/${TAG}.box.txt"
exit "$RC"
