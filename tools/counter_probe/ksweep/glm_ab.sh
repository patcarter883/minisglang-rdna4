#!/usr/bin/env bash
# glm_ab.sh — serve A/B for the GLM decode fixes, TIMING ONLY (no profiler, no counters).
#
# Two legs, N repeats each, medians. Each leg is a FULL serve boot of MODEL=glm TP=2 with a PINNED
# --num-pages, driven by drive_kphases.py at bs=1/5/6. MODE=base means rocprofv3 is NOT attached:
# rocprofv3 costs ~50 ms/step and lands in the gaps, so a wall/step taken under it is not a serving
# time. This script produces the denominator; the kernel shares come from a separate profiled run.
#
# PROVENANCE IS ASSERTED PER LEG, not assumed — a merged-but-inert change has happened twice.
#   * the engine worktree SHA and dirty-count are printed by serve_ktrace.sh into the leg log;
#   * the engage ledger ($TAG-engage.txt) must contain the CANDIDATE's new arm and must NOT contain
#     it on the base leg;
#   * the KERNEL source baked into each image is diffed against the worktree it claims to come from,
#     because KERNELS_REF is only a cache-buster label and the real source is a COPY of whatever was
#     on disk at build time.
#
# ONE exclusive 2-card lease for the whole run — TP=2 needs both, and an A/B whose two legs ran
# beside different neighbours is not an A/B.
#
#   gpu-lease -n 2 -- bash tools/counter_probe/ksweep/glm_ab.sh
set -uo pipefail
BASE_WT=${BASE_WT:-/home/pat/code/minisgl-rdna4-glmbase}
CAND_WT=${CAND_WT:-/home/pat/code/minisgl-rdna4-glmfix}
BASE_IMG=${BASE_IMG:-minisgl-rdna4:glmab-base}
CAND_IMG=${CAND_IMG:-minisgl-rdna4:glmab-cand}
REPS=${REPS:-3}
RES=${RES:-$CAND_WT/tools/counter_probe/results/glm_ab}
mkdir -p "$RES"
MASTER=$RES/glm_ab.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

say "=== glm_ab start  reps=$REPS ==="
say "base: wt=$BASE_WT img=$BASE_IMG sha=$(git -C "$BASE_WT" rev-parse --short HEAD) dirty=$(git -C "$BASE_WT" status --porcelain | wc -l)"
say "cand: wt=$CAND_WT img=$CAND_IMG sha=$(git -C "$CAND_WT" rev-parse --short HEAD) dirty=$(git -C "$CAND_WT" status --porcelain | wc -l)"

# --- image/source provenance: does the image hold the kernel source it claims to? ----------------
for pair in "base:$BASE_IMG:/home/pat/code/rdna4-hip-kernels-glmbase" \
            "cand:$CAND_IMG:/home/pat/code/rdna4-hip-kernels-glmfix"; do
  leg=${pair%%:*}; rest=${pair#*:}; img=${rest%%:*}; kwt=${rest#*:}
  got=$(docker run --rm --entrypoint bash "$img" -lc 'md5sum /opt/rdna4-hip-kernels/mla/mla_rocm/mla_attend.h' | awk '{print $1}')
  want=$(md5sum "$kwt/mla/mla_rocm/mla_attend.h" | awk '{print $1}')
  say "provenance $leg: image mla_attend.h md5=${got:-MISSING} worktree md5=$want"
done

leg_run() {  # tag wt img rep
  local tag=$1 wt=$2 img=$3 rep=$4
  say "--- leg=$tag rep=$rep ---"
  TAG="${tag}-r${rep}" MODE=base LOOP=default MODEL=glm TP=2 SPEC=none \
    CONC=${CONC:-6} GRAPH_BS=${CONC:-6} MEM_RATIO=${MEM_RATIO:-0.86} \
    NUM_PAGES=${NUM_PAGES:-3072} MINISGL_KV_FP8=1 \
    WARM=16 TOK=${TOK:-400} MLIST=1,5,6 PREQ=20 PWORDS=1400 PTOK=3 \
    MINISGL_IMAGE="$img" WT="$wt" RESULTS="$RES" \
    bash "$wt/tools/counter_probe/ksweep/serve_ktrace.sh" >>"$MASTER" 2>&1
  say "--- leg=$tag rep=$rep rc=$? ---"
}

for r in $(seq 1 "$REPS"); do
  # INTERLEAVED, not blocked: a slow drift (thermals, another agent's neighbour job) that lands
  # entirely inside one block would be read as an effect. Alternating splits it across both legs.
  leg_run base "$BASE_WT" "$BASE_IMG" "$r"
  leg_run cand "$CAND_WT" "$CAND_IMG" "$r"
done

say "=== engage ledgers ==="
for f in "$RES"/*-engage.txt; do say "--- $(basename "$f")"; sed 's/^/    /' "$f" | tee -a "$MASTER"; done

python3 "$CAND_WT/tools/counter_probe/ksweep/glm_ab_report.py" --results "$RES" | tee -a "$MASTER"
say "=== glm_ab done ==="
