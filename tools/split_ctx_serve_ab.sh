#!/usr/bin/env bash
# split_ctx_serve_ab.sh — served decode A/B for the caller-supplied split_ctx fix.
#
# The fix changes which attention kernel the K+1 spec-verify graph dispatches, so it can move served
# throughput. This measures that, on the two models the box actually serves, as an INTERLEAVED
# median-of-N A/B: base = engine fdc482a8 + an image holding kernels a45a14d, cand = the fix.
#
# Reuses serve_ktrace.sh (MODE=base, no profiler — rocprofv3 costs ~50 ms/step and lands in the gaps,
# so a wall taken under it is not a serving time) and glm_ab_report.py (medians + the run-to-run
# spread, so an effect that does not clear the noise is called NOISE).
#
# PROVENANCE IS ASSERTED, not assumed, and on the file that CHANGED: KERNELS_REF is only a
# cache-buster label, the real source is a COPY of whatever was on disk at build time, so each
# image's attn_prefill_paged kernel source is md5'd against the worktree it claims to come from.
#
#   gpu-lease -n 2 --timeout 14400 -- env MODEL=qwen35b-awq SPEC=mtp bash tools/split_ctx_serve_ab.sh
set -uo pipefail
BASE_WT=${BASE_WT:-/home/pat/code/minisgl-rdna4-splitbase}
CAND_WT=${CAND_WT:-/home/pat/code/minisgl-rdna4-splitctx}
BASE_KWT=${BASE_KWT:-/home/pat/code/rdna4-hip-kernels}
CAND_KWT=${CAND_KWT:-/home/pat/code/rdna4-hip-kernels-splitctx}
BASE_IMG=${BASE_IMG:-minisgl-rdna4:glmab-cand}
CAND_IMG=${CAND_IMG:-minisgl-rdna4:splitctx}
REPS=${REPS:-2}
MODEL=${MODEL:-glm}
SPEC=${SPEC:-none}
MLIST=${MLIST:-1,5,6}
CONC=${CONC:-6}
RES=${RES:-$CAND_WT/tools/counter_probe/results/split_ctx_ab_${MODEL}_${SPEC}}
rm -rf "$RES"; mkdir -p "$RES"
MASTER=$RES/ab.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

KSRC=attn_prefill_paged/attn_prefill_paged_rocm/attn_prefill_paged_kernels.hip

say "=== split_ctx serve A/B  model=$MODEL spec=$SPEC mlist=$MLIST conc=$CONC reps=$REPS ==="
say "base: wt=$BASE_WT sha=$(git -C "$BASE_WT" rev-parse --short HEAD) dirty=$(git -C "$BASE_WT" status --porcelain | wc -l) img=$BASE_IMG"
say "cand: wt=$CAND_WT sha=$(git -C "$CAND_WT" rev-parse --short HEAD) dirty=$(git -C "$CAND_WT" status --porcelain | wc -l) img=$CAND_IMG"

for pair in "base|$BASE_IMG|$BASE_KWT" "cand|$CAND_IMG|$CAND_KWT"; do
  leg=${pair%%|*}; rest=${pair#*|}; img=${rest%%|*}; kwt=${rest#*|}
  got=$(docker run --rm --entrypoint bash "$img" -lc "md5sum /opt/rdna4-hip-kernels/$KSRC" 2>/dev/null | awk '{print $1}')
  want=$(md5sum "$kwt/$KSRC" | awk '{print $1}')
  say "provenance $leg: image=${got:-MISSING} worktree=$want"
  [ -n "$got" ] && [ "$got" = "$want" ] || { say "ABORT: $leg image does not hold the kernel source it claims"; exit 3; }
done
# ... and the two legs must NOT be the same kernel, or this is a run against itself.
bm=$(md5sum "$BASE_KWT/$KSRC" | awk '{print $1}'); cm=$(md5sum "$CAND_KWT/$KSRC" | awk '{print $1}')
[ "$bm" != "$cm" ] || { say "ABORT: base and cand hold the SAME kernel source — this would be a null A/B"; exit 3; }

leg_run() {  # tag wt img rep
  local tag=$1 wt=$2 img=$3 rep=$4
  say "--- leg=$tag rep=$rep ---"
  TAG="${tag}-r${rep}" MODE=base LOOP=default MODEL="$MODEL" TP=2 SPEC="$SPEC" \
    CONC="$CONC" GRAPH_BS="$CONC" MEM_RATIO=${MEM_RATIO:-0.86} \
    NUM_PAGES=${NUM_PAGES:-3072} MINISGL_KV_FP8=${MINISGL_KV_FP8:-1} \
    WARM=16 TOK=${TOK:-400} MLIST="$MLIST" PREQ=20 PWORDS=1400 PTOK=3 \
    MINISGL_IMAGE="$img" WT="$wt" RESULTS="$RES" \
    bash "$wt/tools/counter_probe/ksweep/serve_ktrace.sh" >>"$MASTER" 2>&1
  say "--- leg=$tag rep=$rep rc=$? ---"
}

for r in $(seq 1 "$REPS"); do
  # INTERLEAVED, not blocked: a thermal drift that landed entirely inside one block would read as
  # an effect.
  leg_run base "$BASE_WT" "$BASE_IMG" "$r"
  leg_run cand "$CAND_WT" "$CAND_IMG" "$r"
done

say "=== engage ledgers (did the same kernels fire on both legs?) ==="
for f in "$RES"/*-engage.txt; do say "--- $(basename "$f")"; sed 's/^/    /' "$f" | tee -a "$MASTER"; done
say "=== report ==="
python3 "$CAND_WT/tools/counter_probe/ksweep/glm_ab_report.py" --results "$RES" \
  --band "$MODEL TP=2, SPEC=$SPEC, CONC=$CONC, pinned --num-pages ${NUM_PAGES:-3072}" 2>&1 | tee -a "$MASTER"
say "=== done. results=$RES"
