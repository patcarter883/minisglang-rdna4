#!/usr/bin/env bash
# run_all.sh — the WHOLE GPU side of the per-kernel sweep, in ONE exclusive lease.
#
# The two cards are shared across every repo on this box and `profile_standard` is a GLOBAL,
# PER-CARD, PERSISTENT setting — so the counter stage cannot be allowed to run while another agent
# times anything. Holding one exclusive `-n 2` lease for the entire session is what makes that safe,
# and it is why every stage lives in this one script instead of in separate invocations.
#
# STAGE ORDER, and why it is this order:
#   1-6  TRACE legs, perf level AUTO. Times taken here are real serving times.
#   7    ISOLATED replay, perf level AUTO. Must be at auto so its times are comparable with 1-6 —
#        that comparison IS the primary deliverable (isolated vs in-serve).
#   8    COUNTERS, perf level PROFILE_STANDARD. Pins clocks non-boost, so NOTHING timed in 1-7 may
#        be compared against a timestamp taken here. Counters are ratios and byte counts, which are
#        clock-independent; times are not. Last, and restored by trap.
#
#   GPU_LEASE_WEDGE_WATCH=0 gpu-lease -n 2 -- bash tools/counter_probe/ksweep/run_all.sh
set -uo pipefail
WT=${WT:-/home/pat/code/minisgl-rdna4-ksweep}
KSW=$WT/tools/counter_probe/ksweep
RES=${RES:-$WT/tools/counter_probe/results/ksweep}
mkdir -p "$RES"
MASTER=$RES/run_all.log
STAGES=${STAGES:-1,2,3,4,5,6,7,8}

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }
want() { [[ ",$STAGES," == *",$1,"* ]]; }

say "=== ksweep run_all start ==="
say "engine=$(cd "$WT" && git rev-parse --short HEAD) lease: ROCR=${ROCR_VISIBLE_DEVICES:-?} HIP=${HIP_VISIBLE_DEVICES:-?} LEASE_ROCR=${LEASE_ROCR_DEVICES:-?}"
say "stages=$STAGES"

# --- the concurrency the pair can actually boot ---------------------------------------------------
# max-running-requests is 6 on these 16 GB cards, but the 35B is a GDN HYBRID: 30 of 40 layers
# reserve a per-sequence RECURRENT STATE up front (~31 MB/seq/card at TP=2) on top of the graph
# buffers, so max-running trades against the KV pool far more steeply than on a dense-KV model.
# CONC is discovered once, by BOOTING, and then pinned for every later stage — an A/B across two
# different concurrencies would not be an A/B.
CONC_USE=${CONC_USE:-6}

stage_trace() {   # tag mode loop model conc
  local tag=$1 mode=$2 loop=$3 model=$4 conc=$5
  say "--- stage: $tag mode=$mode loop=$loop model=$model conc=$conc ---"
  TAG="$tag" MODE="$mode" LOOP="$loop" MODEL="$model" CONC="$conc" GRAPH_BS="$conc" \
    RESULTS="$RES" WT="$WT" \
    bash "$KSW/serve_ktrace.sh" >>"$MASTER" 2>&1
  local rc=$?
  say "--- stage $tag rc=$rc ---"
  # Parse IMMEDIATELY, not at the end. A later stage that dies (an OOM, a wedged card, a lease
  # timeout) must not take the analysis of an already-successful trace with it — and a parse failure
  # is much cheaper to notice now than after every trace has been collected.
  if [ "$mode" = "prof" ] && [ $rc -eq 0 ]; then
    local rp
    rp=$(ls -dt "$WT"/rpv3-ksweep-"$tag"-* 2>/dev/null | head -1)
    if [ -n "$rp" ]; then
      say "    parsing $rp"
      ( cd "$KSW" && python3 parse_ksweep.py "$rp" --tag "$tag" --out-dir "$RES" ) \
        >"$RES/$tag.kernels.txt" 2>&1
      ( cd "$KSW" && python3 analyze_ksweep.py "$rp" --tag "$tag" --out-dir "$RES" ) \
        >"$RES/$tag.analysis.txt" 2>&1
      say "    parsed -> $RES/$tag.kernels.{txt,csv,json}, $tag.analysis.{txt,json}"
    else
      say "    NO rpv3 dir found for $tag"
    fi
  fi
  return $rc
}

# ==================================================================================================
# 1-2  Qwen3.6-35B-A3B-AWQ, TP=2, SPEC=none, the DEFAULT loop.
#      This model is a GDN hybrid and --gdn-radix defaults ON, which resolves to
#      snapshot_kind="recurrent" and therefore forces the SYNCHRONOUS normal_loop
#      (scheduler.py gates on self._rec_radix). The ROCTx range name in the trace records which
#      loop actually ran, so the claim is self-verifying.
# ==================================================================================================
if want 1; then
  if ! stage_trace qwen-normal prof normal qwen35b-awq "$CONC_USE"; then
    say "CONC=$CONC_USE did not boot; falling back to 5 then 4 and PINNING the result"
    for c in 5 4; do
      if stage_trace qwen-normal prof normal qwen35b-awq "$c"; then CONC_USE=$c; break; fi
    done
  fi
  say "CONC pinned at $CONC_USE for every later stage"
fi
want 2 && stage_trace qwen-normal base normal qwen35b-awq "$CONC_USE"

# ==================================================================================================
# 7  ISOLATED replay, AUTO clocks — run HERE, straight after the first trace, not at the end.
#    It is the primary deliverable and it depends only on stage 1, so it must not be hostage to the
#    four serve boots that follow. Its `evict_bytes` is DERIVED from stage 1's measured elementwise
#    volume (make_iso_shapes.py) rather than chosen — choosing it would tune the experiment until it
#    reproduced the answer being tested for.
# ==================================================================================================
if want 7; then
  say "--- stage 7: isolated replay (auto clocks) ---"
  if [ -f "$RES/qwen-normal.analysis.json" ]; then
    ( cd "$KSW" && python3 make_iso_shapes.py \
        --analysis "$RES/qwen-normal.analysis.json" \
        --template "$KSW/iso_replay_example.json" \
        --out "$RES/iso_shapes.json" ) >>"$MASTER" 2>&1
    say "    iso_shapes.json built from the measured elementwise volume"
  else
    say "    no analysis json yet — iso_shapes.json not derived"
  fi
  if [ -f "$KSW/iso_replay.py" ] && [ -f "$RES/iso_shapes.json" ]; then
    docker run --rm \
      -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
      -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
      --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
      --ipc host --shm-size 16gb \
      -v "$WT":/engine -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
      -e HF_HUB_OFFLINE=1 \
      --entrypoint bash minisgl-rdna4:ksweep-prof -lc \
      'source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate; \
       PYTHONPATH=/opt/kernels:/engine/python:/engine \
       python /engine/tools/counter_probe/ksweep/iso_replay.py \
         --spec /engine/tools/counter_probe/results/ksweep/iso_shapes.json \
         --image minisgl-rdna4:ksweep-prof \
         --out /engine/tools/counter_probe/results/ksweep/iso_replay.csv' \
      >>"$MASTER" 2>&1
    say "--- stage 7 rc=$? ---"
  else
    say "stage 7 SKIPPED: missing $KSW/iso_replay.py or $RES/iso_shapes.json"
  fi
fi

# ==================================================================================================
# 3-4  The LOOP A/B. --no-gdn-radix drops the recurrent snapshot store, so the plan resolves to
#      cache_type="naive"/snapshot_kind="" and the zero-sync overlap_loop runs instead. This costs
#      prefix reuse on GDN traffic — that is the price of the experiment, and it is why the pinned
#      --num-pages matters: the snapshot store is a VRAM reservation subtracted from the KV pool, so
#      without pinning, the two arms would differ in pool depth as well as in loop.
# ==================================================================================================
want 3 && stage_trace qwen-overlap prof overlap qwen35b-awq "$CONC_USE"
want 4 && stage_trace qwen-overlap base overlap qwen35b-awq "$CONC_USE"

# ==================================================================================================
# 5-6  GLM-4.7-Flash-AWQ, TP=2, SPEC=none. MLA + MoE, NO recurrent state and no sliding window, so
#      resolve_prefix_cache returns snapshot_kind="" and it runs overlap_loop NATIVELY. That makes
#      the two models a natural control for the loop question as well as a second kernel mix.
# ==================================================================================================
want 5 && stage_trace glm-overlap prof default glm "$CONC_USE"
want 6 && stage_trace glm-overlap base default glm "$CONC_USE"

# ==================================================================================================
# 8  COUNTERS at profile_standard, ROCm 7.14. LAST, because it pins the clocks.
#    BOTH conditions are required: 7.2.1 hangs on --pmc and the host 7.2.4 aborts, so the container
#    must be 7.14; and RDNA4's default `auto` perf level GATES the perfmon clock, so counters read a
#    hard zero with rc=0 and a well-formed CSV. Restored from a trap on every exit path.
# ==================================================================================================
if want 8; then
  say "--- stage 8: cache-gap probe (timing @auto, then counters @profile_standard) ---"
  RES="$RES" WT="$WT" bash "$KSW/cache_gap.sh" >>"$MASTER" 2>&1
  say "--- stage 8 rc=$? ---"
fi

say "=== ksweep run_all done ==="
for c in /sys/class/drm/card*/device; do
  [ -f "$c/power_dpm_force_performance_level" ] || continue
  slot=$(sed -n 's/^PCI_SLOT_NAME=//p' "$c/uevent" 2>/dev/null)
  case "$slot" in 0000:03:00.0|0000:07:00.0)
    say "final perf_level $(basename "$(dirname "$c")") [$slot] = $(cat "$c/power_dpm_force_performance_level")" ;;
  esac
done
