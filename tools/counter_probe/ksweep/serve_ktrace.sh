#!/usr/bin/env bash
# serve_ktrace.sh — per-kernel GPU trace of a REAL minisgl serve, at served decode (M=1,5,6) and at
# prefill, for ONE (model, loop) configuration.
#
# WHAT IT PRODUCES
#   rocprofv3 --kernel-trace --marker-trace over four ROCTx collection windows. The kernel trace
#   carries, PER DISPATCH: Start/End timestamp, Grid_Size_{X,Y,Z}, Workgroup_Size_{X,Y,Z},
#   VGPR_Count, Accum_VGPR_Count, SGPR_Count, Scratch_Size, LDS_Block_Size. That is enough to compute
#   workgroup count, waves, and register-limited occupancy for EVERY kernel on the served path
#   WITHOUT hardware counters — which matters because `--pmc` cannot run against this serve image
#   (7.2.1 hangs on --pmc; counters need 7.14; a .so built in one image will not load in the other).
#
# WHY NOT THE TORCH PROFILER: the decode path is GRAPH-CAPTURED, so kineto records the graph LAUNCH
# and zero kernels. rocprofv3 sees GPU-side dispatches including graph replays.
#
# MODE=prof   rocprofv3 attached          -> the kernel intervals + launch geometry (the numerator)
# MODE=base   same image, no profiler     -> the TRUE wall/step (the denominator of any idle %)
# Reporting an idle % from the profiled leg alone is circular: rocprofv3 costs ~50 ms/step while a
# window is open and it lands entirely in the inter-kernel gaps.
#
# THE LOOP A/B. `--gdn-radix` (default ON for a GDN/CCA hybrid) forces the SYNCHRONOUS normal_loop —
# scheduler/scheduler.py gates on `self._rec_radix`. LOOP=overlap appends `--no-gdn-radix`, which
# drops the snapshot store and lets the zero-sync overlap_loop run. The ROCTx range NAME records
# which loop actually ran (`normal_step#N` vs `overlap_step#N`), so this is self-verifying rather
# than assumed. GLM has no recurrent state and runs overlap_loop natively.
#
#   gpu-lease -n 2 -- bash tools/counter_probe/ksweep/serve_ktrace.sh
set -uo pipefail
WT=${WT:-/home/pat/code/minisgl-rdna4-ksweep}
MODE=${MODE:-prof}
LOOP=${LOOP:-default}
TAG=${TAG:-run}
RESULTS=${RESULTS:-$WT/tools/counter_probe/results/ksweep}
mkdir -p "$RESULTS"

RUN_ID=${RUN_ID:-$$$(od -An -N2 -tu2 </dev/urandom | tr -dc "0-9")}
# A UNIQUE host port per run. A previous run whose engine died but whose PID 1 is still alive keeps
# its published port; the next run then fails to bind and dies instantly with an EMPTY log, which
# reads like a boot bug rather than a leftover.
PORT=${PORT:-$((23000 + RUN_ID % 900))}
URL=http://127.0.0.1:$PORT
OUT=$RESULTS/$TAG-$MODE-$RUN_ID.log
: > "$OUT"

export COMPOSE_PROJECT_NAME="ksw${TAG}${MODE}${RUN_ID}"
export MINISGL_HOST_PORT=$PORT
export MODEL=${MODEL:-qwen35b-awq} SPEC=${SPEC:-none} TP=${TP:-2} ATTN=${ATTN:-hip}
export CONC=${CONC:-6} GRAPH_BS=${GRAPH_BS:-6}
export MEM_RATIO=${MEM_RATIO:-0.86}
export MINISGL_KV_FP8=${MINISGL_KV_FP8:-1}
# Pin the KV pool. Left to auto-sizing it moves with every other knob (CONC, graph buckets, the
# snapshot-store reservation), so the loop A/B would compare two different pool depths as well as
# two loops. Pinning it makes the pool a CONSTANT of the experiment.
NUM_PAGES=${NUM_PAGES:-3072}
EXTRA="--num-pages $NUM_PAGES"
[ "$LOOP" = "overlap" ] && EXTRA="$EXTRA --no-gdn-radix"
[ "$LOOP" = "normal" ]  && EXTRA="$EXTRA --gdn-radix"
export EXTRA_ARGS="$EXTRA ${EXTRA_ARGS_APPEND:-}"

WARM=${WARM:-16}; TOK=${TOK:-500}; MLIST=${MLIST:-1,5,6}
PREQ=${PREQ:-100}; PWORDS=${PWORDS:-1400}; PTOK=${PTOK:-3}
YMLF="$WT/docker-compose.ksweep-$TAG-$MODE-$RUN_ID.yml"

if [ "$MODE" = "prof" ]; then
  IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:ksweep-prof}
  RPDIR=rpv3-ksweep-$TAG-$RUN_ID
  rm -rf "${WT:?}/$RPDIR"; mkdir -p "$WT/$RPDIR"
  export MINISGL_ROCTX=1
  # Window placement is ARITHMETIC over loop iterations; drive_kphases.py's phases have fixed token
  # counts for exactly this reason, and it PRINTS its own iteration estimate per phase so the
  # placement can be checked against what actually happened rather than trusted.
  #   warmup 1..17 | bs1 18..518 | bs5 519..1024 | bs6 1025..1531 | prefill 1532..1931
  # The format is START:LENGTH, not start:end (scheduler.py parses `(a, a+b)`). Writing it as
  # start:end silently produces windows several times too long that straddle phase boundaries, and
  # the result looks like a successful collection of the wrong batch size.
  export MINISGL_ROCTX_WINDOWS=${MINISGL_ROCTX_WINDOWS:-120:200,620:200,1130:200,1600:200}
  # Must fire INSIDE the last phase and AFTER its window closes, or the loop blocks idle forever on
  # receive_msg and the trace is never flushed.
  export MINISGL_EXIT_AFTER_STEPS=${MINISGL_EXIT_AFTER_STEPS:-1860}
  CMD="exec rocprofv3 --kernel-trace --marker-trace --selected-regions --stats --output-format csv -d /engine/$RPDIR -o ksweep -- /engine/tools/serve.sh"
else
  # The SAME image as the profiled leg, minus the rocprofv3 wrapper. A control on a different image
  # would compare a wall measured on one kernel package against busy time measured on another.
  IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:ksweep-prof}
  RPDIR=""
  CMD="exec /engine/tools/serve.sh"
fi

cat > "$YMLF" <<YML
services:
  serve:
    container_name: ksw-$TAG-$MODE-$RUN_ID
    # restart:unless-stopped is right for a SERVER and wrong for a measurement: a boot that fails the
    # KV-pool assert gets resurrected, crash-loops on the GPU, and keeps the published port. Fail once.
    restart: "no"
    command: ["$CMD"]
YML

DC=(docker compose -f "$WT/docker-compose.yml" -f "$YMLF" --profile serve)
down() { ( cd "$WT" && MINISGL_IMAGE="$IMAGE" "${DC[@]}" down >/dev/null 2>&1 ); }
CLKPID=""
cleanup() { [ -n "$CLKPID" ] && kill "$CLKPID" 2>/dev/null; down; rm -f "$YMLF"; }
trap cleanup EXIT INT TERM
down

{
  echo "run=$RUN_ID tag=$TAG mode=$MODE loop=$LOOP image=$IMAGE port=$PORT"
  echo "engine_commit=$(cd "$WT" && git rev-parse HEAD) dirty=$(cd "$WT" && git status --porcelain | wc -l)"
  echo "lease: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset} LEASE_ROCR_DEVICES=${LEASE_ROCR_DEVICES:-unset}"
  echo "serve: MODEL=$MODEL SPEC=$SPEC TP=$TP CONC=$CONC GRAPH_BS=$GRAPH_BS MEM_RATIO=$MEM_RATIO ATTN=$ATTN KV_FP8=$MINISGL_KV_FP8"
  echo "extra: $EXTRA_ARGS"
  echo "roctx: WINDOWS=${MINISGL_ROCTX_WINDOWS:-n/a} EXIT_AFTER_STEPS=${MINISGL_EXIT_AFTER_STEPS:-n/a}"
  echo "drive: warm=$WARM tok=$TOK mlist=$MLIST preq=$PREQ pwords=$PWORDS ptok=$PTOK"
  echo "cmd:   $CMD"
  # The perf level the numbers were taken at. NEVER mixed with profile_standard: that pins clocks to
  # a fixed non-boost state and a time taken there is not comparable to an auto-mode time.
  for c in /sys/class/drm/card*/device; do
    [ -f "$c/power_dpm_force_performance_level" ] || continue
    slot=$(sed -n 's/^PCI_SLOT_NAME=//p' "$c/uevent" 2>/dev/null)
    case "$slot" in 0000:03:00.0|0000:07:00.0)
      echo "perf_level $(basename "$(dirname "$c")") [$slot] = $(cat "$c/power_dpm_force_performance_level")" ;;
    esac
  done
} | tee -a "$OUT"

( cd "$WT" && env MINISGL_IMAGE="$IMAGE" "${DC[@]}" up -d 2>&1 | tail -5 ) | tee -a "$OUT"
C=$( cd "$WT" && MINISGL_IMAGE="$IMAGE" "${DC[@]}" ps -qa serve )
echo "container=$C" | tee -a "$OUT"

# "container Up" is NOT ready. Watch the container AND the health endpoint AND the log for a
# traceback: the TP ranks can raise and exit while the container's bash (and, under the profiler,
# rocprofv3) stays up, so "still running" is not "still booting" — waiting on it burns the whole boot
# timeout AND holds the GPU lease the entire time.
boot_ok=0
for _ in $(seq 1 260); do
  if curl -s --max-time 3 "$URL/v1/models" >/dev/null 2>&1; then boot_ok=1; break; fi
  if [ -z "$C" ]; then break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null)" != "true" ]; then break; fi
  if docker logs "$C" 2>&1 | grep -aqE "^(RuntimeError|AssertionError|torch\.|OSError|ValueError)|Traceback \(most recent"; then
    echo "boot: engine raised while PID 1 stayed alive — not waiting out the timeout" | tee -a "$OUT"
    break
  fi
  sleep 3
done
echo "boot_ok=$boot_ok" | tee -a "$OUT"
if [ "$boot_ok" != 1 ]; then
  echo "=== BOOT FAILED — engine log tail ===" | tee -a "$OUT"
  docker logs "$C" 2>&1 | tail -50 | tee -a "$OUT"
  exit 1
fi

# The ENGAGE LEDGER: which custom HIP kernels actually fired, straight from the rank-0 log. This is
# the answer to "profile what is dispatched, not what exists in the tree" — a whole day once went
# into a dense W4A8 GEMM that this checkpoint never dispatches.
docker logs "$C" 2>&1 | grep -a "hip-engage" | sed 's/.*\[hip-engage\] //' | sort -u \
  > "$RESULTS/$TAG-engage.txt"
echo "=== engage ledger ($(wc -l < "$RESULTS/$TAG-engage.txt") kernels) ===" | tee -a "$OUT"
cat "$RESULTS/$TAG-engage.txt" | tee -a "$OUT"

# Clock/power sampling for the duration of the drive. One of the candidate explanations for the
# isolated-vs-in-serve gap is that a SUSTAINED two-card serve runs at a lower clock than a short
# single-card bench — which would be a UNIFORM multiplier across every kernel, a signature that
# distinguishes it from any per-kernel cause. Cheap to record, impossible to recover later.
( while true; do
    echo "$(date +%s.%N) $(rocm-smi --showgpuclocks --showpower --csv 2>/dev/null | tr '\n' '|')"
    sleep 2
  done ) > "$RESULTS/$TAG-$MODE-$RUN_ID.clocks.txt" 2>&1 &
CLKPID=$!

python3 "$WT/tools/counter_probe/ksweep/drive_kphases.py" \
  --url "$URL" --out "$RESULTS/$TAG-$MODE-$RUN_ID.phases.json" \
  --warmup-tokens "$WARM" --tok "$TOK" --m-list "$MLIST" \
  --prefill-reqs "$PREQ" --prefill-words "$PWORDS" --prefill-tokens "$PTOK" \
  2>&1 | tee -a "$OUT"

kill "$CLKPID" 2>/dev/null; CLKPID=""

if [ "$MODE" = "prof" ]; then
  # No signals anywhere: wait for the engine to hit its own bound and exit, which is what flushes.
  echo "=== waiting for the container to exit on its own (trace flush) ===" | tee -a "$OUT"
  for _ in $(seq 1 240); do
    [ "$(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null)" = "true" ] || break
    sleep 3
  done
  docker logs "$C" 2>&1 | grep -aiE "EXIT_AFTER_STEPS|Opened result|output generation|rocprof" | tail -8 | tee -a "$OUT"
  echo "=== output files ===" | tee -a "$OUT"
  find "$WT/$RPDIR" -type f -size +0 | tee -a "$OUT"
fi
echo "=== engine log tail ===" | tee -a "$OUT"
docker logs "$C" 2>&1 | grep -aiE "error|Traceback|out of memory|OOM|kv pool|num_pages|prefix cache|snapshot" | tail -20 | tee -a "$OUT"
echo "OUT=$OUT RPDIR=${RPDIR:-none}" | tee -a "$OUT"
