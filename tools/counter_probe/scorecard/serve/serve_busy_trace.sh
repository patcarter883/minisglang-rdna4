#!/usr/bin/env bash
# GPU-BUSY vs WALL on a REAL minisgl decode serve (Qwen3.6-35B-A3B-AWQ, TP=2, SPEC=none).
#
# WHAT THIS ANSWERS
#   "bs=1 decode is inter-kernel-gap bound, ~68% idle" and "serving is overhead-bound at every batch
#   size". Both are statements about the fraction of the step the GPU is executing kernels, so they
#   need GPU-side dispatch timestamps for a REAL serve — not a microbench, and not a torch profile.
#
# WHY NOT THE TORCH PROFILER. Two independent reasons, either one fatal:
#   1. It inflates this workload. tools/parse_decode_trace.py over the recorded
#      tools/loads/qwen35b_mtp_decode.pt.trace.json.gz reports 161.9 ms wall and 35.3 ms GPU-busy per
#      step against a real served step of ~11 ms — the "busy" alone is 3x the whole real step, so any
#      idle fraction derived from it is an artifact of the instrument.
#   2. The decode path is GRAPH-CAPTURED. A kineto trace records the graph LAUNCH and zero kernels,
#      so the in-graph dispatches — which are the entire question — are invisible.
#   rocprofv3 --kernel-trace sees GPU-side dispatches INCLUDING those replayed from a captured graph,
#   at much lower overhead. That is why it is the instrument here.
#
# TWO LEGS, and both are needed:
#   MODE=base  stock image, NO profiler   -> the TRUE wall/step (the denominator of "idle %")
#   MODE=prof  lean-prof + rocprofv3      -> the kernel intervals (the numerator)
# Reporting an idle % from the profiled run alone would be circular: it would be measured against a
# wall that the instrument itself stretched. The base leg is what says whether it did.
#
# CONSTRAINTS INHERITED FROM tools/propose_rocprof.sh (all load-bearing, none optional):
#   * IMAGE must have torch's vendored librocprofiler-sdk.so pointed at the ROCm one, or the process
#     core-dumps in rocprofiler_configure. build_prof_image.sh makes that image from TODAY's lean,
#     so the baked /opt/kernels matches the hot-mounted engine (a stale one fails at BOOT, not build).
#   * rocprofv3 is LD_PRELOADed into the ENGINE process, so ANY signal aborts the trace write. The
#     only shutdown that flushes is MINISGL_EXIT_AFTER_STEPS, which makes run_forever RETURN.
#   * Unique COMPOSE_PROJECT_NAME *and* unique container_name per run: docker-compose.yml pins
#     container_name to ${LEASE_NAME}-serve, identical for every run on the same cards, so the
#     project name alone cannot stop one run's `down` from killing another's container.
#
# NO HARDWARE COUNTERS. --pmc hangs on the ROCm 7.2.1 serve image (gfx1201). --kernel-trace only, at
# the normal auto perf level; power_dpm_force_performance_level is not touched.
#
#   gpu-lease -n 2 -- bash tools/counter_probe/scorecard/serve/serve_busy_trace.sh
set -uo pipefail
WT=/home/pat/code/minisgl-rdna4-cscore
MODE=${MODE:-prof}
RESULTS=$WT/tools/counter_probe/results/serve
mkdir -p "$RESULTS"

RUN_ID=${RUN_ID:-$$$(od -An -N2 -tu2 </dev/urandom | tr -dc "0-9")}
# A UNIQUE host port per run, not a fixed one. A previous run whose engine died but whose PID 1 is
# still alive keeps its published port, and the next run's container then fails to bind and dies
# instantly with an EMPTY log — which looks like a boot bug rather than a leftover. Twice.
PORT=${PORT:-$((22000 + RUN_ID % 900))}
URL=http://127.0.0.1:$PORT
OUT=$RESULTS/${MODE}-$RUN_ID.log
: > "$OUT"

export COMPOSE_PROJECT_NAME="cscore${MODE}${RUN_ID}"
export MINISGL_HOST_PORT=$PORT
# The served configuration under test. SPEC=none is the point: with the model's default (mtp) this
# would measure spec decode, which is a different step with a different kernel mix.
export MODEL=qwen35b-awq SPEC=none TP=2 ATTN=hip
# CONC=8 does NOT boot on this pair: the engine dies in _determine_num_pages with "Not enough memory
# for KV cache after reserving recurrent state / draft model / CUDA-graph buffers". This is a GDN
# HYBRID — 30 of its 40 layers hold a per-sequence RECURRENT STATE reserved for max-running-requests
# up front (~31 MB/seq/card at TP=2), on top of the graph buffers, so max-running trades against the
# KV pool far more steeply than on a dense-KV model. 4 at ratio 0.86 boots.
export CONC=${CONC:-4} GRAPH_BS=${GRAPH_BS:-4}
export MEM_RATIO=${MEM_RATIO:-0.86}

# fp8 KV, the production default (compose sets MINISGL_KV_FP8=1). Left alone deliberately: this is
# meant to measure the SERVED config. It is only servable because the profiling image is now rebuilt
# from today's `lean` (build_prof_image.sh) rather than being a hand-built image from three days ago
# whose baked /opt/kernels still had the per-TENSOR store_kv signature.
export MINISGL_KV_FP8=${MINISGL_KV_FP8:-1}

WARM=${WARM:-16}; TOKA=${TOKA:-600}; TOKB=${TOKB:-600}; MB=${MB:-4}
YMLF="$WT/docker-compose.cscore-$MODE-$RUN_ID.yml"

if [ "$MODE" = "prof" ]; then
  IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean-prof-today}
  RPDIR=rpv3-busy-$RUN_ID
  rm -rf "${WT:?}/$RPDIR"; mkdir -p "$WT/$RPDIR"
  # Window placement is ARITHMETIC over loop iterations, which is why the driver's phases have fixed
  # token counts (see drive_windows.py). warmup 16 -> iters ~1..18; phase A (bs=1, TOKA) -> ~19..620;
  # phase B (bs=8, TOKB) -> ~621..1230. Windows sit well inside each phase so a few iterations of
  # drift (chunked prefill, a radix hit, an idle wakeup) cannot push a window into the wrong phase.
  export MINISGL_ROCTX=1
  export MINISGL_ROCTX_WINDOWS=${MINISGL_ROCTX_WINDOWS:-200:200,800:200}
  # Must fire INSIDE phase B and AFTER the second window closes, or the loop blocks idle forever on
  # receive_msg and the trace is never flushed.
  export MINISGL_EXIT_AFTER_STEPS=${MINISGL_EXIT_AFTER_STEPS:-1100}
  CMD="exec rocprofv3 --kernel-trace --marker-trace --selected-regions --stats --output-format csv -d /engine/$RPDIR -o busy -- /engine/tools/serve.sh"
else
  # The SAME image as the profiled leg, minus the rocprofv3 wrapper. Running the control on
  # stock `lean` would compare a wall measured on one kernel package against busy time
  # measured on another — the two images are different vintages and bake different
  # /opt/kernels. Matched by construction is the only way this comparison means anything.
  IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean-prof-today}
  RPDIR=""
  CMD="exec /engine/tools/serve.sh"
fi

cat > "$YMLF" <<YML
services:
  serve:
    container_name: cscore-$MODE-$RUN_ID
    # The serve service is restart:unless-stopped, which is right for a SERVER and wrong for a
    # measurement. A boot that fails the KV-pool assert gets resurrected, crash-loops on the GPU, and
    # keeps the published port — so the NEXT run cannot bind and reports a boot failure with an empty
    # log, which reads like a different bug entirely. Fail once, visibly.
    restart: "no"
    command: ["$CMD"]
YML

DC=(docker compose -f "$WT/docker-compose.yml" -f "$YMLF" --profile serve)
down() { ( cd "$WT" && MINISGL_IMAGE="$IMAGE" "${DC[@]}" down >/dev/null 2>&1 ); }
trap 'down; rm -f "$YMLF"' EXIT INT TERM
down

{
  echo "run=$RUN_ID mode=$MODE image=$IMAGE port=$PORT"
  echo "commit=$(cd "$WT" && git rev-parse HEAD) dirty=$(cd "$WT" && git status --porcelain | wc -l)"
  echo "lease: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset} LEASE_ROCR_DEVICES=${LEASE_ROCR_DEVICES:-unset}"
  echo "serve: MODEL=$MODEL SPEC=$SPEC TP=$TP CONC=$CONC GRAPH_BS=$GRAPH_BS MEM_RATIO=$MEM_RATIO ATTN=$ATTN KV_FP8=$MINISGL_KV_FP8"
  echo "roctx: WINDOWS=${MINISGL_ROCTX_WINDOWS:-n/a} EXIT_AFTER_STEPS=${MINISGL_EXIT_AFTER_STEPS:-n/a}"
  echo "drive: warm=$WARM tokA=$TOKA tokB=$TOKB M_B=$MB"
  echo "cmd:   $CMD"
  # The perf level the numbers were taken at. Never mixed with profile_standard.
  for c in 0 1; do
    p=/sys/class/drm/card$c/device/power_dpm_force_performance_level
    [ -r "$p" ] && echo "perf_level card$c=$(cat "$p")"
  done
} | tee -a "$OUT"

# Do NOT swallow `up`'s stderr: a port clash or an image problem is reported HERE and nowhere else.
( cd "$WT" && env MINISGL_IMAGE="$IMAGE" "${DC[@]}" up -d 2>&1 | tail -5 ) | tee -a "$OUT"
C=$( cd "$WT" && MINISGL_IMAGE="$IMAGE" "${DC[@]}" ps -qa serve )
echo "container=$C" | tee -a "$OUT"

# Fail FAST on a boot that dies. A 35B TP=2 boot takes minutes, so the driver's ready-timeout has to
# be generous — but a KV-pool assert or an OOM kills the container in ~90 s and there is then nothing
# to wait for. Watch the container, not just the port, and bail with the reason.
boot_ok=0
for _ in $(seq 1 200); do
  if curl -s --max-time 3 "$URL/v1/models" >/dev/null 2>&1; then boot_ok=1; break; fi
  if [ -z "$C" ]; then break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null)" != "true" ]; then break; fi
  # A dead ENGINE behind a live PID 1. The TP ranks can raise and exit while the container's bash
  # (and, under the profiler, rocprofv3) stays up, so "container still running" is NOT "still
  # booting" — waiting on it burns the whole boot timeout AND holds the GPU lease the entire time.
  if docker logs "$C" 2>&1 | grep -aqE "^(RuntimeError|AssertionError|torch\.|OSError|ValueError)|Traceback \(most recent"; then
    echo "boot: engine raised while PID 1 stayed alive — not waiting out the timeout" | tee -a "$OUT"
    break
  fi
  sleep 3
done
echo "boot_ok=$boot_ok" | tee -a "$OUT"
if [ "$boot_ok" != 1 ]; then
  echo "=== BOOT FAILED — engine log tail ===" | tee -a "$OUT"
  docker logs "$C" 2>&1 | tail -40 | tee -a "$OUT"
  exit 1
fi

python3 "$WT/tools/counter_probe/scorecard/serve/drive_windows.py" \
  --url "$URL" --out "$RESULTS/${MODE}-$RUN_ID.phases.json" \
  --warmup-tokens "$WARM" --phase-a-tokens "$TOKA" --phase-b-tokens "$TOKB" --phase-b-m "$MB" \
  2>&1 | tee -a "$OUT"

if [ "$MODE" = "prof" ]; then
  # No signals anywhere: wait for the engine to hit its own bound and exit, which is what flushes.
  echo "=== waiting for the container to exit on its own (trace flush) ===" | tee -a "$OUT"
  for _ in $(seq 1 200); do
    [ "$(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null)" = "true" ] || break
    sleep 3
  done
  docker logs "$C" 2>&1 | grep -aiE "EXIT_AFTER_STEPS|Opened result|output generation|rocprof" | tail -8 | tee -a "$OUT"
  echo "=== output files ===" | tee -a "$OUT"
  find "$WT/$RPDIR" -type f -size +0 | tee -a "$OUT"
else
  docker logs "$C" 2>&1 | tail -5 | tee -a "$OUT"
fi
echo "=== engine log tail ===" | tee -a "$OUT"
docker logs "$C" 2>&1 | grep -aiE "error|Traceback|out of memory|OOM|kv pool|num_pages" | tail -12 | tee -a "$OUT"
echo "OUT=$OUT RPDIR=${RPDIR:-none}" | tee -a "$OUT"
