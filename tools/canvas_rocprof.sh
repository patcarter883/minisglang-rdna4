#!/usr/bin/env bash
# canvas_rocprof.sh — decompose a DiffusionGemma BLOCK-DIFFUSION canvas step by GPU kernel and phase.
#
# Two legs in ONE lease, because the two questions need opposite instruments and the answer only
# means something if both are taken off the same boot config:
#
#   leg `base`  — no profiler. MINISGL_CANVAS_TIMING=1 gives the host-side fwd/sampler/soft_embed
#                 split and the served tok/s. This is TODAY'S BASELINE, and it is the number every
#                 share below is a share OF. It also captures the [hip-engage] ledger, which is the
#                 only way to know WHICH kernels the checkpoint actually dispatches (a dense-GEMM
#                 conclusion drawn without it is how a day went into a tile model for a path that
#                 never ran).
#   leg `trace` — rocprofv3 --kernel-trace --marker-trace --selected-regions. The canvas loop only
#                 just got ROCTx markers (scheduler/diffusion.py: canvas_step / canvas_fwd /
#                 canvas_sampler / canvas_soft_embed / canvas_encode); before them --selected-regions
#                 never opened a window and a block-diffusion serve traced EMPTY.
#
# WHY NOT ONE LEG. MINISGL_CANVAS_TIMING costs two torch.cuda.synchronize() per step, which serialises
# exactly the overlap the trace is meant to measure; and rocprofv3 inflates small kernels more than
# large ones. Each leg is therefore run with the OTHER instrument off, and the trace is used for
# SHARES while the timing leg is used for the absolute step time.
#
# WHY A BOUNDED EXIT AND NO SIGNALS: see tools/propose_rocprof.sh. rocprofv3 is LD_PRELOADed into the
# engine process, so any signal lands in its error handler and aborts the write. The trace comes out
# only on a normal interpreter teardown -> MINISGL_EXIT_AFTER_STEPS.
#
# MINISGL_SWA_RADIX is PINNED (default 0) on BOTH legs and reported. It is ON by default and it
# page-splits every prefill and lets a warm prompt skip the encoder entirely — both move wall-clock
# per request without moving per-denoising-step cost. 0 is also the documented workaround for the
# unresolved chunked-encoder divergence on partial prefix hits (docs §D3).
#
#   gpu-lease -n 2 --timeout 7200 -- bash tools/canvas_rocprof.sh
set -uo pipefail
WT=${WT:-/home/pat/code/minisgl-rdna4-dgprof}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:post-tile-prof}
DG_MODEL=${DG_MODEL:-cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4}
SCRATCH=${SCRATCH:-$HOME/.cache/minisgl-perf}
SWA_RADIX=${SWA_RADIX:-0}
CONC=${CONC:-1}
# A canvas BLOCK is 256 tokens wide and costs k~14-21 denoising steps, so ONE 256-token request is
# only ~17 canvas steps -- and `_rtx_step` counts canvas steps, not loop iterations (idle iterations
# never reach `_canvas_step`). A window placed at step 60 off a single request would therefore never
# open, and the trace would be empty for the SECOND time, from a different cause. The trace leg
# drives REPS sequential requests so the step counter reaches the window.
# STEPS MUST BE REACHABLE, and that is a stricter requirement than it looks. `_bounded_exit_reached`
# counts scheduler-loop ITERATIONS, and the canvas loop BLOCKS in `receive_msg` whenever nothing is
# runnable -- so idle time does not accumulate iterations. The count is therefore ~= (canvas steps) +
# (prefills), and MEASURED here: 5 requests produced 26+12+19+9+19 = 85 canvas steps. A bound of 250
# was never reached, the engine never returned, rocprofv3 never ran its destructor, and the trap tore
# the container down -- which is precisely the abort-without-writing failure propose_rocprof.sh warns
# about. The trace came back EMPTY for the third distinct reason in one session. Bound it BELOW what
# the drive loop actually produces.
STEPS=${STEPS:-75}           # scheduler-loop iterations before the engine returns (trace leg)
RTX_SKIP=${RTX_SKIP:-20}     # past the first block: its encoder pass and cold caches
RTX_STEPS=${RTX_STEPS:-40}
REPS=${REPS:-6}              # sequential requests on the trace leg
MAXTOK=${MAXTOK:-256}
LEGS=${LEGS:-base,trace}   # comma list; a trace re-run should not re-pay for the base boot
NUM_PAGES=${NUM_PAGES:-}     # pinned on BOTH legs once known; empty => engine sizes it, and we read
                             # the size back out of the log so the trace leg can pin the SAME pool.
mkdir -p "$SCRATCH"
RUN_ID=${RUN_ID:-$$$(od -An -N2 -tu2 </dev/urandom | tr -dc "0-9")}
OUT=${OUT:-$SCRATCH/canvas_rocprof-$RUN_ID.txt}
RPDIR=cvout-$RUN_ID
: > "$OUT"
rm -rf "${WT:?}/$RPDIR"; mkdir -p "$WT/$RPDIR"
export COMPOSE_PROJECT_NAME="minisglcv$RUN_ID"

say() { echo "$@" | tee -a "$OUT"; }
say "=== canvas_rocprof run=$RUN_ID image=$IMAGE model=$DG_MODEL"
say "=== engine=$(cd "$WT" && git rev-parse --short HEAD)  swa_radix=$SWA_RADIX conc=$CONC"

# ---------------------------------------------------------------------------------------------
# The per-run compose override. container_name AND project must both be unique: docker-compose.yml
# pins container_name to ${LEASE_NAME}-serve, which is IDENTICAL for every run on the same cards, so
# COMPOSE_PROJECT_NAME alone cannot stop one run's `down` trap killing another run's container.
# ---------------------------------------------------------------------------------------------
YMLF="$WT/docker-compose.canvas-$RUN_ID.yml"
write_yml() {  # $1 = command line, $2... = extra "KEY: val" env lines
  local cmd="$1"; shift
  { echo "services:"
    echo "  serve:"
    echo "    container_name: minisglcv-$RUN_ID"
    echo "    command: [\"$cmd\"]"
    echo "    environment:"
    echo "      MINISGL_SWA_RADIX: \"$SWA_RADIX\""
    for kv in "$@"; do echo "      $kv"; done
  } > "$YMLF"
}
DC() { ( cd "$WT" && env MINISGL_IMAGE="$IMAGE" docker compose -f docker-compose.yml -f "$YMLF" --profile serve "$@" ); }
down() { DC down >/dev/null 2>&1; }
trap 'down; rm -f "$YMLF"' EXIT INT TERM

# The driving client. Reports per-request tok/s and the aggregate; the canvas has no greedy mode so
# a sha is recorded for provenance only, never as an identity gate (uniform-noise canvas + a
# multinomial every step => two boots differ no matter what is pinned).
drive() {  # $1 = n requests, $2 = max_tokens
  python3 - "$1" "$2" <<'PY' 2>&1
import hashlib, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
n, maxtok = int(sys.argv[1]), int(sys.argv[2])
PROMPTS = [
    "Write one paragraph explaining why the sky is blue.",
    "List three differences between a list and a tuple in Python.",
    "What is the capital of Australia, and why was it chosen?",
    "Explain in one paragraph what a hash table is and when it is the wrong choice.",
]
def one(i):
    body = json.dumps({
        "model": "m", "max_tokens": maxtok, "temperature": 0.0, "stream": False,
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
    }).encode()
    rq = urllib.request.Request("http://localhost:1919/v1/chat/completions", body,
                                {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(rq, timeout=900))
    except Exception as e:
        return f"req{i}: FAILED {type(e).__name__}: {e}"
    dt = time.perf_counter() - t0
    txt = r["choices"][0]["message"]["content"]
    ct = r.get("usage", {}).get("completion_tokens", 0)
    return (f"req{i}: {ct} tok in {dt:.2f}s = {ct/dt if dt else 0:.1f} tok/s "
            f"sha={hashlib.sha256(txt.encode()).hexdigest()[:16]}")
t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=n) as ex:
    outs = list(ex.map(one, range(n)))
print("\n".join(outs))
print(f"WALL {time.perf_counter()-t0:.2f}s")
PY
}

wait_ready() {  # "Container Up" is NOT ready: poll the HTTP health AND require non-zero VRAM, and
                # bail the moment the log shows a traceback rather than burning the whole timeout.
  local i c; c="minisglcv-$RUN_ID"
  for i in $(seq 1 400); do
    if [ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" != "true" ]; then
      say "!! container not running after ~$((i*3))s"; docker logs "$c" 2>&1 | tail -40 | tee -a "$OUT"; return 1
    fi
    if docker logs "$c" 2>&1 | grep -qaE "Traceback \(most recent call last\)|RuntimeError|torch.OutOfMemoryError"; then
      say "!! traceback in the log"; docker logs "$c" 2>&1 | grep -aA20 "Traceback" | tail -50 | tee -a "$OUT"; return 1
    fi
    if curl -s --max-time 3 http://localhost:1919/health >/dev/null 2>&1 \
       || curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1; then
      say "== ready after ~$((i*3))s"; return 0
    fi
    sleep 3
  done
  say "!! readiness timeout"; docker logs "$c" 2>&1 | tail -40 | tee -a "$OUT"; return 1
}

EXTRA=""
[ -n "$NUM_PAGES" ] && EXTRA="--num-pages $NUM_PAGES"

# =============================================================================================
if [[ ",$LEGS," == *",base,"* ]]; then
say ""; say "############ LEG base — served baseline + canvas timing + the hip-engage ledger"
# =============================================================================================
write_yml "exec /engine/tools/serve.sh" "MINISGL_CANVAS_TIMING: \"1\""
down
export MODEL="$DG_MODEL" SPEC=none TP=2 CONC="$CONC" GRAPH_BS="$CONC" EXTRA_ARGS="$EXTRA"
unset MINISGL_ROCTX MINISGL_EXIT_AFTER_STEPS
DC up -d >/dev/null 2>&1
if wait_ready; then
  # [canvas-timing] reports CUMULATIVE averages every 10 steps, so the marginal step is the
  # difference of two consecutive report points. One 256-token request is only ~17 canvas steps =~2
  # report points, which is not enough to difference a steady state out of a boot transient. Four
  # measured requests give ~7-10 points.
  say "--- warmup"; drive 1 64 >>"$OUT" 2>&1
  for _r in 1 2 3 4; do
    say "--- measured $_r"; drive 1 "$MAXTOK" | tee -a "$OUT"
  done
  say ""; say "--- [canvas-timing] (cumulative; difference consecutive points for the marginal step)"
  docker logs "minisglcv-$RUN_ID" 2>&1 | grep -a "\[canvas-timing\]" | tail -12 | tee -a "$OUT"
  say ""; say "--- [hip-engage] ledger: WHICH kernels this checkpoint actually dispatches"
  docker logs "minisglcv-$RUN_ID" 2>&1 | grep -oaE "\[hip-engage\] .*" | sort -u | tee -a "$OUT"
  say ""; say "--- KV pool / cache plan / canvas config"
  docker logs "minisglcv-$RUN_ID" 2>&1 | grep -aiE "num_pages|num-pages|kv pool|KV cache|prefix cache|canvas|page_size" \
    | grep -av "canvas-timing" | head -25 | tee -a "$OUT"
  say ""; say "--- [canvas] per-block k (denoising steps actually realised)"
  docker logs "minisglcv-$RUN_ID" 2>&1 | grep -oaE "\[canvas\] uid.*" | tail -12 | tee -a "$OUT"
fi
down
fi

# =============================================================================================
if [[ ",$LEGS," == *",trace,"* ]]; then
say ""; say "############ LEG trace — rocprofv3 kernel+marker trace, marker-gated to the canvas step"
# =============================================================================================
write_yml "exec rocprofv3 --kernel-trace --marker-trace --selected-regions --stats --output-format csv -d /engine/$RPDIR -o canvas -- /engine/tools/serve.sh" \
  "MINISGL_ROCTX: \"1\"" "MINISGL_ROCTX_SKIP: \"$RTX_SKIP\"" "MINISGL_ROCTX_STEPS: \"$RTX_STEPS\"" \
  "MINISGL_EXIT_AFTER_STEPS: \"$STEPS\""
down
DC up -d >/dev/null 2>&1
if wait_ready; then
  say "--- driving $REPS sequential requests (the engine exits MID-request at its step bound, by"
  say "    design: we want the trace, not the answer). Sequential, not concurrent, so the canvas"
  say "    step measured is the bs=1 step the baseline leg timed."
  for _r in $(seq 1 "$REPS"); do
    drive 1 "$MAXTOK" >>"$OUT" 2>&1
    docker inspect -f '{{.State.Running}}' "minisglcv-$RUN_ID" 2>/dev/null | grep -q true || break
  done
  say "--- waiting for the engine to hit its bound and exit on its own (no signals)"
  for _ in $(seq 1 200); do
    [ "$(docker inspect -f '{{.State.Running}}' "minisglcv-$RUN_ID" 2>/dev/null)" = "true" ] || break
    sleep 3
  done
  docker logs "minisglcv-$RUN_ID" 2>&1 | grep -aiE "EXIT_AFTER_STEPS|Opened result|output generation" | tail -6 | tee -a "$OUT"
fi
say "--- trace files"
find "$WT/$RPDIR" -type f -size +0 2>/dev/null | tee -a "$OUT"
cp -r "$WT/$RPDIR" "$SCRATCH/" 2>/dev/null
fi
say ""; say "=== done. out=$OUT  trace=$SCRATCH/$RPDIR"
