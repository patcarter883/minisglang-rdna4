#!/usr/bin/env bash
# diffusiongemma_generate.sh — boot the block-diffusion serve and get a real generation out.
#
# This is the milestone gate for the canvas execution mode: not a kernel probe, not a parity
# fixture, but the whole ENCODE -> DENOISE xk -> COMMIT -> re-ENCODE cycle running in the scheduler
# and producing text a human can read. Three things are measured, in this order of importance:
#
#   1. COHERENCE. Does it produce language? A canvas that attends the wrong keys still emits fluent
#      tokens, so the read is the gate, and the per-block log lines below are the evidence for how
#      it got there.
#   2. k — the realised denoising steps per committed block. The entire cost case for block
#      diffusion turns on this and NOTHING else can produce it: an isolated kernel bench gives the
#      per-step cost, tok/s hides it (a block emits all 256 tokens at once), and the reference's own
#      `tokens_per_forward` is measured on a different stack. Scraped from the [canvas] log lines
#      the scheduler emits at every commit.
#   3. THE AUTOREGRESSIVE SIBLING IS UNAFFECTED. Same image, same worktree, gemma-4 at TP=2 —
#      because "the canvas path did not disturb the AR path" is a claim about the SERVE, not about
#      a test file. Set AR=0 to skip.
#
# Runs INSIDE the container, under ONE foreground `gpu-lease -n 2` (17 GB int4 needs both cards).
# The image must be one whose /opt/kernels has head_dim 512 (minisgl-rdna4:gemma4) — the 5
# full-attention layers have no kernel otherwise and every canvas batch dies in the dispatch.
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1

DG_MODEL="${DG_MODEL:-cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4}"
AR_MODEL="${AR_MODEL:-cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4}"
RUN_AR="${AR:-1}"
PORT=1919
RES=/engine/_diffusiongemma_results.txt
: > "$RES"
SERVER_PID=0
LOG=""

start_server() {  # $1 = model, $2 = tag, $3.. = extra env assignments
  local model="$1" tag="$2"; shift 2
  LOG="/engine/_dg_$tag.log"
  # setsid: the server + its TP workers form their own process group so stop_server can kill the
  # WHOLE group. A plain kill leaks the workers, which keep the torch.distributed port bound and
  # make the NEXT phase fail with an unrelated-looking bind error.
  #
  # SWA-radix OFF for the canvas phase: its window snapshot/restore is taken at autoregressive
  # commit points and addresses the ring at the pre-canvas stride. Re-validating it against a
  # canvas is its own piece of work; leaving it on would silently seed a stale window.
  env "$@" MODEL="$model" TP=2 SPEC=none PORT=$PORT CONC=1 \
    setsid bash /engine/tools/serve.sh >"$LOG" 2>&1 &
  SERVER_PID=$!
  echo "== phase $tag: $model (pid=$SERVER_PID log=$LOG)"
}

wait_ready() {
  local i
  for i in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "!! server exited early"; tail -60 "$LOG"; return 1
    fi
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "== ready after ~$((i*3))s"; return 0
    fi
    sleep 3
  done
  echo "!! readiness timeout"; tail -60 "$LOG"; return 1
}

stop_server() {
  [ "$SERVER_PID" -gt 0 ] || return 0
  kill -TERM -"$SERVER_PID" 2>/dev/null
  local i; for i in $(seq 1 40); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -"$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=0
}
trap stop_server EXIT

ask() {  # $1 = prompt, $2 = max_tokens -> prints the completion, then a TIMING: line
  python - "$1" "$2" <<'PY'
import json, sys, time, urllib.request
prompt, maxtok = sys.argv[1], int(sys.argv[2])
body = json.dumps({
    "model": "x", "max_tokens": maxtok, "temperature": 0.0,
    "messages": [{"role": "user", "content": prompt}],
}).encode()
req = urllib.request.Request("http://localhost:1919/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
t0 = time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    dt = time.perf_counter() - t0
    print(d["choices"][0]["message"]["content"])
    # Non-streaming on purpose: a block-diffusion request is OPAQUE until its block commits, so
    # inter-token latency is not a thing that exists here. End-to-end wall time per emitted token is
    # the only number comparable against the autoregressive sibling.
    n = d.get("usage", {}).get("completion_tokens", 0)
    print(f"TIMING: {n} tokens in {dt:.2f}s = {n/max(dt,1e-9):.1f} tok/s = "
          f"{dt/max(n,1)*1000:.1f} ms/token")
except Exception as e:  # a failure here is a result, not a crash — record it and keep going
    print(f"<<REQUEST FAILED: {type(e).__name__}: {e}>>")
PY
}

PROMPTS=(
  "Write one paragraph explaining why the sky is blue."
  "List three differences between a list and a tuple in Python."
  "What is the capital of Australia, and why was it chosen?"
)

# ============================ phase 1: block diffusion ============================
start_server "$DG_MODEL" canvas MINISGL_SWA_RADIX=0
if wait_ready; then
  {
    echo "=========================== BLOCK DIFFUSION ==========================="
    echo "model: $DG_MODEL"
  } >> "$RES"
  for i in "${!PROMPTS[@]}"; do
    echo "-- prompt $i: ${PROMPTS[$i]}"
    out=$(ask "${PROMPTS[$i]}" 256)
    {
      echo
      echo "--- prompt $i: ${PROMPTS[$i]}"
      echo "$out"
    } >> "$RES"
  done
  # k, straight off the scheduler's own per-commit log lines.
  {
    echo
    echo "--- realised denoising steps per block (k) ---"
    grep -o '\[canvas\].*' "$LOG" || echo "(no [canvas] lines — no block committed)"
    echo
    python - "$LOG" <<'PY'
import re, sys, statistics
ks, emitted = [], []
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"\[canvas\].*steps=(\d+)/(\d+) emitted=(\d+)", line)
    if m:
        ks.append(int(m.group(1))); emitted.append(int(m.group(3)))
if ks:
    print(f"blocks={len(ks)}  k: min={min(ks)} median={statistics.median(ks)} max={max(ks)} "
          f"mean={statistics.mean(ks):.1f} of {m.group(2)}")
    print(f"tokens emitted per block: {emitted}")
    print(f"forwards per emitted token = {sum(ks)/max(sum(emitted),1):.3f}  "
          f"(the autoregressive sibling is exactly 1.000)")
else:
    print("no committed blocks parsed")
PY
  } >> "$RES"
fi
stop_server

# ============================ phase 2: the AR guard ============================
if [ "$RUN_AR" = "1" ]; then
  start_server "$AR_MODEL" ar
  if wait_ready; then
    {
      echo
      echo "=========================== AUTOREGRESSIVE GUARD ==========================="
      echo "model: $AR_MODEL"
    } >> "$RES"
    # The SAME prompts at the SAME 256-token budget as phase 1. Anything else is not a comparison:
    # both numbers include TTFT, so they are only commensurate at equal output length.
    for i in "${!PROMPTS[@]}"; do
      echo "-- AR prompt $i"
      out=$(ask "${PROMPTS[$i]}" 256)
      { echo; echo "--- prompt $i: ${PROMPTS[$i]}"; echo "$out"; } >> "$RES"
    done
  fi
  stop_server
fi

echo
echo "================ RESULTS ($RES) ================"
cat "$RES"
