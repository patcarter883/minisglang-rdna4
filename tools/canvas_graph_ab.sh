#!/usr/bin/env bash
# canvas_graph_ab.sh — the cudagraph-captured canvas step against the eager one, and the canvas
# path's concurrency curve. One boot per leg, same worktree, same image, same prompts.
#
# WHY A DEDICATED HARNESS RATHER THAN diffusiongemma_generate.sh. That script answers "does block
# diffusion generate, and how does it compare to the autoregressive sibling". This one answers three
# different questions, each of which needs the serve booted twice with one knob moved:
#
#   1. DID CAPTURE ENGAGE, AND IS IT THE SAME COMPUTATION? "Faster" is not the claim — "faster and
#      identical" is; see the byte-identity note below for where that second half comes from.
#   2. WHERE DOES A CANVAS STEP'S TIME GO? MINISGL_CANVAS_TIMING=1 splits each step into
#      fwd_issue / fwd_tail / sampler / soft_embed. A step whose issue time dominates is host-bound
#      and is exactly what capture removes; a step dominated by fwd_tail is compute-bound and capture
#      cannot help it. tok/s alone cannot tell those apart, and the difference decides what to fix.
#   3. WHERE DOES CONCURRENCY SATURATE? A canvas step already batches every in-flight block into ONE
#      forward, so bs>1 is not new machinery — but nobody had measured it.
#
# ON BYTE-IDENTITY, and why it is NOT scraped from the completions. This path has no greedy mode:
# the canvas starts as uniform noise over the whole 262144-token vocabulary and every denoising step
# draws a multinomial, so `temperature 0 / top_p 1 / top_k 1` pins nothing and two boots return
# different text no matter what. Worse, capture itself perturbs the process RNG (CUDAGraph capture
# reserves a generator offset), so even one shared seed would not make the two legs comparable.
# The identity claim therefore rests on the IN-PROCESS check the engine performs: the first two
# canvas replays of every serve also run the eager forward on the same inputs and log
# `max|delta|` over the whole backbone hidden state (GraphRunner.replay_canvas). That is a stronger
# statement than matching text — it compares the computation, not a sample from it — and it can
# fail. Set IDENT_SEED=<n> on a tree that carries `SamplingParams.seed` to also pin the completions.
#
# Runs INSIDE the container under ONE foreground `gpu-lease -n 2`. The image must be one whose
# /opt/kernels has head_dim 512 (minisgl-rdna4:gemma4).
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1

DG_MODEL="${DG_MODEL:-cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4}"
PORT=1919
RES=/engine/_canvas_graph_ab.txt
: > "$RES"
SERVER_PID=0
LOG=""

start_server() {  # $1 = tag, $2 = CONC, $3 = GRAPH_BS
  local tag="$1" conc="$2" gbs="$3"
  LOG="/engine/_cga_$tag.log"
  # setsid: the server + its TP workers form their own process group so stop_server can kill the
  # WHOLE group. A plain kill leaks the workers, which keep the torch.distributed port bound and make
  # the NEXT leg fail with an unrelated-looking bind error.
  #
  # MINISGL_SWA_RADIX=0 on BOTH legs, so the only difference between them is graph capture. It is on
  # by default and it changes two things this measurement would otherwise confound: the prefix cache
  # becomes snapshot-capable, which page-splits EVERY prefill (PrefillAdder._add_one_req,
  # `is_recurrent_radix`) into a body + sub-page tail, and a warm prompt then skips the encoder pass
  # entirely. Both move wall-clock per request without moving the per-denoising-step cost this A/B is
  # about, and it is also the configuration the 72.3/81.1 baseline was measured under.
  env MINISGL_CANVAS_TIMING=1 MINISGL_SWA_RADIX=0 MODEL="$DG_MODEL" TP=2 SPEC=none PORT=$PORT \
    CONC="$conc" GRAPH_BS="$gbs" \
    setsid bash /engine/tools/serve.sh >"$LOG" 2>&1 &
  SERVER_PID=$!
  echo "== leg $tag: conc=$conc graph_bs=$gbs (pid=$SERVER_PID log=$LOG)"
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

# ---- the client: N concurrent requests, aggregate tok/s, per-request sha256 of the completion ----
drive() {  # $1 = concurrency, $2 = max_tokens, $3 = endpoint (chat|completions), $4 = seed|-
  python - "$1" "$2" "$3" "$4" <<'PY'
import hashlib, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

conc, maxtok, endpoint, seed = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
PROMPTS = [
    "Write one paragraph explaining why the sky is blue.",
    "List three differences between a list and a tuple in Python.",
    "What is the capital of Australia, and why was it chosen?",
    "Explain in one paragraph what a hash table is and when it is the wrong choice.",
]

def one(i):
    p = PROMPTS[i % len(PROMPTS)]
    if endpoint == "chat":
        url, body = "/v1/chat/completions", {
            "model": "x", "max_tokens": maxtok, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
            "messages": [{"role": "user", "content": p}],
        }
    else:
        url, body = "/v1/completions", {
            "model": "x", "max_tokens": maxtok, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
            "prompt": p,
        }
        if seed != "-":
            body["seed"] = int(seed) + i
    req = urllib.request.Request(
        f"http://localhost:1919{url}", json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    dt = time.perf_counter() - t0
    ch = d["choices"][0]
    txt = ch["message"]["content"] if endpoint == "chat" else ch["text"]
    return dt, d.get("usage", {}).get("completion_tokens", 0), txt

t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=conc) as ex:
    res = list(ex.map(one, range(conc)))
wall = time.perf_counter() - t0
tot = sum(n for _, n, _ in res)
for i, (dt, n, txt) in enumerate(res):
    h = hashlib.sha256(txt.encode()).hexdigest()[:16]
    print(f"  req{i}: {n} tok in {dt:.2f}s = {n/max(dt,1e-9):.1f} tok/s  sha={h}")
print(f"AGGREGATE conc={conc}: {tot} tok in {wall:.2f}s = {tot/max(wall,1e-9):.1f} tok/s")
PY
}

leg() {  # $1 = tag, $2 = CONC, $3 = GRAPH_BS
  local tag="$1"
  start_server "$@"
  wait_ready || { stop_server; return 1; }
  {
    echo
    echo "=================== leg: $tag (conc=$2 graph_bs=$3) ==================="
    echo "--- completion shape (/v1/completions, 128 tok, seed=${IDENT_SEED:--}) ---"
  } >> "$RES"
  drive 1 128 completions "${IDENT_SEED:--}" >> "$RES" 2>&1
  { echo "--- throughput (chat, 256 tok) ---"; } >> "$RES"
  for _ in 1 2; do drive 1 256 chat - >> "$RES" 2>&1; done
  if [ "$2" -gt 1 ]; then
    { echo "--- concurrency (chat, 256 tok) ---"; } >> "$RES"
    for c in 1 2 4; do
      [ "$c" -le "$2" ] || continue
      drive "$c" 256 chat - >> "$RES" 2>&1
    done
  fi
  {
    echo "--- capture engagement + per-step split (from $LOG) ---"
    grep -o 'canvas graphs captured.*\|Capturing CANVAS.*\|canvas CUDA graph:.*' "$LOG" | head -5
    grep -o '\[canvas-graph\].*' "$LOG" | head -4
    grep -o '\[canvas-timing\].*' "$LOG" | tail -4
    grep -o '\[canvas\] uid.*' "$LOG" | head -20
  } >> "$RES"
  stop_server
}

leg eager  "${CONC_EAGER:-1}" 0
leg graph  "${CONC_GRAPH:-1}" 1
[ "${CONC_SWEEP:-4}" -gt 1 ] && leg graph_conc "${CONC_SWEEP:-4}" "${CONC_SWEEP:-4}"

echo
echo "================ RESULTS ($RES) ================"
cat "$RES"
