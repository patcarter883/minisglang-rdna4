#!/usr/bin/env bash
# Inner half of tools/midband_serve_ab.sh — runs INSIDE the container.
#
# Two SERVED legs of the same model at TP=2, identical but for the mid-band dispatch rule:
#   old  = the whole mid-band (gemv_max < M < 64) on prefill_wmma  (what this change replaces)
#   new  = wmma_tiled_tuned except the measured wide-N corner
# Each leg reports (a) the AR guard — one stream, which is M=1 decode + M>>64 prefill and must
# therefore be NEUTRAL — and (b) a CONCURRENT leg whose decode M lands IN the mid-band, the only
# place the change can move an e2e number at all.
#
# VISIBILITY (this harness already lost one GPU window by not having it). A dead serve and a slow
# serve look identical from outside, so:
#   * the serve's stdout is TEE'd — it reaches `docker logs` as well as the log file. The previous
#     version redirected it to a file only, so a hang showed up as a container with one line of
#     output and nothing to diagnose from.
#   * no `setsid`. It made the recorded $PID a parent that exits immediately, so the liveness check
#     was watching the wrong process and could neither see a crash nor clean one up.
#   * readiness is bounded, prints progress, and on failure dumps the log tail AND rocm-smi — GPU at
#     ~0% with no log progress is the discriminator between hung and slow.
#   * every leg ends by reaping the whole minisgl process tree, pass or fail. A leg that crashes and
#     leaves ranks alive makes the NEXT leg hang on the distributed rendezvous, which is a hang with
#     an entirely misleading cause.
#
# Provenance is asserted, not assumed: each leg greps its own log for the [hip-engage] lines naming
# the dense arms that actually fired. Same set in both legs => the A/B compared the rule to itself.
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

MODEL="${MB_MODEL:?}"
TP="${MB_TP:-2}"
CONC="${MB_CONC:-20}"
MAXTOK="${MB_MAXTOK:-256}"
MEM="${MB_MEM:-0.86}"
READY_S="${MB_READY_S:-240}"

# PORT, and why it is not hardcoded. minisgl's TCPStore rendezvous binds PORT+1, and a leg that
# leaves a rank behind (or another agent cycling a serve on this box) keeps it bound. The failure is
# a TP=2 HALF-DEATH: rank 1 dies on EADDRINUSE, rank 0 loads its weights anyway and then blocks
# forever on the first collective at ~7% GPU -- a serve that is indistinguishable from a slow boot
# from the outside and can NEVER become ready. So: pick a pair nothing holds, and re-check per leg.
pick_port() {
  local p start="${1:-${MB_PORT_BASE:-11900}}"
  for p in $(seq "$start" 2 $((start + 200))); do
    if ! ss -tan 2>/dev/null | grep -qE ":($p|$((p+1)))[[:space:]]"; then echo "$p"; return 0; fi
  done
  echo "!! no free port pair from $start" >&2; return 1
}

reap() {
  pkill -f 'python -m minisgl' 2>/dev/null
  pkill -f 'minisgl' 2>/dev/null
  sleep 8
  pkill -9 -f 'python -m minisgl' 2>/dev/null
  pkill -9 -f 'minisgl' 2>/dev/null
  # WAIT for the processes to be gone, rather than sleeping a guess. The previous fixed 12s was
  # shorter than it took the rendezvous socket to release, which is what put leg 2 into EADDRINUSE.
  local i
  for i in $(seq 1 30); do
    pgrep -f 'python -m minisgl' >/dev/null 2>&1 || break
    sleep 2
  done
  sleep 5
}

set_rule() {  # $1 = old|new
  # The A/B is ONE unambiguous, reversible edit at the head of the mid-band block: either the shipped
  # three-way rule runs, or every mid-band shape short-circuits to prefill_wmma (the old rule).
  # Matching the whole body by text broke the moment the body gained a third arm — and a patcher that
  # silently no-ops turns the A/B into new-vs-itself — so this asserts the anchor is present exactly
  # once and refuses to run otherwise.
  python - "$1" <<'PYEOF'
import pathlib, sys

mode = sys.argv[1]
p = pathlib.Path("/engine/python/minisgl/quant/kernels.py")
s = p.read_text()

NEW = '    if n is None:\n        return "wmma_tiled_tuned"\n'
OLD = '    if True:  # A/B LEG: the OLD rule -- the WHOLE mid-band on prefill_wmma\n        return "prefill_wmma"\n'

cn, co = s.count(NEW), s.count(OLD)
assert cn + co == 1, f"A/B anchor is not unique (new={cn} old={co}) -- refusing to patch"
s = s.replace(NEW if cn else OLD, NEW if mode == "new" else OLD)
p.write_text(s)
print(f"[prov] mid-band rule now: {mode}", flush=True)
PYEOF
}

bench() {  # $1 = tag
  MB_TAG="$1" python - <<'PY'
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

tag = os.environ["MB_TAG"]
BASE = f"http://localhost:{os.environ['MB_PORT']}/v1/chat/completions"
PROMPT = ("Write a detailed technical explanation of how a tensor-parallel inference engine "
          "splits attention and MLP weights across two GPUs, and what has to be all-reduced.")
MAXTOK = int(os.environ.get("MB_MAXTOK", "256"))
CONC = int(os.environ.get("MB_CONC", "20"))


def one(i, maxtok):
    body = {"model": "m",
            "messages": [{"role": "user",
                          "content": f"{PROMPT} (variant {i}: focus on point {i % 7})"}],
            "max_tokens": maxtok, "temperature": 0.0, "stream": False}
    req = urllib.request.Request(BASE, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["usage"]["completion_tokens"]


one(99, 32)  # warmup: graph replay + lazy alloc; never counted

# AR guard: ONE stream. Decode is M=1 (decode_gemv) and prefill is M>>64 (wmma_tiled_tuned), so the
# mid-band rule cannot touch it -- which is exactly why it is the guard.
best = None
for r in range(3):  # repeat: a single serve tok/s reading is not stable to better than a few %
    t0 = time.time(); n = one(r, MAXTOK); w = time.time() - t0
    print(f"RESULT {tag} ar_guard rep{r} tokens={n} wall={w:.2f}s tok_s={n/w:.2f}", flush=True)
    best = max(best or 0, n / w)
print(f"RESULT {tag} ar_guard BEST tok_s={best:.2f}", flush=True)

# The band the change actually touches: CONC concurrent streams -> decode M = running batch.
t0 = time.time()
with ThreadPoolExecutor(CONC) as ex:
    ns = list(ex.map(lambda i: one(i, MAXTOK), range(CONC)))
w = time.time() - t0
print(f"RESULT {tag} conc{CONC} tokens={sum(ns)} wall={w:.2f}s tok_s={sum(ns)/w:.2f}", flush=True)
PY
}

run_leg() {  # $1 = old|new  $2 = leg index (gives each leg a DISJOINT port window)
  local MODE="$1" IDX="${2:-0}" LOG="/engine/_surface/_midband_serve_$1.log" i
  reap
  local PORT; PORT=$(pick_port $(( ${MB_PORT_BASE:-11900} + IDX * 40 ))) || return 1
  set_rule "$MODE" || return 1
  echo "== leg $MODE: launching serve (TP=$TP conc=$CONC mem=$MEM port=$PORT/$((PORT+1)))"
  MODEL="$MODEL" TP="$TP" SPEC=none PORT="$PORT" CONC="$CONC" MEM_RATIO="$MEM" \
    bash /engine/tools/serve.sh 2>&1 | tee "$LOG" &
  local t0=$SECONDS
  while true; do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
    if (( SECONDS - t0 > READY_S )); then
      echo "!! leg $MODE NOT READY after ${READY_S}s -- dumping state"
      echo "---- log tail ----"; tail -30 "$LOG"
      echo "---- rocm-smi ----"; rocm-smi --showuse 2>&1 | head -20
      reap; return 1
    fi
    if ! pgrep -f 'python -m minisgl' >/dev/null 2>&1 && (( SECONDS - t0 > 25 )); then
      echo "!! leg $MODE: no minisgl process alive -- it CRASHED, not slow"
      tail -30 "$LOG"; reap; return 1
    fi
    # A rank that lost the rendezvous can never be waited out; the survivor deadlocks at the first
    # collective with its weights already resident. Abort on the traceback, do not count to 240.
    if grep -qE "EADDRINUSE|DistNetworkError|AssertionError|CUDA out of memory|HIP out of memory" \
         "$LOG" 2>/dev/null; then
      echo "!! leg $MODE: fatal line in the serve log -- aborting instead of waiting out the timeout"
      grep -nE "EADDRINUSE|DistNetworkError|AssertionError|out of memory" "$LOG" | tail -5
      tail -20 "$LOG"; rocm-smi --showuse 2>&1 | head -12; reap; return 1
    fi
    (( (SECONDS - t0) % 30 == 0 )) && echo "   ... waiting $((SECONDS - t0))s"
    sleep 3
  done
  echo "== leg $MODE READY after $((SECONDS - t0))s"
  MB_MAXTOK="$MAXTOK" MB_CONC="$CONC" MB_PORT="$PORT" bench "$MODE"
  echo "-- PROVENANCE leg $MODE: dense arms that ENGAGED"
  grep -o 'mmq_fp8_gemm([a-z0-9_+]*)' "$LOG" | sort | uniq -c
  reap
}

# ONE LEG PER CONTAINER INVOCATION. Running both legs in one container leaked the first leg's VRAM
# into the second: `reap` returns when the processes are gone from `pgrep`, but the KFD allocations
# are only guaranteed released when the process space itself goes away, and leg 2 hit
# `OutOfMemoryError ... 780 MiB free` on a 16 GiB card that should have been empty. Container exit is
# the only reliable barrier, so the caller runs this script once per leg.
rc=0
run_leg "${MB_LEG:?set MB_LEG=new|old}" "${MB_LEG_IDX:-0}" || rc=1
set_rule new   # leave the worktree on the shipped rule however the leg ended
reap
exit $rc
