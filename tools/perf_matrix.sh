#!/usr/bin/env bash
# MATCHED PERF MATRIX — baseline d276137c vs perf/propose-owindow-capture, and BOTH against PLAIN.
#
# WHAT THIS MEASURES AND WHY IT IS SHAPED THIS WAY
#
# One image (minisgl-rdna4:lean), engine hot-mounted at /engine, so the ONLY difference between a
# base leg and a new leg is which worktree is mounted. Provenance is therefore asserted by md5'ing
# the MOUNTED source from inside the container (spec/width.py and spec/capture.py are ABSENT at
# d276137c — that absence is the strongest provenance signal available), never by an env var.
#
# TWO PASSES, because the instrument perturbs the thing it measures:
#   PASS A (clean)  MINISGL_SPEC_DEBUG=1 only. SPEC_DEBUG is pure host-side counters. This pass is
#                   where TRUE tok/s comes from. ms/step is derived as wall/steps using the step
#                   counter out of the [spec] line — a real per-step time with no sync overhead.
#   PASS B (attrib) + MINISGL_SPEC_TIMING=1. That costs FOUR cuda syncs per step, so its tok/s is
#                   NOT production tok/s and is not reported as such. It exists only to split the
#                   step into propose / stage / verify-forward / accept.
# Reporting tok/s off a timing-instrumented run would understate every leg; reporting propose ms
# without the syncs is impossible. Hence two passes.
#
# TRUE tok/s = sum(usage.completion_tokens) over NON-streaming requests / wall. Never SSE chunks:
# under spec ONE chunk carries a whole accepted block (that bug once read 39.7 tok/s as 14.9).
#
# NREQ=1 AND NREQ=8 on every leg: a CONC=1-only A/B on this box has already misread a real +20%
# lever as flat 0.0%, and the M<=16 verify cliff is by construction a bs=1 effect.
#
# TWO PROMPT CLASSES: a ~95-token instruction and a >=3k-token real code prompt. An accept-len is
# meaningless without its max_tokens and prompt class, so both are recorded with every row.
#
# Usage:  gpu-lease -n 2 -- bash tools/perf_matrix.sh <passA|passB> <tag> <worktree> [more legs...]
set -uo pipefail

# NOT a per-session /tmp scratchpad. That default cost this investigation both its crash dump and its
# long-prompt fixture to a reboot, and because the fixture's SIZE was never recorded, the workload
# that crashed 3/3 became unreconstructable — a regenerated lighter one then passed 8/8, which is
# indistinguishable from "fixed". Durable path, fixture DERIVED from a repo file, size echoed below.
SCRATCH=${SCRATCH:-$HOME/.cache/minisgl-perf}
mkdir -p "$SCRATCH"
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
LONGCODE=${LONGCODE:-$SCRATCH/longcode.txt}
OUT=${OUT:-$SCRATCH/perf_matrix.txt}

NEW_WT=${NEW_WT:-/home/pat/code/minisgl-rdna4-propose}
BASE_WT=${BASE_WT:-/home/pat/code/minisgl-rdna4-perfbase}

# Regenerate the ">=3k-token real code prompt" class from the module the prompt text describes.
# NOTE the size is load-bearing: 12288 B (~3.1k tok) x8 fits, ~45k B (~11k tok) x8 OOM'd the seed
# prefill before the _spec_seed_fits gate. Any run that changes this must say so in its result.
[ -s "$LONGCODE" ] || head -c 12288 "$NEW_WT/python/minisgl/engine/graph.py" > "$LONGCODE"

export MODEL="${MODEL:-laguna}" TP="${TP:-2}" CONC="${CONC:-8}" GRAPH_BS="${GRAPH_BS:-8}"
WORKLOADS=${WORKLOADS:-"short:384:1 short:384:8 long:1600:1 long:1600:8"}

down_all() {
  ( cd "$NEW_WT"  && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 )
  ( cd "$BASE_WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 )
}
trap down_all EXIT INT TERM

wait_ready() { for _ in $(seq 1 400); do
  curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

# ---------------------------------------------------------------------------------------------
# One workload: NREQ concurrent NON-STREAMING greedy requests, report TRUE tok/s from usage.
drive() { PROMPTCLASS="$1" MAXTOK="$2" NREQ="$3" LONGCODE="$LONGCODE" python3 - <<'PY'
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE = "http://localhost:1919"
MODEL = json.loads(urllib.request.urlopen(f"{BASE}/v1/models", timeout=60).read())["data"][0]["id"]
CLS, MT, NREQ = os.environ["PROMPTCLASS"], int(os.environ["MAXTOK"]), int(os.environ["NREQ"])
SHORT = ("Write a Python function that reverses a singly linked list in place, then explain how it "
         "works, why it is O(n) time and O(1) space, and what happens on an empty list and on a "
         "single-node list. Then show how you would test it.")
if CLS == "long":
    code = open(os.environ["LONGCODE"]).read()
    PROMPT = ("Here is a module from a CUDA-graph-capturing LLM inference engine.\n\n```python\n"
              + code + "\n```\n\nReview this code. Explain what the graph capture path does, "
              "identify the invariants a caller must uphold, and point out anything that would "
              "break if the batch size or sequence length changed between capture and replay.")
else:
    PROMPT = SHORT
def run(i):
    # NREQ>1 uses distinct suffixes so the requests are not one shared radix prefix — a single
    # shared prefix would make 8 concurrent requests measure prefix-cache hits, not decode.
    txt = PROMPT if NREQ == 1 else f"{PROMPT}\n\n(Answer variant {i}: focus on point {i + 1}.)"
    b = {"model": MODEL, "messages": [{"role": "user", "content": txt}], "max_tokens": MT,
         "temperature": 0.0, "top_p": 1.0, "seed": 1234, "stream": False}
    r = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=3600).read())
    u = d["usage"]
    return u["completion_tokens"], u["prompt_tokens"]
if os.environ.get("WARMUP"):
    # Discarded. One request so the first TIMED workload does not pay lazy allocations, the first
    # graph replay, or a cold radix root. Deliberately a DIFFERENT string from every timed prompt so
    # it cannot seed a prefix-cache hit for them.
    b = {"model": MODEL, "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
         "max_tokens": 16, "temperature": 0.0, "seed": 7, "stream": False}
    r = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    urllib.request.urlopen(r, timeout=600).read()
    raise SystemExit
t = time.perf_counter()
with ThreadPoolExecutor(NREQ) as ex:
    res = list(ex.map(run, range(NREQ)))
w = time.perf_counter() - t
n = sum(c for c, _ in res)
print(f"    prompt_tokens={res[0][1]} completion_tokens={n} wall={w:.3f}s TPS={n / w:.2f}")
PY
}

# ---------------------------------------------------------------------------------------------
# Cumulative [spec]/[spec-timing] counters at this instant. Both lines print RUNNING MEANS over a
# monotonically growing step count, so a WINDOW is recovered by differencing two snapshots — that
# is the only way to attribute steps to one workload rather than to the whole boot.
snap() { ( cd "$1" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
  | grep -aE "\[spec\] step=|\[spec-timing\] step=" | tail -40 | python3 -c '
import re, sys
spec = tim = ""
for ln in sys.stdin:
    if "[spec] step=" in ln: spec = ln
    if "[spec-timing] step=" in ln: tim = ln
def g(s, k, cast=float):
    m = re.search(k, s)
    return cast(m.group(1)) if m else None
out = {}
if spec:
    out["n"] = g(spec, r"\[spec\] step=(\d+)", int)
    out["A"] = g(spec, r"draft_accepted=(\d+)/", int)
    out["P"] = g(spec, r"draft_accepted=\d+/(\d+)", int)
    out["e"] = g(spec, r"emitted/step=([0-9.]+)")
    out["r"] = g(spec, r"reqs/step=([0-9.]+)")
if tim:
    out["tn"] = g(tim, r"\[spec-timing\] step=(\d+)", int)
    for k in ("propose", "stage", "forward", "accept", "total"):
        out[k] = g(tim, k + r"=([0-9.]+)ms")
print(" ".join(f"{k}={v}" for k, v in out.items() if v is not None))
' ; }

# Difference two snapshots into per-window means. Emitted/reqs are RUNNING MEANS over a growing n,
# so the cumulative totals are mean*n and the window total is the difference of those. An ABSENT
# snapshot means the counters were still at zero, which is a real value (they start at 0 at boot),
# not a missing one.
#
# ms/step is NOT wall/(logged steps): [spec] only prints every 50 steps, so the last partial block
# of a workload is invisible and wall/logged_steps would over-report the step time by up to 50 steps'
# worth. Both RATIOS in that line (accept-len, accept-rate) are quantization-robust, so ms/step is
# reconstructed from them and the exact token count instead:
#     steps = completion_tokens / (accept-len * NREQ)   ->   ms/step = wall*1000 / steps
# For a PLAIN leg there is no [spec] line at all and accept-len is exactly 1.0 by definition.
delta() { S1="$1" S2="$2" WALL="$3" TOKENS="$4" NREQ="$5" python3 - <<'PY'
import os
def parse(s):
    d = {}
    for tok in s.split():
        k, _, v = tok.partition("=")
        d[k] = float(v)
    return d
a, b = parse(os.environ["S1"]), parse(os.environ["S2"])
wall, tokens, nreq = float(os.environ["WALL"]), float(os.environ["TOKENS"]), float(os.environ["NREQ"])
out, acc = [], None
if "n" in b and b["n"] > a.get("n", 0):
    an = a.get("n", 0.0)
    dn = b["n"] - an
    dE = b["e"] * b["n"] - a.get("e", 0.0) * an
    dR = b["r"] * b["n"] - a.get("r", 0.0) * an
    dA, dP = b["A"] - a.get("A", 0.0), b["P"] - a.get("P", 0.0)
    out.append(f"logged-steps={int(dn)}")
    if dR > 0:
        acc = dE / dR                      # emitted tokens per REQUEST per step == 1 + accepted
        out.append(f"accept-len={acc:.3f}")
    if dP > 0:
        out.append(f"accept-rate={dA / dP:.3f}")
elif "n" not in b:
    acc = 1.0                              # PLAIN decode: one token per request per step
    out.append("accept-len=1.000 (plain)")
if acc and acc > 0 and tokens > 0:
    steps = tokens / (acc * nreq)
    out.append(f"steps~={steps:.0f}")
    out.append(f"ms/step={wall * 1000 / steps:.2f}")
if "tn" in b and b["tn"] > a.get("tn", 0):
    at = a.get("tn", 0.0)
    dt = b["tn"] - at
    for k in ("propose", "stage", "forward", "accept", "total"):
        if k in b:
            out.append(f"{k}={(b[k] * b['tn'] - a.get(k, 0.0) * at) / dt:.2f}ms")
print("    " + "  ".join(out))
PY
}

leg() {  # leg <tag> <worktree>
  local tag=$1 wt=$2
  echo "=== LEG $tag  pass=$PASS  MODEL=$MODEL SPEC=${SPEC:-default} K=${SPEC_K:-default} TP=$TP CONC=$CONC ===" | tee -a "$OUT"
  down_all
  ( cd "$wt" && env MINISGL_IMAGE="$IMAGE" docker compose --profile serve up -d >/dev/null 2>&1 )
  if ! wait_ready; then
    echo "  NEVER BECAME READY" | tee -a "$OUT"
    ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs --tail 40 2>&1 ) \
      | grep -aiE "error|assert|traceback|oom|memory" | tail -12 | tee -a "$OUT"
    down_all; return 1
  fi
  local c; c=$( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -q serve )
  # PROVENANCE — which BUILD is mounted? width.py/capture.py are ABSENT at d276137c.
  echo -n "  PROV sha=$( cd "$wt" && git rev-parse --short HEAD ) md5(dflash/capture/width/sched)= " | tee -a "$OUT"
  docker exec "$c" sh -c 'for f in spec/dflash.py spec/capture.py spec/width.py scheduler/scheduler.py; do
      if [ -f "/engine/python/minisgl/$f" ]; then md5sum "/engine/python/minisgl/$f" | cut -c1-8 | tr "\n" " "; else printf "ABSENT "; fi; done' \
    2>/dev/null | tee -a "$OUT"; echo "" | tee -a "$OUT"
  echo -n "  ENV witness: " | tee -a "$OUT"
  docker exec "$c" sh -c 'tr "\0" "\n" < /proc/1/environ | grep -E "^MINISGL_SPEC_(TIMING|DEBUG|SAMPLED)=|^MINISGL_KV_FP8=|^MINISGL_DFLASH_DDTREE=" | tr "\n" " "' \
    2>/dev/null | tee -a "$OUT"; echo "" | tee -a "$OUT"
  ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
    | grep -aE "ADAPTIVE verify width|PROPOSE graphs|DFlash Laguna drafter|drafter:|\[serve\] spec" \
    | sed 's/^[^|]*| *//' | tail -6 | sed 's/^/  BOOT /' | tee -a "$OUT"

  WARMUP=1 drive short 16 1 >/dev/null 2>&1   # discarded; see the WARMUP note in drive()
  for w in $WORKLOADS; do
    local cls mt nq; IFS=: read -r cls mt nq <<< "$w"
    echo "  -- workload=$cls max_tokens=$mt NREQ=$nq" | tee -a "$OUT"
    local s1 s2 t0 t1 line tok
    s1=$(snap "$wt")
    t0=$(date +%s.%N)
    line=$(drive "$cls" "$mt" "$nq" 2>&1)
    t1=$(date +%s.%N)
    echo "$line" | tee -a "$OUT"
    s2=$(snap "$wt")
    tok=$(sed -n 's/.*completion_tokens=\([0-9]*\).*/\1/p' <<< "$line"); tok=${tok:-0}
    delta "$s1" "$s2" "$(echo "$t1 - $t0" | bc)" "$tok" "$nq" 2>&1 | tee -a "$OUT"
    # A request that fails MID-RUN means the scheduler died. Without the server-side traceback that
    # is just a client RemoteDisconnected, which says nothing — dump the engine's own error lines.
    if [[ "$tok" == "0" ]]; then
      echo "  *** WORKLOAD FAILED — engine log tail:" | tee -a "$OUT"
      ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs --tail 400 2>&1 ) \
        | grep -aiE "error|assert|traceback|out of memory|oom|died unexpectedly|File \"|Runtime" \
        | tail -30 | sed 's/^/  ERR /' | tee -a "$OUT"
    fi
  done
  ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
    | grep -aE "\[spec-timing\] step=|\[spec\] step=" | tail -3 | sed 's/^[^|]*| *//' | sed 's/^/  TAIL /' | tee -a "$OUT"
  down_all
  echo "" | tee -a "$OUT"
}

PASS=$1; shift
case "$PASS" in
  passA) export MINISGL_SPEC_DEBUG=1; unset MINISGL_SPEC_TIMING ;;
  passB) export MINISGL_SPEC_DEBUG=1 MINISGL_SPEC_TIMING=1 ;;
  *) echo "usage: perf_matrix.sh <passA|passB> <tag> <worktree> ..." >&2; exit 2 ;;
esac

while [[ $# -gt 0 ]]; do
  t=$1; w=$2; shift 2
  leg "$t" "$w"
done
echo "== MATRIX SEGMENT COMPLETE ($PASS) ==" | tee -a "$OUT"
