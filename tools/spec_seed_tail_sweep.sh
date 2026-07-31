#!/usr/bin/env bash
# DFLASH PROMPT-PREFILL SEED — TAIL SWEEP (extends tools/spec_seed_ab.sh, commit 5bd8d5a7).
#
# The A/B compared two worktrees (seed off vs seed at tail=528). 341c4df0 made the tail a runtime
# knob (MINISGL_DFLASH_SEED_TAIL; 0 = seeding entirely off, i.e. the pre-change baseline ON THE SAME
# BINARY), so the whole sweep runs from ONE tree — no cross-tree provenance problem, and every cell
# differs only in an integer.
#
# One boot per TAIL. Requests, in order (uid 1..5):
#   warm  (16 tok)         — not scored
#   CODE1 (3571-tok real code prompt, MAXTOK)   — LONG regime, cold prefix
#   CODE2 (same prompt again)                   — LONG regime, radix prefix-cache HIT
#   SHORT1 (~95 tok, SHORTTOK)                  — SHORT regime, cold prefix
#   SHORT2 (same again)                         — SHORT regime, prefix-cache HIT
#
# Env: TAIL (int, or "none" for the --spec-algorithm none PLAIN leg), DBG, MAXTOK, SHORTTOK, K, TP.
# Config is PRODUCTION and IDENTICAL to spec_seed_ab.sh so the two banks are comparable:
#   MINISGL_KV_FP8=1, --cache-type radix + MINISGL_SWA_RADIX=1, MINISGL_SPEC_MHA_PAGED=1,
#   --cuda-graph-max-bs 8, --max-running-requests 1, K=15, TP=2, temperature 0, --memory-ratio 0.93.
set -uo pipefail
source /app/.venv/bin/activate 2>/dev/null || source /opt/venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1
export MINISGL_KV_FP8="${MINISGL_KV_FP8:-1}"
export MINISGL_SPEC_MHA_PAGED="${MINISGL_SPEC_MHA_PAGED:-1}"
export MINISGL_SWA_RADIX="${MINISGL_SWA_RADIX:-1}"

MODEL="${MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
DRAFT="${DRAFT:-poolside/Laguna-XS-2.1-DFlash-NVFP4}"
K="${K:-15}"; TP="${TP:-2}"; MAXTOK="${MAXTOK:-1600}"; SHORTTOK="${SHORTTOK:-384}"
DBG="${DBG:-2}"; GRAPHBS="${GRAPHBS:-8}"; TAIL="${TAIL:-64}"
PORT=21958

SPEC=dflash
if [ "$TAIL" = "none" ]; then
  SPEC=none; LEG="plain"
  unset MINISGL_DFLASH_SEED_TAIL
else
  LEG="t$TAIL"
  export MINISGL_DFLASH_SEED_TAIL="$TAIL"
fi
LOG=/engine/tools/seedtail.$LEG.server.log
SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; sleep 4; }
trap stop EXIT

echo "################ LEG=$LEG  SPEC=$SPEC  TAIL=${MINISGL_DFLASH_SEED_TAIL:-<unset/none>}  DBG=$DBG  K=$K ################"
echo "-- PROVENANCE (from inside the container) --"
echo -n "  git rev-parse HEAD         : "; git -C /engine rev-parse HEAD 2>/dev/null || echo "(no git)"
echo -n "  git status --porcelain py/ : "; git -C /engine status --porcelain -- python | tr '\n' ' ' ; echo
echo -n "  /engine/python digest      : "; find /engine/python -name '*.py' -print0 | sort -z | xargs -0 md5sum | md5sum
echo -n "  dflash.py md5              : "; md5sum /engine/python/minisgl/spec/dflash.py
echo -n "  prompt file md5            : "; md5sum /engine/tools/spec_seed_prompt.txt
echo -n "  /opt/minisgl/python digest : "; find /opt/minisgl/python -name '*.py' -print0 2>/dev/null | sort -z | xargs -0 md5sum 2>/dev/null | md5sum
echo -n "  /opt/kernels        digest : "; find /opt/kernels -name '*.so' -print0 2>/dev/null | sort -z | xargs -0 md5sum 2>/dev/null | md5sum
echo -n "  MINISGL_DFLASH_SEED_TAIL   : "; echo "${MINISGL_DFLASH_SEED_TAIL:-<unset>}"

SPEC_ARGS=(--spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$K")
[ "$SPEC" = "none" ] && SPEC_ARGS=(--spec-algorithm none)

boot(){
  setsid env MINISGL_SPEC_DEBUG="$DBG" \
    PYTHONPATH=/opt/kernels:/engine/python:/engine python -m minisgl \
    --model "$MODEL" --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size "$TP" --disable-pynccl \
    --cache-type radix --attention-backend hip --page-size 16 \
    --cuda-graph-max-bs "$GRAPHBS" --max-running-requests 1 \
    --memory-ratio "${MEMR:-0.93}" \
    "${SPEC_ARGS[@]}" \
    > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 400); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "SERVER DIED"; tail -100 "$LOG"; exit 1; }
    grep -qE "Traceback \(most recent call last\)|AssertionError" "$LOG" && \
      { echo "SCHEDULER DIED (traceback in log)"; tail -60 "$LOG"; exit 1; }
    sleep 3
  done
  echo "server not ready"; tail -100 "$LOG"; exit 1
}

echo "[boot] launching..."; boot; echo "[boot] ready."
echo "-- SEED PROVENANCE (the cell is VOID if this does not show the intended tail) --"
grep -nE "prompt-prefill draft seed ENABLED|prefill_seed=|DFlash Laguna drafter" "$LOG" | head -6
echo -n "  seed-ENABLED line count = "; grep -c "prompt-prefill draft seed ENABLED" "$LOG"

PORT=$PORT MAXTOK=$MAXTOK SHORTTOK=$SHORTTOK LEG=$LEG python - <<'PY'
import json, os, time, urllib.request, hashlib
PORT=os.environ["PORT"]; MAXTOK=int(os.environ["MAXTOK"]); SHORTTOK=int(os.environ["SHORTTOK"])
LEG=os.environ["LEG"]
BASE=f"http://127.0.0.1:{PORT}"
CODE=open("/engine/tools/spec_seed_prompt.txt").read()
SHORT=("Write a complete, production-quality C++ source file implementing a B-tree with configurable "
       "order, supporting insert with node splitting, delete with merging, point lookup, and an "
       "in-order range scan iterator. Include the full class definition and all method bodies. "
       "Output only code.")
def run(p, mt, tag):
    b={"model":"m","messages":[{"role":"user","content":p}],"max_tokens":mt,
       "temperature":0.0,"stream":False}
    r=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(b).encode(),
                             headers={"Content-Type":"application/json"})
    t=time.perf_counter(); d=json.loads(urllib.request.urlopen(r,timeout=7200).read())
    w=time.perf_counter()-t; u=d.get("usage",{})
    msg=d["choices"][0]["message"]
    txt=(msg.get("reasoning_content") or "")+"\n<<<CONTENT>>>\n"+(msg.get("content") or "")
    open(f"/engine/tools/seedtail.{LEG}.{tag}.txt","w").write(txt)
    print(f"REQ {tag}: prompt_tok={u.get('prompt_tokens')} completion_tok={u.get('completion_tokens')} "
          f"wall={w:.2f}s TRUE_tok/s={u.get('completion_tokens',0)/w:.2f} "
          f"finish={d['choices'][0].get('finish_reason')} out_chars={len(txt)} "
          f"md5={hashlib.md5(txt.encode()).hexdigest()[:8]}", flush=True)
run("hi", 16, "warm")
run(CODE, MAXTOK, "CODE1")
run(CODE, MAXTOK, "CODE2")
run(SHORT, SHORTTOK, "SHORT1")
run(SHORT, SHORTTOK, "SHORT2")
PY

echo "############ P-BUCKET / DIVERGENCE ANALYSIS  leg=$LEG ############"
LOG="$LOG" python - <<'PY'
import os, re, collections
log=os.environ["LOG"]
pat=re.compile(r"\[spec-dbg\] uid=(\d+) c0=(\d+) dev=(\d+) conf=(-?\d+) k=(\d+) n=(-?\d+) emit=\[(.*?)\]")
by=collections.defaultdict(list)
for line in open(log, errors="ignore"):
    m=pat.search(line)
    if m:
        uid,c0,dev,conf,k,n,emit=m.groups()
        ne=len([x for x in emit.split(",") if x.strip()])
        by[int(uid)].append((int(c0),int(k),int(n),ne))
BUCK=[(0,32),(32,64),(64,128),(128,256),(256,512),(512,1024),(1024,10**9)]
if not by: print("  (no [spec-dbg] lines — DBG!=2 or spec off)")
for uid in sorted(by):
    rows=sorted(by[uid])
    if len(rows)<5: continue
    c0min=rows[0][0]
    tot_e=sum(r[3] for r in rows); tot_n=sum(r[2] for r in rows); tot_k=sum(r[1] for r in rows)
    print(f"\n=== uid={uid}  steps={len(rows)}  prompt_len={c0min}  emitted={tot_e}")
    print(f"    ACCEPT-LEN emitted/steps = {tot_e/len(rows):.3f}   accepted/step = {tot_n/len(rows):.3f}"
          f"   rate accepted/drafted = {tot_n/max(tot_k,1):.3f}")
    h=collections.Counter(r[2] for r in rows); tot=len(rows)
    print(f"    P(n=0)={h.get(0,0)/tot:.3f}  P(full n=k)={sum(1 for r in rows if r[2]>=r[1] and r[1]>0)/tot:.3f}")
    print("    accept-len vs P:")
    for lo,hi in BUCK:
        sel=[r for r in rows if lo <= r[0]-c0min < hi]
        if not sel: continue
        e=sum(r[3] for r in sel); n=sum(r[2] for r in sel); kk=sum(r[1] for r in sel)
        print(f"      P in [{lo:>5},{hi if hi<10**9 else 'inf':>5}): steps={len(sel):>4}  "
              f"emitted/step={e/len(sel):.3f}  rate={n/max(kk,1):.3f}")
PY
stop
echo "[done] leg=$LEG"
