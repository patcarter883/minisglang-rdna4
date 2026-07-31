#!/usr/bin/env bash
# DFlash CONDITIONING diagnostic (Laguna).
#
# One boot, MINISGL_SPEC_DEBUG=2 so the engine emits, per verify step:
#   [dflash-dbg] uid=.. anchor=.. base_pos=.. B=.. k=.. draft=[...]      (the drafter's top-1 chain)
#   [spec-dbg]   uid=.. c0=.. dev=.. conf=.. k=.. n=.. emit=[...]        (n = WHERE the chain diverged)
# then post-processes the log into (a) a divergence-position histogram and (b) accept-len as a
# function of the number of GENERATED tokens already in the drafter's aux prefix (P = c0 - c0_min).
#
# Env: CTXW (MINISGL_DFLASH_CTX_WINDOW, 0 = default/unbounded), LEG, MAXTOK
set -uo pipefail
source /app/.venv/bin/activate 2>/dev/null || source /opt/venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1
export MINISGL_KV_FP8="${MINISGL_KV_FP8:-0}"

MODEL="${MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
DRAFT="${DRAFT:-poolside/Laguna-XS-2.1-DFlash-NVFP4}"
K="${K:-15}"; TP="${TP:-2}"; LEG="${LEG:-L1}"; CTXW="${CTXW:-0}"; MAXTOK="${MAXTOK:-1600}"
PORT=21955
LOG=/engine/tools/spec_divergence.$LEG.server.log
SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; sleep 3; }
trap stop EXIT

echo "################ LEG=$LEG  CTXW=$CTXW  K=$K ################"
echo -n "  /opt/minisgl/python digest : "; find /opt/minisgl/python -name '*.py' -print0 2>/dev/null | sort -z | xargs -0 md5sum 2>/dev/null | md5sum
echo -n "  /opt/kernels        digest : "; find /opt/kernels -name '*.so' -print0 2>/dev/null | sort -z | xargs -0 md5sum 2>/dev/null | md5sum
echo -n "  /engine/python digest      : "; find /engine/python -name '*.py' -print0 | sort -z | xargs -0 md5sum | md5sum
echo -n "  dflash.py md5              : "; md5sum /engine/python/minisgl/spec/dflash.py
echo -n "  scheduler.py md5           : "; md5sum /engine/python/minisgl/scheduler/scheduler.py
echo -n "  MHA_PAGED discriminator    : "; grep -c MINISGL_SPEC_MHA_PAGED /engine/python/minisgl/engine/engine.py

boot(){
  setsid env MINISGL_SPEC_DEBUG=2 MINISGL_DFLASH_CTX_WINDOW="$CTXW" \
    PYTHONPATH=/opt/kernels:/engine/python:/engine python -m minisgl \
    --model "$MODEL" --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size "$TP" --disable-pynccl \
    --cache-type naive --attention-backend hip --page-size 16 \
    --cuda-graph-max-bs 0 --max-running-requests 1 \
    --memory-ratio "${MEMR:-0.90}" \
    --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$K" \
    > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "SERVER DIED"; tail -80 "$LOG"; exit 1; }
    sleep 3
  done
  echo "server not ready"; tail -80 "$LOG"; exit 1
}

echo "[boot] launching..."; boot; echo "[boot] ready."
echo "-- ENGINE ARGV --"; for p in $(pgrep -f "python -m minisgl" | head -2); do echo -n "  pid $p : "; tr '\0' ' ' < /proc/$p/cmdline; echo; done
echo "-- drafter build line --"; grep -iE "DFlash Laguna drafter|spec-decode" "$LOG" | head -8

PORT=$PORT MAXTOK=$MAXTOK python - <<'PY'
import json, os, time, urllib.request
PORT=os.environ["PORT"]; MAXTOK=int(os.environ["MAXTOK"])
BASE=f"http://127.0.0.1:{PORT}"
CODE=("Write a complete, production-quality C++ source file implementing a B-tree with configurable "
      "order, supporting insert with node splitting, delete with merging, point lookup, and an "
      "in-order range scan iterator. Include the full class definition and all method bodies. "
      "Output only code.")
REPT=("Count from 1 to 400, writing each number on its own line, like:\n1\n2\n3\n")
def run(p, mt, tag):
    b={"model":"m","messages":[{"role":"user","content":p}],"max_tokens":mt,
       "temperature":0.0,"stream":False}
    r=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(b).encode(),
                             headers={"Content-Type":"application/json"})
    t=time.perf_counter(); d=json.loads(urllib.request.urlopen(r,timeout=3600).read())
    w=time.perf_counter()-t; u=d.get("usage",{})
    print(f"REQ {tag}: prompt_tok={u.get('prompt_tokens')} completion_tok={u.get('completion_tokens')} "
          f"wall={w:.2f}s tok/s={u.get('completion_tokens',0)/w:.2f} "
          f"finish={d['choices'][0].get('finish_reason')}", flush=True)
run("hi", 32, "warm")
run(CODE, MAXTOK, "CODE")
run(REPT, MAXTOK, "REPETITIVE")
PY

echo "############ DIVERGENCE ANALYSIS  leg=$LEG ctxw=$CTXW ############"
LOG="$LOG" python - <<'PY'
import os, re, collections
log=os.environ["LOG"]
pat=re.compile(r"\[spec-dbg\] uid=(\d+) c0=(\d+) dev=(\d+) conf=(-?\d+) k=(\d+) n=(-?\d+) emit=\[(.*?)\]")
dpat=re.compile(r"\[dflash-dbg\] uid=(\d+) anchor=(-?\d+) base_pos=(\d+) B=(\d+) k=(\d+) draft=\[(.*?)\]")
by=collections.defaultdict(list); drafts=collections.defaultdict(dict)
for line in open(log, errors="ignore"):
    m=pat.search(line)
    if m:
        uid,c0,dev,conf,k,n,emit=m.groups()
        ne=len([x for x in emit.split(",") if x.strip()])
        by[int(uid)].append((int(c0),int(k),int(n),ne)); continue
    m=dpat.search(line)
    if m:
        uid,anchor,bp,B,k,dr=m.groups()
        drafts[int(uid)][int(bp)]=[int(x) for x in dr.split(",") if x.strip()]
BUCK=[(0,32),(32,64),(64,128),(128,256),(256,512),(512,1024),(1024,10**9)]
for uid in sorted(by):
    rows=sorted(by[uid])
    if len(rows)<5: continue
    c0min=rows[0][0]
    tot_e=sum(r[3] for r in rows); tot_n=sum(r[2] for r in rows)
    tot_k=sum(r[1] for r in rows)
    print(f"\n=== uid={uid}  steps={len(rows)}  c0_min(prompt_len)={c0min}  "
          f"gen_span={rows[-1][0]-c0min}  emitted={tot_e} accepted={tot_n} drafted={tot_k}")
    print(f"    accept-len emitted/steps = {tot_e/len(rows):.3f}   accepted/step = {tot_n/len(rows):.3f}"
          f"   accept RATE accepted/drafted = {tot_n/max(tot_k,1):.3f}")
    h=collections.Counter(r[2] for r in rows); tot=len(rows)
    print("    divergence-position histogram  n=#accepted drafts before first mismatch:")
    print("      " + "  ".join(f"n={i}:{h.get(i,0)}({100*h.get(i,0)/tot:.0f}%)" for i in range(0,16) if h.get(i,0)))
    print(f"    P(n=0 immediate divergence) = {h.get(0,0)/tot:.3f}   P(full accept n=k) = "
          f"{sum(1 for r in rows if r[2]>=r[1] and r[1]>0)/tot:.3f}")
    print("    accept-len vs P (# generated tokens already in the drafter aux prefix):")
    for lo,hi in BUCK:
        sel=[r for r in rows if lo <= r[0]-c0min < hi]
        if not sel: continue
        e=sum(r[3] for r in sel); n=sum(r[2] for r in sel); kk=sum(r[1] for r in sel)
        print(f"      P in [{lo:>5},{hi if hi<10**9 else 'inf':>5}): steps={len(sel):>4}  "
              f"emitted/step={e/len(sel):.3f}  accepted/step={n/len(sel):.3f}  "
              f"rate={n/max(kk,1):.3f}")
    print("    first 6 steps (drafts vs accepted count):")
    for c0,k,n,ne in rows[:6]:
        d=drafts.get(uid,{}).get(c0)
        print(f"      P={c0-c0min:<4} k={k} n={n} emitted={ne} draft={d}")
PY
stop
echo "[done] leg=$LEG"
