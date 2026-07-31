#!/usr/bin/env bash
# PROMPT-PREFILL DRAFT SEED — A/B harness (adapted from spec_dflash_divergence.sh).
#
# Tests the falsifiable prediction of CONTINUANCE §11.6: with the drafter blind to the prompt the
# `P<64` accept-len bucket stays ~3.5 no matter how long the prompt is; with the seed in it must
# jump, because the drafter's aux prefix no longer starts empty.
#
# The two legs are two WORKTREES mounted at /engine (compose mounts `.`), NOT an env flag:
#   BASELINE  /home/pat/code/minisgl-rdna4-seedbase  @ 35382357 (parent, no seed)
#   CANDIDATE /home/pat/code/minisgl-rdna4-specod    @ 67bbdfb2 (seed on)
# so provenance is asserted from inside the container (md5 of the two changed files + the two boot
# log lines that exist only on the candidate).
#
# Boot config is PRODUCTION, not Phase 1a's: MINISGL_KV_FP8=1, --cache-type radix,
# MINISGL_SPEC_MHA_PAGED=1, --cuda-graph-max-bs 8, --max-running-requests 1, K=15, TP=2, greedy.
# That also closes §11.6's stated coverage hole (served acceptance was never measured).
#
# Requests, one boot:  warm / CODE (long real-code prompt) / CODE again (radix prefix-cache HIT) /
#                      SHORT control (~95 tok).  Every completion is written to
#                      /engine/tools/spec_seed.$LEG.<tag>.txt for the byte-identity gate.
#
# Env: LEG (name), SPEC (dflash|none), DBG (0|1|2), MAXTOK, SHORTTOK, K, TP, MEMR
set -uo pipefail
source /app/.venv/bin/activate 2>/dev/null || source /opt/venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1
export MINISGL_KV_FP8="${MINISGL_KV_FP8:-1}"
export MINISGL_SPEC_MHA_PAGED="${MINISGL_SPEC_MHA_PAGED:-1}"
# Laguna is SWA-hybrid, so `--cache-type radix` is silently DOWNGRADED to naive unless this is on
# (scheduler.py:148). tools/serve.sh defaults it to 1 for SWA models, so production = 1; without it
# the repeat-request prefix-cache-hit seeding path the change also enables is never exercised.
export MINISGL_SWA_RADIX="${MINISGL_SWA_RADIX:-1}"

MODEL="${MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
DRAFT="${DRAFT:-poolside/Laguna-XS-2.1-DFlash-NVFP4}"
K="${K:-15}"; TP="${TP:-2}"; LEG="${LEG:-L1}"; MAXTOK="${MAXTOK:-1600}"
SHORTTOK="${SHORTTOK:-384}"; SPEC="${SPEC:-dflash}"; DBG="${DBG:-2}"; GRAPHBS="${GRAPHBS:-8}"
PORT=21957
LOG=/engine/tools/spec_seed.$LEG.server.log
SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; sleep 3; }
trap stop EXIT

echo "################ LEG=$LEG  SPEC=$SPEC  DBG=$DBG  K=$K  KV_FP8=$MINISGL_KV_FP8  graphbs=$GRAPHBS ################"
echo "-- PROVENANCE (from inside the container) --"
echo -n "  git rev-parse HEAD         : "; git -C /engine rev-parse HEAD 2>/dev/null || echo "(no git)"
echo -n "  git status --porcelain py/ : "; git -C /engine status --porcelain -- python | tr '\n' ' ' ; echo
echo -n "  /engine/python digest      : "; find /engine/python -name '*.py' -print0 | sort -z | xargs -0 md5sum | md5sum
echo -n "  dflash.py md5              : "; md5sum /engine/python/minisgl/spec/dflash.py
echo -n "  scheduler.py md5           : "; md5sum /engine/python/minisgl/scheduler/scheduler.py
echo -n "  spec/base.py md5           : "; md5sum /engine/python/minisgl/spec/base.py
echo -n "  prompt file md5            : "; md5sum /engine/tools/spec_seed_prompt.txt
echo -n "  SEED discriminator (grep)  : "; grep -c "prefill_aux_tail" /engine/python/minisgl/spec/base.py
echo -n "  /opt/minisgl/python digest : "; find /opt/minisgl/python -name '*.py' -print0 2>/dev/null | sort -z | xargs -0 md5sum 2>/dev/null | md5sum
echo -n "  /opt/kernels        digest : "; find /opt/kernels -name '*.so' -print0 2>/dev/null | sort -z | xargs -0 md5sum 2>/dev/null | md5sum

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
    # The launcher outlives a dead scheduler subprocess, so polling the parent pid is not enough —
    # without this the boot loop spins for 20 min holding both cards.
    grep -qE "Traceback \(most recent call last\)|AssertionError" "$LOG" && \
      { echo "SCHEDULER DIED (traceback in log)"; tail -60 "$LOG"; exit 1; }
    sleep 3
  done
  echo "server not ready"; tail -100 "$LOG"; exit 1
}

echo "[boot] launching..."; boot; echo "[boot] ready."
echo "-- ENGINE ARGV --"; for p in $(pgrep -f "python -m minisgl" | head -2); do echo -n "  pid $p : "; tr '\0' ' ' < /proc/$p/cmdline; echo; done
echo "-- SEED / drafter boot lines (present ONLY on the candidate) --"
grep -nE "prompt-prefill draft seed ENABLED|prefill_seed=|DFlash Laguna drafter|spec-decode" "$LOG" | head -12
echo -n "  seed-line count = "; grep -c "prompt-prefill draft seed ENABLED" "$LOG"

PORT=$PORT MAXTOK=$MAXTOK SHORTTOK=$SHORTTOK LEG=$LEG python - <<'PY'
import json, os, time, urllib.request
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
    # BOTH fields: the reasoning parser routes most of this model's output to reasoning_content, so
    # comparing only `content` would silently compare two empty strings and call it byte-identical.
    txt=(msg.get("reasoning_content") or "")+"\n<<<CONTENT>>>\n"+(msg.get("content") or "")
    open(f"/engine/tools/spec_seed.{LEG}.{tag}.txt","w").write(txt)
    print(f"REQ {tag}: prompt_tok={u.get('prompt_tokens')} completion_tok={u.get('completion_tokens')} "
          f"wall={w:.2f}s TRUE_tok/s={u.get('completion_tokens',0)/w:.2f} "
          f"finish={d['choices'][0].get('finish_reason')} out_chars={len(txt)} "
          f"md5={__import__('hashlib').md5(txt.encode()).hexdigest()}", flush=True)
run("hi", 16, "warm")
run(CODE, MAXTOK, "CODE1")
run(CODE, MAXTOK, "CODE2")     # same boot, same prompt -> radix prefix-cache HIT
run(SHORT, SHORTTOK, "SHORT")
PY

echo "-- spec counters --"; grep -E "^\[spec\]|mean accept-len" "$LOG" | tail -6
echo "-- prefix-cache / radix lines --"; grep -iE "prefix cache|SWA-radix|SWA-hybrid" "$LOG" | tail -6

echo "############ P-BUCKET / DIVERGENCE ANALYSIS  leg=$LEG ############"
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
if not by:
    print("  (no [spec-dbg] lines — DBG!=2 or spec off)")
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
    print("      " + "  ".join(f"n={i}:{h.get(i,0)}({100*h.get(i,0)/tot:.0f}%)" for i in range(0,17) if h.get(i,0)))
    print(f"    P(n=0 immediate divergence) = {h.get(0,0)/tot:.3f}   P(full accept n=k) = "
          f"{sum(1 for r in rows if r[2]>=r[1] and r[1]>0)/tot:.3f}")
    print("    accept-len vs P (# GENERATED tokens already emitted for this req):")
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
