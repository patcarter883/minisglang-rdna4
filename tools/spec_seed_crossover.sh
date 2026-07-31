#!/usr/bin/env bash
# DFLASH PROMPT-PREFILL SEED — PROMPT-LENGTH CROSSOVER.
#
# The tail sweep measures tail at two prompt lengths (95 / 3571) and finds opposite signs. This leg
# locates the sign flip IN PROMPT LENGTH: one boot per TAIL, the SAME code prompt truncated to
# {3571, 2048, 1024, 512, 256, 128} tokens, max_tokens fixed at 512 in every cell (accept-len on this
# drafter is a function of generation length, so it must be held constant across the sweep).
#
# Truncation is from the FRONT (each shorter prompt is a strict token PREFIX of the longer ones) and
# the lengths are issued LONGEST FIRST, so every cell after the first is a radix prefix-cache HIT and
# prefill cost is normalised away instead of scaling with the axis under test.
#
# Env: TAIL (int, 0 = seeding off), DBG, GENTOK.
set -uo pipefail
source /app/.venv/bin/activate 2>/dev/null || source /opt/venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1
export MINISGL_KV_FP8="${MINISGL_KV_FP8:-1}"
export MINISGL_SPEC_MHA_PAGED="${MINISGL_SPEC_MHA_PAGED:-1}"
export MINISGL_SWA_RADIX="${MINISGL_SWA_RADIX:-1}"

MODEL="${MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
DRAFT="${DRAFT:-poolside/Laguna-XS-2.1-DFlash-NVFP4}"
K="${K:-15}"; TP="${TP:-2}"; DBG="${DBG:-2}"; GRAPHBS="${GRAPHBS:-8}"; TAIL="${TAIL:-64}"
GENTOK="${GENTOK:-512}"
PORT=21959

SPEC=dflash
if [ "$TAIL" = "none" ]; then SPEC=none; LEG="xplain"; unset MINISGL_DFLASH_SEED_TAIL
else LEG="xt$TAIL"; export MINISGL_DFLASH_SEED_TAIL="$TAIL"; fi
LOG=/engine/tools/seedx.$LEG.server.log
SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; sleep 4; }
trap stop EXIT

echo "################ CROSSOVER LEG=$LEG SPEC=$SPEC TAIL=${MINISGL_DFLASH_SEED_TAIL:-<unset>} GENTOK=$GENTOK ################"
echo -n "  dflash.py md5 : "; md5sum /engine/python/minisgl/spec/dflash.py
echo -n "  prompt md5    : "; md5sum /engine/tools/spec_seed_prompt.txt

SPEC_ARGS=(--spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$K")
[ "$SPEC" = "none" ] && SPEC_ARGS=(--spec-algorithm none)
boot(){
  setsid env MINISGL_SPEC_DEBUG="$DBG" \
    PYTHONPATH=/opt/kernels:/engine/python:/engine python -m minisgl \
    --model "$MODEL" --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size "$TP" --disable-pynccl \
    --cache-type radix --attention-backend hip --page-size 16 \
    --cuda-graph-max-bs "$GRAPHBS" --max-running-requests 1 \
    --memory-ratio "${MEMR:-0.93}" "${SPEC_ARGS[@]}" > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 400); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "SERVER DIED"; tail -60 "$LOG"; exit 1; }
    grep -qE "Traceback \(most recent call last\)|AssertionError" "$LOG" && { echo "SCHEDULER DIED"; tail -60 "$LOG"; exit 1; }
    sleep 3
  done
  echo "server not ready"; tail -60 "$LOG"; exit 1
}
echo "[boot] launching..."; boot; echo "[boot] ready."
echo "-- SEED PROVENANCE --"
grep -nE "prompt-prefill draft seed ENABLED|prefill_seed=" "$LOG" | head -4
echo -n "  seed-ENABLED line count = "; grep -c "prompt-prefill draft seed ENABLED" "$LOG"

PORT=$PORT GENTOK=$GENTOK LEG=$LEG MODEL=$MODEL python - <<'PY'
import json, os, time, urllib.request, hashlib
from transformers import AutoTokenizer
PORT=os.environ["PORT"]; GEN=int(os.environ["GENTOK"]); LEG=os.environ["LEG"]
BASE=f"http://127.0.0.1:{PORT}"
tok=AutoTokenizer.from_pretrained(os.environ["MODEL"])
CODE=open("/engine/tools/spec_seed_prompt.txt").read()
ids=tok(CODE, add_special_tokens=False)["input_ids"]
print(f"full prompt = {len(ids)} raw tokens", flush=True)
def run(p, tag):
    b={"model":"m","messages":[{"role":"user","content":p}],"max_tokens":GEN,
       "temperature":0.0,"stream":False}
    r=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(b).encode(),
                             headers={"Content-Type":"application/json"})
    t=time.perf_counter(); d=json.loads(urllib.request.urlopen(r,timeout=7200).read())
    w=time.perf_counter()-t; u=d.get("usage",{})
    print(f"REQ {tag}: prompt_tok={u.get('prompt_tokens')} completion_tok={u.get('completion_tokens')} "
          f"wall={w:.2f}s TRUE_tok/s={u.get('completion_tokens',0)/w:.2f} "
          f"finish={d['choices'][0].get('finish_reason')}", flush=True)
run("hi", "warm")
# LONGEST FIRST so every later (shorter) prompt is a prefix-cache HIT.
for n in (100000, 2048, 1024, 512, 256, 128):
    sub = tok.decode(ids[:n]) if n < len(ids) else CODE
    run(sub, f"P{min(n,len(ids))}")
PY

echo "############ ACCEPT-LEN PER REQUEST  leg=$LEG ############"
LOG="$LOG" python - <<'PY'
import os, re, collections
log=os.environ["LOG"]
pat=re.compile(r"\[spec-dbg\] uid=(\d+) c0=(\d+) dev=(\d+) conf=(-?\d+) k=(\d+) n=(-?\d+) emit=\[(.*?)\]")
by=collections.defaultdict(list)
for line in open(log, errors="ignore"):
    m=pat.search(line)
    if m:
        uid,c0,dev,conf,k,n,emit=m.groups()
        by[int(uid)].append((int(c0),int(k),int(n),len([x for x in emit.split(",") if x.strip()])))
for uid in sorted(by):
    rows=sorted(by[uid])
    if len(rows)<5: continue
    c0min=rows[0][0]; e=sum(r[3] for r in rows); n=sum(r[2] for r in rows); kk=sum(r[1] for r in rows)
    print(f"  uid={uid} prompt_len={c0min:>5} steps={len(rows):>4} emitted={e:>5} "
          f"ACCEPT-LEN={e/len(rows):.3f} rate={n/max(kk,1):.3f}")
PY
stop
echo "[done] leg=$LEG"
