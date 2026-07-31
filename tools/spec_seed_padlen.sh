#!/usr/bin/env bash
# DFLASH SEED — PROMPT-LENGTH CROSSOVER, held-task design.
#
# Truncating the code prompt to intermediate lengths (spec_seed_crossover.sh) does NOT isolate the
# length axis: each truncation asks for a DIFFERENT continuation and accept-len swings 3.0-8.0 inside
# one leg, swamping the seed effect. This leg fixes the task instead: the SAME 95-token B-tree
# instruction is always the LAST thing in the prompt, preceded by N tokens of filler taken from the
# code file. So the requested output is constant, the seeded tail always covers the instruction, and
# the only thing that moves is how many prompt tokens sit in front of it.
#
# Env: TAIL (int, 0 = seeding off, "none" = --spec-algorithm none), DBG, GENTOK.
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
GENTOK="${GENTOK:-384}"
PORT=21960

SPEC=dflash
if [ "$TAIL" = "none" ]; then SPEC=none; LEG="pplain"; unset MINISGL_DFLASH_SEED_TAIL
else LEG="pt$TAIL"; export MINISGL_DFLASH_SEED_TAIL="$TAIL"; fi
LOG=/engine/tools/seedpad.$LEG.server.log
SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; sleep 4; }
trap stop EXIT

echo "################ PADLEN LEG=$LEG SPEC=$SPEC TAIL=${MINISGL_DFLASH_SEED_TAIL:-<unset>} GENTOK=$GENTOK ################"
echo -n "  dflash.py md5 : "; md5sum /engine/python/minisgl/spec/dflash.py

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
FILLER=open("/engine/tools/spec_seed_prompt.txt").read()
fids=tok(FILLER, add_special_tokens=False)["input_ids"]
TASK=("Write a complete, production-quality C++ source file implementing a B-tree with configurable "
      "order, supporting insert with node splitting, delete with merging, point lookup, and an "
      "in-order range scan iterator. Include the full class definition and all method bodies. "
      "Output only code.")
def run(p, tag):
    b={"model":"m","messages":[{"role":"user","content":p}],"max_tokens":GEN,
       "temperature":0.0,"stream":False}
    r=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(b).encode(),
                             headers={"Content-Type":"application/json"})
    t=time.perf_counter(); d=json.loads(urllib.request.urlopen(r,timeout=7200).read())
    w=time.perf_counter()-t; u=d.get("usage",{}); m=d["choices"][0]["message"]
    txt=(m.get("reasoning_content") or "")+"|"+(m.get("content") or "")
    print(f"REQ {tag}: prompt_tok={u.get('prompt_tokens')} completion_tok={u.get('completion_tokens')} "
          f"wall={w:.2f}s TRUE_tok/s={u.get('completion_tokens',0)/w:.2f} "
          f"finish={d['choices'][0].get('finish_reason')} md5={hashlib.md5(txt.encode()).hexdigest()[:8]}",
          flush=True)
run("hi", "warm")
# Filler FIRST, task LAST. Longest first so the shared filler prefix is a radix cache hit after #1.
for n in (3400, 2048, 1024, 512, 256, 128, 0):
    pad = tok.decode(fids[:n]) if n else ""
    run((pad + "\n\n" + TASK) if n else TASK, f"PAD{n}")
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
    c0=rows[0][0]; e=sum(r[3] for r in rows); n=sum(r[2] for r in rows); kk=sum(r[1] for r in rows)
    def acc(lim):
        sel=[r for r in rows if r[0]-c0 < lim]
        return sum(x[3] for x in sel)/len(sel) if sel else float('nan')
    print(f"  uid={uid} prompt_len={c0:>5} steps={len(rows):>4} ALL={e/len(rows):.3f} "
          f"P<128={acc(128):.3f} P<64={acc(64):.3f} rate={n/max(kk,1):.3f}")
PY
stop
echo "[done] leg=$LEG"
