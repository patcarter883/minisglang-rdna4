#!/usr/bin/env bash
# DFlash persistent-draft-KV validation — LEAN image, 35B MXFP4 target, TP=2, CUDA-graph verify.
# Measures SUSTAINED decode tok/s on a long deterministic generation ("count to 400"), for:
#   none              -> no-spec baseline (CUDA graph on)              => the ~48 tok/s bar to beat
#   dflash persist=1  -> NEW persistent per-req draft KV (default)     => should beat baseline
#   dflash persist=0  -> OLD full-context re-feed (MINISGL_DFLASH_PERSIST_KV=0) => the ~29 tok/s "before"
# and confirms LOSSLESSNESS (greedy dflash text == baseline text, byte-identical).
# Runs inside minisgl-rdna4:lean (canonical kernels /opt/kernels, venv /opt/venv, NO vllm), --attn hip,
# MINISGL_KV_FP8=1. Driven under a 2-card lease by run_spec_dflash_persist.sh.
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
python -c "
import importlib
for m in ('moe_hip','tail_hip','gdn_hip','attn_hip','attn_decode','attn_prefill_paged'):
    try: importlib.import_module(m); print(f'[setup] {m} OK')
    except Exception as e: print(f'[setup] {m} MISSING ({e.__class__.__name__})')
"

MODEL="${MODEL:-pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4}"
DRAFT="${DRAFT:-z-lab/Qwen3.6-35B-A3B-DFlash}"
TP="${TP:-2}"; PORT="${PORT:-21077}"; MEMRATIO="${MEMRATIO:-0.82}"
MAXRUN="${MAXRUN:-4}"; MAXTOK="${MAXTOK:-700}"; NUM_DRAFT="${NUM_DRAFT:-8}"
# CONFIGS = space-separated  algo:persist  tuples. algo=none -> baseline; algo=dflash -> spec (persist 0|1).
CONFIGS="${CONFIGS:-none:1 dflash:1 dflash:0}"
TAG="${TAG:-dflash_persist}"
RES="/engine/tools/spec_${TAG}.tsv"; : > "$RES"
OUTDIR=/engine/tools

SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1=algo $2=persist ; sets $LOG
  local algo="$1" persist="$2"
  LOG="/engine/tools/spec_${TAG}.${algo}.p${persist}.log"
  local spec="" graph="--graph 16"
  local env_extra="MINISGL_MOE_SCATTER=0 MINISGL_KV_FP8=1"
  local pynccl=""; [ "$TP" -gt 1 ] && pynccl="--disable-pynccl"
  if [ "$algo" != "none" ]; then
    spec="--spec-algorithm $algo --spec-num-draft $NUM_DRAFT --spec-draft-model-path $DRAFT"
    env_extra="$env_extra MINISGL_SPEC_DEBUG=1 MINISGL_DFLASH_PERSIST_KV=$persist"
  fi
  echo "[boot $algo persist=$persist] -> $LOG"
  setsid env PYTHONPATH=/opt/kernels:/engine/python:/engine $env_extra python -m minisgl \
    --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" $graph $pynccl \
    --memory-ratio "$MEMRATIO" --max-running-requests "$MAXRUN" \
    --attention-backend hip $spec > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 500); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "[boot $algo p=$persist] DIED:"; tail -40 "$LOG"; return 1; }
    grep -q "Traceback (most recent call last)" "$LOG" && { echo "[boot $algo p=$persist] CRASH:"; tail -40 "$LOG"; return 1; }
    sleep 3
  done; echo "[boot $algo p=$persist] timeout:"; tail -40 "$LOG"; return 1
}

probe(){ PORT=$PORT MAXTOK=$MAXTOK OUT=$1 python - <<'PY'
import json,os,time,urllib.request
PORT=os.environ["PORT"]; MAXTOK=int(os.environ["MAXTOK"]); OUT=os.environ["OUT"]
prompt=("Count from 1 to 400, listing every integer in order separated by commas, "
        "like: 1, 2, 3, 4, 5, and so on. Do not stop until you reach 400.")
def gen():
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":MAXTOK,"enable_thinking":False,
                     "messages":[{"role":"user","content":prompt}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,
                             headers={"Content-Type":"application/json"})
    t=time.time(); d=json.load(urllib.request.urlopen(r,timeout=600)); el=time.time()-t
    txt=d["choices"][0]["message"]["content"]
    n=(d.get("usage") or {}).get("completion_tokens") or 0
    return n, el, txt
n0,e0,txt=gen()  # measured (single long deterministic run = sustained decode)
print(f"PROBE toks_per_s={n0/e0:.2f} completion_tokens={n0} elapsed={e0:.2f}")
json.dump(txt, open(OUT,"w"))
PY
}

for cfg in $CONFIGS; do
  IFS=: read -r algo persist <<< "$cfg"
  echo "===== boot $algo persist=$persist ====="
  OUTJSON="$OUTDIR/spec_${TAG}.${algo}.p${persist}.txt.json"
  if boot "$algo" "$persist"; then
    out=$(probe "$OUTJSON"); echo "$out"
    tps=$(echo "$out" | sed -n 's/.*toks_per_s=\([0-9.]*\).*/\1/p')
    ctok=$(echo "$out" | sed -n 's/.*completion_tokens=\([0-9]*\).*/\1/p')
    accline=$(grep -E "\[spec\]" "$LOG" | tail -1)
    eps=$(echo "$accline" | sed -n 's/.*emitted\/step=\([0-9.]*\).*/\1/p')
    acc=$(echo "$accline" | sed -n 's/.*accept_rate=\([0-9.]*\).*/\1/p')
    printf "%s\tpersist=%s\t%s\t%s\t%s\t%s\n" "$algo" "$persist" "${tps:-NA}" "${ctok:-NA}" "${eps:-NA}" "${acc:-NA}" >> "$RES"
  else
    printf "%s\tpersist=%s\tBOOT_FAIL\t-\t-\t-\n" "$algo" "$persist" >> "$RES"
  fi
  stop; sleep 2
done

echo "===== LOSSLESSNESS: baseline(none:1) vs dflash(persist=1) ====="
python - "$OUTDIR/spec_${TAG}.none.p1.txt.json" "$OUTDIR/spec_${TAG}.dflash.p1.txt.json" <<'PY'
import json,sys,os
def load(p): return json.load(open(p)) if os.path.exists(p) else None
a=load(sys.argv[1]); b=load(sys.argv[2])
if a is None or b is None:
    print("LOSSLESS CHECK: SKIPPED (missing output)"); sys.exit(0)
m = a==b
print(f"  baseline len={len(a)} dflash len={len(b)}  MATCH={m}")
if not m:
    for j,(x,y) in enumerate(zip(a,b)):
        if x!=y: print(f"    diverge@{j}: base={a[max(0,j-30):j+10]!r} dflash={b[max(0,j-30):j+10]!r}"); break
    else: print("    (one is a prefix of the other)")
print("DFLASH LOSSLESSNESS:", "PASS" if m else "MISMATCH")
PY

echo "===== RESULTS  model=$MODEL  K=$NUM_DRAFT ====="
printf "algo\tpersist\ttoks/s\tcompl_tok\temit/step\taccept\n"; cat "$RES"
echo "[done]"
