#!/usr/bin/env bash
# DFlash persistent-KV: (1) clean LOSSLESSNESS on naturally-terminating prompts (baseline vs dflash,
# byte-exact), and (2) LONG-CONTEXT perf where the O(P) re-feed the persistent KV removes actually
# bites (persist=1 vs persist=0 at ~4k generated tokens). Lean image, 35B MXFP4, TP=2, graph verify.
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
python -c "import moe_hip,tail_hip,gdn_hip,attn_hip,attn_decode,attn_prefill_paged;print('[setup] hip pkgs OK')" || exit 1

MODEL="${MODEL:-pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4}"
DRAFT="${DRAFT:-z-lab/Qwen3.6-35B-A3B-DFlash}"
TP="${TP:-2}"; PORT="${PORT:-21078}"; MEMRATIO="${MEMRATIO:-0.82}"
MAXRUN="${MAXRUN:-4}"; NUM_DRAFT="${NUM_DRAFT:-8}"; LONGTOK="${LONGTOK:-4000}"
OUTDIR=/engine/tools; RES="$OUTDIR/spec_dflash_longctx.tsv"; : > "$RES"

SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1=algo $2=persist ; sets $LOG
  local algo="$1" persist="$2"
  LOG="$OUTDIR/spec_dflash_longctx.${algo}.p${persist}.log"
  local spec="" env_extra="MINISGL_MOE_SCATTER=0 MINISGL_KV_FP8=1"
  local pynccl=""; [ "$TP" -gt 1 ] && pynccl="--disable-pynccl"
  if [ "$algo" != "none" ]; then
    spec="--spec-algorithm $algo --spec-num-draft $NUM_DRAFT --spec-draft-model-path $DRAFT"
    env_extra="$env_extra MINISGL_SPEC_DEBUG=1 MINISGL_DFLASH_PERSIST_KV=$persist"
  fi
  echo "[boot $algo persist=$persist] -> $LOG"
  setsid env PYTHONPATH=/opt/kernels:/engine/python:/engine $env_extra python -m minisgl \
    --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" --graph 16 $pynccl \
    --memory-ratio "$MEMRATIO" --max-running-requests "$MAXRUN" --attention-backend hip $spec > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 500); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "[boot $algo p=$persist] DIED:"; tail -40 "$LOG"; return 1; }
    grep -q "Traceback (most recent call last)" "$LOG" && { echo "CRASH:"; tail -40 "$LOG"; return 1; }
    sleep 3
  done; echo "timeout"; tail -40 "$LOG"; return 1
}

# Short, naturally-terminating prompts (finish on EOS well under the cap) -> clean byte-exact compare.
lossless_probe(){ PORT=$PORT OUT=$1 python - <<'PY'
import json,os,urllib.request
PORT=os.environ["PORT"]; OUT=os.environ["OUT"]
prompts=["What is the capital of France? Answer in one short sentence.",
         "Name the first four planets from the Sun, comma-separated.",
         "Write one sentence about why the sky appears blue.",
         "What is 12 times 12? Give just the number."]
res=[]
for p in prompts:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":160,"enable_thinking":False,
                     "messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(r,timeout=300))
    res.append({"text":d["choices"][0]["message"]["content"],
                "fr":d["choices"][0].get("finish_reason"),
                "n":(d.get("usage") or {}).get("completion_tokens")})
json.dump(res,open(OUT,"w"))
for i,x in enumerate(res): print(f"  [{i}] fr={x['fr']} n={x['n']} {x['text'][:70]!r}")
PY
}

# One long deterministic generation -> pushes context to ~LONGTOK so the O(P) re-feed matters.
long_probe(){ PORT=$PORT LONGTOK=$LONGTOK python - <<'PY'
import json,os,time,urllib.request
PORT=os.environ["PORT"]; LONGTOK=int(os.environ["LONGTOK"])
prompt=("Count from 1 to 2000, listing every integer in order separated by commas "
        "(1, 2, 3, 4, ...). Do not skip any number and do not stop early.")
body=json.dumps({"model":"m","temperature":0.0,"max_tokens":LONGTOK,"enable_thinking":False,
                 "messages":[{"role":"user","content":prompt}]}).encode()
r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
t=time.time(); d=json.load(urllib.request.urlopen(r,timeout=900)); el=time.time()-t
n=(d.get("usage") or {}).get("completion_tokens") or 0
print(f"LONG toks_per_s={n/el:.2f} completion_tokens={n} elapsed={el:.2f}")
PY
}

echo "===== baseline (none) — lossless reference + long-ctx bar ====="
boot none 1 && { lossless_probe "$OUTDIR/ll.none.json"; base_long=$(long_probe); echo "$base_long"; } ; stop; sleep 2

echo "===== dflash persist=1 — lossless check + long-ctx ====="
boot dflash 1 && {
  lossless_probe "$OUTDIR/ll.dflash.json"
  p1_long=$(long_probe); echo "$p1_long"
  acc=$(grep -E "\[spec\]" "$LOG" | tail -1)
} ; stop; sleep 2

echo "===== dflash persist=0 — long-ctx (the O(P) re-feed) ====="
boot dflash 0 && { p0_long=$(long_probe); echo "$p0_long"; acc0=$(grep -E "\[spec\]" "$LOG" | tail -1); } ; stop; sleep 2

echo "===== LOSSLESSNESS (baseline vs dflash persist=1, byte-exact on terminating prompts) ====="
python - "$OUTDIR/ll.none.json" "$OUTDIR/ll.dflash.json" <<'PY'
import json,sys,os
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); ok=True
for i,(x,y) in enumerate(zip(a,b)):
    m=(x["text"]==y["text"]); ok&=m
    print(f"  prompt[{i}] MATCH={m} (base fr={x['fr']} n={x['n']} / dflash fr={y['fr']} n={y['n']})")
    if not m: print(f"    base={x['text']!r}\n    dfl ={y['text']!r}")
print("DFLASH LOSSLESSNESS:", "PASS" if ok else "MISMATCH")
PY

echo "===== LONG-CONTEXT tok/s (K=$NUM_DRAFT, ~${LONGTOK} tok) ====="
echo "baseline: $base_long"
echo "persist=1: ${p1_long:-NA}   ${acc:-}"
echo "persist=0: ${p0_long:-NA}   ${acc0:-}"
echo "[done]"
