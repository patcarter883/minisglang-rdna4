#!/usr/bin/env bash
# DFlash full-context A/B (in-container, LEAN image). Boots the z-lab Qwen3.5-4B DFlash drafter on the
# Qwen3.5-4B target TWICE — legacy 1-token aux feed (MINISGL_DFLASH_FULLCTX=0, expect ~0.33 accept-len)
# vs the full-context fix (=1) — under MINISGL_SPEC_DEBUG=1, and reports [spec] mean accept-len + the
# generated text (coherence + fc0-vs-fc1 losslessness spot-check). Eager (--graph 0).
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || true
# /opt/kernels FIRST (canonical HIP kernels), then the mounted engine source. APPEND, don't override.
export PYTHONPATH=/opt/kernels:/engine/python:/engine

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
DRAFT="${DRAFT:-z-lab/Qwen3.5-4B-DFlash}"
PORT="${PORT:-21955}"; MEM="${MEM:-0.72}"; K="${K:-15}"
# Representative serving config: low concurrency UNDER GRAPH CAPTURE (the graph-capture-required rule;
# eager numbers are not the served config). --graph N captures up to bs N.
GRAPH="${GRAPH:-4}"; MAXREQ="${MAXREQ:-4}"
OUTDIR=/engine/tools

SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ local log="$1"; shift
  setsid env PYTHONPATH=/opt/kernels:/engine/python:/engine MINISGL_MOE_SCATTER=0 MINISGL_SPEC_DEBUG=1 "$@" \
    python -m minisgl --model "$MODEL" --tensor-parallel-size 1 --port "$PORT" --graph "$GRAPH" --attn hip \
    --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$K" \
    --memory-ratio "$MEM" --max-running-requests "$MAXREQ" > "$log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && { echo "[boot] ready"; return 0; }
    kill -0 "$SRV" 2>/dev/null || { echo "[boot] DIED:"; tail -40 "$log"; return 1; }; sleep 3
  done; echo "[boot] not ready:"; tail -50 "$log"; return 1
}
probe(){ PORT=$PORT OUT=$1 python - <<'PY'
import json,os,urllib.request
PORT,OUT=os.environ["PORT"],os.environ["OUT"]
prompts=[
 "Explain in a short paragraph why the sky appears blue during the day.",
 "Write a brief step-by-step recipe for making a simple omelette.",
 "Summarize the plot of Romeo and Juliet in three sentences.",
 "What is the difference between a list and a tuple in Python? Answer concisely.",
 "Describe the water cycle in a few sentences.",
 "Give three tips for writing clear commit messages.",
]
res=[]
for p in prompts:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":200,"messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    res.append(json.load(urllib.request.urlopen(r,timeout=300))["choices"][0]["message"]["content"])
json.dump(res,open(OUT,"w"))
print("  sample[0]:", res[0][:160].replace("\n"," "))
PY
}

for FC in 0 1; do
  LOG="$OUTDIR/dflash_ab.fc$FC.log"
  echo "===================== MINISGL_DFLASH_FULLCTX=$FC (K=$K) ====================="
  boot "$LOG" env MINISGL_DFLASH_FULLCTX=$FC || { echo "[fc$FC] boot failed"; continue; }
  probe "$OUTDIR/dflash_ab.fc$FC.json"
  stop
  al=$(grep -oE "\[spec\] mean accept-len=[0-9.]+ over [0-9]+ reqs" "$LOG" | tail -1)
  echo "[fc$FC] ${al:-<no [spec] accept-len logged — increase max_tokens/prompts>}"
done

echo "===================== losslessness (fc0 vs fc1 text) ====================="
python - "$OUTDIR/dflash_ab.fc0.json" "$OUTDIR/dflash_ab.fc1.json" <<'PY'
import json,sys
try:
    a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
except FileNotFoundError as e:
    print("  (missing output:", e, ")"); raise SystemExit
for i,(x,y) in enumerate(zip(a,b)):
    print(f"  prompt[{i}]: {'MATCH' if x==y else 'DIFF'} (len {len(x)} vs {len(y)})")
PY
echo "[done]"
