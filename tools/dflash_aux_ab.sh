#!/usr/bin/env bash
# DFlash aux-capture A/B (in-container, LEAN). FULLCTX=1 fixed; loops MINISGL_AUX_POSTMLP: 0 = legacy
# residual-only capture (missing the captured layer's MLP add), 1 = full post-layer stream (x+residual
# = z-lab hidden_states[lid+1], what the drafter was trained on). Reports [spec] mean accept-len +
# ap0-vs-ap1 losslessness. Tests whether the aux representation was the acceptance slack.
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"; DRAFT="${DRAFT:-z-lab/Qwen3.5-4B-DFlash}"
PORT="${PORT:-21966}"; MEM="${MEM:-0.72}"; K="${K:-15}"; GRAPH="${GRAPH:-0}"; MAXREQ="${MAXREQ:-4}"
OUTDIR=/engine/tools
SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT
boot(){ local log="$1"; shift
  setsid env PYTHONPATH=/opt/kernels:/engine/python:/engine MINISGL_MOE_SCATTER=0 MINISGL_SPEC_DEBUG=1 \
    MINISGL_DFLASH_FULLCTX=1 "$@" \
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
prompts=["Explain in a short paragraph why the sky appears blue during the day.",
 "Write a brief step-by-step recipe for making a simple omelette.",
 "Summarize the plot of Romeo and Juliet in three sentences.",
 "What is the difference between a list and a tuple in Python? Answer concisely.",
 "Describe the water cycle in a few sentences.","Give three tips for writing clear commit messages."]
res=[]
for p in prompts:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":200,"messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    res.append(json.load(urllib.request.urlopen(r,timeout=300))["choices"][0]["message"]["content"])
json.dump(res,open(OUT,"w")); print("  sample[0]:", res[0][:140].replace("\n"," "))
PY
}
for AP in 0 1; do
  LOG="$OUTDIR/dflash_aux.ap$AP.log"
  echo "===================== MINISGL_AUX_POSTMLP=$AP (FULLCTX=1, K=$K) ====================="
  boot "$LOG" env MINISGL_AUX_POSTMLP=$AP || { echo "[ap$AP] boot failed"; continue; }
  probe "$OUTDIR/dflash_aux.ap$AP.json"; stop
  echo "[ap$AP] $(grep -oE "\[spec\] mean accept-len=[0-9.]+ over [0-9]+ reqs" "$LOG" | tail -1)"
done
echo "===================== losslessness (ap0 vs ap1 text) ====================="
python - "$OUTDIR/dflash_aux.ap0.json" "$OUTDIR/dflash_aux.ap1.json" <<'PY'
import json,sys
try: a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
except FileNotFoundError as e: print("  (missing:",e,")"); raise SystemExit
for i,(x,y) in enumerate(zip(a,b)): print(f"  prompt[{i}]: {'MATCH' if x==y else 'DIFF'} (len {len(x)} vs {len(y)})")
PY
echo "[done]"
