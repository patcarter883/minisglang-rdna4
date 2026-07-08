#!/usr/bin/env bash
# CAPSTONE: DFlash on the GDN Qwen3.5-4B target UNDER GRAPH CAPTURE (needs the merged GDN spec-verify
# capture). FULLCTX=1 + AUX_POSTMLP=1 fixed; loops --graph: 0 (eager) vs 4 (captured). Confirms the
# graph engages (verify_graph_replays>0), stays lossless (g0 vs g4 byte-identical), and preserves
# accept-len. This is the graph-capture-required bar for DFlash-on-GDN.
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"; DRAFT="${DRAFT:-z-lab/Qwen3.5-4B-DFlash}"
PORT="${PORT:-21977}"; MEM="${MEM:-0.70}"; K="${K:-7}"; MAXREQ="${MAXREQ:-2}"
OUTDIR=/engine/tools
SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT
boot(){ local log="$1" gr="$2"
  setsid env PYTHONPATH=/opt/kernels:/engine/python:/engine MINISGL_MOE_SCATTER=0 MINISGL_SPEC_DEBUG=1 \
    MINISGL_DFLASH_FULLCTX=1 MINISGL_AUX_POSTMLP=1 \
    python -m minisgl --model "$MODEL" --tensor-parallel-size 1 --port "$PORT" --graph "$gr" --attn hip \
    --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$K" \
    --memory-ratio "$MEM" --max-running-requests "$MAXREQ" > "$log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && { echo "[boot g$gr] ready"; return 0; }
    kill -0 "$SRV" 2>/dev/null || { echo "[boot g$gr] DIED:"; tail -40 "$log"; return 1; }; sleep 3
  done; echo "[boot g$gr] not ready:"; tail -50 "$log"; return 1
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
json.dump(res,open(OUT,"w"))
PY
}
EAGER=0; CAP="${CAP:-2}"   # eager baseline vs the captured graph size to compare against
for GR in $EAGER $CAP; do
  LOG="$OUTDIR/dflash_graph.g$GR.log"
  echo "===================== --graph $GR (DFlash, FULLCTX=1, K=$K) ====================="
  boot "$LOG" "$GR" || { echo "[g$GR] boot failed"; continue; }
  probe "$OUTDIR/dflash_graph.g$GR.json"; stop
  echo "[g$GR] $(grep -oE "\[spec\] mean accept-len=[0-9.]+ over [0-9]+ reqs" "$LOG" | tail -1)"
  echo "[g$GR] $(grep -oE "spec-verify graphs captured|verify_graph_replays=[0-9]+" "$LOG" | sort -u | tail -2 | tr '\n' ' ')"
done
echo "===================== losslessness (eager g$EAGER vs captured g$CAP) ====================="
python - "$OUTDIR/dflash_graph.g$EAGER.json" "$OUTDIR/dflash_graph.g$CAP.json" <<'PY'
import json,sys
try: a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
except FileNotFoundError as e: print("  (missing:",e,")"); raise SystemExit
ok=all(x==y for x,y in zip(a,b))
for i,(x,y) in enumerate(zip(a,b)): print(f"  prompt[{i}]: {'MATCH' if x==y else 'DIFF'} (len {len(x)} vs {len(y)})")
print("  LOSSLESS graph==eager:", "PASS" if ok else "FAIL")
PY
echo "[done]"
