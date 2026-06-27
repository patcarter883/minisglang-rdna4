#!/usr/bin/env bash
# Controls whether the GDN spec VERIFY forward is bit-exact vs plain decode at qlen=K+1, on the SAME
# 5 prompts the DFlash run uses. The DFlash run diverged on prompts 0,4 even at num_draft=5; this
# pins whether that is a GDN-verify property (ngram FORCE_N0 diverges identically) or DFlash-specific.
#   A. baseline (spec off)
#   B. ngram FORCE_N0 num_draft=5  (qlen=6 verify, accept none -> must == baseline if GDN verify exact)
#   C. ngram FORCE_N0 num_draft=3  (qlen=4 verify; smaller block)
set -uo pipefail
source /app/.venv/bin/activate
mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"; PORT="${PORT:-21945}"; MEM="${MEM:-0.72}"
LOG=/engine/tools/spec_gdn_qlen.server.log; OUTDIR=/engine/tools
SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT
boot(){ local label="$1"; shift
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 GDN_HIP_WMMA_PREFILL="${WMMA:-1}" "$@" \
    --model "$MODEL" --tensor-parallel-size 1 --port "$PORT" --graph 0 --attn hip \
    --memory-ratio "$MEM" --max-running-requests 4 > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && { echo "[launch:$label] ready"; return 0; }
    kill -0 "$SRV" 2>/dev/null || { echo "[launch:$label] DIED:"; tail -40 "$LOG"; exit 1; }; sleep 3
  done; echo "[$label] not ready"; tail -40 "$LOG"; exit 1; }
probe(){ PORT=$PORT OUT=$1 python - <<'PY'
import json,os,urllib.request
PORT,OUT=os.environ["PORT"],os.environ["OUT"]
prompts=["The capital of France is","Q: What is 17 multiplied by 4? A:",
 "Repeat exactly five times: the cat sat on the mat.",
 "List: apple banana cherry apple banana cherry apple banana cherry apple banana",
 "Count up: one two three four five six seven eight nine ten one two three four five"]
res=[]
for p in prompts:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":96,"messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    res.append(json.load(urllib.request.urlopen(r,timeout=240))["choices"][0]["message"]["content"])
json.dump(res,open(OUT,"w"))
PY
}
diffb(){ python - "$OUTDIR/q.base.json" "$1" "$2" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); ok=True
for i,(x,y) in enumerate(zip(a,b)):
    m=x==y; ok&=m; print(f"  [{sys.argv[3]}] prompt[{i}]: {'MATCH' if m else 'DIFF'}")
print(f"  [{sys.argv[3]}] LOSSLESS:", "PASS" if ok else "MISMATCH")
PY
}
echo "===== A. baseline ====="; boot base env MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl; probe "$OUTDIR/q.base.json"; stop
echo "===== B. ngram FORCE_N0 num_draft=5 (qlen=6) ====="
boot ng5 env MINISGL_SPEC_FORCE_N0=1 python -m minisgl --spec-algorithm ngram --spec-num-draft 5 --spec-ngram-max 3
probe "$OUTDIR/q.ng5.json"; stop; diffb "$OUTDIR/q.ng5.json" ngram5-N0
echo "===== C. ngram FORCE_N0 num_draft=3 (qlen=4) ====="
boot ng3 env MINISGL_SPEC_FORCE_N0=1 python -m minisgl --spec-algorithm ngram --spec-num-draft 3 --spec-ngram-max 3
probe "$OUTDIR/q.ng3.json"; stop; diffb "$OUTDIR/q.ng3.json" ngram3-N0
echo "[done]"
