#!/usr/bin/env bash
# DFlash losslessness DIAGNOSTIC. The main run showed COHERENT but NOT bit-identical output on a GDN
# target. verify_greedy is lossless by construction, so a mismatch means the VERIFY FORWARD's
# per-position argmax (or the GDN per-token-state install) is not bit-stable for the draft batch
# composition. This isolates where:
#   A. baseline                    (spec off, reference)
#   B. dflash FORCE_N0 num_draft=7 (stage+verify 7 drafts, accept NONE -> must == baseline; if it
#                                   DIFFS, the bug is the verify forward / GDN state at qlen=8)
#   C. ngram FORCE_N0 num_draft=7  (same qlen=8 verify, but n-gram drafts; controls for DFlash)
#   D. ngram accept  num_draft=7   (n-gram full accept at qlen=8; ngram was bit-exact at num_draft=5)
set -uo pipefail
source /app/.venv/bin/activate
mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
PYTHONPATH=/engine/python:/engine python -c \
  "import gdn_hip, moe_hip, tail_hip, attn_decode, attn_hip, attn_prefill_paged; print('[setup] hip pkgs OK')" \
  || { echo '[setup] hip import FAILED'; exit 1; }

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
DRAFT="${DRAFT:-z-lab/Qwen3.5-4B-DFlash}"
PORT="${PORT:-21944}"; MEM="${MEM:-0.72}"; LOG=/engine/tools/spec_dflash_diag.server.log
OUTDIR=/engine/tools

SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ local label="$1"; shift
  echo "[launch:$label] $* -> $LOG"
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 GDN_HIP_WMMA_PREFILL="${WMMA:-1}" "$@" \
    --model "$MODEL" --tensor-parallel-size 1 --port "$PORT" --graph 0 --attn hip \
    --memory-ratio "$MEM" --max-running-requests 4 > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && { echo "[launch:$label] ready"; return 0; }
    kill -0 "$SRV" 2>/dev/null || { echo "[launch:$label] DIED:"; tail -40 "$LOG"; exit 1; }; sleep 3
  done; echo "[launch:$label] not ready:"; tail -50 "$LOG"; exit 1
}
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
diff_to_base(){ python - "$OUTDIR/diag.base.json" "$1" "$2" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); ok=True
for i,(x,y) in enumerate(zip(a,b)):
    m=x==y; ok&=m
    print(f"  [{sys.argv[3]}] prompt[{i}]: {'MATCH' if m else 'DIFF'}")
print(f"  [{sys.argv[3]}] LOSSLESS:", "PASS" if ok else "MISMATCH")
PY
}

echo "===== A. baseline ====="
boot base env MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl
probe "$OUTDIR/diag.base.json"; stop

echo "===== B. dflash FORCE_N0 num_draft=7 ====="
boot dfn0 env MINISGL_SPEC_FORCE_N0=1 python -m minisgl \
  --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft 7
probe "$OUTDIR/diag.dfn0.json"; stop
diff_to_base "$OUTDIR/diag.dfn0.json" dflash-N0

echo "===== C. ngram FORCE_N0 num_draft=7 ====="
boot ngn0 env MINISGL_SPEC_FORCE_N0=1 python -m minisgl \
  --spec-algorithm ngram --spec-num-draft 7 --spec-ngram-max 3
probe "$OUTDIR/diag.ngn0.json"; stop
diff_to_base "$OUTDIR/diag.ngn0.json" ngram-N0

echo "===== D. ngram accept num_draft=7 ====="
boot ngacc env MINISGL_SPEC_DEBUG=1 python -m minisgl \
  --spec-algorithm ngram --spec-num-draft 7 --spec-ngram-max 3
probe "$OUTDIR/diag.ngacc.json"; stop
diff_to_base "$OUTDIR/diag.ngacc.json" ngram-accept
echo "[done]"
