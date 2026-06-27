#!/usr/bin/env bash
# Native MTP self-speculation validation on Qwen3.5-4B (qwen3_5 GDN-hybrid, TP=1). Runs INSIDE
# vllm22-w4a8:combined under a 1-card lease. The model's appended next-token head (mtp.* — a single
# FULL-ATTENTION layer + fc + pre-norms, reusing the target embed + tied lm_head) is loaded and run
# autoregressively as the draft (--spec-algorithm mtp). The MTP draft is full attention (NOT GDN),
# so it never touches GDN state; the backbone verify over the 32-layer GDN stack uses the
# per-token-state verify kernel (already lossless). Verifies:
#   1. COHERENCE with MTP drafting on;
#   2. acceptance rate (MTP should accept FAR more than n-gram's ~17%).
set -uo pipefail
source /app/.venv/bin/activate
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
PYTHONPATH=/engine/python:/engine python -c \
  "import gdn_hip, moe_hip, tail_hip, attn_decode, attn_hip, attn_prefill_paged; print('[setup] hip pkgs OK')" \
  || { echo '[setup] hip import FAILED'; exit 1; }

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"; PORT=21945; LOG=/engine/tools/mtp_qwen35.server.log
SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1 = extra env assignment
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 GDN_HIP_WMMA_PREFILL="${WMMA:-1}" $1 python -m minisgl \
    --model "$MODEL" --tensor-parallel-size 1 --port $PORT --graph 0 --attn hip \
    --memory-ratio 0.85 --max-running-requests 4 \
    --spec-algorithm mtp --spec-num-draft "${KDRAFT:-4}" > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "[boot] DIED:"; tail -50 "$LOG"; exit 1; }; sleep 3
  done; echo "[boot] not ready:"; tail -50 "$LOG"; exit 1
}
probe(){ PORT=$PORT OUT=$1 python - <<'PY'
import json,os,urllib.request
PORT,OUT=os.environ["PORT"],os.environ["OUT"]
prompts=[
 "The capital of France is",
 "Q: What is 17 multiplied by 4? A:",
 "Repeat exactly five times: the cat sat on the mat.",
 "List: apple banana cherry apple banana cherry apple banana cherry apple banana",
 "Count up: one two three four five six seven eight nine ten one two three four five",
]
res=[]
for p in prompts:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":96,"messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    txt=json.load(urllib.request.urlopen(r,timeout=240))["choices"][0]["message"]["content"]
    res.append(txt); print(f"\n>>> {p[:55]!r}\n<<< {txt[:160]!r}")
json.dump(res,open(OUT,"w"))
PY
}

echo "===== MTP SPEC (Qwen3.5-4B, native next-token head as draft) ====="
boot "MINISGL_SPEC_DEBUG=1"; probe /engine/tools/mtp_qwen35.spec.json
echo "[mtp] acceptance:"; grep -E "\[spec\]" "$LOG" | tail -3; stop

if [ "${RUN_N0:-0}" = "1" ]; then
  echo "===== FORCE_N0 reference (1 token/step) ====="
  boot "MINISGL_SPEC_FORCE_N0=1"; probe /engine/tools/mtp_qwen35.n0.json; stop
  python - /engine/tools/mtp_qwen35.spec.json /engine/tools/mtp_qwen35.n0.json <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); ok=True
for i,(x,y) in enumerate(zip(a,b)):
    m=x==y; ok&=m; print(f"  prompt[{i}]: {'MATCH' if m else 'DIFF'}  (len {len(x)} vs {len(y)})")
print("\nMTP ACCEPT/ROLLBACK LOSSLESS:", "PASS" if ok else "FAIL")
PY
fi
echo "[done]"
