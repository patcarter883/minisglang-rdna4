#!/usr/bin/env bash
# Spec-decode validation on a GDN-HYBRID model (Qwen3.5-4B, qwen3_5: 3-in-4 linear-attention +
# 1-in-4 full-attention, TP=1). Runs INSIDE vllm22-w4a8:combined under a 1-card lease. Exercises the
# GDN per-token-state verify kernel (gdn_prefill_verify / causal_conv1d_fwd_verify): the verify
# captures conv+ssm state after each token; the scheduler installs the accepted-prefix state directly
# (no snapshot, no 2x re-advance). Verifies:
#   1. coherence with spec on;
#   2. BIT-EXACTNESS: SPEC (multi-token accept) == FORCE_N0 (1 token/step) — both go through the GDN
#      per-token-state verify path (bit-stable recurrent kernels, independent of GDN_HIP_WMMA_PREFILL),
#      so identical output proves the captured-state install is exact (a wrong install would corrupt
#      the stream). This is now BIT-IDENTICAL (was only "coherent, fp-drifting" under re-advance).
set -uo pipefail
source /app/.venv/bin/activate
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
PYTHONPATH=/engine/python:/engine python -c \
  "import gdn_hip, moe_hip, tail_hip, attn_decode, attn_hip, attn_prefill_paged; print('[setup] hip pkgs OK')" \
  || { echo '[setup] hip import FAILED'; exit 1; }

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"; PORT=21941; LOG=/engine/tools/spec_gdn.server.log
SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1 = extra env assignment
  # GDN_HIP_WMMA_PREFILL=${WMMA:-1}: only affects the PROMPT prefill kernel. The spec VERIFY now
  # uses the dedicated per-token-state recurrent kernel (gdn_prefill_verify) regardless of this flag,
  # so spec==n0 is bit-exact even at WMMA=1 (the old re-advance path needed WMMA=0 to be bit-stable).
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 GDN_HIP_WMMA_PREFILL="${WMMA:-1}" $1 python -m minisgl \
    --model "$MODEL" --tensor-parallel-size 1 --port $PORT --graph 0 --attn hip \
    --memory-ratio 0.85 --max-running-requests 4 \
    --spec-algorithm ngram --spec-num-draft 5 --spec-ngram-max 3 > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "[boot] DIED:"; tail -40 "$LOG"; exit 1; }; sleep 3
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
    res.append(txt); print(f"\n>>> {p[:55]!r}\n<<< {txt[:150]!r}")
json.dump(res,open(OUT,"w"))
PY
}

echo "===== SPEC (GDN per-token-state verify + accepted-prefix install) ====="
boot "MINISGL_SPEC_DEBUG=1"; probe /engine/tools/gdn.spec.json
echo "[spec] acceptance:"; grep -E "\[spec\]" "$LOG" | tail -3; stop

echo "===== FORCE_N0 (1 token/step, same GDN per-token-state verify) ====="
boot "MINISGL_SPEC_FORCE_N0=1"; probe /engine/tools/gdn.n0.json; stop

echo "===== DIFF spec vs n0 (must match -> GDN state rollback lossless) ====="
python - /engine/tools/gdn.spec.json /engine/tools/gdn.n0.json <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); ok=True
for i,(x,y) in enumerate(zip(a,b)):
    m=x==y; ok&=m; print(f"  prompt[{i}]: {'MATCH' if m else 'DIFF'}  (len {len(x)} vs {len(y)})")
    if not m:
        for j,(cx,cy) in enumerate(zip(x,y)):
            if cx!=cy: print(f"    diverge@{j}: spec={x[max(0,j-20):j+20]!r} n0={y[max(0,j-20):j+20]!r}"); break
print("\nGDN STATE ROLLBACK LOSSLESS:", "PASS" if ok else "FAIL")
PY
echo "[done]"
