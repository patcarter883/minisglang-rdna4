#!/usr/bin/env bash
# Rigorous losslessness check for the spec-decode accept/commit/rollback logic. Greedy spec decode
# must reproduce sequential greedy decode THROUGH THE SAME (verify) kernel — so we compare:
#   SPEC      : multi-token accept (real speculative decoding)
#   FORCE_N0  : same staging+verify forward, but commit only the bonus (1 token/step)
# Both use the extend/verify kernel, so any kernel-vs-baseline numerics cancel out. Identical output
# proves the acceptance + KV rollback is lossless. Prompts are repetitive to stress multi-token accepts.
set -uo pipefail
source /app/.venv/bin/activate
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"; PORT=21937; LOG=/engine/tools/spec_lossless.server.log
SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1=extra env assignment
  setsid env PYTHONPATH=/engine/python:/engine $1 python -m minisgl \
    --model "$MODEL" --tensor-parallel-size 1 --port $PORT --graph 0 --memory-ratio 0.8 \
    --max-running-requests 4 --spec-algorithm ngram --spec-num-draft 6 --spec-ngram-max 3 > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 200); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo DIED; tail -30 "$LOG"; exit 1; }; sleep 3
  done; echo "not ready"; tail -40 "$LOG"; exit 1
}
probe(){ PORT=$PORT OUT=$1 python - <<'PY'
import json,os,urllib.request
PORT,OUT=os.environ["PORT"],os.environ["OUT"]
prompts=[
 "Repeat exactly five times: the cat sat on the mat.",
 "Count: one two three four five six seven eight nine ten one two three four five",
 "List: apple banana cherry apple banana cherry apple banana cherry apple banana",
 "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the",
 "a b c d e f g a b c d e f g a b c d e f g a b c d e f g a b c",
]
res=[]
for p in prompts:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":128,"messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    res.append(json.load(urllib.request.urlopen(r,timeout=180))["choices"][0]["message"]["content"])
json.dump(res,open(OUT,"w"))
PY
}

echo "===== run SPEC (multi-token accept) ====="
boot "MINISGL_SPEC_DEBUG=1"; probe /engine/tools/ll.spec.json
echo "[spec] acceptance:"; grep -E "\[spec\]" "$LOG" | tail -3; stop

echo "===== run FORCE_N0 (1 token/step, same verify kernel) ====="
boot "MINISGL_SPEC_FORCE_N0=1"; probe /engine/tools/ll.n0.json; stop

echo "===== DIFF spec vs n0 (must be identical -> accept/rollback lossless) ====="
python - /engine/tools/ll.spec.json /engine/tools/ll.n0.json <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); ok=True
for i,(x,y) in enumerate(zip(a,b)):
    m=x==y; ok&=m; print(f"  prompt[{i}]: {'MATCH' if m else 'DIFF'}  (len {len(x)} vs {len(y)})")
    if not m:
        for j,(cx,cy) in enumerate(zip(x,y)):
            if cx!=cy: print(f"    diverge@{j}: spec={x[max(0,j-20):j+20]!r} n0={y[max(0,j-20):j+20]!r}"); break
print("\nACCEPT/ROLLBACK LOSSLESS:", "PASS" if ok else "FAIL")
PY
echo "[done]"
