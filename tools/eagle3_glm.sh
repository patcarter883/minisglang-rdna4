#!/usr/bin/env bash
# EAGLE3 draft-model spec-decode validation on GLM-4.7-Flash (glm4_moe_lite, AWQ, MLA) at TP=2. Runs
# INSIDE vllm22-w4a8:combined under a 2-card lease. The SEPARATE EAGLE3 draft checkpoint
# (thoughtworks/GLM-4.7-Flash-Eagle3 — a 1-layer Llama GQA head over 3 captured GLM aux layers,
# compressed 32000 draft vocab) is loaded and run as a LINEAR CHAIN draft (--spec-algorithm eagle3
# --spec-draft-model-path ...). Verifies:
#   1. COHERENCE with EAGLE3 drafting on (absorbed multi-query mla_verify backbone verify);
#   2. acceptance rate (EAGLE3 should accept high);
#   3. (optional, RUN_N0=1) losslessness: SPEC == FORCE_N0 (1 token/step, same verify kernel).
# Eager (--graph 0), MLA backend auto-forced, page_size stays 16.
set -uo pipefail
source /app/.venv/bin/activate
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
PYTHONPATH=/engine/python:/engine python -c "import mla_hip, moe_hip, tail_hip, swiglu_hip; print('[setup] hip pkgs OK')" \
  || { echo '[setup] hip import FAILED'; exit 1; }

MODEL="${MODEL:-QuantTrio/GLM-4.7-Flash-AWQ}"
DRAFT="${DRAFT:-thoughtworks/GLM-4.7-Flash-Eagle3}"
PORT=21947; LOG=/engine/tools/eagle3_glm.server.log
SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1 = extra env assignment
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 $1 python -m minisgl \
    --model "$MODEL" --tensor-parallel-size 2 --port $PORT --graph 0 --disable-pynccl \
    --memory-ratio 0.85 --max-running-requests 4 \
    --spec-algorithm eagle3 --spec-draft-model-path "$DRAFT" \
    --spec-num-draft "${KDRAFT:-5}" > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "[boot] DIED:"; tail -60 "$LOG"; exit 1; }; sleep 3
  done; echo "[boot] not ready:"; tail -60 "$LOG"; exit 1
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

echo "===== EAGLE3 SPEC (GLM, separate draft checkpoint as linear chain) ====="
boot "MINISGL_SPEC_DEBUG=1"; probe /engine/tools/eagle3_glm.spec.json
echo "[eagle3] acceptance:"; grep -E "\[spec\]" "$LOG" | tail -3
grep -iE "eagle|draft" "$LOG" | grep -iv "spec-algorithm\|spec-draft" | head -6; stop

echo "===== (optional) FORCE_N0 reference (1 token/step) ====="
if [ "${RUN_N0:-0}" = "1" ]; then
  boot "MINISGL_SPEC_FORCE_N0=1"; probe /engine/tools/eagle3_glm.n0.json; stop
  python - /engine/tools/eagle3_glm.spec.json /engine/tools/eagle3_glm.n0.json <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); ok=True
for i,(x,y) in enumerate(zip(a,b)):
    m=x==y; ok&=m; print(f"  prompt[{i}]: {'MATCH' if m else 'DIFF'}  (len {len(x)} vs {len(y)})")
print("\nEAGLE3 ACCEPT/ROLLBACK LOSSLESS:", "PASS" if ok else "FAIL")
PY
fi
echo "[done]"
