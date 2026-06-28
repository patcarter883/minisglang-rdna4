#!/usr/bin/env bash
# MLA spec-decode CUDA-graph validation on GLM-4.7-Flash (EAGLE3, TP=2). Runs INSIDE
# vllm22-w4a8:combined under a 2-card lease. Boots the EAGLE3 spec serve EAGER (--graph 0) then
# GRAPH-captured (--graph 8) and checks:
#   1. the verify graphs actually capture (boot banner "spec-verify graphs captured");
#   2. CORRECTNESS: graph output == eager output BYTE-FOR-BYTE (the graph just replays the same
#      verify forward — precomputed verify indices == _verify_indices — so it must match exactly);
#   3. UPLIFT: graph tokens/sec > eager (the eager per-step launch latency over the 47-layer MLA+MoE
#      verify forward is collapsed to one replay).
set -uo pipefail
source /app/.venv/bin/activate
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
python -c "import mla_hip, moe_hip, tail_hip, swiglu_hip; print('[setup] hip pkgs OK')" || { echo FAIL; exit 1; }

MODEL="${MODEL:-QuantTrio/GLM-4.7-Flash-AWQ}"; PORT=21971
DRAFT="${DRAFT:-thoughtworks/GLM-4.7-Flash-Eagle3}"; K="${K:-6}"
LOG=/engine/tools/mla_spec_graph.server.log
SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1 = --graph value
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 MINISGL_SPEC_DEBUG=1 \
    MINISGL_SPEC_TIMING=1 MINISGL_SPEC_PREFILL_SEED=1 python -m minisgl \
    --model "$MODEL" --tensor-parallel-size 2 --port $PORT --graph "$1" --disable-pynccl \
    --memory-ratio 0.80 --max-running-requests 4 \
    --spec-algorithm eagle3 --spec-draft-model-path "$DRAFT" --spec-num-draft "$K" > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 400); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "[boot] DIED:"; tail -40 "$LOG"; return 1; }; sleep 3
  done; echo "[boot] timeout:"; tail -40 "$LOG"; return 1
}
probe(){ PORT=$PORT OUT=$1 python - <<'PY'
import json,os,time,urllib.request
PORT,OUT=os.environ["PORT"],os.environ["OUT"]
cohere=["The capital of France is","Q: What is 17 times 4? A:","Write one sentence about the ocean."]
res=[]
for p in cohere:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":64,"messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    res.append(json.load(urllib.request.urlopen(r,timeout=180))["choices"][0]["message"]["content"])
    print(f"  <<< {res[-1][:90]!r}")
# timed tok/s (batch=1, 3x256 after warmup)
tp="Explain in detail how a binary search tree works, including insertion, lookup, and traversal."
def gen():
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":256,"messages":[{"role":"user","content":tp}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t=time.time(); d=json.load(urllib.request.urlopen(r,timeout=300)); el=time.time()-t
    n=(d.get("usage") or {}).get("completion_tokens") or len(d["choices"][0]["message"]["content"].split())
    return n, el
gen(); ns=els=0.0
for _ in range(3): n,e=gen(); ns+=n; els+=e
print(f"  TOKS_PER_S={ns/els:.2f}")
json.dump(res,open(OUT,"w"))
PY
}

echo "===== EAGER (--graph 0) ====="
boot 0; echo "[eager] coherence + tok/s:"; probe /engine/tools/mla_spec_graph.eager.json
grep -E "\[spec\]|\[spec-timing\]" "$LOG" | tail -3; stop

echo "===== GRAPH (--graph 8) ====="
boot 8; echo "[graph] verify-capture banner:"; grep -iE "spec-verify graphs captured|Capturing spec-verify" "$LOG" | head -2
echo "[graph] coherence + tok/s:"; probe /engine/tools/mla_spec_graph.graph.json
grep -E "\[spec\]|\[spec-timing\]" "$LOG" | tail -3; stop

echo "===== compare (graph vs eager) ====="
python - /engine/tools/mla_spec_graph.eager.json /engine/tools/mla_spec_graph.graph.json <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); ok=all(x==y for x,y in zip(a,b))
for i,(x,y) in enumerate(zip(a,b)): print(f"  prompt[{i}]: {'MATCH' if x==y else 'DIFF'} ({len(x)} vs {len(y)})")
print("GRAPH==EAGER (lossless replay):", "PASS" if ok else "FAIL")
PY
echo "[done]"
