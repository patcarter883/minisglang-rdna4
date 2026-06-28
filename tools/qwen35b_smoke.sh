#!/usr/bin/env bash
# Qwen3.6-35B-A3B-AWQ (compressed-tensors W4A16 MoE) coherence smoke — INSIDE vllm22-w4a8:combined,
# TP=2. Validates the new compressed-tensors expert path end-to-end: boots the base model (no spec),
# generates a few prompts, prints the text. Garbage => the int4 unpack/packing-order is wrong;
# coherent => the W4A16 MoE adapter is correct. Then a quick MTP boot (the sweep's draft head).
set -uo pipefail
source /app/.venv/bin/activate
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
python -c "import gdn_hip, moe_hip, tail_hip, mla_hip; print('[setup] hip pkgs OK')" || { echo FAIL; exit 1; }

MODEL="${MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"; PORT=21961; TP="${TP:-2}"
LOG=/engine/tools/qwen35b_smoke.server.log
SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1 = extra spec args
  local pynccl=""; [ "$TP" -gt 1 ] && pynccl="--disable-pynccl"
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 $2 python -m minisgl \
    --model "$MODEL" --tensor-parallel-size "$TP" --port $PORT --graph "${GRAPH:-0}" $pynccl \
    --memory-ratio 0.82 --max-running-requests 4 --attention-backend hip $1 > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 400); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "[boot] DIED:"; tail -40 "$LOG"; return 1; }; sleep 3
  done; echo "[boot] timeout:"; tail -40 "$LOG"; return 1
}
probe(){ PORT=$PORT python - <<'PY'
import json,os,urllib.request
PORT=os.environ["PORT"]
for p in ["The capital of France is","Q: What is 17 times 4? A:",
          "Write one sentence about the ocean.","List three primary colors:"]:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":48,
                     "messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,
                             headers={"Content-Type":"application/json"})
    try:
        txt=json.load(urllib.request.urlopen(r,timeout=180))["choices"][0]["message"]["content"]
    except Exception as e:
        txt=f"<ERROR {e}>"
    print(f"\n>>> {p!r}\n<<< {txt[:200]!r}")
PY
}

echo "===== Qwen35B base (no spec) coherence ====="
boot "" "" && probe; stop
echo
echo "===== Qwen35B MTP (--spec-algorithm mtp --spec-num-draft 3) ====="
boot "--spec-algorithm mtp --spec-num-draft 3" "MINISGL_SPEC_DEBUG=1" && probe
echo "[mtp] accept:"; grep -E "\[spec\]" "$LOG" | tail -2; stop
echo "[done]"
