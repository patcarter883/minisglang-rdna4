#!/usr/bin/env bash
# INNER script (runs inside minisgl-rdna4:lean). Boots the dense Qwen3.6-27B GDN-hybrid
# (compressed-tensors W4A16, quantized GDN in_proj) at TP=2 and probes coherence. Launch via
# tools/run_q27_lean.sh under a 2-card lease.
set -uo pipefail
source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
python -c "import gdn_hip, moe_hip, tail_hip, attn_decode, attn_hip, attn_prefill_paged, w4a8_fp8_wmma; print('[setup] hip pkgs OK')" \
  || { echo '[setup] hip import FAILED'; exit 1; }

MODEL="${MODEL:-cyankiwi/Qwen3.6-27B-AWQ-INT4}"; PORT="${PORT:-21962}"; LOG=/engine/tools/q27_validate.server.log
SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

echo "[boot] starting $MODEL TP=${TP:-2} attn=${ATTN:-hip} graph=${GRAPH:-0}"
setsid env PYTHONPATH=/opt/kernels:/engine/python:/engine MINISGL_MOE_SCATTER=0 \
  python -m minisgl --model "$MODEL" --tensor-parallel-size "${TP:-2}" --port "$PORT" \
  --graph "${GRAPH:-0}" --attn "${ATTN:-hip}" --dtype "${DTYPE:-bfloat16}" \
  --memory-ratio "${MEMRATIO:-0.82}" \
  --max-running-requests "${MAXRUN:-8}" > "$LOG" 2>&1 &
SRV=$!
for _ in $(seq 1 400); do
  python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && break
  kill -0 "$SRV" 2>/dev/null || { echo "[boot] DIED:"; tail -60 "$LOG"; exit 1; }; sleep 3
done
kill -0 "$SRV" 2>/dev/null || { echo "[boot] not ready:"; tail -60 "$LOG"; exit 1; }
echo "[boot] ready"; echo "[vram]"; rocm-smi --showmeminfo vram 2>/dev/null | grep -iE "GPU\[|Used" | head

PORT=$PORT python - <<'PY'
import json,os,urllib.request
PORT=os.environ["PORT"]
prompts=[
 "The capital of France is",
 "What is 17 multiplied by 24? /no_think",
 "Write a short two-line poem about the sea. /no_think",
 "List the first five prime numbers. /no_think",
]
for p in prompts:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":420,
                     "messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,
                             headers={"Content-Type":"application/json"})
    try:
        txt=json.load(urllib.request.urlopen(r,timeout=240))["choices"][0]["message"]["content"]
    except Exception as e:
        txt=f"<ERROR {e}>"
    print(f"\n>>> {p!r}\n<<< {txt!r}")
PY
echo; echo "[vram-after]"; rocm-smi --showmeminfo vram 2>/dev/null | grep -iE "GPU\[|Used" | head
stop
