#!/usr/bin/env bash
# MoE atomic-scatter decode coherence smoke — runs INSIDE minisgl-rdna4:lean, launched by
# run_bench_window.sh's container recipe (see the invocation in docs/CONTINUANCE_opt_loop.md).
#
# WHY THIS EXISTS SEPARATELY FROM THE BENCH
# The fused decode gemm2 (mmq_fp8_moe_gemm_scatter / moe_gemm_splitk_scatter) reduces the top_k
# experts with atomicAdd into a shared fp32 (M,K). That makes the result ORDER-DEPENDENT: the sum is
# fp32-associative-only, so it is NOT bit-exact vs the gather_reduce path and cannot be gated as
# such. A tok/s table cannot see this — 128 tokens of fluent garbage counts as 128 tokens at 0 fails.
# So the scatter path must be judged by reading the generated text, at the SAME --graph setting the
# benchmark used (capture is exactly what was claimed impossible here).
#
#   MMS=1 GRAPH=16 bash tools/moe_scatter_coherence.sh
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate

MODEL="${MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"
TP="${TP:-2}"
PORT="${PORT:-21962}"
GRAPH="${GRAPH:-16}"
# Same names run_bench_window.sh already forwards, so a coherence run and a bench run are configured
# identically (INNER=/engine/tools/moe_scatter_coherence.sh is the only difference).
MMS="${MOE_SCATTER:-1}"                  # MINISGL_MOE_SCATTER
SPLITK="${MINISGL_MOE_SPLITK:-}"         # >=2 routes the M==1 scatter to moe_splitk_hip
LOG=/engine/tools/moe_scatter_coherence.server.log

SRV=""
stop() { [ -n "$SRV" ] || return 0
  kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

echo "[boot] scatter=$MMS splitk=${SPLITK:-0} graph=$GRAPH tp=$TP"
local_pynccl=""; [ "$TP" -gt 1 ] && local_pynccl="--disable-pynccl"
setsid env MINISGL_MOE_SCATTER="$MMS" MINISGL_MOE_SPLITK="$SPLITK" python -m minisgl \
  --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" --graph "$GRAPH" $local_pynccl \
  --memory-ratio 0.82 --max-running-requests 4 --attention-backend hip > "$LOG" 2>&1 &
SRV=$!
for _ in $(seq 1 400); do
  python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && break
  kill -0 "$SRV" 2>/dev/null || { echo "[boot] DIED:"; tail -40 "$LOG"; exit 1; }
  sleep 3
done
python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null \
  || { echo "[boot] timeout:"; tail -40 "$LOG"; exit 1; }

# Which decode gemm2 actually ran. The [hip-engage] line is the only proof the scatter was reached
# rather than silently falling through to gather_reduce (the M<=2 gate is easy to miss).
echo "[engaged]"; grep -E "hip-engage.*(scatter|gather_reduce)" "$LOG" | sort -u
echo "[capture]"; grep -icE "capturing (cuda )?graph" "$LOG" | sed 's/^/  capture log lines: /'

PORT=$PORT python - <<'PY'
import json, os, urllib.request
PORT = os.environ["PORT"]
PROMPTS = [
    "The capital of France is",
    "Q: What is 17 times 4? A:",
    "Write one sentence about the ocean.",
    "Count from 1 to 10, separated by commas.",
    "Explain in two sentences why matrix multiplication is memory-bandwidth bound at batch size 1.",
]
for p in PROMPTS:
    body = json.dumps({"model": "m", "temperature": 0.0, "max_tokens": 64,
                       "messages": [{"role": "user", "content": p}]}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                              headers={"Content-Type": "application/json"})
    try:
        txt = json.load(urllib.request.urlopen(r, timeout=180))["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        txt = f"<ERROR {e}>"
    print(f"\n>>> {p!r}\n<<< {txt[:300]!r}", flush=True)
PY
stop
echo "[done] server log: $LOG"
