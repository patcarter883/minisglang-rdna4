#!/usr/bin/env bash
# Runs INSIDE vllm22-w4a8:combined. Boots GLM-4.7-Flash (glm4_moe_lite, AWQ) on TP=2 with the
# engine-forced MLA backend (do NOT pass --attention-backend: is_mla forces 'mla' + page_size 16),
# EAGER (--graph 0; MLA cudagraph capture is not implemented yet), probes a few greedy chat
# completions for coherence, then kills the whole server process group. CPU-validated already
# (glm4_moe_lite_build_smoke.py, TP=1 + TP=2). Needs a 2-card lease.
set -uo pipefail
source /app/.venv/bin/activate

MODEL="${MODEL:-QuantTrio/GLM-4.7-Flash-AWQ}"
TP="${TP:-2}"
PORT="${PORT:-21919}"
MEMRATIO="${MEMRATIO:-0.85}"
MAXRUN="${MAXRUN:-8}"
GRAPH="${GRAPH:-0}"                 # cuda_graph_max_bs; 0 = eager, >0 = capture decode graphs
MMS="${MOE_SCATTER:-0}"            # MINISGL_MOE_SCATTER: 0 = graph-safe gather_reduce MoE decode
KV_FP8="${KV_FP8:-0}"             # 1 -> MINISGL_KV_FP8 (e4m3 latent KV cache, ~2x tokens)
MAXSEQ="${MAXSEQ:-}"              # --max-seq-len-override (max context per request)
LOG=/engine/tools/glm_smoke.server.log
maxseq_flag=""; [ -n "$MAXSEQ" ] && maxseq_flag="--max-seq-len-override $MAXSEQ"

echo "[setup] server deps ..."
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
PYTHONPATH=/engine/python:/engine python -c "import mla_hip, moe_hip, tail_hip, swiglu_hip; print('[setup] hip pkgs import OK')" \
  || { echo '[setup] hip pkg import FAILED'; exit 1; }

SRV=""
stop() {
  [ -n "$SRV" ] || return 0
  echo "[stop] terminating server process group -$SRV ..."
  kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null
  wait "$SRV" 2>/dev/null; SRV=""
}
trap stop EXIT

pynccl=""; [ "$TP" -gt 1 ] && pynccl="--disable-pynccl"
mode="eager"; [ "$GRAPH" -gt 0 ] && mode="graph(max_bs=$GRAPH, moe_scatter=$MMS)"
kvf=""; [ "$KV_FP8" = "1" ] && kvf="MINISGL_KV_FP8=1"
echo "[launch] $MODEL TP=$TP $mode (mla) kv_fp8=$KV_FP8 maxseq=${MAXSEQ:-default} memratio=$MEMRATIO -> $LOG"
# NO --attention-backend (is_mla forces 'mla'). --graph 0 = eager; >0 captures the decode graph
# (MoE-decode uses the graph-safe gather_reduce path via MINISGL_MOE_SCATTER).
setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER="$MMS" $kvf python -m minisgl \
  --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" --graph "$GRAPH" \
  $pynccl --memory-ratio "$MEMRATIO" --max-running-requests "$MAXRUN" $maxseq_flag \
  > "$LOG" 2>&1 &
SRV=$!
ready=0
for _ in $(seq 1 300); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null; then
    ready=1; echo "[launch] ready"; break
  fi
  kill -0 "$SRV" 2>/dev/null || { echo "[launch] server PID $SRV DIED:"; tail -50 "$LOG"; exit 1; }
  sleep 3
done
[ "$ready" = 1 ] || { echo "[launch] NOT ready in time:"; tail -60 "$LOG"; exit 1; }

echo "[serve-log] backend / page_size / KV alloc / capture:"
grep -iE "overrid|page.?size|Allocating .* KV|CUDA graph|Capturing|captur" "$LOG" | head -10 || true

# Concurrency report: KV pool tokens / max context = how many full-context requests fit at once.
KVTOK=$(grep -oE "Allocating [0-9]+ tokens" "$LOG" | head -1 | grep -oE "[0-9]+")
if [ -n "$KVTOK" ] && [ -n "$MAXSEQ" ]; then
  echo "===== CONCURRENCY (kv_fp8=$KV_FP8, max_context=$MAXSEQ) ====="
  echo "  KV pool: $KVTOK tokens   |   max context/req: $MAXSEQ"
  python - "$KVTOK" "$MAXSEQ" "$MAXRUN" <<'PY'
import sys
kvtok, ctx, maxrun = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
by_mem = kvtok // ctx
print(f"  concurrent full-context requests (KV-limited): {by_mem}")
print(f"  --max-running-requests cap: {maxrun}")
print(f"  => effective concurrency at {ctx}-token context: {min(by_mem, maxrun)}")
print(f"  (at shorter avg context the scheduler admits more; KV is shared dynamically)")
PY
fi

echo "===== coherence probe (greedy) ====="
PORT="$PORT" python - <<'PY'
import json, os, urllib.request
PORT = os.environ["PORT"]
prompts = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue.",
    "Q: What is 17 multiplied by 4? A:",
    "Write a haiku about the ocean.",
]
ok = True
for p in prompts:
    body = json.dumps({
        "model": "glm", "temperature": 0.0, "max_tokens": 64,
        "messages": [{"role": "user", "content": p}],
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        out = json.load(urllib.request.urlopen(req, timeout=120))
        txt = out["choices"][0]["message"]["content"]
    except Exception as e:
        txt = f"<ERROR: {e!r}>"; ok = False
    print(f"\n>>> {p}\n<<< {txt!r}")
print("\nPROBE:", "RESPONDED" if ok else "ERROR")
PY
echo "[done]"
