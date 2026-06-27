#!/usr/bin/env bash
# Runs INSIDE vllm22-w4a8:combined under a 1-card lease. Validates the n-gram speculative-decode
# MVP on an MHA model (Qwen3-0.6B, head_dim 128, Triton-free native-HIP path):
#   1. boots a BASELINE serve (spec OFF, eager, overlap disabled) and records greedy outputs;
#   2. boots a SPEC serve (--spec-algorithm ngram, eager sync loop) and records greedy outputs
#      + acceptance stats (MINISGL_SPEC_DEBUG=1);
#   3. diffs the two — greedy spec decode must reproduce baseline text.
# Both run --graph 0 (eager) so the only delta is the spec verify path.
set -uo pipefail
source /app/.venv/bin/activate

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-21929}"
MEMRATIO="${MEMRATIO:-0.80}"
MAXRUN="${MAXRUN:-8}"
NUM_DRAFT="${NUM_DRAFT:-4}"
NGRAM_MAX="${NGRAM_MAX:-3}"
OUTDIR=/engine/tools
LOG="$OUTDIR/spec_smoke.server.log"

echo "[setup] server deps ..."
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
PYTHONPATH=/engine/python:/engine python -c \
  "import attn_decode, attn_hip, attn_prefill_paged, tail_hip; print('[setup] MHA hip pkgs import OK')" \
  || { echo '[setup] hip pkg import FAILED'; exit 1; }

SRV=""
stop() {
  [ -n "$SRV" ] || return 0
  kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null
  wait "$SRV" 2>/dev/null; SRV=""
}
trap stop EXIT

boot() {  # $1 = label, $2... = extra env+args passed verbatim to the launcher
  local label="$1"; shift
  echo "[launch:$label] $* -> $LOG"
  setsid env PYTHONPATH=/engine/python:/engine "$@" \
    --model "$MODEL" --tensor-parallel-size 1 --port "$PORT" --graph 0 \
    --memory-ratio "$MEMRATIO" --max-running-requests "$MAXRUN" \
    > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 200); do
    if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null; then
      echo "[launch:$label] ready"; return 0
    fi
    kill -0 "$SRV" 2>/dev/null || { echo "[launch:$label] DIED:"; tail -40 "$LOG"; exit 1; }
    sleep 3
  done
  echo "[launch:$label] NOT ready:"; tail -60 "$LOG"; exit 1
}

probe() {  # $1 = output json path
  PORT="$PORT" OUT="$1" python - <<'PY'
import json, os, urllib.request
PORT, OUT = os.environ["PORT"], os.environ["OUT"]
prompts = [
    "The capital of France is",
    "Q: What is 17 multiplied by 4? A:",
    "Repeat this sentence exactly three times: The quick brown fox jumps over the lazy dog.",
    "Continue the pattern: 2 4 6 8 10 12 14 16 18 20 22 24",
    "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n    return",
]
res = []
for p in prompts:
    body = json.dumps({"model": "m", "temperature": 0.0, "max_tokens": 96,
                       "messages": [{"role": "user", "content": p}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"]["content"]
    res.append(txt)
    print(f"\n>>> {p[:60]!r}\n<<< {txt[:160]!r}")
json.dump(res, open(OUT, "w"))
PY
}

echo "===== BASELINE (spec off, eager, overlap disabled) ====="
boot baseline env MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl
probe "$OUTDIR/spec_smoke.baseline.json"
stop

echo "===== SPEC (ngram, num_draft=$NUM_DRAFT, ngram_max=$NGRAM_MAX) ====="
boot spec env MINISGL_SPEC_DEBUG=1 python -m minisgl \
  --spec-algorithm ngram --spec-num-draft "$NUM_DRAFT" --spec-ngram-max "$NGRAM_MAX"
probe "$OUTDIR/spec_smoke.spec.json"
echo "[spec-log] acceptance stats:"
grep -E "\[spec\]" "$LOG" | tail -8 || echo "  (no [spec] lines — drafts may never have been accepted)"
grep -iE "overrid|page.?size|spec" "$LOG" | grep -iE "page.?size|spec|overrid" | head -6 || true
stop

echo "===== DIFF (greedy spec must match baseline) ====="
python - "$OUTDIR/spec_smoke.baseline.json" "$OUTDIR/spec_smoke.spec.json" <<'PY'
import json, sys
b = json.load(open(sys.argv[1])); s = json.load(open(sys.argv[2]))
ok = True
for i, (x, y) in enumerate(zip(b, s)):
    match = x == y
    ok &= match
    print(f"  prompt[{i}]: {'MATCH' if match else 'DIFF'}")
    if not match:
        # show first divergence
        for j, (cx, cy) in enumerate(zip(x, y)):
            if cx != cy:
                print(f"    diverge@char {j}: base={x[max(0,j-20):j+20]!r}  spec={y[max(0,j-20):j+20]!r}")
                break
        else:
            print(f"    len base={len(x)} spec={len(y)} (one is a prefix of the other)")
print("\nSPEC LOSSLESSNESS:", "PASS (identical greedy output)" if ok else "MISMATCH")
PY
echo "[done]"
