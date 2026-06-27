#!/usr/bin/env bash
# Runs INSIDE vllm22-w4a8:combined under a 1-card lease. Validates the TARGET HIDDEN-STATE CAPTURE
# seam (Engine.forward_verify(return_hidden=True) + *ForCausalLM.forward(return_hidden=True) +
# set_capture_layers) on an MHA model (Qwen3-0.6B), the foundation for MTP/EAGLE3/DFlash draft heads.
#
#   1. PLAIN-SPEC serve (--spec-algorithm ngram, capture OFF) -> greedy outputs (the reference: this
#      already runs through the verify kernel, so it is the correct losslessness oracle — NOT a plain
#      -decode baseline, whose fp differs from the verify kernel per SPEC_DECODE.md).
#   2. CAPTURE-PROBE serve: --spec-algorithm ngram with MINISGL_SPEC_CAPTURE_PROBE=<layer ids>. This
#      swaps in _CaptureProbeProposer, which drafts EXACTLY like n-gram (so output is unchanged) but
#      ALSO declares needs_last_hidden + capture_layer_ids. The engine then runs forward_verify with
#      return_hidden=True and feeds last_hidden [hidden] / aux_hidden [num_layers, hidden] back into
#      ProposeContext; the probe asserts those shapes and prints [capture-probe] lines.
#   3. Asserts: (a) capture output == plain-spec output (turning capture ON must NOT change drafting —
#      the verify forward returns the same logits whether or not hidden is also returned); (b) at
#      least one [capture-probe] OK line was logged (the seam actually delivered hidden states).
set -uo pipefail
source /app/.venv/bin/activate

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PORT="${PORT:-21931}"
MEMRATIO="${MEMRATIO:-0.80}"
MAXRUN="${MAXRUN:-8}"
NUM_DRAFT="${NUM_DRAFT:-4}"
NGRAM_MAX="${NGRAM_MAX:-3}"
CAPTURE_LAYERS="${CAPTURE_LAYERS:-0,13,27}"   # 3 decoder-layer ids (EAGLE3-style aux capture)
OUTDIR=/engine/tools
LOG="$OUTDIR/spec_capture.server.log"

echo "[setup] server deps ..."
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1

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
    "Continue the pattern: 2 4 6 8 10 12 14 16 18 20 22 24",
    "Repeat this sentence exactly three times: The quick brown fox jumps over the lazy dog.",
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

echo "===== PLAIN-SPEC (ngram, capture OFF) — losslessness reference ====="
boot plainspec env python -m minisgl \
  --spec-algorithm ngram --spec-num-draft "$NUM_DRAFT" --spec-ngram-max "$NGRAM_MAX"
probe "$OUTDIR/spec_capture.baseline.json"
stop

echo "===== CAPTURE-PROBE (ngram + hidden-state capture, layers=$CAPTURE_LAYERS) ====="
boot capture env MINISGL_SPEC_DEBUG=1 MINISGL_SPEC_CAPTURE_PROBE="$CAPTURE_LAYERS" python -m minisgl \
  --spec-algorithm ngram --spec-num-draft "$NUM_DRAFT" --spec-ngram-max "$NGRAM_MAX"
probe "$OUTDIR/spec_capture.spec.json"
echo "[capture-log] probe shape assertions:"
grep -E "\[capture-probe\]" "$LOG" | head -8 || echo "  (NONE — seam delivered NO hidden states!)"
NPROBE=$(grep -cE "\[capture-probe\].*OK" "$LOG" || true)
stop

echo "===== DIFF (capture output must match plain-spec, both via the verify kernel) ====="
python - "$OUTDIR/spec_capture.baseline.json" "$OUTDIR/spec_capture.spec.json" "$NPROBE" <<'PY'
import json, sys
b = json.load(open(sys.argv[1])); s = json.load(open(sys.argv[2])); nprobe = int(sys.argv[3])
ok = True
for i, (x, y) in enumerate(zip(b, s)):
    match = x == y
    ok &= match
    print(f"  prompt[{i}]: {'MATCH' if match else 'DIFF'}")
    if not match:
        for j, (cx, cy) in enumerate(zip(x, y)):
            if cx != cy:
                print(f"    diverge@char {j}: base={x[max(0,j-20):j+20]!r}  spec={y[max(0,j-20):j+20]!r}")
                break
        else:
            print(f"    len base={len(x)} spec={len(y)} (one is a prefix of the other)")
print(f"\nCAPTURE PROBE OK lines: {nprobe}")
print("CAPTURE-NEUTRAL:", "PASS (capture ON == capture OFF output)" if ok else "MISMATCH")
print("CAPTURE SEAM:", "PASS (hidden states delivered + shapes valid)" if nprobe > 0 else "FAIL (no hidden states)")
print("\nOVERALL:", "PASS" if (ok and nprobe > 0) else "FAIL")
PY
echo "[done]"
