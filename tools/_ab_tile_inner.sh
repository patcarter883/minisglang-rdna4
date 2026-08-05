#!/usr/bin/env bash
# In-container leg of the dense-tile/act-quant SERVE A/B. Boots the production serve config once
# (native HIP attention + cudagraph capture), asserts PROVENANCE (which kernels/engine are actually
# loaded), then runs tools/ab_tile_serve_bench.py for --reps repeats.
#
# Byte-identical in both legs' worktrees. What differs is only the image (/opt/kernels) and the
# mounted /engine.
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate

MODEL="${MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"
TP="${TP:-2}"; PORT="${PORT:-1919}"; MEMRATIO="${MEMRATIO:-0.82}"
MAXRUN="${MAXRUN:-24}"; GRAPH="${GRAPH:-16}"; ATTN="${ATTN:-hip}"
LEG="${LEG:-unknown}"; REPS="${REPS:-3}"
# PIN the KV pool. Left to auto-sizing the pool is whatever VRAM survives the model + the
# recurrent-radix/GDN/graph reservations, which on a 16 GB card is the SCRAPS (~0.02 GiB) — and the
# two legs' kernel packages differ in code-object footprint by enough to swing that remainder 42%
# (2448 vs 1424 servable tokens, measured). A different pool changes which requests are ADMITTED,
# so the A/B would compare admission policy, not GEMM cost. --num-pages makes the pool a constant of
# the experiment; at page_size=1 it is exactly a token count.
NUM_PAGES="${NUM_PAGES:-16384}"
# --no-gdn-radix. The recurrent-radix snapshot store CLONES per-request GDN state during serving,
# from the torch allocator, on top of its own accounting reservation — so it grows with request
# history and it killed both Qwen legs at the same point in rep 3 with
# `torch.OutOfMemoryError ... gdn_state.clone_slot` once the KV pool was pinned. Two reasons it must
# be off here, not just tuned around: it is a stochastic, history-dependent allocator inside a
# measurement of GEMM cost, and this bench deliberately cache-BUSTS every prompt, so the store can
# never register a hit anyway. Off, it is pure removed variance. Inert for MLA (GLM).
OUT="/engine/tools/_ab_tile_out"; mkdir -p "$OUT"

echo "######## PROVENANCE [$LEG] ########"
python - <<'PY'
import hashlib, os, sys
sys.path.insert(0, "/opt/kernels")
import fp8_wmma, tail_hip
print("[prov] fp8_wmma.dense_tile_explain :", hasattr(fp8_wmma, "dense_tile_explain"))
print("[prov] fp8_wmma.moe_tile_choose    :", hasattr(fp8_wmma, "moe_tile_choose"))
print("[prov] tail_hip.rms_norm_quant     :", hasattr(tail_hip, "rms_norm_quant"))
for p in ["/opt/rdna4-hip-kernels/fp8_wmma/fp8_wmma_rocm/w4a8_fp8_wmma_kernel.hip",
          "/opt/rdna4-hip-kernels/fp8_wmma/fp8_wmma_rocm/moe_kernel.hip",
          "/opt/rdna4-hip-kernels/tail/tail_rocm/tail_kernels.hip"]:
    h = hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "MISSING"
    print(f"[prov] md5 {h}  {os.path.basename(p)}")
print("[prov] tile_select.h present       :",
      os.path.exists("/opt/rdna4-hip-kernels/fp8_wmma/fp8_wmma_rocm/tile_select.h"))
import minisgl.quant.kernels as K
print("[prov] engine _pick_dense_kernel arms:",
      sorted({l.split('"')[1] for l in open(K.__file__) if l.strip().startswith('return "')}))
import minisgl.layers.norm as N
print("[prov] engine RMSNormFused.forward_quant:", hasattr(N.RMSNormFused, "forward_quant"))
PY
echo "###################################"

python -c "import gdn_hip, moe_hip, tail_hip, mla_hip, fp8_wmma; print('[setup] hip pkgs import OK')" \
  || { echo '[setup] hip pkg import FAILED'; exit 1; }

LOG="$OUT/${LEG}.server.log"
pynccl=""; [ "$TP" -gt 1 ] && pynccl="--disable-pynccl"
echo "[launch] $LEG attn=$ATTN graph=$GRAPH tp=$TP -> $LOG"
setsid python -m minisgl \
  --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" --host 0.0.0.0 --graph "$GRAPH" \
  --attention-backend "$ATTN" $pynccl --memory-ratio "$MEMRATIO" \
  --num-pages "$NUM_PAGES" --max-running-requests "$MAXRUN" --no-gdn-radix > "$LOG" 2>&1 &
SRV=$!
stop() {
  [ -n "${SRV:-}" ] || return 0
  kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""
}
trap stop EXIT

ready=0
for _ in $(seq 1 400); do
  if python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null; then
    ready=1; echo "[launch] ready"; break
  fi
  kill -0 "$SRV" 2>/dev/null || { echo "[launch] server DIED:"; tail -40 "$LOG"; exit 1; }
  if grep -qE '^Process minisgl-|torch\.OutOfMemoryError|RuntimeError: No HIP GPUs|CUDA error:' "$LOG" 2>/dev/null; then
    echo "[launch] worker CRASHED:"; tail -40 "$LOG"; exit 1
  fi
  sleep 3
done
[ "$ready" = 1 ] || { echo "[launch] not ready:"; tail -40 "$LOG"; exit 1; }
grep -iE 'captur|cuda.?graph' "$LOG" | head -3

python /engine/tools/ab_tile_serve_bench.py --url "http://127.0.0.1:$PORT" --label "$LEG" \
  --reps "$REPS" --out "$OUT/${LEG}.json"

# Which dense/MoE kernel arms the SERVE actually dispatched (the engine's `engaged()` ledger), so
# the A/B can say the arm changed, not just that the number changed.
grep -oE 'fp8_wmma\.[a-z_0-9]+\([a-z_0-9+]+' "$LOG" | sort | uniq -c | sort -rn | head -20
stop
echo "[done] $LEG"
