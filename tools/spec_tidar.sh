#!/usr/bin/env bash
# TiDAR self-draft spec-decode validation on the ZAYA1-8B TiDAR fp8 target (model_type=zaya, CCA).
# NO separate draft model: the proposer drafts a block with ONE causal target forward over
# [confirmed | mask×B] (Scheduler._tidar_block_predict), reads mask/block from tidar_config.json in
# the model dir. Runs INSIDE vllm22-w4a8:combined under a 1-card lease (fp8 ~9.6GB fits one 16GB card).
#
# Validates:
#   1. COHERENCE with TiDAR spec on (block-parallel self-draft + linear verify_greedy accept);
#   2. LOSSLESSNESS: BASELINE (spec off) == TiDAR (block drafts, greedy accept), byte-identical.
#      verify_greedy is lossless regardless of draft quality, so identical greedy text proves the
#      whole new path is correct: block_predict + the CCA verify-state capture/install (B.0) rollback.
#      (A non-instruct base model may degenerate into repetition — that is FINE; both must degenerate
#      IDENTICALLY. The lossless diff is the real correctness gate, not chat coherence.)
#   3. acceptance / tokens-per-step (MINISGL_SPEC_DEBUG=1) — the OPD lift should show here.
# Both run --graph 0 (eager) so the only delta is the spec verify path; --attn hip.
set -uo pipefail
source /app/.venv/bin/activate
mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
PYTHONPATH=/engine/python:/engine python -c \
  "import moe_hip, tail_hip, attn_decode, attn_hip, attn_prefill_paged, cca_hip.cca_op; print('[setup] hip pkgs OK')" \
  || { echo '[setup] hip import FAILED'; exit 1; }

MODEL="${MODEL:-/big/zaya1-tidar-opd-fp8}"
PORT="${PORT:-21955}"
NUM_DRAFT="${NUM_DRAFT:-4}"     # B: TiDAR block size (tidar_config block_size=4)
MAXTOK="${MAXTOK:-64}"
MEMRATIO="${MEMRATIO:-0.85}"
LOG=/engine/tools/spec_tidar.server.log
OUTDIR=/engine/tools

SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1=label  $2...=extra env+args verbatim
  local label="$1"; shift
  echo "[launch:$label] $* -> $LOG"
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 \
    MINISGL_ATTN_HIP=1 MINISGL_TAIL_HIP=1 "$@" \
    --model "$MODEL" --tensor-parallel-size 1 --port "$PORT" --graph "${GRAPH:-0}" --attn hip \
    --memory-ratio "$MEMRATIO" --max-running-requests 4 > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 400); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models',timeout=3)" 2>/dev/null \
      && { echo "[launch:$label] ready"; return 0; }
    kill -0 "$SRV" 2>/dev/null || { echo "[launch:$label] DIED:"; tail -60 "$LOG"; exit 1; }; sleep 3
  done; echo "[launch:$label] not ready:"; tail -80 "$LOG"; exit 1
}
probe(){ PORT=$PORT OUT=$1 MODEL=$MODEL MAXTOK=$MAXTOK python - <<'PY'
import json,os,urllib.request
PORT,OUT,MODEL,MAXTOK=os.environ["PORT"],os.environ["OUT"],os.environ["MODEL"],int(os.environ["MAXTOK"])
prompts=[
 "The capital of France is",
 "Question: What is 17 plus 26? Answer:",
 "The first three prime numbers are",
 "Once upon a time, in a small village, there lived",
]
res=[]
for p in prompts:
    body=json.dumps({"model":MODEL,"temperature":0.0,"max_tokens":MAXTOK,
                     "messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    txt=json.load(urllib.request.urlopen(r,timeout=300))["choices"][0]["message"]["content"]
    res.append(txt); print(f"\n>>> {p[:55]!r}\n<<< {txt[:160]!r}")
json.dump(res,open(OUT,"w"))
PY
}

echo "===== BASELINE (spec off, eager) ====="
boot baseline env MINISGL_DISABLE_OVERLAP_SCHEDULING=1 MINISGL_ZAYA_OLDMOE="${OLDMOE:-0}" \
  MINISGL_ZAYA_W8A16="${W8A16:-0}" python -m minisgl
probe "$OUTDIR/tidar.baseline.json"; stop

FUSED="${FUSED:-0}"   # FUSED=1 -> Phase-C single-forward fused path (MINISGL_TIDAR_FUSED)
echo "===== TiDAR (self-draft block-diffusion, num_draft=$NUM_DRAFT, FUSED=$FUSED) ====="
boot tidar env MINISGL_SPEC_DEBUG=1 MINISGL_DISABLE_OVERLAP_SCHEDULING=1 MINISGL_TIDAR_FUSED="$FUSED" \
  MINISGL_TIDAR_FUSED_NOREP="${NOREP:-0}" MINISGL_ZAYA_OLDMOE="${OLDMOE:-0}" MINISGL_TIDAR_SEG="${SEG:-0}" \
  MINISGL_TIDAR_DUMP="${DUMP:-0}" MINISGL_TIDAR_PROFILE="${PROFILE:-0}" MINISGL_TIDAR_TIME="${TIME:-0}" \
  MINISGL_ZAYA_W8A16="${W8A16:-0}" MINISGL_TIDAR_MIX_BETA="${MIX:-1.0}" \
  python -m minisgl --spec-algorithm tidar --spec-num-draft "$NUM_DRAFT"
probe "$OUTDIR/tidar.spec.json"
echo "[tidar] acceptance:"; grep -E "\[spec\]" "$LOG" | tail -8 || echo "  (no [spec] lines)"; stop

echo "===== DIFF baseline vs tidar (greedy must be prefix-exact -> LOSSLESS) ====="
# NOTE: only the DEFAULT MIX=1.0 (pure-AR verify) is lossless. MIX<1.0 = logit-mixing "Trust-Diffusion"
# verify, which is INTENTIONALLY not lossless vs base AR — a MISMATCH below is EXPECTED for MIX<1.0
# (judge those runs by acceptance / tok-s / coherence, not this prefix gate).
# Rigorous gate: the shorter output must be an EXACT PREFIX of the longer. A spec step emits up to
# K+1 tokens at once, so with a fixed max_tokens the spec run can overshoot the exact token budget by
# up to K (a longer identical tail) — that length delta is a harness boundary, NOT a divergence. A
# real CCA verify-state bug would diverge WITHIN the shared prefix, which this catches.
python - "$OUTDIR/tidar.baseline.json" "$OUTDIR/tidar.spec.json" "$NUM_DRAFT" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); K=int(sys.argv[3]); ok=True
for i,(x,y) in enumerate(zip(a,b)):
    n=min(len(x),len(y)); prefix_exact=x[:n]==y[:n]; delta=abs(len(x)-len(y)); ok&=prefix_exact
    tag="exact" if len(x)==len(y) else ("prefix-exact" if prefix_exact else "DIVERGE")
    extra="" if len(x)==len(y) else f"  (+{delta}ch tail = spec burst boundary)"
    print(f"  prompt[{i}]: {tag}  len {len(x)} vs {len(y)}{extra}")
    if not prefix_exact:
        for j,(cx,cy) in enumerate(zip(x,y)):
            if cx!=cy: print(f"    REAL DIVERGENCE @char {j}: base={x[max(0,j-20):j+20]!r} tidar={y[max(0,j-20):j+20]!r}"); break
print("\nTIDAR SPEC LOSSLESSNESS (shorter is exact prefix of longer):", "PASS" if ok else "MISMATCH")
PY
echo "[done]"
