#!/usr/bin/env bash
# DFlash block-diffusion spec-decode validation on the GDN-HYBRID target Qwen3.5-4B (qwen3_5).
# Draft = z-lab/Qwen3.5-4B-DFlash (tied-vocab, 6 Qwen3 GQA layers, 8 captured target layers, block=16).
# Runs INSIDE vllm22-w4a8:combined under a 1-card lease.
#
# Validates:
#   1. COHERENCE with DFlash spec on (block-parallel drafting + linear verify_greedy accept);
#   2. LOSSLESSNESS: BASELINE (spec off) == DFLASH (block drafts, greedy accept). verify_greedy is
#      lossless regardless of draft quality, so identical greedy text proves the block-diffusion
#      proposer + capture/aux plumbing + GDN per-token-state verify rollback are all correct.
#   3. acceptance/tokens-per-step (MINISGL_SPEC_DEBUG=1) — DFlash claims high block acceptance.
# Both run --graph 0 (eager) so the only delta is the spec verify path; --attn hip (capture-capable).
set -uo pipefail
source /app/.venv/bin/activate
# Isolated writable Triton cache copy (never corrupt the shared production cache). --attn hip is
# Triton-free for the hot path, so this only covers any incidental JIT.
mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
PYTHONPATH=/engine/python:/engine python -c \
  "import gdn_hip, moe_hip, tail_hip, attn_decode, attn_hip, attn_prefill_paged; print('[setup] hip pkgs OK')" \
  || { echo '[setup] hip import FAILED'; exit 1; }

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
DRAFT="${DRAFT:-z-lab/Qwen3.5-4B-DFlash}"
PORT="${PORT:-21943}"
NUM_DRAFT="${NUM_DRAFT:-7}"
LOG=/engine/tools/spec_dflash.server.log
OUTDIR=/engine/tools

SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1=label  $2=memratio  $3...=extra env+args verbatim
  local label="$1"; local memratio="$2"; shift 2
  echo "[launch:$label] (memratio=$memratio) $* -> $LOG"
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 GDN_HIP_WMMA_PREFILL="${WMMA:-1}" "$@" \
    --model "$MODEL" --tensor-parallel-size 1 --port "$PORT" --graph 0 --attn hip \
    --memory-ratio "$memratio" --max-running-requests 4 > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && { echo "[launch:$label] ready"; return 0; }
    kill -0 "$SRV" 2>/dev/null || { echo "[launch:$label] DIED:"; tail -50 "$LOG"; exit 1; }; sleep 3
  done; echo "[launch:$label] not ready:"; tail -60 "$LOG"; exit 1
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

# DFlash boots a SEPARATE ~0.9GB draft trunk AFTER the engine grabs the KV cache, so the verify
# serve runs a lower memory-ratio to leave room (target ~8GB + draft ~0.9GB on a 16GB card). The
# baseline matches that ratio so KV-cache size (and thus any prefix behaviour) is identical.
MEMRATIO_BASE="${MEMRATIO_BASE:-0.85}"
MEMRATIO_DFLASH="${MEMRATIO_DFLASH:-0.72}"

echo "===== BASELINE (spec off, eager) ====="
boot baseline "$MEMRATIO_DFLASH" env MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl
probe "$OUTDIR/dflash.baseline.json"; stop

echo "===== DFLASH (block-diffusion, num_draft=$NUM_DRAFT) ====="
boot dflash "$MEMRATIO_DFLASH" env MINISGL_SPEC_DEBUG=1 python -m minisgl \
  --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$NUM_DRAFT"
probe "$OUTDIR/dflash.spec.json"
echo "[dflash] acceptance:"; grep -E "\[spec\]" "$LOG" | tail -5 || echo "  (no [spec] lines)"; stop

echo "===== DIFF baseline vs dflash (greedy must match -> lossless) ====="
python - "$OUTDIR/dflash.baseline.json" "$OUTDIR/dflash.spec.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); ok=True
for i,(x,y) in enumerate(zip(a,b)):
    m=x==y; ok&=m; print(f"  prompt[{i}]: {'MATCH' if m else 'DIFF'}  (len {len(x)} vs {len(y)})")
    if not m:
        for j,(cx,cy) in enumerate(zip(x,y)):
            if cx!=cy: print(f"    diverge@{j}: base={x[max(0,j-20):j+20]!r} dflash={y[max(0,j-20):j+20]!r}"); break
        else: print(f"    (one is a prefix of the other)")
print("\nDFLASH SPEC LOSSLESSNESS:", "PASS" if ok else "MISMATCH")
PY
echo "[done]"
