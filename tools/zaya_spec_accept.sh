#!/usr/bin/env bash
# Container-side accept-len + losslessness harness for ZAYA spec-decode (DFlash OR TiDAR).
# Generic: the two wrappers (run_zaya_dflash_accept.sh / run_zaya_tidar_accept.sh) set the env and
# docker-run this on minisgl-rdna4:lean with the LIVE kernels repo mounted at /kernels, so the run
# exercises whatever batch-invariant verify kernels are currently in /home/pat/code/rdna4-hip-kernels.
#
# THE MEASUREMENT: mean accept-len / emitted-per-step (MINISGL_SPEC_DEBUG=1) + greedy losslessness
# (baseline spec-off text must be a byte-exact prefix of the spec-on text). accept-len is the number
# that the verify-M determinism bug depresses; this harness is how we read the delta once the
# batch-invariant kernels land. Runs eager by default (GRAPH=0) so the ONLY variable is the verify
# path; set GRAPH=8 for the prod tok/s follow-up after accept-len recovers.
#
# Env contract (set by the wrapper):
#   MODEL       target checkpoint dir (RXF for dflash, fp8 TiDAR for tidar)
#   SPEC_ALGO   dflash | tidar
#   DRAFT       DFlash drafter ckpt dir (dflash only; empty for tidar self-draft)
#   NUM_DRAFT   block / draft width (dflash: match the drafter's trained num_spec; tidar: block_size)
#   GRAPH       cuda-graph-max-bs (0 = eager, default; 8 = prod perf pass)
#   MEMRATIO    KV memory ratio (leave headroom for the DFlash draft trunk)
#   GENTOK      max_tokens per probe generation
#   KV_FP8 MOE_SCATTER CACHE   serve knobs mirroring the deployed RXF serve
set -uo pipefail
source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/kernels/_kernels:/engine/python:/engine

MODEL="${MODEL:?set MODEL}"; SPEC_ALGO="${SPEC_ALGO:?set SPEC_ALGO}"
DRAFT="${DRAFT:-}"; NUM_DRAFT="${NUM_DRAFT:-4}"; GRAPH="${GRAPH:-0}"
MEMRATIO="${MEMRATIO:-0.85}"; GENTOK="${GENTOK:-96}"; PORT="${PORT:-21977}"
KV_FP8="${KV_FP8:-1}"; MOE_SCATTER="${MOE_SCATTER:-0}"; CACHE="${CACHE:-naive}"
LOG=/engine/tools/zaya_spec_accept.server.log
OUTDIR=/engine/tools

echo "[accept] kernels HEAD: $(git -C /kernels rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[accept] SPEC_ALGO=$SPEC_ALGO MODEL=$MODEL DRAFT=${DRAFT:-<self>} NUM_DRAFT=$NUM_DRAFT GRAPH=$GRAPH"
python -c "import cca_hip.cca_op, moe_hip, tail_hip, attn_decode, attn_hip, attn_prefill_paged; print('[accept] hip pkgs OK')" \
  || { echo '[accept] hip import FAILED'; exit 1; }

SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1=label  $2...=extra env+args verbatim
  local label="$1"; shift
  echo "[launch:$label] $* -> $LOG"
  setsid env PYTHONPATH="$PYTHONPATH" MINISGL_MOE_SCATTER="$MOE_SCATTER" MINISGL_KV_FP8="$KV_FP8" \
    MINISGL_ATTN_HIP=1 MINISGL_TAIL_HIP=1 "$@" \
    --model "$MODEL" --tensor-parallel-size 1 --port "$PORT" --graph "$GRAPH" \
    --attention-backend hip --page-size 16 --cache-type "$CACHE" --disable-pynccl \
    --memory-ratio "$MEMRATIO" --max-running-requests 4 > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 400); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models',timeout=3)" 2>/dev/null \
      && { echo "[launch:$label] ready"; return 0; }
    kill -0 "$SRV" 2>/dev/null || { echo "[launch:$label] DIED:"; tail -70 "$LOG"; exit 1; }; sleep 3
  done; echo "[launch:$label] not ready:"; tail -80 "$LOG"; exit 1
}

probe(){ PORT=$PORT OUT=$1 MODEL=$MODEL GENTOK=$GENTOK python - <<'PY'
import json,os,urllib.request
PORT,OUT,MODEL,GENTOK=os.environ["PORT"],os.environ["OUT"],os.environ["MODEL"],int(os.environ["GENTOK"])
prompts=[
 "The capital of France is",
 "Question: What is 17 plus 26? Answer:",
 "The first five prime numbers are",
 "Explain in two sentences how a bicycle stays upright.",
 "Count up from one to twenty:",
]
res=[]
for p in prompts:
    body=json.dumps({"model":MODEL,"temperature":0.0,"max_tokens":GENTOK,
                     "messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,
                             headers={"Content-Type":"application/json"})
    txt=json.load(urllib.request.urlopen(r,timeout=300))["choices"][0]["message"]["content"]
    res.append(txt); print(f"\n>>> {p[:50]!r}\n<<< {txt[:140]!r}")
json.dump(res,open(OUT,"w"))
PY
}

echo "===== BASELINE (spec off) ====="
boot baseline env MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl
probe "$OUTDIR/zaya_accept.baseline.json"; stop

echo "===== SPEC ($SPEC_ALGO, num_draft=$NUM_DRAFT) ====="
if [ "$SPEC_ALGO" = "dflash" ]; then
  [ -n "$DRAFT" ] || { echo "[accept] dflash requires DRAFT"; exit 1; }
  boot spec env MINISGL_SPEC_DEBUG=1 MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl \
    --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$NUM_DRAFT"
else
  boot spec env MINISGL_SPEC_DEBUG=1 MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl \
    --spec-algorithm tidar --spec-num-draft "$NUM_DRAFT"
fi
probe "$OUTDIR/zaya_accept.spec.json"
echo "[accept] acceptance lines:"
grep -E "\[spec\]|emitted/step|accept-len|draft_accepted" "$LOG" | tail -8 || echo "  (no acceptance lines)"
stop

echo "===== LOSSLESSNESS (baseline must be exact prefix of spec) ====="
python - "$OUTDIR/zaya_accept.baseline.json" "$OUTDIR/zaya_accept.spec.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); ok=True
for i,(x,y) in enumerate(zip(a,b)):
    n=min(len(x),len(y)); pref=x[:n]==y[:n]; ok&=pref
    tag="exact" if len(x)==len(y) else ("prefix-exact" if pref else "DIVERGE")
    print(f"  prompt[{i}]: {tag}  len {len(x)} vs {len(y)}")
    if not pref:
        for j,(cx,cy) in enumerate(zip(x,y)):
            if cx!=cy: print(f"    diverge@char {j}: base={x[max(0,j-20):j+20]!r} spec={y[max(0,j-20):j+20]!r}"); break
print("\nSPEC LOSSLESSNESS (greedy):", "PASS" if ok else "MISMATCH")
PY
echo "[accept] done"
