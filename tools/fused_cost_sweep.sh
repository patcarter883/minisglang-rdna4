#!/usr/bin/env bash
# STEP-0.5 fused cost sweep: boot the AR baseline ONCE, then sweep the fused path over
# {flat,seg} × B∈{2,4}, capturing median tok/s + the per-step cost breakdown ([spec-fused-time]).
# Answers: does small-B and/or the CPU-mask fix bring fused within reach of AR (25 tok/s)?
#   gpu-lease -n 1 -- bash tools/run_fused_cost_sweep.sh
set -uo pipefail
source /app/.venv/bin/activate
mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1

ORIG="${ORIG:-/models/ZAYA1-8B-fp8}"
DIFF="${DIFF:-/big/zaya1-tidar-opd-fp8}"
PORT="${PORT:-21977}"
GENTOK="${GENTOK:-256}"; NRUNS="${NRUNS:-2}"; MEMRATIO="${MEMRATIO:-0.85}"
LOG=/engine/tools/fused_sweep.server.log

SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ local label="$1"; local model="$2"; shift 2
  echo "[launch:$label] $* -> $LOG"
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 MINISGL_ATTN_HIP=1 MINISGL_TAIL_HIP=1 "$@" \
    --model "$model" --tensor-parallel-size 1 --port "$PORT" --graph 0 --attn hip \
    --memory-ratio "$MEMRATIO" --max-running-requests 4 > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 400); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models',timeout=3)" 2>/dev/null \
      && { echo "[launch:$label] ready"; return 0; }
    kill -0 "$SRV" 2>/dev/null || { echo "[launch:$label] DIED:"; tail -40 "$LOG"; return 1; }; sleep 3
  done; echo "[launch:$label] not ready"; tail -40 "$LOG"; return 1; }

bench(){ PORT=$PORT MODEL=$2 GENTOK=$GENTOK NRUNS=$NRUNS LABEL=$1 python - <<'PY'
import json,os,time,statistics,urllib.request
PORT,MODEL,GENTOK,NRUNS,LABEL=os.environ["PORT"],os.environ["MODEL"],int(os.environ["GENTOK"]),int(os.environ["NRUNS"]),os.environ["LABEL"]
PROMPT="Write a detailed explanation of how a computer works, starting from transistors."
def gen():
    body=json.dumps({"model":MODEL,"temperature":0.0,"max_tokens":GENTOK,"messages":[{"role":"user","content":PROMPT}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); resp=json.load(urllib.request.urlopen(r,timeout=600)); dt=time.perf_counter()-t0
    ct=resp.get("usage",{}).get("completion_tokens") or GENTOK
    return dt,ct
gen()
rates=[]
for i in range(NRUNS):
    dt,ct=gen(); rates.append(ct/dt)
med=statistics.median(rates)
print(f"  [{LABEL}] MEDIAN = {med:.1f} tok/s")
json.dump({"median_tok_s":med},open(f"/engine/tools/sweep.{LABEL}.json","w"))
PY
}

echo "===== AR baseline (orig ZAYA, spec off) ====="
boot orig "$ORIG" env MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl && bench orig "$ORIG"; stop

# sweep: label  SEG  NUM_DRAFT
for cfg in "flatB4 0 4" "segB4 1 4" "flatB2 0 2" "segB2 1 2"; do
  set -- $cfg; label=$1; segv=$2; nd=$3
  echo "===== FUSED $label (SEG=$segv B=$nd) ====="
  boot "$label" "$DIFF" env MINISGL_SPEC_DEBUG=1 MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
    MINISGL_TIDAR_FUSED=1 MINISGL_TIDAR_SEG="$segv" MINISGL_ZAYA_OLDMOE=1 MINISGL_TIDAR_TIME=1 \
    python -m minisgl --spec-algorithm tidar --spec-num-draft "$nd" || { stop; continue; }
  bench "$label" "$DIFF"
  echo "  [$label] cost/accept:"; grep -E "spec-fused-time|spec-fused\]" "$LOG" | tail -2 | sed -E 's/\x1b\[[0-9;]*m//g'
  stop
done

echo "===== SWEEP SUMMARY ====="
python - <<'PY'
import json,os
def rd(l):
    try: return json.load(open(f"/engine/tools/sweep.{l}.json"))["median_tok_s"]
    except Exception: return None
ar=rd("orig"); print(f"  AR baseline            : {ar} tok/s" if ar else "  AR: FAIL")
for l in ["flatB4","segB4","flatB2","segB2"]:
    v=rd(l); print(f"  fused {l:8s}         : {v} tok/s  ({v/ar:.2f}x AR)" if v and ar else f"  fused {l}: FAIL")
PY
echo "[done]"
