#!/usr/bin/env bash
# NORTH-STAR v0 falsifier: can the CCA/GDN ZAYA decode be cudagraph-captured, and does it recover the
# ~2/3 dispatch overhead the eager A/B implies (ZAYA-8B AR 42ms/token eager vs ~14ms BW floor)?
# Boots the ZAYA AR path (spec OFF) at --graph 0 (eager) then --graph N (captured), benches clean ITL,
# and reports whether every custom HIP decode kernel still fires under capture ([hip-engage] lines).
#   gpu-lease -n 1 -- bash tools/run_cca_graph_v0.sh
set -uo pipefail
source /app/.venv/bin/activate
mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1

ORIG="${ORIG:-/models/ZAYA1-8B-fp8}"
PORT="${PORT:-21988}"; GENTOK="${GENTOK:-256}"; NRUNS="${NRUNS:-3}"; MEMRATIO="${MEMRATIO:-0.85}"
GRAPHS="${GRAPHS:-0 8}"
LOG=/engine/tools/cca_graph_v0.server.log

SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ local label="$1"; local g="$2"
  echo "[launch:$label] --graph $g -> $LOG"
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 MINISGL_ATTN_HIP=1 MINISGL_TAIL_HIP=1 \
    MINISGL_ZAYA_W8A16=1 MINISGL_HIP_ENGAGE_LOG=1 MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
    python -m minisgl --model "$ORIG" --tensor-parallel-size 1 --port "$PORT" --graph "$g" --attn hip \
    --memory-ratio "$MEMRATIO" --max-running-requests 4 > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 500); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models',timeout=3)" 2>/dev/null \
      && { echo "[launch:$label] ready"; return 0; }
    kill -0 "$SRV" 2>/dev/null || { echo "[launch:$label] DIED:"; tail -40 "$LOG"; return 1; }; sleep 3
  done; echo "[launch:$label] not ready"; tail -40 "$LOG"; return 1; }

bench(){ PORT=$PORT MODEL=$ORIG GENTOK=$GENTOK NRUNS=$NRUNS LABEL=$1 python - <<'PY'
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
print(f"  [{LABEL}] MEDIAN = {med:.1f} tok/s  ({1000/med:.1f} ms/tok)")
json.dump({"median_tok_s":med},open(f"/engine/tools/ccagraph.{LABEL}.json","w"))
PY
}

for g in $GRAPHS; do
  echo "===== ZAYA AR  --graph $g ====="
  boot "g$g" "$g" || { stop; continue; }
  bench "g$g"
  echo "  [g$g] graph state:"; grep -E 'CUDA graph|Capturing CUDA graphs|Free GPU memory (before|after) capturing' "$LOG" | sed -E 's/\x1b\[[0-9;]*m//g; s/^.*INFO *//' | head -4
  echo "  [g$g] HIP kernels engaged:"; grep -E '\[hip-engage\]' "$LOG" | sed -E 's/\x1b\[[0-9;]*m//g; s/^.*INFO *//' | sort -u
  stop
done

echo "===== v0 SUMMARY ====="
python - <<'PY'
import json,glob,os
def rd(l):
    try: return json.load(open(f"/engine/tools/ccagraph.{l}.json"))["median_tok_s"]
    except Exception: return None
base=rd("g0")
print(f"  eager (--graph 0) : {base} tok/s" if base else "  g0 FAIL")
for f in sorted(glob.glob("/engine/tools/ccagraph.g*.json")):
    l=os.path.basename(f)[9:-5]
    if l=="g0": continue
    v=rd(l)
    print(f"  graph ({l})       : {v} tok/s  ({v/base:.2f}x eager)" if v and base else f"  {l} FAIL")
PY
echo "[done]"
