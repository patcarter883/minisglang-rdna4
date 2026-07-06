#!/usr/bin/env bash
# FUNCTIONAL A/B: original AR ZAYA1-8B  vs  the TiDAR diffusion model (self-draft spec-decode).
# Both are the SAME ZAYA CCA+MoE arch at fp8 on ONE 16GB card; the only delta is AR decode vs
# TiDAR block-draft + lossless verify. Measures single-stream DECODE THROUGHPUT (tok/s) with a
# warmup + median-of-N sustained generations (NOT the banned 2-request delta), plus a coherence
# sample. Runs INSIDE vllm22-w4a8:combined under a 1-card lease.
#
#   A: original ZAYA (AR)          -> /models/ZAYA1-8B-fp8, plain serve (1 token / forward)
#   B: TiDAR diffusion (spec)      -> /big/zaya1-tidar-opd-fp8, --spec-algorithm tidar (block draft)
# Honest ceiling: the two-forward B is ~parity (2 forwards, ~2 tok/step); the fused single-forward
# (Phase C) is the ~2x — projected offline, not in this runner.
set -uo pipefail
source /app/.venv/bin/activate
mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1

ORIG="${ORIG:-/models/ZAYA1-8B-fp8}"
DIFF="${DIFF:-/big/zaya1-tidar-opd-fp8}"
PORT="${PORT:-21966}"
NUM_DRAFT="${NUM_DRAFT:-4}"
GENTOK="${GENTOK:-256}"   # tokens per timed generation (decode-dominated)
NRUNS="${NRUNS:-3}"       # timed runs; median reported
MEMRATIO="${MEMRATIO:-0.85}"
LOG=/engine/tools/ab_zaya.server.log
OUTDIR=/engine/tools

SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1=label  $2=model  $3...=extra env+args
  local label="$1"; local model="$2"; shift 2
  echo "[launch:$label] model=$model $* -> $LOG"
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 \
    MINISGL_ATTN_HIP=1 MINISGL_TAIL_HIP=1 "$@" \
    --model "$model" --tensor-parallel-size 1 --port "$PORT" --graph 0 --attn hip \
    --memory-ratio "$MEMRATIO" --max-running-requests 4 > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 400); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models',timeout=3)" 2>/dev/null \
      && { echo "[launch:$label] ready"; return 0; }
    kill -0 "$SRV" 2>/dev/null || { echo "[launch:$label] DIED:"; tail -60 "$LOG"; exit 1; }; sleep 3
  done; echo "[launch:$label] not ready:"; tail -80 "$LOG"; exit 1
}

# Single-stream decode throughput: warmup then NRUNS timed generations; report MEDIAN tok/s + sample.
bench(){ PORT=$PORT MODEL=$2 GENTOK=$GENTOK NRUNS=$NRUNS LABEL=$1 python - <<'PY'
import json,os,time,statistics,urllib.request
PORT,MODEL,GENTOK,NRUNS,LABEL=os.environ["PORT"],os.environ["MODEL"],int(os.environ["GENTOK"]),int(os.environ["NRUNS"]),os.environ["LABEL"]
PROMPT="Write a detailed explanation of how a computer works, starting from transistors."
def gen():
    body=json.dumps({"model":MODEL,"temperature":0.0,"max_tokens":GENTOK,
                     "messages":[{"role":"user","content":PROMPT}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.perf_counter(); resp=json.load(urllib.request.urlopen(r,timeout=600)); dt=time.perf_counter()-t0
    ct=resp.get("usage",{}).get("completion_tokens") or GENTOK
    return dt, ct, resp["choices"][0]["message"]["content"]
gen()  # warmup (discard)
rates=[]; sample=""
for i in range(NRUNS):
    dt,ct,txt=gen(); rates.append(ct/dt); sample=txt
    print(f"  [{LABEL}] run{i+1}: {ct} tok in {dt:.2f}s = {ct/dt:.1f} tok/s")
med=statistics.median(rates)
print(f"  [{LABEL}] MEDIAN decode throughput = {med:.1f} tok/s  (over {NRUNS} runs, {GENTOK} tok each)")
print(f"  [{LABEL}] sample: {sample[:140]!r}")
json.dump({"median_tok_s":med,"rates":rates,"sample":sample},open(f"/engine/tools/ab.{LABEL}.json","w"))
PY
}

echo "============================================================"
echo "  A: ORIGINAL ZAYA (AR decode, spec off)"
echo "============================================================"
boot orig "$ORIG" env MINISGL_DISABLE_OVERLAP_SCHEDULING=1 MINISGL_ZAYA_W8A16="${W8A16:-0}" \
  python -m minisgl
bench orig "$ORIG"; stop

echo "============================================================"
echo "  B: TiDAR DIFFUSION (self-draft spec, num_draft=$NUM_DRAFT)"
echo "============================================================"
boot tidar "$DIFF" env MINISGL_SPEC_DEBUG=1 MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
  MINISGL_TIDAR_FUSED="${FUSED:-0}" MINISGL_TIDAR_SEG="${SEG:-0}" MINISGL_ZAYA_OLDMOE="${OLDMOE:-0}" \
  MINISGL_TIDAR_TIME="${TIME:-0}" MINISGL_ZAYA_W8A16="${W8A16:-0}" \
  python -m minisgl --spec-algorithm tidar --spec-num-draft "$NUM_DRAFT"
bench tidar "$DIFF"
echo "  [tidar] acceptance (emitted/step = tokens per 2-forward step):"
grep -E "\[spec\]" "$LOG" | tail -4 || echo "    (no [spec] lines)"; stop

echo "============================================================"
echo "  A/B SUMMARY"
echo "============================================================"
python - <<'PY'
import json
o=json.load(open("/engine/tools/ab.orig.json")); t=json.load(open("/engine/tools/ab.tidar.json"))
oa,tb=o["median_tok_s"],t["median_tok_s"]
import os
mode = "FUSED 1-fwd" if os.environ.get("FUSED") == "1" else "2-fwd"
print(f"  A original ZAYA (AR)          : {oa:6.1f} tok/s")
print(f"  B TiDAR diffusion ({mode:>10s}): {tb:6.1f} tok/s   ({tb/oa:.2f}x vs A)")
print(f"\n  NOTE: FUSED=1 measures the single-forward path (0b gate: must be >1.0x A to justify TiDAR")
print(f"  over base AR); the 2-fwd path is ~parity-or-slower (2 forwards).")
PY
echo "[done]"
