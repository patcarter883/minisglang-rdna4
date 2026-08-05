#!/usr/bin/env bash
# Inner half of tools/quant_m_invariance_serve_run.sh — runs INSIDE the container.
# Two legs: the shipped int4 gemv cap (8) and the candidate cap (16). Everything else identical.
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1
export MINISGL_KV_FP8=0

MODEL="${QMINV_MODEL:?}"
TP="${QMINV_TP:-2}"
NCONC="${QMINV_N:-12}"
MAXTOK="${QMINV_MAXTOK:-24}"
KFILE=/engine/python/minisgl/quant/kernels.py

set_cap() {  # $1 = new _W4A8_GEMV_MAX_INT4
  python - "$1" <<'PY'
import re, sys, pathlib
cap = int(sys.argv[1])
p = pathlib.Path("/engine/python/minisgl/quant/kernels.py")
s = p.read_text()
s2, n = re.subn(r"^_W4A8_GEMV_MAX_INT4 = \d+$", f"_W4A8_GEMV_MAX_INT4 = {cap}", s, flags=re.M)
assert n == 1, f"cap patch matched {n} sites"
p.write_text(s2)
print(f"[prov] source now has _W4A8_GEMV_MAX_INT4 = {cap}")
PY
}

run_leg() {  # $1 = cap, $2 = tag
  local CAP="$1" TAG="$2" LOG="/engine/_qminv_serve_$2.log"
  set_cap "$CAP" || return 1
  MODEL="$MODEL" TP="$TP" SPEC=none PORT=1919 CONC="$NCONC" GRAPH_BS=0 CACHE_TYPE=naive \
    setsid bash /engine/tools/serve.sh >"$LOG" 2>&1 &
  local PID=$!
  local i
  for i in $(seq 1 150); do
    if ! kill -0 "$PID" 2>/dev/null; then echo "!! serve exited early"; tail -40 "$LOG"; return 1; fi
    curl -sf http://localhost:1919/health >/dev/null 2>&1 && break
    sleep 3
  done
  if ! curl -sf http://localhost:1919/health >/dev/null 2>&1; then
    echo "!! readiness timeout"; tail -40 "$LOG"; kill -- -"$PID" 2>/dev/null; return 1
  fi
  echo "== leg $TAG READY (cap=$CAP)"
  python /engine/tools/quant_m_invariance_serve.py --port 1919 --n "$NCONC" \
    --max-tokens "$MAXTOK" --out "/engine/_qminv_serve_$TAG.txt"
  echo "-- PROVENANCE: dense arms engaged this leg:"
  grep -o 'mmq_fp8_gemm([a-z0-9_+]*)' "$LOG" | sort | uniq -c
  kill -- -"$PID" 2>/dev/null
  sleep 15
}

run_leg 8 cap8
run_leg 16 cap16
set_cap 8
