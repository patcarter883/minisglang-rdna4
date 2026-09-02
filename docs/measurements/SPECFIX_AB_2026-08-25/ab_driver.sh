#!/usr/bin/env bash
# Two-leg spec-fix A/B: baseline worktree (2af6af51) vs fix worktree (2fba9ae3), same image,
# same cards, sequential legs under ONE gpu-lease held by the caller. Args: MODEL SPEC [EXTRA_ENV...]
# Per leg: compose up from the leg's worktree (mounts IT at /engine), 3-branch ready wait,
# provenance assert (fix-only symbol in the MOUNTED scheduler + import path), bench, banner+engage
# capture, compose down. Results land in $ABDIR.
set -uo pipefail
ABDIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:?model alias}"; SPEC="${2:?spec}"
BASE_WT=/home/pat/code/minisgl-rdna4-abbase
FIX_WT=/home/pat/code/minisgl-rdna4-specreview
PROJ=lease-specab; CNAME=specab-serve
TAG="${AB_TAG:-${MODEL}_${SPEC}}"

cleanup() { (cd "$FIX_WT" && LEASE_NAME=specab docker compose -p $PROJ --profile serve down >/dev/null 2>&1) || true; }
trap cleanup EXIT INT TERM

wait_ready() { # 3-branch: port ready / container exited / fatal in logs
  for _ in $(seq 1 240); do
    curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0
    st=$(docker inspect -f '{{.State.Status}}' "$CNAME" 2>/dev/null || echo gone)
    [[ "$st" == "exited" || "$st" == "gone" ]] && { echo "CONTAINER $st"; docker logs "$CNAME" 2>&1 | tail -30; return 1; }
    if docker logs "$CNAME" 2>&1 | grep -qE "Traceback|AssertionError|CUDA out of memory|RuntimeError"; then
      echo "FATAL IN LOGS"; docker logs "$CNAME" 2>&1 | grep -B2 -A12 -E "Traceback|AssertionError" | tail -40; return 1
    fi
    sleep 5
  done
  echo "TIMEOUT"; return 1
}

run_leg() { # NAME WORKTREE EXPECT_FIXSYM(0|1)
  local name="$1" wt="$2" expect="$3"
  local log="$ABDIR/${TAG}_${name}.log"
  echo "=== LEG $name wt=$wt $(git -C "$wt" log --oneline -1) ===" | tee -a "$log"
  cd "$wt"
  LEASE_NAME=specab MINISGL_IMAGE=minisgl-rdna4:lean MODEL="$MODEL" SPEC="$SPEC" \
    docker compose -p $PROJ --profile serve up -d serve >>"$log" 2>&1 || { echo "UP FAILED"; return 1; }
  if ! wait_ready >>"$log" 2>&1; then echo "LEG $name BOOT FAILED (see $log)"; cleanup; return 1; fi
  # --- provenance: the MOUNTED tree is what this leg imports, and it is the right tree ---
  local impath fixsym
  impath=$(docker exec "$CNAME" python -c "import minisgl.scheduler.scheduler as s; print(s.__file__)" 2>/dev/null)
  fixsym=$(docker exec "$CNAME" grep -c "_fused_route_ok" /engine/python/minisgl/scheduler/scheduler.py 2>/dev/null | head -1); fixsym="${fixsym:-0}"
  echo "PROVENANCE[$name] import=$impath fixsym=$fixsym expect=$expect" | tee -a "$log"
  [[ "$impath" == /engine/python/* ]] || { echo "PROVENANCE FAIL: import not from /engine"; cleanup; return 1; }
  if [[ "$expect" == "1" && "$fixsym" -lt 1 ]] || [[ "$expect" == "0" && "$fixsym" -ge 1 ]]; then
    echo "PROVENANCE FAIL: fixsym=$fixsym expect=$expect"; cleanup; return 1
  fi
  docker logs "$CNAME" 2>&1 | grep -E '\[serve\]' | tee -a "$log" | sed "s/^/  banner[$name] /"
  # --- bench ---
  python3 "$ABDIR/${BENCH_CLIENT:-bench_client.py}" http://127.0.0.1:1919 "$ABDIR/${TAG}_${name}.json" 2>&1 | tee -a "$log"
  # --- engage ledger (docker logs may not carry engine output; record what we get) ---
  docker logs "$CNAME" 2>&1 | grep -E "hip-engage|accept" | sort -u > "$ABDIR/${TAG}_${name}.engage" || true
  wc -l "$ABDIR/${TAG}_${name}.engage" | tee -a "$log"
  cleanup; sleep 5
}

run_leg base "$BASE_WT" 0 || exit 1
run_leg fix  "$FIX_WT" 1 || exit 1
echo "=== A/B $TAG complete ==="
for l in base fix; do
  echo "--- $l ---"; python3 -c "import json;d=json.load(open('$ABDIR/${TAG}_${l}.json'));print('median',d['median_tok_s'],'spread',d['spread_tok_s']);print({k:v for k,v in d['metrics_delta'].items() if v})" 2>/dev/null
done
