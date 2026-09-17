#!/usr/bin/env bash
# Variant legs at the tuned M=1 dspark operating point (tau=0): base+CONF_TAU=0 vs fix+MAX_BS=1
# (fix serve.sh defaults tau to 0 when the spec gate is 1). Same protocol as ab_driver.sh.
set -uo pipefail
ABDIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=qwen38-27b-int4; SPEC=dspark
PROJ=lease-specab; CNAME=specab-serve; TAG=qwen38_dspark_tau0

cleanup() { (cd /home/pat/code/minisgl-rdna4-specreview && LEASE_NAME=specab docker compose -p $PROJ --profile serve down >/dev/null 2>&1) || true; }
trap cleanup EXIT INT TERM

wait_ready() {
  for _ in $(seq 1 240); do
    curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0
    st=$(docker inspect -f '{{.State.Status}}' "$CNAME" 2>/dev/null || echo gone)
    [[ "$st" == "exited" || "$st" == "gone" ]] && { echo "CONTAINER $st"; docker logs "$CNAME" 2>&1 | tail -30; return 1; }
    docker logs "$CNAME" 2>&1 | grep -qE "Traceback|AssertionError|CUDA out of memory" && { echo "FATAL IN LOGS"; docker logs "$CNAME" 2>&1 | tail -40; return 1; }
    sleep 5
  done
  echo TIMEOUT; return 1
}

run_leg() { # NAME WORKTREE EXPECT ENVPAIRS...
  local name="$1" wt="$2" expect="$3"; shift 3
  local log="$ABDIR/${TAG}_${name}.log"
  echo "=== LEG $name wt=$wt $(git -C "$wt" log --oneline -1 | head -c 60) env=$* ===" | tee -a "$log"
  cd "$wt"
  env "$@" LEASE_NAME=specab MINISGL_IMAGE=minisgl-rdna4:lean MODEL="$MODEL" SPEC="$SPEC" \
    docker compose -p $PROJ --profile serve up -d serve >>"$log" 2>&1 || { echo "UP FAILED"; return 1; }
  wait_ready >>"$log" 2>&1 || { echo "LEG $name BOOT FAILED (see $log)"; cleanup; return 1; }
  local fixsym
  fixsym=$(docker exec "$CNAME" grep -c "_fused_route_ok" /engine/python/minisgl/scheduler/scheduler.py 2>/dev/null | tr -d '[:space:]')
  echo "PROVENANCE[$name] fixsym=$fixsym expect=$expect" | tee -a "$log"
  { [[ "$expect" == "1" && "${fixsym:-0}" -ge 1 ]] || [[ "$expect" == "0" && "${fixsym:-0}" -eq 0 ]]; } \
    || { echo "PROVENANCE FAIL"; cleanup; return 1; }
  docker logs "$CNAME" 2>&1 | grep -E 'dspark: conf|spec sampled' | sed "s/^/  banner[$name] /"
  python3 "$ABDIR/bench_client.py" http://127.0.0.1:1919 "$ABDIR/${TAG}_${name}.json" 2>&1 | tee -a "$log"
  docker logs "$CNAME" 2>&1 | grep -E "hip-engage|\[spec\] mean" | sort -u > "$ABDIR/${TAG}_${name}.engage" || true
  cleanup; sleep 5
}

run_leg base_tau0 /home/pat/code/minisgl-rdna4-abbase 0 MINISGL_DSPARK_CONF_TAU=0 || exit 1
run_leg fix_bs1  /home/pat/code/minisgl-rdna4-specreview 1 MINISGL_SPEC_MAX_BS=1 || exit 1
echo "=== $TAG complete ==="
for l in base_tau0 fix_bs1; do
  echo "--- $l ---"; python3 -c "import json;d=json.load(open('$ABDIR/${TAG}_${l}.json'));print('median',d['median_tok_s'],'spread',d['spread_tok_s'])"
done
