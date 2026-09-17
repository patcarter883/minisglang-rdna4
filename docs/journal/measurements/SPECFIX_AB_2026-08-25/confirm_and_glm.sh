#!/usr/bin/env bash
# 1) Confirm leg: fix worktree, DEFAULT SPEC=dspark launch (serve.sh should now resolve
#    MAX_BS=1 -> tau=0 and land ~36.7 tok/s). 2) GLM EAGLE3 multi-turn cache-hit A/B.
set -uo pipefail
ABDIR="$(cd "$(dirname "$0")" && pwd)"
PROJ=lease-specab; CNAME=specab-serve
FIX_WT=/home/pat/code/minisgl-rdna4-specreview

cleanup() { (cd "$FIX_WT" && LEASE_NAME=specab docker compose -p $PROJ --profile serve down >/dev/null 2>&1) || true; }
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

echo "=== CONFIRM LEG: fix default dspark ($(git -C $FIX_WT log --oneline -1 | head -c 50)) ==="
cd "$FIX_WT"
LEASE_NAME=specab MINISGL_IMAGE=minisgl-rdna4:lean MODEL=qwen38-27b-int4 SPEC=dspark \
  docker compose -p $PROJ --profile serve up -d serve >"$ABDIR/confirm.log" 2>&1 || { echo "UP FAILED"; exit 1; }
wait_ready >>"$ABDIR/confirm.log" 2>&1 || { echo "CONFIRM BOOT FAILED"; tail -20 "$ABDIR/confirm.log"; exit 1; }
docker logs "$CNAME" 2>&1 | grep -E 'dspark: conf' | sed 's/^/  banner /'
python3 "$ABDIR/bench_client.py" http://127.0.0.1:1919 "$ABDIR/qwen38_dspark_fixdefault.json"
cleanup; sleep 5

echo "=== GLM EAGLE3 MULTI-TURN A/B ==="
AB_TAG=glm_eagle3 BENCH_CLIENT=bench_multiturn.py "$ABDIR/ab_driver.sh" glm eagle3
