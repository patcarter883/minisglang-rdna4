#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; TOKS="${TOKS:-256}"; OUT="$REPO/tools/_img_ab_results.txt"
wait_ready(){ for _ in $(seq 1 200); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
run(){ local img="$1" label="$2"
  echo "=== $label ($img) ===" | tee -a "$OUT"
  LEASE_NAME=imgab MINISGL_IMAGE="$img" \
    gpu-lease -n 2 --detach --name imgab -- docker compose -p lease-imgab --profile serve up -d >/dev/null 2>&1
  if ! wait_ready; then echo "  BOOT FAILED" | tee -a "$OUT"
    docker compose -p lease-imgab --profile serve logs --tail 25 2>&1 | sed 's/^/    /' | tee -a "$OUT"
    docker compose -p lease-imgab --profile serve down >/dev/null 2>&1; sleep 5; return 1; fi
  python3 "$REPO/tools/_hostloop_driver.py" "$TOKS" "$REPO/tools/_img_out_$label.txt" 2>&1 | sed 's/^/  /' | tee -a "$OUT"
  docker compose -p lease-imgab --profile serve down >/dev/null 2>&1; sleep 5; }
: > "$OUT"
run minisgl-rdna4:lean      stale_jul23
run minisgl-rdna4:lean-cur  current_kernels
echo "=== done ===" | tee -a "$OUT"
