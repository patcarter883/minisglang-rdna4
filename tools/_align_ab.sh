#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; TOKS="${TOKS:-256}"; OUT="$REPO/tools/_align_ab_results.txt"
wait_ready(){ for _ in $(seq 1 180); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
run(){ local pkg="$1" label="$2"
  echo "=== $label  (moe_hip=$pkg) ===" | tee -a "$OUT"
  LEASE_NAME=alignab MOE_HIP_PKG="$pkg" gpu-lease -n 2 --detach --name alignab -- \
    docker compose -p lease-alignab --profile serve up -d >/dev/null 2>&1
  if ! wait_ready; then echo "  BOOT FAILED" | tee -a "$OUT"
    docker compose -p lease-alignab --profile serve logs --tail 25 2>&1 | sed 's/^/    /' | tee -a "$OUT"
    docker compose -p lease-alignab --profile serve down >/dev/null 2>&1; return 1; fi
  python3 "$REPO/tools/_hostloop_driver.py" "$TOKS" "$REPO/tools/_align_out_$label.txt" 2>&1 | sed 's/^/  /' | tee -a "$OUT"
  docker compose -p lease-alignab --profile serve down >/dev/null 2>&1; sleep 5; }
: > "$OUT"
run /home/pat/code/rdna4-hip-kernels-alignbase/moe/torch-ext/moe_hip baseline_serial
run /home/pat/code/rdna4-hip-kernels-moealign/moe/torch-ext/moe_hip parallel_scan
echo "=== done ===" | tee -a "$OUT"
