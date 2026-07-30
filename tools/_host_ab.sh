#!/usr/bin/env bash
# Host-overhead A/B at the NEW operating point (kernel 11.04 ms, host 2.61 ms = 19%).
# --no-gdn-radix drops the recurrent-radix prefix cache, which is what forces the SYNCHRONOUS
# normal_loop (scheduler.py:720). Measured only +1.4% back when kernels were 84% of the step;
# host is now 19%, so retest.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; TOKS="${TOKS:-256}"; OUT="$REPO/tools/_host_ab_results.txt"
PKG=/home/pat/code/rdna4-hip-kernels-mmax/fp8_wmma/torch-ext/fp8_wmma
wait_ready(){ for _ in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
cleanup(){ docker compose -p lease-hab --profile serve down >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
run(){ local args="$1" label="$2"
  echo "=== $label (extra args: '$args') ===" | tee -a "$OUT"
  LEASE_NAME=hab MINISGL_IMAGE=minisgl-rdna4:lean FP8_WMMA_PKG="$PKG" \
    MINISGL_GDN_PROJ_GEMV=1 MINISGL_MINV_DECODE_GEMV=1 MINISGL_EXTRA_ARGS="$args" \
    gpu-lease -n 2 --detach --name hab -- docker compose -p lease-hab --profile serve up -d >/dev/null 2>&1
  if ! wait_ready; then echo "  BOOT FAILED" | tee -a "$OUT"
    docker compose -p lease-hab --profile serve logs --tail 12 2>&1 | sed 's/^/    /' | tee -a "$OUT"; cleanup; sleep 5; return 1; fi
  python3 "$REPO/tools/_hostloop_driver.py" "$TOKS" "$REPO/tools/_host_out_$label.txt" 2>&1 | grep tok/s | sed 's/^/  /' | tee -a "$OUT"
  cleanup; sleep 5; }
: > "$OUT"
run ""               sync_loop_default
run "--no-gdn-radix" overlap_loop
echo "=== done ===" | tee -a "$OUT"
