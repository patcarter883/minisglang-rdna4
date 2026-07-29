#!/usr/bin/env bash
# Re-measure the session's wins on the REAL serving path (sampled: temp 1.0 / top_k 20 / top_p 0.95,
# the checkpoint's generation_config) rather than greedy. Greedy takes an argmax; sampled runs the
# top-k/top-p sampler, which is work no greedy measurement ever included.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; TOKS="${TOKS:-256}"; OUT="$REPO/tools/_sampled_ab_results.txt"
PKG=/home/pat/code/rdna4-hip-kernels-mmax/fp8_wmma/torch-ext/fp8_wmma
[ -f "$PKG/_ops.py" ] || { echo "FATAL: $PKG not built"; exit 1; }
wait_ready(){ for _ in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
cleanup(){ docker compose -p lease-sab --profile serve down >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
run(){ local gd="$1" mv="$2" label="$3"
  echo "=== $label (GDN_PROJ=$gd MINV_DECODE=$mv) ===" | tee -a "$OUT"
  LEASE_NAME=sab MINISGL_IMAGE=minisgl-rdna4:lean-b000dd0 FP8_WMMA_PKG="$PKG" \
    MINISGL_GDN_PROJ_GEMV="$gd" MINISGL_MINV_DECODE_GEMV="$mv" \
    gpu-lease -n 2 --detach --name sab -- docker compose -p lease-sab --profile serve up -d >/dev/null 2>&1
  if ! wait_ready; then echo "  BOOT FAILED" | tee -a "$OUT"; cleanup; sleep 5; return 1; fi
  GREEDY=1 python3 "$REPO/tools/_hostloop_driver.py" "$TOKS" "$REPO/tools/_s_greedy_$label.txt" 2>&1 \
    | grep tok/s | sed 's/^/  GREEDY  /' | tee -a "$OUT"
  python3 "$REPO/tools/_hostloop_driver.py" "$TOKS" "$REPO/tools/_s_sampled_$label.txt" 2>&1 \
    | grep tok/s | sed 's/^/  SAMPLED /' | tee -a "$OUT"
  cleanup; sleep 5; }
: > "$OUT"
run 0 0 baseline
run 1 1 both_gemv
echo "=== done ===" | tee -a "$OUT"
