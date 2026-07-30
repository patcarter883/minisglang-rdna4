#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; TOKS="${TOKS:-256}"; OUT="$REPO/tools/_bylane_ab_results.txt"
wait_ready(){ for _ in $(seq 1 200); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
run(){ local v="$1" label="$2"
  echo "=== $label (VLLM_W4A8_MOE_GEMV_BYLANE='$v') ===" | tee -a "$OUT"
  LEASE_NAME=blab MINISGL_IMAGE=minisgl-rdna4:lean VLLM_W4A8_MOE_GEMV_BYLANE="$v" \
    gpu-lease -n 2 --detach --name blab -- docker compose -p lease-blab --profile serve up -d >/dev/null 2>&1
  if ! wait_ready; then echo "  BOOT FAILED" | tee -a "$OUT"
    docker compose -p lease-blab --profile serve logs --tail 20 2>&1 | sed 's/^/    /' | tee -a "$OUT"
    docker compose -p lease-blab --profile serve down >/dev/null 2>&1; sleep 5; return 1; fi
  python3 "$REPO/tools/_hostloop_driver.py" "$TOKS" "$REPO/tools/_bylane_out_$label.txt" 2>&1 | sed 's/^/  /' | tee -a "$OUT"
  docker compose -p lease-blab --profile serve down >/dev/null 2>&1; sleep 5; }
: > "$OUT"
run ""  bylane_off
run "1" bylane_on
echo "=== done ===" | tee -a "$OUT"
