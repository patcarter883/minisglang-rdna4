#!/usr/bin/env bash
# A/B the cost of the M-invariant WMMA GEMM (layers/minv.py) on bs=1 decode.
#
#   MINISGL_MINV_GEMM=1  (default, shipped)  -> every unquantized Linear uses dense_gemm_rd
#   MINISGL_MINV_GEMM=0  (diagnostic only)   -> F.linear / rocBLAS
#
# DIAGNOSTIC ONLY: =0 breaks the M-invariance that prefix caching / chunked prefill / spec-verify
# depend on. This measures the prize; it is not a shippable config.
#
# Run from the WORKTREE dir so compose's `.:/engine` mount is the isolated tree.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
TOKS="${TOKS:-256}"
OUT="${OUT:-$REPO/tools/_minv_ab_results.txt}"

wait_ready() {
  for _ in $(seq 1 180); do
    if curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1; then return 0; fi
    sleep 5
  done
  return 1
}

run_cfg() {
  local minv="$1" name="$2"
  echo "=== config: MINISGL_MINV_GEMM=$minv ($name) ===" | tee -a "$OUT"

  LEASE_NAME="minvab" MINISGL_MINV_GEMM="$minv" \
    gpu-lease -n 2 --detach --name minvab -- \
    docker compose -p lease-minvab --profile serve up -d >/dev/null 2>&1

  if ! wait_ready; then
    echo "  BOOT FAILED — last 30 log lines:" | tee -a "$OUT"
    docker compose -p lease-minvab --profile serve logs --tail 30 2>&1 | sed 's/^/    /' | tee -a "$OUT"
    docker compose -p lease-minvab --profile serve down >/dev/null 2>&1
    return 1
  fi

  # confirm the flag actually reached the engine
  local seen
  seen=$(docker exec "${LEASE_NAME:-minvab}-serve" printenv MINISGL_MINV_GEMM 2>/dev/null || echo "?")
  echo "  container MINISGL_MINV_GEMM=$seen" | tee -a "$OUT"

  python3 "$REPO/tools/_hostloop_driver.py" "$TOKS" "$REPO/tools/_minv_ab_out_$name.txt" 2>&1 \
    | sed 's/^/  /' | tee -a "$OUT"

  docker compose -p lease-minvab --profile serve down >/dev/null 2>&1
  sleep 5
}

: > "$OUT"
echo "minv A/B  model=${MINISGL_MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}  TP=2  bs=1  max_tokens=$TOKS" | tee -a "$OUT"
run_cfg 1 minv_on
run_cfg 0 minv_off
echo "=== done ===" | tee -a "$OUT"
