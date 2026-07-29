#!/usr/bin/env bash
# bs=1 decode A/B matrix for the minisgl<->vllm24 gap hunt.
# Each config is a "<label>|<env assignments>|<extra minisgl args>" triple.
# Run from the WORKTREE dir so compose's `.:/engine` mount is the isolated tree.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
TOKS="${TOKS:-256}"
OUT="${OUT:-$REPO/tools/_decode_ab_results.txt}"

CONFIGS=(
  "baseline||"
  "minv_block_m16|MINISGL_MINV_BLOCK_M=16|"
  "no_gdn_radix||--no-gdn-radix"
)

wait_ready() { for _ in $(seq 1 180); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }

: > "$OUT"
echo "decode A/B  model=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit  TP=2  bs=1  max_tokens=$TOKS" | tee -a "$OUT"
for cfg in "${CONFIGS[@]}"; do
  label="${cfg%%|*}"; rest="${cfg#*|}"; envs="${rest%%|*}"; args="${rest#*|}"
  echo "=== $label  env=[$envs] args=[$args] ===" | tee -a "$OUT"

  # shellcheck disable=SC2086
  env $envs LEASE_NAME=decab MINISGL_EXTRA_ARGS="$args" \
    gpu-lease -n 2 --detach --name decab -- \
    docker compose -p lease-decab --profile serve up -d >/dev/null 2>&1

  if ! wait_ready; then
    echo "  BOOT FAILED:" | tee -a "$OUT"
    docker compose -p lease-decab --profile serve logs --tail 25 2>&1 | sed 's/^/    /' | tee -a "$OUT"
    docker compose -p lease-decab --profile serve down >/dev/null 2>&1; sleep 5; continue
  fi
  python3 "$REPO/tools/_hostloop_driver.py" "$TOKS" "$REPO/tools/_decode_ab_out_$label.txt" 2>&1 \
    | sed 's/^/  /' | tee -a "$OUT"
  docker compose -p lease-decab --profile serve down >/dev/null 2>&1
  sleep 5
done
echo "=== done ===" | tee -a "$OUT"
