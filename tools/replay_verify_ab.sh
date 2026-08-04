#!/usr/bin/env bash
# RULE-4 serve validation + A/B for the ReplaySSM SPEC VERIFY (ring append + O(1) rollback).
#
# Two legs, IDENTICAL engine source (both mount this worktree at /engine) and identical serve
# config; the ONLY difference is the baked kernel package:
#   ctl  = kernels WITHOUT gdn_verify_replay -> the engine falls back to the materialising
#          gdn_prefill_verify (flush before, per-token ssm scratch, invalidate after, host scatter)
#   rvk  = kernels WITH it                   -> the draft window is appended to the ring and the
#          accept is a cursor rewind; no ssm scratch is allocated at all
# That is a provenance difference, not a flag, and each leg ASSERTS which verify it took by grepping
# [hip-engage] — an A/B whose control silently ran the candidate is the classic way to measure
# new-vs-itself.
#
# Recorded per leg: which verify engaged, the KV pool (the freed per-token scratch should show up as
# pool tokens), decode tok/s + accepted tokens per chunk, and the acceptance suite.
#
# Run from the worktree root under a 2-card lease:
#   gpu-lease -n 2 -- bash tools/replay_verify_ab.sh
set -uo pipefail

CTL_IMAGE="${CTL_IMAGE:-minisgl-rdna4:comb}"
RVK_IMAGE="${RVK_IMAGE:-minisgl-rdna4:rvk}"
OUT="${OUT:-/home/pat/fixtures/minisgl-kv-calib/replay_verify_ab}"
PORT="${PORT:-1919}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen35b-awq}"
SPEC="${SPEC:-mtp}"
SPEC_K="${SPEC_K:-4}"
CONC="${CONC:-4}"
M_LIST="${M_LIST:-1,4}"
# 0.86, not the 0.80 default: MTP does not boot at 0.80 on this model (the recurrent-radix snapshot
# store takes 0.38 GiB of a ~1 GiB post-weights budget). Both legs get the same ratio.
MEM_RATIO="${MEM_RATIO:-0.86}"
ACC_REPS="${ACC_REPS:-2}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
PROJECT="${PROJECT:-minisgl-replayverify-ab}"
mkdir -p "$OUT"

boot() {  # $1 leg  $2 image
  local leg="$1" img="$2"
  echo "== boot leg=$leg image=$img mem=$MEM_RATIO spec=$SPEC k=$SPEC_K =="
  MINISGL_IMAGE="$img" MODEL="$MODEL_ALIAS" SPEC="$SPEC" SPEC_K="$SPEC_K" TP=2 CONC="$CONC" \
    ATTN=hip MEM_RATIO="$MEM_RATIO" MINISGL_HOST_PORT="$PORT" LEASE_NAME="rv-$leg" \
    COMPOSE_PROJECT_NAME="$PROJECT" docker compose --profile serve up -d >/dev/null || return 1
  local t0=$SECONDS
  while (( SECONDS - t0 < BOOT_TIMEOUT )); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "   ready after $((SECONDS-t0))s"; return 0; }
    docker logs "rv-$leg-serve" 2>&1 | grep -q "AssertionError\|Traceback" && { echo "   CRASHED"; return 1; }
    sleep 5
  done
  echo "   TIMEOUT after ${BOOT_TIMEOUT}s"; return 1
}

teardown() { docker rm -f "rv-$1-serve" >/dev/null 2>&1; }

run_leg() {  # $1 leg  $2 image  $3 expect_replay(0|1)
  local leg="$1" img="$2" want="$3"
  boot "$leg" "$img" || { docker logs "rv-$leg-serve" 2>&1 | tail -40; teardown "$leg"; return 1; }
  docker logs "rv-$leg-serve" > "$OUT/$leg.boot.log" 2>&1

  grep -oE "hip-engage\] gdn_hip\.[a-z0-9_]+" "$OUT/$leg.boot.log" | sort -u > "$OUT/$leg.engage.txt"
  echo "-- GDN kernels engaged ($leg) --"; sed 's/^/   /' "$OUT/$leg.engage.txt"
  local got=0
  grep -q "gdn_verify_replay" "$OUT/$leg.engage.txt" && got=1
  if [[ "$got" != "$want" ]]; then
    echo "PROVENANCE MISMATCH leg=$leg wanted replay-verify=$want got=$got"; teardown "$leg"; return 1
  fi
  # the control must have taken the materialising verify, not simply skipped verifying
  if [[ "$want" == "0" ]] && ! grep -q "gdn_prefill_verify" "$OUT/$leg.engage.txt"; then
    echo "PROVENANCE MISMATCH leg=$leg: control ran NEITHER verify kernel"; teardown "$leg"; return 1
  fi
  echo "   provenance OK (replay-verify engaged = $got)"

  grep -E "Allocating .* tokens for KV|Reserved |Free memory after" "$OUT/$leg.boot.log" \
    | sort -u > "$OUT/$leg.mem.txt"
  echo "-- pool sizing ($leg) --"; sed 's/^/   /' "$OUT/$leg.mem.txt"

  echo "-- acceptance ($leg), $ACC_REPS rep(s) --"
  : > "$OUT/$leg.acceptance.txt"
  for rep in $(seq 1 "$ACC_REPS"); do
    python3 tools/serve_acceptance.py --base "http://127.0.0.1:$PORT" --conc "$CONC" \
      >> "$OUT/$leg.acceptance.txt" 2>&1
  done
  grep -E "RESULT:|FAILED:" "$OUT/$leg.acceptance.txt" | sed 's/^/   /'

  echo "-- decode bench ($leg) --"
  python3 tools/serve_matrix_bench.py --url "http://127.0.0.1:$PORT" --label "$leg" \
    --m "$M_LIST" --workloads decode --decode-tokens 256 > "$OUT/$leg.bench.txt" 2>&1
  grep -E "^\s*[0-9]+\s+[0-9]" "$OUT/$leg.bench.txt" | sed 's/^/   /'
  teardown "$leg"
}

run_leg ctl "$CTL_IMAGE" 0 || echo "LEG ctl FAILED"
run_leg rvk "$RVK_IMAGE" 1 || echo "LEG rvk FAILED"

echo
echo "=================== SUMMARY (mem=$MEM_RATIO, spec=$SPEC k=$SPEC_K) ==================="
for leg in ctl rvk; do
  echo "--- $leg ---"
  grep -E "Allocating .* tokens for KV" "$OUT/$leg.mem.txt" 2>/dev/null | head -1
  grep -E "RESULT:" "$OUT/$leg.acceptance.txt" 2>/dev/null
  grep -E "^\s*[0-9]+\s+[0-9]" "$OUT/$leg.bench.txt" 2>/dev/null
done
