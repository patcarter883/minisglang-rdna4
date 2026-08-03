#!/usr/bin/env bash
# RULE-4 serve validation + decode A/B for ReplaySSM on the GDN path.
#
# Two legs, IDENTICAL engine source (both mount this worktree at /engine) and identical serve config;
# the ONLY difference is the baked kernel package:
#   replay  = kernels WITH gdn_decode_conv_gated_replay  -> the engine allocates the ring and takes it
#   control = kernels WITHOUT it                          -> ring is None, the pre-change decode ladder
# That is a real provenance difference, not a flag, and each leg ASSERTS which path it took by
# grepping the [hip-engage] log for the op name (an A/B whose control silently ran the candidate is
# the classic way to measure new-vs-itself).
#
# Run under a 2-card lease from the worktree root:
#   gpu-lease -n 2 -- bash tools/replay_serve_ab.sh
set -uo pipefail

REPLAY_IMAGE="${REPLAY_IMAGE:-minisgl-rdna4:replayssm}"
CTL_IMAGE="${CTL_IMAGE:-minisgl-rdna4:replayctl}"
OUT="${OUT:-/tmp/replay_ab}"
PORT="${PORT:-1919}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen35b-awq}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
mkdir -p "$OUT"

preflight() {  # an ABI-mismatched .so shows up as a 0%-GPU wedge at readiness timeout, not a build error
  local img="$1"
  echo "== preflight: import the kernel packages in $img =="
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
    -e ROCR_VISIBLE_DEVICES="${LEASE_ROCR_DEVICES:-0}" -e HIP_VISIBLE_DEVICES="${LEASE_HIP_DEVICES:-0}" \
    --entrypoint bash "$img" -lc '
      python - <<PY
import torch, gdn_hip as g
print("torch", torch.__version__, "hip", torch.version.hip, "cuda-avail", torch.cuda.is_available())
print("has fused replay:", hasattr(g, "gdn_decode_conv_gated_replay"),
      "| ring L:", getattr(g, "REPLAY_RING_LEN", None))
import fp8_wmma, attn_hip  # the other serve-path packages must load in the SAME image
print("kernel packages import OK")
PY' || return 1
}

boot() {
  local leg="$1" img="$2"
  echo "== boot leg=$leg image=$img =="
  MINISGL_IMAGE="$img" MODEL="$MODEL_ALIAS" SPEC=none TP=2 CONC=4 ATTN=hip \
    MINISGL_HOST_PORT="$PORT" LEASE_NAME="replay-$leg" COMPOSE_PROJECT_NAME="replay-$leg" \
    docker compose --profile serve up -d >/dev/null 2>&1 || return 1
  local t0=$SECONDS
  while (( SECONDS - t0 < BOOT_TIMEOUT )); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "   ready after $((SECONDS - t0))s"; return 0
    fi
    sleep 5
  done
  echo "   TIMEOUT after ${BOOT_TIMEOUT}s"; return 1
}

teardown() {
  local leg="$1" img="$2"
  MINISGL_IMAGE="$img" COMPOSE_PROJECT_NAME="replay-$leg" LEASE_NAME="replay-$leg" \
    docker compose --profile serve down >/dev/null 2>&1
}

run_leg() {
  local leg="$1" img="$2" want_replay="$3"
  preflight "$img" | tee "$OUT/$leg.preflight.txt" || { echo "PREFLIGHT FAILED"; return 1; }
  boot "$leg" "$img" || { docker logs "replay-$leg-serve" 2>&1 | tail -60; teardown "$leg" "$img"; return 1; }

  # provenance: which decode rung actually engaged
  docker logs "replay-$leg-serve" 2>&1 | grep -E "hip-engage.*gdn" | sort -u > "$OUT/$leg.engage.txt"
  echo "-- engaged GDN kernels ($leg) --"; cat "$OUT/$leg.engage.txt"
  if grep -q "gdn_decode_conv_gated_replay" "$OUT/$leg.engage.txt"; then got=1; else got=0; fi
  if [[ "$got" != "$want_replay" ]]; then
    echo "PROVENANCE MISMATCH leg=$leg wanted replay=$want_replay got=$got"
    teardown "$leg" "$img"; return 1
  fi
  echo "   provenance OK (replay engaged = $got)"

  # VRAM accounting: the ring must show up in the reserved recurrent state, not silently over-commit
  docker logs "replay-$leg-serve" 2>&1 | grep -E "Reserved .* recurrent state|Allocating .* tokens for KV|Free memory after" \
    | sort -u > "$OUT/$leg.mem.txt"
  echo "-- pool sizing ($leg) --"; cat "$OUT/$leg.mem.txt"

  echo "-- coherence ($leg) --"
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
      "model":"x","max_tokens":220,"temperature":0,"top_p":1.0,
      "messages":[{"role":"user","content":"In three short paragraphs, explain what a Gated Delta Net is and how its recurrent state differs from standard attention. Then list three practical consequences for serving."}]
    }' | tee "$OUT/$leg.coherence.json" | python -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])'

  echo "-- decode bench ($leg) --"
  python tools/serve_matrix_bench.py --url "http://127.0.0.1:$PORT" --label "$leg" \
      --m 1,2,4 --workloads decode --decode-tokens 256 2>&1 | tee "$OUT/$leg.bench.txt"
  teardown "$leg" "$img"
}

run_leg control "$CTL_IMAGE" 0 || exit 1
run_leg replay  "$REPLAY_IMAGE" 1 || exit 1
echo
echo "=================== SUMMARY ==================="
for leg in control replay; do echo "--- $leg ---"; grep -E "decode|tok/s|TPOT" "$OUT/$leg.bench.txt"; done
