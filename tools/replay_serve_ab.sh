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
CONC="${CONC:-4}"
M_LIST="${M_LIST:-1,2,4}"
REPS="${REPS:-1}"
PROJECT="${PROJECT:-minisgl-rdna4-replayserve}"
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
  # ONE compose project for both legs, reusing the network this worktree already owns: the box's
  # docker address pool is fully subnetted by other agents' abandoned compose networks, so creating a
  # per-leg network fails with "all predefined address pools have been fully subnetted". The legs run
  # sequentially and are told apart by container_name (LEASE_NAME), so one project is enough.
  MINISGL_IMAGE="$img" MODEL="$MODEL_ALIAS" SPEC=none TP=2 CONC="$CONC" ATTN=hip \
    MINISGL_HOST_PORT="$PORT" LEASE_NAME="replay-$leg" COMPOSE_PROJECT_NAME="$PROJECT" \
    docker compose --profile serve up -d || return 1
  local t0=$SECONDS
  while (( SECONDS - t0 < BOOT_TIMEOUT )); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "   ready after $((SECONDS - t0))s"; return 0
    fi
    sleep 5
  done
  echo "   TIMEOUT after ${BOOT_TIMEOUT}s"; return 1
}

teardown() {   # by container, NOT `compose down`: down deletes the shared network we are reusing
  local leg="$1" img="$2"
  docker rm -f "replay-$leg-serve" >/dev/null 2>&1
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
  : > "$OUT/$leg.bench.txt"
  for rep in $(seq 1 "$REPS"); do
    echo "[rep $rep]" | tee -a "$OUT/$leg.bench.txt"
    python tools/serve_matrix_bench.py --url "http://127.0.0.1:$PORT" --label "$leg-r$rep" \
        --m "$M_LIST" --workloads decode --decode-tokens 256 2>&1 | tee -a "$OUT/$leg.bench.txt"
  done
  teardown "$leg" "$img"
}

run_leg control "$CTL_IMAGE" 0 || exit 1
run_leg replay  "$REPLAY_IMAGE" 1 || exit 1
echo
echo "=================== SUMMARY ==================="
for leg in control replay; do
  echo "--- $leg ---"
  grep -E "^\s+[0-9]+\s+[0-9]" "$OUT/$leg.bench.txt"
done
