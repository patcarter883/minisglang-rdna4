#!/usr/bin/env bash
# Boot an arm, wait for readiness, run the tool-call degeneration probe, tear down. Two arms:
# q4e (Qwen3.8-Flash-Next, the suspect) and qwen36 (GDN-without-PLE, the control that separates
# "this engine" from "the q4e-only composite prefix-cache path").
#
#   tools/toolcall_degen_ab.sh q4e
#   tools/toolcall_degen_ab.sh qwen36
#   tools/toolcall_degen_ab.sh both
#
# READINESS IS /v1/models, NEVER a short completion — a thinking model's first 8 tokens never leave
# the <think> span, so a probe that reads `content` can never pass and costs a whole boot.
set -uo pipefail
cd "$(dirname "$0")/.."

ARM="${1:-q4e}"
REPS="${REPS:-4}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"
LOGDIR="${LOGDIR:-/home/pat/fixtures/minisgl-toolcall-degen}"
mkdir -p "$LOGDIR"

# ARM_LABEL overrides the fixture label without changing the lease/container name, so two runs of
# the SAME arm from DIFFERENT source trees (e.g. pre-fix vs fixed worktree) land in separate,
# self-identifying fixtures instead of two directories that only a timestamp tells apart.
# EXTRA_ARGS is forwarded to the serve verbatim (serve.sh appends it last and it wins).
run_arm() {
  local model="$1" expect="$2" name="$3" extra="${4:-}"
  local label="${ARM_LABEL:-$name}"
  local log="$LOGDIR/$(date +%Y%m%d-%H%M%S)-${ARM_LABEL:-$name}-serve.log"
  echo "=== arm $label: booting MODEL=$model EXTRA_ARGS='$extra' (log: $log)"
  echo "=== source tree: $PWD  (commit $(git rev-parse --short HEAD 2>/dev/null || echo '?'))"
  MODEL="$model" EXTRA_ARGS="$extra" gpu-lease -n 2 --detach --name "$name" -- \
      docker compose --profile serve up -d || { echo "boot failed"; return 1; }

  local container="lease-$name-serve"
  echo "=== waiting up to ${READY_TIMEOUT}s for /v1/models"
  local t0=$SECONDS
  until curl -sf http://127.0.0.1:1919/v1/models >/dev/null 2>&1; do
    if (( SECONDS - t0 > READY_TIMEOUT )); then
      echo "TIMEOUT after $((SECONDS-t0))s — dumping the tail and giving up"
      docker logs --tail 60 "$container" 2>&1 | tee -a "$log"
      docker compose -p "lease-$name" --profile serve down 2>/dev/null
      return 1
    fi
    if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
      echo "container exited during boot — log tail:"
      docker logs --tail 60 "$container" 2>&1 | tee -a "$log"
      return 1
    fi
    sleep 10
  done
  echo "=== ready in $((SECONDS-t0))s"

  # Snapshot the boot log BEFORE probing: the probe greps it for the composite prefix-cache line.
  docker logs "$container" > "$log" 2>&1
  grep -m1 "recurrent-radix" "$log" || echo "  (no recurrent-radix line in the boot log)"

  python3 tools/toolcall_degen_probe.py \
      --arm "$label" --expect-model "$expect" --reps "$REPS" --serve-log "$log"
  local rc=$?

  echo "=== arm $label: tearing down"
  docker compose -p "lease-$name" --profile serve down 2>/dev/null
  return $rc
}

case "$ARM" in
  q4e)    run_arm q4e        "Qwen3.8-Flash-Next" q4edeg ;;
  qwen36) run_arm qwen35b-awq "Qwen3.6"           q36deg ;;
  # SAME CHECKPOINT, suspect path OFF. The qwen36 arm changes the MODEL as well as the path, so a
  # q4e-vs-qwen36 gap cannot separate "this NVFP4 checkpoint is worse at tool arguments" from "the
  # q4e-only composite prefix-cache path corrupts state". This one moves only the path.
  noradix) run_arm q4e "Qwen3.8-Flash-Next" q4edeg "--no-gdn-radix" ;;
  both)   run_arm q4e "Qwen3.8-Flash-Next" q4edeg; run_arm qwen35b-awq "Qwen3.6" q36deg ;;
  *)      echo "usage: $0 {q4e|qwen36|noradix|both}   (ARM_LABEL=<name> to label the fixture)"; exit 2 ;;
esac
