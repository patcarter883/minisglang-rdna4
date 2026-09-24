#!/usr/bin/env bash
# Driver for tools/glm_drafter_attn_parity.py: parity once, then one (drafter, bs, ring case) per
# PROCESS (the timing graphs are never destroyed in-process — see timing() there). All runs share ONE
# lease, i.e. one card, recorded in the output. EVERY docker run is hard-bounded: a hung graph replay
# (host blocked in replay, GPU idle) fails the case with rc=124, the container is killed, and the
# driver moves on — a hang is never retried. Other crashes get one retry.
#   REPO=<clean worktree> OUT=docs/journal/measurements/<dir>/parity_and_timing.txt \
#     gpu-lease -n 1 -- bash tools/glm_drafter_attn_bench.sh [part ...]
set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
IMAGE="${IMAGE:-minisgl-rdna4:specfix-20260924}"
OUT="${OUT:?set OUT to a path relative to REPO}"
CASE_TIMEOUT="${CASE_TIMEOUT:-420}"
PARTS_LIST=("${@:-mtp eagle3}"); PARTS_LIST=(${PARTS_LIST[*]})
run() {
  local name="glmattn-bench-$$-$RANDOM"
  timeout -k 5 "$CASE_TIMEOUT" docker run --rm --name "$name" \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
    --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -v "$REPO":/engine -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
    -e PYTHONPATH=/engine/python:/opt/kernels -e OUT="/engine/$OUT" -e ARMS="${ARMS:-old,new,new_ctl}" -e ROUNDS="${ROUNDS:-5}" "$@" \
    --entrypoint bash "$IMAGE" -lc "python /engine/tools/glm_drafter_attn_parity.py; rc=\$?; chown $(id -u):$(id -g) /engine/$OUT; exit \$rc"
  local rc=$?
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then docker kill "$name" >/dev/null 2>&1; rc=124; fi
  return $rc
}
echo "# $(date -Is) image=$IMAGE repo=$(git -C "$REPO" rev-parse --short HEAD) ROCR=$ROCR_VISIBLE_DEVICES parts=${PARTS_LIST[*]}" >> "$REPO/$OUT"
if [ -z "${SKIP_PARITY:-}" ]; then
  run -e TIMING=0; rc=$?; [ $rc -ne 0 ] && echo "[bench] parity rc=$rc" >> "$REPO/$OUT"
fi
for part in "${PARTS_LIST[@]}"; do
  for bs in ${BS_LIST:-1 4}; do
    for case in ${CASE_LIST:-512/512 2048/2048 8192/8192 512/64}; do
      for attempt in 1 2; do
        run -e PARITY=0 -e PARTS=$part -e BS=$bs -e CASES=$case; rc=$?
        [ $rc -eq 0 ] && break
        echo "[bench] $part bs=$bs $case attempt $attempt rc=$rc$([ $rc -eq 124 ] && echo ' (TIMEOUT: hung; not retried)')" >> "$REPO/$OUT"
        [ $rc -eq 124 ] && break
      done
    done
  done
done
