#!/usr/bin/env bash
# Drive tests/qwen4exp_offload_serve_test.py in the serve image.
#
# BOTH CARDS ARE EXPOSED BY DEFAULT (`ROCR_VISIBLE_DEVICES=0,1`, `HIP_VISIBLE_DEVICES` deliberately
# UNSET), because the harness now takes `--tp`. Every qwen4_exp run before this one was TP=1 on card
# 0 — not by choice but because `-e ROCR_VISIBLE_DEVICES=0` here made TP=2 impossible to even
# attempt, while `tools/serve.sh` defaults to TP=2 and every other model in this repo serves there.
# `ROCR=0` pins a run back to card 0 when a single-card comparison is the point.
# ROCm device 2 is the Ryzen iGPU and must NEVER appear in that list.
#
# The GPU lease is WAIVED for this task; never run two of these at once. At `--tp 2` ONE invocation
# takes BOTH cards, so a concurrent second run is not "sharing" — it is two jobs on one card.
# PYTHONPATH is APPENDED to, never replaced: the bare form drops /opt/kernels and the repo root's
# __init__.py-less HIP dirs import as EMPTY namespace packages.
#
# Arguments are forwarded to the harness through `bash -s --`, NOT interpolated into a `-lc` string:
# a `--quality-prompt "a sentence with spaces"` re-splits on every word under interpolation and
# argparse rejects it three minutes into a boot.
#
# CHUNK_MIB is a FEASIBILITY term, not a tuning knob, and the two are not the same size of mistake.
# The host arena packs next-fit and a region may never straddle a chunk, so a chunk that is not a
# whole multiple of the per-layer row set abandons the remainder of every chunk. Measured on this
# checkpoint at TP=2 (`tools/offload/plan_chunk_sweep.py`, 48 layers, 12 device layers): the rows are
# 400/200/100/50 MiB, i.e. exactly 750 MiB per layer, so CHUNK_MIB=750 reserves 26.367 GiB/rank with
# ZERO waste while 768 reserves 27.000 and 3072 also 27.000 (+0.633 GiB/rank = +1.27 GiB/node) and
# 1024 reserves 36.000 (+9.63 GiB/rank — one layer per chunk). Pick the chunk from the row sizes.
#
# FLOOR_GIB is the `MemAvailable` the capacity gate refuses to eat into. 12 is the DEFAULT and the
# policy P3b's ceilings were measured under; it is exposed only so a run can be told to probe BELOW
# it deliberately, which is the only way to find this box's real 2-rank pinning ceiling
# (`host_capacity.PINNED_CEILING_BYTES[2]` is still P3b's un-remeasured 62 GiB and its own comment
# calls it suspect). Anything other than 12 is a MEASUREMENT operating point that must be recorded
# next to the number it produced — never a way to wave an infeasible plan through.
set -euo pipefail

REPO="${REPO:-/home/pat/code/minisgl-rdna4-offload}"
IMAGE="${IMAGE:-minisgl-rdna4:m1b-20260903}"
TEST="${TEST:-/engine/tests/qwen4exp_offload_serve_test.py}"

docker run --rm -i \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e ROCR_VISIBLE_DEVICES="${ROCR:-0,1}" \
  -v "$REPO":/engine \
  -v /home/pat/.cache/hf-q4e:/model:ro \
  -v /home/pat/.cache/hf-ple:/ple:ro \
  -e HF_HUB_OFFLINE=1 \
  -e MINISGL_PLE_META_FILES=/ple/model-bf16-00010.safetensors \
  -e MINISGL_WEIGHT_ARENA_CHUNK_MIB="${CHUNK_MIB:-3072}" \
  -e MINISGL_WEIGHT_ARENA_FLOOR_GIB="${FLOOR_GIB:-12}" \
  -e TEST="$TEST" \
  --entrypoint bash "$IMAGE" -s -- "$@" <<'INNER'
set -euo pipefail
export MINISGL_PLE_FILES="$(ls /ple/model-plefp8-*.safetensors | paste -sd:)"
export PYTHONPATH=/engine/python:/opt/kernels
exec python "$TEST" "$@"
INNER
