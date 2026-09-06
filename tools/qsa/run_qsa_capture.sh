#!/usr/bin/env bash
# Drive tests/qwen4exp_qsa_capture_test.py in the serve image on ONE leased card.
#
# The 4-layer subset builds UNQUANTIZED (docs/measurements/QSA_INDEXER.md §3: a randomly-filled
# NVFP4 build reaches the MoE kernel with fp8 scales the real loader would have cast to fp16), so
# only `qsa_index` has to be fresh here — but it has to be fresh, and it has to be THE FRESH ONE:
# a missing qsa_index does not crash, `build_qsa_runtime` catches the ImportError and the
# full-attention layers run DENSE, i.e. the capture-identity gate would pass against the very path
# this feature replaces. The provenance block below asserts the resolved module PATH and prints the
# .so sha256 before the harness starts.
#
# /opt/kernels STAYS on PYTHONPATH, appended last: dropping it turns this repo's __init__.py-less
# HIP directories into empty namespace packages that import clean with zero attributes.
#
# MINISGL_MOE_G2FUSE=0 is not optional and not a tuning knob: the decode gemm2 atomic scatter is
# documented non-bit-exact, so a bit-identity gate run with it on is measuring the SCATTER.
#
# THE LEASE IS NOT TAKEN HERE. Call this THROUGH `gpu-lease -n 1 --`; it forwards the arbiter's
# composed pair verbatim (hand-setting HIP/ROCR is how a card-1 assignment becomes "No HIP GPUs").
set -euo pipefail

REPO="${REPO:-/home/pat/code/minisgl-rdna4-qsa}"
KERN="${KERN:-/home/pat/code/rdna4-hip-kernels-qsa}"
IMAGE="${IMAGE:-minisgl-rdna4:m1b-20260903}"
OUTDIR="${OUTDIR:?set OUTDIR to a writable host dir}"
TAG="${TAG:?set TAG}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"

mkdir -p "$OUTDIR"
CNAME="qsa-capture-${TAG}-$$"
cleanup() { docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

timeout --signal=KILL "$RUN_TIMEOUT" docker run --rm -i --name "$CNAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}" \
  -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}" \
  -v "$REPO":/engine \
  -v "$KERN":/kern:ro \
  -v "$OUTDIR":/out \
  -v /home/pat/.cache/hf-q4e:/model:ro \
  -e HF_HUB_OFFLINE=1 \
  -e MINISGL_MOE_G2FUSE=0 \
  --entrypoint bash "$IMAGE" -s -- "$@" <<'INNER'
set -euo pipefail
export PYTHONPATH=/engine/python:/kern/qsa_index/torch-ext:/opt/kernels
python - <<'PY'
import hashlib, os
import qsa_index
p = os.path.dirname(qsa_index.__file__)
assert p.startswith("/kern/qsa_index"), f"qsa_index resolved to {p}, NOT the fresh build"
so = [f for f in os.listdir(p) if f.endswith(".so")][0]
print(f">> qsa_index: {os.path.join(p, so)}\n>> sha256 "
      f"{hashlib.sha256(open(os.path.join(p, so),'rb').read()).hexdigest()}", flush=True)
PY
exec python /engine/tests/qwen4exp_qsa_capture_test.py "$@"
INNER
