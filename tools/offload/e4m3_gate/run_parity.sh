#!/usr/bin/env bash
# Record the e4m3-merge parity dump against ONE build. Called twice per build so the SAME-BUILD
# floor is measured, not assumed — `mmq_fp8_moe_gemm_scatter` is an atomic reduction whose order
# varies, and a cross-build delta on such a tensor is not evidence.
#
#   BUILD=<torch-ext dir>  OUTDIR=<host dir>  bash run_parity.sh
#
# ONE card (ROCR=0). Compile-here-run-here: the .so under test was built in this same image.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
BUILD="${BUILD:?set BUILD to a torch-ext dir}"
OUTDIR="${OUTDIR:?set OUTDIR}"
IMAGE="${IMAGE:-minisgl-rdna4:m1b-20260903}"
mkdir -p "$OUTDIR"

docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e ROCR_VISIBLE_DEVICES=0 \
  -v "$BUILD":/build:ro \
  -v "$S":/probe:ro \
  -v "$OUTDIR":/out \
  -e PARITY_OUT=/out \
  -e PARITY_SKIP="${PARITY_SKIP:-}" -e PARITY_ONLY="${PARITY_ONLY:-}" \
  --entrypoint bash "$IMAGE" -lc '
    set -euo pipefail
    export PYTHONPATH=/build
    python -c "import fp8_wmma,os;p=os.path.dirname(fp8_wmma.__file__);assert p.startswith(\"/build\"),p;print(\">> fp8_wmma from\",p)"
    python /probe/parity_e4m3_merge.py'
