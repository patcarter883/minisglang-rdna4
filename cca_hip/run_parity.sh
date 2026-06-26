#!/usr/bin/env bash
# Build the vendored cca_hip kernel + run its standalone parity tests under a 1-card lease.
set -euo pipefail
LEASE=/home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh
WT=/home/pat/code/minisgl-rdna4
"$LEASE" -n 1 -- bash -c '
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
    -v '"$WT"':/engine \
    -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
    --entrypoint bash vllm22-w4a8:combined -lc "
      source /app/.venv/bin/activate
      cd /engine/cca_hip
      echo === BUILD ===
      GPU_ARCHS=gfx1201 python setup.py build_ext --inplace 2>&1 | grep -iE \"error generated|FAILED|copying.*\.so\" | head
      echo === PARITY ===
      for t in test_cca_kernel.py test_cca_decode_qk.py test_cca_prefill_qk.py test_cca_mixed_qk.py; do
        echo \"--- \$t ---\"; python \$t 2>&1 | tail -8
      done"'
