#!/usr/bin/env bash
# seed_kv KV-parity (batched seed_buffered vs autoregressive step_masked) for the prompt-prefill
# draft-KV seed. step_masked attends on the attn_decode HIP kernel, so this runs in the SERVE image
# (kernels at /opt/kernels), mounting a CLEAN worktree (REPO=<worktree>; never the shared tree).
# Launch UNDER the shared lease (single card):
#   REPO=<worktree> gpu-lease -n 1 -- bash tools/run_seed_kv_parity.sh
set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
IMAGE="${IMAGE:-minisgl-rdna4:specfix-20260924}"
echo "[run_seed_kv_parity] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset} REPO=$REPO IMAGE=$IMAGE"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -v "$REPO":/engine \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/engine/python:/opt/kernels \
  --entrypoint bash "$IMAGE" -lc '
    set -e
    CKPT=$(ls -d /root/.cache/huggingface/hub/models--thoughtworks--GLM-4.7-Flash-Eagle3/snapshots/*/ | head -1)
    echo "[run_seed_kv_parity] EAGLE3_CKPT=$CKPT"
    EAGLE3_CKPT="$CKPT" python /engine/tools/seed_kv_parity.py
    EAGLE3_CKPT="$CKPT" python /engine/tools/eagle3_parity.py
  '
echo "[run_seed_kv_parity] exited rc=$?"
