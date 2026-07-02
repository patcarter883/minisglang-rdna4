#!/usr/bin/env bash
# In-engine RSA validation: ONE 1-card lease, hard-timeout-wrapped docker run, isolated worktree.
# Serves Zaya on 1919 and exercises a PLAIN call + an RSA call (rsa:{n,k,t,...}) on the same port.
set -uo pipefail

LEASE=gpu-lease
REPO=${REPO:-/home/pat/code/minisgl-rdna4-rsa-engine}
MODEL=/models/ZAYA1-8B-fp8
IMAGE=vllm22-w4a8:combined
PHASE_TO=${PHASE_TO:-520}

$LEASE -n 1 -- bash -s <<OUTER
set -uo pipefail
echo "[rsa-harness] HIP=\$HIP_VISIBLE_DEVICES ROCR=\$ROCR_VISIBLE_DEVICES"
timeout --signal=KILL ${PHASE_TO} docker run --rm \\
  --device /dev/kfd --device /dev/dri --group-add video \\
  --security-opt seccomp=unconfined --security-opt label=disable \\
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \\
  -e HIP_VISIBLE_DEVICES="\$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="\$ROCR_VISIBLE_DEVICES" \\
  -e MINISGL_ATTN_HIP=1 -e MINISGL_TAIL_HIP=1 \\
  -v ${REPO}:/engine \\
  -v /home/pat/models:/models:ro \\
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \\
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \\
  --entrypoint bash ${IMAGE} -lc "
    source /app/.venv/bin/activate
    pip install -q msgpack pyzmq prompt_toolkit accelerate 2>/dev/null
    PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 \\
      python /engine/tools/rsa_validate_client.py --model ${MODEL} --port 1919 --graph 16 \\
        --rsa-n 4 --rsa-k 2 --rsa-t 2 --rsa-max-tokens 192 --rsa-tail-tokens 256"
echo "[rsa-harness] docker rc=\$?"
OUTER
