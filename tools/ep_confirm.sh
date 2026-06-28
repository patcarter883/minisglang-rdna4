#!/usr/bin/env bash
# EP graph-mode end-to-end confirm: GRAPH ON (production path), prefill token-count fix in place.
# ONE 2-card lease, ONE docker run, hard-timeout wrapped. Mounts the ISOLATED worktree, never $PWD.
# Confirms: boots + captures graphs WITH EP collectives in-graph, coherent output, no hang.
set -uo pipefail

LEASE=/home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh
REPO=${REPO:-/home/pat/code/minisgl-rdna4-zaya-dp-ep}
MODEL=/models/ZAYA1-8B-fp8
IMAGE=vllm22-w4a8:combined
GRAPH=${GRAPH:-16}
CONC=${CONC:-4}
MAXTOK=${MAXTOK:-32}
PHASE_TO=${PHASE_TO:-520}
REQ_TO=${REQ_TO:-90}
BOOT_TO=${BOOT_TO:-360}
EPFLAG=${EPFLAG:---enable-ep}
TAG=${TAG:-EPCONF}

$LEASE -n 2 -- bash -s <<OUTER
set -uo pipefail
echo "[ep-confirm] HIP=\$HIP_VISIBLE_DEVICES ROCR=\$ROCR_VISIBLE_DEVICES GRAPH=${GRAPH} EPFLAG=${EPFLAG}"
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
      python /engine/tools/dp_validate_client.py \\
        --dp 2 --model ${MODEL} --conc ${CONC} --max-tokens ${MAXTOK} \\
        --port 1919 --graph ${GRAPH} --boot-timeout ${BOOT_TO} --req-timeout ${REQ_TO} \\
        --tag ${TAG} ${EPFLAG}"
echo "[ep-confirm] docker rc=\$?"
OUTER
