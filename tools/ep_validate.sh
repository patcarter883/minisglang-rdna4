#!/usr/bin/env bash
# EP (expert-parallel) validation harness (HOST side). ONE 2-card lease; each phase is its own
# hard-timeout-wrapped docker run so a DEADLOCKED EP lockstep TIMES OUT -> FAIL (never hangs the box).
#
# EP shards the 16 ZAYA experts across the 2 dp replicas (8/rank) and combines via all_gather +
# masked-local w8a8_moe + all_reduce, INSIDE the captured decode graph (common-bs lockstep agreed
# OUTSIDE the graph). This validates: (a) it boots + serves coherently UNDER GRAPH CAPTURE, (b)
# greedy-parity vs the DP-only replicated-expert path, (c) the A/B throughput/KV headline.
#
# Phases:
#   EP2     dp=2 --enable-ep graph ON  -> the headline (EP collectives inside the captured graph).
#   DP2     dp=2            graph ON  -> A/B reference (experts replicated, no EP collectives).
#   EP2NG   dp=2 --enable-ep graph OFF -> eager EP (bring-up isolation), only if EP2 fails (auto).
#
# Knobs (env): CONC MAXTOK GRAPH PHASE_TO REQ_TO BOOT_TO RUN_EP2NG ATTN_HIP TAIL_HIP
set -uo pipefail

LEASE=gpu-lease
# Source isolation: mount the ISOLATED per-task worktree, NEVER the shared $PWD checkout (a concurrent
# agent edits that tree mid-run -> the container reads torn code). Override with REPO=... if needed.
REPO=${REPO:-/home/pat/code/minisgl-rdna4-zaya-dp-ep}
MODEL=/models/ZAYA1-8B-fp8
IMAGE=vllm22-w4a8:combined
GRAPH=${GRAPH:-16}
CONC=${CONC:-64}
MAX_RUNNING=${MAX_RUNNING:-0}
MAXTOK=${MAXTOK:-256}
PHASE_TO=${PHASE_TO:-480}
REQ_TO=${REQ_TO:-120}
BOOT_TO=${BOOT_TO:-360}
RUN_EP2NG=${RUN_EP2NG:-1}
ATTN_HIP=${ATTN_HIP:-1}
TAIL_HIP=${TAIL_HIP:-1}

$LEASE -n 2 -- bash -s <<OUTER
set -uo pipefail
echo "[harness] HIP_VISIBLE_DEVICES=\$HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=\$ROCR_VISIBLE_DEVICES"
echo "[harness] CONC=${CONC} MAX_RUNNING=${MAX_RUNNING} MAXTOK=${MAXTOK} GRAPH=${GRAPH} PHASE_TO=${PHASE_TO} REQ_TO=${REQ_TO} ATTN_HIP=${ATTN_HIP} TAIL_HIP=${TAIL_HIP}"

# run_phase <label> <dp> <graph> <ep:0|1>
run_phase() {
  label=\$1; dp=\$2; graph=\$3; ep=\$4
  epflag=""; [ "\$ep" = "1" ] && epflag="--enable-ep"
  echo "================ PHASE \$label (dp=\$dp graph=\$graph ep=\$ep, timeout ${PHASE_TO}s) ================"
  timeout --signal=KILL ${PHASE_TO} docker run --rm \\
    --device /dev/kfd --device /dev/dri --group-add video \\
    --security-opt seccomp=unconfined --security-opt label=disable \\
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \\
    -e HIP_VISIBLE_DEVICES="\$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="\$ROCR_VISIBLE_DEVICES" \\
    -e MINISGL_ATTN_HIP=${ATTN_HIP} -e MINISGL_TAIL_HIP=${TAIL_HIP} \\
    -v ${REPO}:/engine \\
    -v /home/pat/models:/models:ro \\
    -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \\
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \\
    --entrypoint bash ${IMAGE} -lc "
      source /app/.venv/bin/activate
      pip install -q msgpack pyzmq prompt_toolkit accelerate 2>/dev/null
      PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 \\
        python /engine/tools/dp_validate_client.py \\
          --dp \$dp --model ${MODEL} --conc ${CONC} --max-tokens ${MAXTOK} \\
          --max-running ${MAX_RUNNING} \\
          --port 1919 --graph \$graph --boot-timeout ${BOOT_TO} --req-timeout ${REQ_TO} \\
          --tag \$label \$epflag"
  rc=\$?
  echo "================ PHASE \$label exit rc=\$rc ================"
  return \$rc
}

# 1) headline: DP=2 + EP, graph ON (EP collectives captured into the decode graph)
run_phase EP2 2 ${GRAPH} 1; r_ep2=\$?

# 2) A/B reference: DP=2, graph ON, experts replicated (no EP)
run_phase DP2 2 ${GRAPH} 0; r_dp2=\$?

# 3) eager EP fallback (bring-up isolation) if the captured EP crashed/deadlocked
r_ep2ng=skip
if [ "${RUN_EP2NG}" = "2" ] || { [ "${RUN_EP2NG}" = "1" ] && [ "\$r_ep2" != "0" ]; }; then
  run_phase EP2NG 2 0 1; r_ep2ng=\$?
fi

echo "[harness] SUMMARY ep2_rc=\$r_ep2 dp2_rc=\$r_dp2 ep2ng_rc=\$r_ep2ng"
OUTER
