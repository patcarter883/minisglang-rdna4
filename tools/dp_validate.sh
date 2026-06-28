#!/usr/bin/env bash
# DP launcher validation harness (HOST side). ONE lease for the whole run; each phase is its own
# hard-timeout-wrapped docker run so a GPU-wedged replica TIMES OUT -> FAIL (never hangs the box).
#
# Phases (decode-dominated by construction: short chat prompt + max_tokens=${MAXTOK:-256}):
#   ISO    dp=1 graph OFF   -> single-card kernel baseline. Tells whether the dp=2 0x1016 HSA
#                              exception is graph-capture vs the attention/MoE kernel itself.
#   DP2    dp=2 graph ON    -> the headline DP case (the one that crashed in attempt 1).
#   DP2NG  dp=2 graph OFF   -> only if DP2 fails; isolates graph-capture from the kernel on 2 cards.
#   DP1    dp=1 graph ON    -> single-replica baseline for the ~2x comparison.
#
# Coherence is driven through /v1/chat/completions (chat template applied) by the client, so the
# instruct model produces judge-able text. Per-request client timeout (REQ_TO) makes a wedged
# replica fail fast; the per-phase docker timeout (PHASE_TO) is the hard backstop.
#
# Knobs (env): CONC MAXTOK GRAPH PHASE_TO REQ_TO BOOT_TO RUN_DP2NG ATTN_HIP TAIL_HIP
set -uo pipefail

LEASE=/home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh
REPO=/home/pat/code/minisgl-rdna4
MODEL=/models/ZAYA1-8B-fp8
IMAGE=vllm22-w4a8:combined
GRAPH=${GRAPH:-16}
# CONC MUST oversubscribe a single replica for DP's ~2x to appear. Each replica is capped at
# MAX_RUNNING (default = GRAPH = one captured decode step), so a dp=1 server SERIALIZES CONC requests
# through bs=GRAPH waves while dp=2 runs two such replicas in PARALLEL. At CONC<=GRAPH a single
# replica batches everything into one step and the second card is idle -> the 2x gate spuriously
# fails (this was the attempt-1 "throughput-shortfall": CONC=16==GRAPH). Default CONC = 4x GRAPH.
CONC=${CONC:-64}
MAX_RUNNING=${MAX_RUNNING:-0}   # per-replica ceiling; 0 -> client uses GRAPH (one captured step)
MAXTOK=${MAXTOK:-256}        # decode-dominated: long generation so DP's steady-state win shows
PHASE_TO=${PHASE_TO:-420}    # per-phase hard timeout (s); boot ~120s + run; tight enough to fail fast
REQ_TO=${REQ_TO:-120}        # per-request client timeout (s)
BOOT_TO=${BOOT_TO:-300}
RUN_DP2NG=${RUN_DP2NG:-1}    # run the dp=2 graph-off fallback only if DP2 crashes (auto), or force=2
ATTN_HIP=${ATTN_HIP:-1}      # MINISGL_ATTN_HIP forwarded into the container (0 = pure-Triton attn)
TAIL_HIP=${TAIL_HIP:-1}      # MINISGL_TAIL_HIP forwarded into the container (0 = torch tail refs)

$LEASE -n 2 -- bash -s <<OUTER
set -uo pipefail
echo "[harness] HIP_VISIBLE_DEVICES=\$HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=\$ROCR_VISIBLE_DEVICES"
echo "[harness] CONC=${CONC} MAX_RUNNING=${MAX_RUNNING} MAXTOK=${MAXTOK} GRAPH=${GRAPH} PHASE_TO=${PHASE_TO} REQ_TO=${REQ_TO} ATTN_HIP=${ATTN_HIP} TAIL_HIP=${TAIL_HIP}"

# run_phase <label> <dp> <graph> [extra-client-args...]
run_phase() {
  label=\$1; dp=\$2; graph=\$3; shift 3
  echo "================ PHASE \$label (dp=\$dp graph=\$graph, timeout ${PHASE_TO}s) ================"
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
          --tag \$label $*"
  rc=\$?
  echo "================ PHASE \$label exit rc=\$rc ================"
  return \$rc
}

# 0) single-card kernel baseline, graph OFF (isolation: graph-capture vs kernel)
run_phase ISO 1 0; r_iso=\$?

# 1) the headline DP=2 case (graph on) — the attempt-1 crash
run_phase DP2 2 ${GRAPH}; r_dp2=\$?

# 2) DP=2 graph OFF fallback — run if DP2 crashed (auto) or RUN_DP2NG=2 (force)
r_dp2ng=skip
if [ "${RUN_DP2NG}" = "2" ] || { [ "${RUN_DP2NG}" = "1" ] && [ "\$r_dp2" != "0" ]; }; then
  run_phase DP2NG 2 0; r_dp2ng=\$?
fi

# 3) single-replica baseline (graph on) for the ~2x comparison
run_phase DP1 1 ${GRAPH}; r_dp1=\$?

echo "[harness] SUMMARY iso_rc=\$r_iso dp2_rc=\$r_dp2 dp2ng_rc=\$r_dp2ng dp1_rc=\$r_dp1"
OUTER
