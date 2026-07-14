#!/usr/bin/env bash
# GPU validation for sampled (rejection-sampling) speculative verify (MINISGL_SPEC_SAMPLED).
#
# PART A (correctness, decisive) — the in-process distributional validator: probs_from_logits vs the
# fused HIP sampler, verify_sampled emits ~ p, greedy reduction. No model, ~seconds. THIS is the gate.
#   gpu-lease -n 1 -- bash tools/run_sampled_spec_validate.sh
#
# PART B (integration smoke) — engage + coherence on a WORKING drafter (drafter QUALITY is irrelevant
# to correctness; sampled-spec is lossless for any drafter). Boot the qwen35b-dflash profile with
# sampled-spec on and send a temp>0 request; confirm spec engages (accept-len > 0) and output is
# coherent + no crash/desync. (Kept as instructions below — it needs the 35B + 2 cards.)
#   gpu-lease -n 2 --detach --name samp -- bash -c '
#     MINISGL_EXTRA_ARGS="" docker compose --profile qwen35b-dflash up -d'   # then set the env:
#   # add to the qwen35b-dflash env:  MINISGL_SPEC_SAMPLED=1
#   curl :1919/v1/completions -d '{"model":"m","prompt":"Explain a hash map.","max_tokens":128,"temperature":0.8,"top_p":0.95}'
#   docker compose --profile qwen35b-dflash logs | grep -E "SAMPLED .* ENABLED|\[spec\] mean accept"
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[samp-validate] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -v "$PWD":/engine \
  -v /home/pat/code/rdna4-hip-kernels:/kernels:ro \
  --entrypoint bash "${MINISGL_IMAGE:-minisgl-rdna4:lean}" -lc '
    set -uo pipefail
    source /app/.venv/bin/activate 2>/dev/null || true
    export PYTHONPATH=/kernels/_kernels:/engine/python:/engine
    python /engine/tools/validate_sampled_spec.py
  '
echo "[samp-validate] exited rc=$?"
