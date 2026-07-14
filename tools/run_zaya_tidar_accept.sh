#!/usr/bin/env bash
# TiDAR-on-ZAYA accept-len + losslessness on the self-draft fp8 TiDAR checkpoint, lean image + LIVE
# kernels mount -> exercises the current batch-invariant verify kernels. Re-measures whether the
# verify-M fix lifts TiDAR acceptance off its floor (two-forward ~0.18 / fused ~0.07). No RXF-TiDAR
# exists (TiDAR is a separately full-FT'd diffusion checkpoint), so this uses the fp8 TiDAR model; the
# verify-M fix is quant-orthogonal (it's attention/GEMM batch-invariance), so this is still a valid read.
#
# Launch under a 1-card lease (fp8 ~9.6GB fits one 16GB card):
#   gpu-lease -n 1 -- bash tools/run_zaya_tidar_accept.sh
#
# Knobs (env): NUM_DRAFT (TiDAR block_size, config=4), FUSED (1 = single-forward fused path; default 0
#   = two-forward, the only lossless mode), GRAPH (0 eager default; 8 = prod tok/s), MEMRATIO.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/big/zaya1-tidar-opd-fp8}"
NUM_DRAFT="${NUM_DRAFT:-4}"; GRAPH="${GRAPH:-0}"; MEMRATIO="${MEMRATIO:-0.85}"; GENTOK="${GENTOK:-96}"
FUSED="${FUSED:-0}"; MIX="${MIX:-1.0}"
CNAME="${LEASE_NAME:-zaya-tidar-accept}-accept"
trap 'docker rm -f "$CNAME" >/dev/null 2>&1 || true' EXIT INT TERM
echo "[tidar-accept] HIP=${HIP_VISIBLE_DEVICES:-unset} model=$MODEL num_draft=$NUM_DRAFT fused=$FUSED graph=$GRAPH"
docker run --rm --name "$CNAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e HF_HUB_OFFLINE=1 \
  -e MODEL="$MODEL" -e SPEC_ALGO=tidar -e NUM_DRAFT="$NUM_DRAFT" \
  -e GRAPH="$GRAPH" -e MEMRATIO="$MEMRATIO" -e GENTOK="$GENTOK" -e KV_FP8=1 -e MOE_SCATTER=0 \
  -e MINISGL_TIDAR_FUSED="$FUSED" -e MINISGL_TIDAR_MIX_BETA="$MIX" \
  -v "$PWD":/engine \
  -v /home/pat/code/rdna4-hip-kernels:/kernels:ro \
  -v /home/pat/code/_big:/big:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash minisgl-rdna4:lean /engine/tools/zaya_spec_accept.sh
echo "[tidar-accept] exited rc=$?"
