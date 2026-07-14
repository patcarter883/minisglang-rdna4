#!/usr/bin/env bash
# DFlash-on-ZAYA accept-len + losslessness against the DEPLOYED RXF serve (ZAYA1-8B-RXF-h32), on the
# lean image with the LIVE kernels repo mounted -> exercises the current batch-invariant verify
# kernels. THE decisive re-measure of the verify-M fix: does mean accept-len recover from ~0.26?
#
# Launch under a 1-card lease (single card = a full DP replica; capture/serve are TP=1-exact):
#   gpu-lease -n 1 -- bash tools/run_zaya_dflash_accept.sh
#
# Knobs (env): DRAFT (drafter ckpt), NUM_DRAFT (match the drafter's trained block), GRAPH (0 eager,
#   default; 8 = prod tok/s pass), MEMRATIO. Default drafter = the vLLM-trained on-policy m4dss-opd-r1
#   (the 0.26-on-minisgl / 1.30-on-vLLM one) so the delta is attributable to the kernel fix ALONE, not
#   a re-distill. To test the ns15 (15-wide) drafter: DRAFT=.../ZAYA1-8B-DFlash-CCA-5L-ns15-ep14 NUM_DRAFT=15.
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/root/.cache/huggingface/ZAYA1-8B-RXF-h32}"
DRAFT_HOST="${DRAFT_HOST:-/home/pat/code/_models/ZAYA1-8B-DFlash-CCA-5L-m4dss-ep14-opd-r1}"
NUM_DRAFT="${NUM_DRAFT:-4}"; GRAPH="${GRAPH:-0}"; MEMRATIO="${MEMRATIO:-0.82}"; GENTOK="${GENTOK:-96}"
MINV="${MINV:-1}"   # 1 = M-invariant dense_gemm FIX (default); 0 = rocBLAS floor (old broken path)
KHEAD="$(git -C /home/pat/code/rdna4-hip-kernels rev-parse --short HEAD 2>/dev/null || echo unknown)"
CNAME="${LEASE_NAME:-zaya-dflash-accept}-accept"
trap 'docker rm -f "$CNAME" >/dev/null 2>&1 || true' EXIT INT TERM
echo "[dflash-accept] HIP=${HIP_VISIBLE_DEVICES:-unset} draft=$DRAFT_HOST num_draft=$NUM_DRAFT graph=$GRAPH"
docker run --rm --name "$CNAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e HF_HUB_OFFLINE=1 \
  -e MODEL="$MODEL" -e SPEC_ALGO=dflash -e DRAFT=/draft -e NUM_DRAFT="$NUM_DRAFT" \
  -e GRAPH="$GRAPH" -e MEMRATIO="$MEMRATIO" -e GENTOK="$GENTOK" -e KV_FP8=1 -e MOE_SCATTER=0 \
  -e MINISGL_MINV_GEMM="$MINV" -e KHEAD="$KHEAD" \
  -e DDTREE="${DDTREE:-1}" -e DDTREE_BUDGET="${DDTREE_BUDGET:-32}" -e SAMPLED="${SAMPLED:-0}" \
  -v "$PWD":/engine \
  -v /home/pat/code/rdna4-hip-kernels:/kernels:ro \
  -v "$DRAFT_HOST":/draft:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash minisgl-rdna4:lean /engine/tools/zaya_spec_accept.sh
echo "[dflash-accept] exited rc=$?"
