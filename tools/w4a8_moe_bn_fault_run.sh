#!/usr/bin/env bash
# Run the fused-MoE-gemm1 BN fault repro in the serve image against THIS worktree, with fp8_wmma
# taken from an isolated KERNELS worktree.
#
#   gpu-lease -n 1 --timeout 1800 -- bash tools/w4a8_moe_bn_fault_run.sh --out /engine/_bnfault.txt
set -uo pipefail
IMG="${TILE_IMG:-minisgl-rdna4:lean}"
WT="${TILE_WT:-/home/pat/code/minisgl-rdna4-wn2}"
KWT="${TILE_KERNELS:-/home/pat/code/rdna4-hip-kernels-wn2}"
TOOL="${TILE_TOOL:-tools/w4a8_moe_bn_fault_repro.py}"

MOUNT_K=()
if [ -n "${KWT}" ]; then
  MOUNT_K=(-v "${KWT}/fp8_wmma/torch-ext/fp8_wmma:/opt/kernels/fp8_wmma:ro")
fi

docker run --rm --name bnfault-$$ \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e MINISGL_HIP_ENGAGE_LOG=0 \
  -e LEASE_ROCR_DEVICES="${LEASE_ROCR_DEVICES:-}" \
  -v "$WT":/engine "${MOUNT_K[@]}" \
  --entrypoint bash "$IMG" -lc \
    'PYTHONPATH=/opt/kernels:/engine/python:/engine exec python /engine/'"${TOOL}"' "$@"' \
    _ "$@"
