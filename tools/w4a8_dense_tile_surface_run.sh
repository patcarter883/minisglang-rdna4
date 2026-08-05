#!/usr/bin/env bash
# Run a w4a8 dense TILE sweep in the serve image against THIS worktree, with the fp8_wmma package
# taken from an isolated KERNELS worktree (the extended tile set is not in the image's /opt/kernels).
#
#   gpu-lease -n 1 --timeout 10800 -- bash tools/w4a8_dense_tile_surface_run.sh \
#       --out /engine/_tile_surface.txt --csv /engine/_tile_surface.csv
#
# TILE_KERNELS may be left empty to run against the image's baked /opt/kernels (shipped tiles only).
set -uo pipefail
IMG="${TILE_IMG:-minisgl-rdna4:lean}"
WT="${TILE_WT:-/home/pat/code/minisgl-rdna4-tile}"
KWT="${TILE_KERNELS:-/home/pat/code/rdna4-hip-kernels-tile}"
TOOL="${TILE_TOOL:-tools/w4a8_dense_tile_surface.py}"

MOUNT_K=()
if [ -n "${KWT}" ]; then
  # Overlay ONLY the fp8_wmma module; every other kernel still comes from the image's build, so the
  # .so ABI stays matched to the image that compiled it.
  MOUNT_K=(-v "${KWT}/fp8_wmma/torch-ext/fp8_wmma:/opt/kernels/fp8_wmma:ro")
fi

docker run --rm --name tilesurface-$$ \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e MINISGL_HIP_ENGAGE_LOG=0 \
  -e TILE_EXCLUSIVE="${TILE_EXCLUSIVE:-0}" -e LEASE_ROCR_DEVICES="${LEASE_ROCR_DEVICES:-}" \
  -v "$WT":/engine "${MOUNT_K[@]}" \
  --entrypoint bash "$IMG" -lc \
    'PYTHONPATH=/opt/kernels:/engine/python:/engine exec python /engine/'"${TOOL}"' "$@"' \
    _ "$@"
