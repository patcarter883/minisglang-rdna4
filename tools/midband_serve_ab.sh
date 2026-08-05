#!/usr/bin/env bash
# Served A/B of the w4a8 dense MID-BAND dispatch rule, old vs new, in the serve image.
#
#   gpu-lease -n 2 --timeout 5400 -- bash tools/midband_serve_ab.sh
#
# TP=2 so it needs both cards. `minisgl-rdna4:gemma4` ONLY — :lean and :gemma4-dgd predate the fp16
# tail kernels and raise "bf16 only" on an fp16 checkpoint.
set -uo pipefail
IMG="${MB_IMG:-minisgl-rdna4:gemma4}"
WT="${MB_WT:-/home/pat/code/minisgl-rdna4-midband}"
mkdir -p "$WT/_surface"
# One container per leg -- see the VRAM note in _midband_serve_inner.sh.
for leg in new:0 old:1; do
MB_LEG="${leg%%:*}"; MB_LEG_IDX="${leg##*:}"
echo "########## LEG $MB_LEG ##########"
docker run --rm --name "midband-serve-$MB_LEG-$$" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 \
  -e MB_MODEL="${MB_MODEL:-cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4}" \
  -e MB_TP="${MB_TP:-2}" -e MB_CONC="${MB_CONC:-20}" -e MB_MAXTOK="${MB_MAXTOK:-256}" \
  -e MB_MEM="${MB_MEM:-0.86}" -e MB_READY_S="${MB_READY_S:-240}" -e PYTHONUNBUFFERED=1 \
  -e MB_LEG="$MB_LEG" -e MB_LEG_IDX="$MB_LEG_IDX" \
  -v "$WT":/engine \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash "$IMG" -lc 'bash /engine/tools/_midband_serve_inner.sh'
sleep 20   # let the container's KFD allocations actually go back to the driver
done
