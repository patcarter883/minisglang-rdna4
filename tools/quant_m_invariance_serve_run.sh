#!/usr/bin/env bash
# Serve-level M-invariance probe (outer half). ONE lease, ONE container; the two legs are in
# tools/_qminv_serve_inner.sh so no heredoc has to survive a nested shell quote.
#
#   gpu-lease -n 2 --timeout 5400 -- bash tools/quant_m_invariance_serve_run.sh
set -uo pipefail
IMG="${QMINV_IMG:-minisgl-rdna4:gemma4}"
WT="${QMINV_WT:-/home/pat/code/minisgl-rdna4-qminv}"
NAME="qminv-serve"

docker rm -f "$NAME" >/dev/null 2>&1
exec docker run --rm --name "$NAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 \
  -e QMINV_MODEL="${QMINV_MODEL:-cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4}" \
  -e QMINV_TP="${QMINV_TP:-2}" -e QMINV_N="${QMINV_N:-12}" -e QMINV_MAXTOK="${QMINV_MAXTOK:-24}" \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  -v "$WT":/engine \
  --entrypoint bash "$IMG" -lc 'bash /engine/tools/_qminv_serve_inner.sh'
