#!/usr/bin/env bash
# P2 — mixed-media grouped MoE GEMM, in the serve image, against THIS worktree.
#
#   bash /home/pat/code/minisgl-rdna4-offload/tools/offload/p2_run.sh
#
# The GPU lease is WAIVED for this development work (explicit user instruction), so this script
# does NOT call gpu-lease. It must therefore be run SERIALLY — never alongside another GPU probe.
#
# Notes that are load-bearing:
#  * ROCR_VISIBLE_DEVICES=0,1 is set here AND re-asserted by the python script itself, and
#    HIP_VISIBLE_DEVICES is deliberately left UNSET. ROCm device 2 is the Ryzen iGPU advertising
#    ~47 GB of GTT; if it enters enumeration it poisons any "biggest free pool" logic. Setting both
#    variables double-filters and breaks whenever card 1 is used (CLAUDE.md).
#  * The WORKTREE is mounted, never $PWD — the shared tree is mutable state with no lock.
#  * Full ROCm device passthrough is mandatory or is_rocm() is False and Triton disables itself.
#  * NO Triton cache is mounted on purpose: this probe compiles nothing through Triton (baked HIP
#    kernels + ATen only), and mounting the shared production cache read-write for a throwaway run
#    is how that cache gets corrupted.
#
# Add --selftest to validate arguments, layout arithmetic, the route builder, the shape classifier,
# the DLPack binding and the JSON schema WITHOUT touching the GPU (drop the --device flags too).
set -uo pipefail

WT="${P2_WT:-/home/pat/code/minisgl-rdna4-offload}"
IMG="${P2_IMG:-minisgl-rdna4:lean}"
TOOL="tools/offload/p2_mixed_media_moe.py"

if [ ! -d "$WT/.git" ] && [ ! -f "$WT/.git" ]; then
  echo "P2_WT=$WT is not a git worktree" >&2; exit 2
fi

docker run --rm --name p2-mixedmedia-$$ \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e ROCR_VISIBLE_DEVICES=0,1 \
  -e HF_HUB_OFFLINE=1 \
  -e PYTHONUNBUFFERED=1 \
  -v "$WT":/engine \
  --entrypoint bash "$IMG" -lc \
    'unset HIP_VISIBLE_DEVICES; PYTHONPATH=/opt/kernels:/engine/python:/engine \
       exec python /engine/'"$TOOL"' --out-dir /engine/docs/measurements/WEIGHT_OFFLOAD_2026-09-02 "$@"' \
    _ "$@"
