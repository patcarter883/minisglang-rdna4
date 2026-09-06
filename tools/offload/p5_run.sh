#!/usr/bin/env bash
# P5 -- torch over a foreign device pointer, IN THE SERVE IMAGE, under graph capture.
#
# Runs tools/offload/p5_torch_foreign_ptr.py inside this repo's serve image with ROCm device
# passthrough, mounting THIS WORKTREE (never $PWD of the shared tree -- source-isolation rule).
#
#   ./tools/offload/p5_run.sh                 # all default legs
#   ./tools/offload/p5_run.sh --selftest      # no GPU: arg + JSON-shape validation, in-image
#   MINISGL_IMAGE=minisgl-rdna4:lean ./tools/offload/p5_run.sh --legs expandable_device
#
# Notes bound by CLAUDE.md / the P5 brief:
#   * ROCR_VISIBLE_DEVICES=0,1 and HIP_VISIBLE_DEVICES deliberately UNSET -- ROCm device 2 is the
#     Ryzen iGPU (47 GB of GTT) and must never enter enumeration. The python script re-forces this.
#   * The GPU lease is WAIVED for this development work by explicit user instruction. Do not run
#     two GPU probes concurrently.
#   * PYTORCH_CUDA_ALLOC_CONF is NOT set here: the python parent composes it per leg (the
#     expandable_segments coexistence question is the point of the probe).
#   * The shared vllm-gfx1201 Triton cache is deliberately NOT mounted. P5 compiles nothing (plain
#     ATen elementwise ops), so there is no autotune to save, and mounting the shared production
#     cache read-write from a throwaway probe is the corruption risk CLAUDE.md warns about.
#     TRITON_CACHE_DIR points at container-local scratch instead.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
IMAGE="${MINISGL_IMAGE:-minisgl-rdna4:lean}"
OUT_REL="docs/measurements/WEIGHT_OFFLOAD_2026-09-02"

if [[ "$REPO" == "/home/pat/code/minisgl-rdna4" ]]; then
  echo "REFUSING: \$REPO is the SHARED tree. Run this from an isolated git worktree." >&2
  exit 3
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "FATAL: image '$IMAGE' not found (set MINISGL_IMAGE)." >&2
  exit 3
fi
mkdir -p "$REPO/$OUT_REL"

IMAGE_ID="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
GIT_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"

echo "[p5_run] image=$IMAGE ($IMAGE_ID)"
echo "[p5_run] worktree=$REPO  ->  /engine   git=$GIT_SHA dirty_files=$GIT_DIRTY"
echo "[p5_run] out=$REPO/$OUT_REL"
# Box state on the HOST side too -- the in-container /proc view of memory is the host's, but
# rocm-smi's per-card view is worth capturing from here as well, before the run perturbs it.
{
  echo "image=$IMAGE id=$IMAGE_ID"
  echo "worktree=$REPO git=$GIT_SHA dirty_files=$GIT_DIRTY"
  date -Is
  rocm-smi --showuse --showmeminfo vram 2>&1 || true
  free -g 2>&1 || true
  grep -E '^(pswpin|pswpout|pgmajfault) ' /proc/vmstat 2>&1 || true
} | tee "$REPO/$OUT_REL/p5_host_box_state.txt" | sed 's/^/[p5_run] /'

# Args are forwarded through a properly quoted string: `$*` would split on spaces and silently
# mangle anything like --legs "a, b".
ARGS_Q=""
if [[ $# -gt 0 ]]; then ARGS_Q="$(printf '%q ' "$@")"; fi

exec docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e ROCR_VISIBLE_DEVICES=0,1 \
  -e MINISGL_P5_ROCR_DEVICES=0,1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/tmp/triton-p5 \
  -e P5_IMAGE="$IMAGE" -e P5_IMAGE_ID="$IMAGE_ID" -e P5_GIT_SHA="$GIT_SHA" \
  -e P5_CHOWN_UID="$(id -u)" -e P5_CHOWN_GID="$(id -g)" \
  -v "$REPO":/engine \
  --entrypoint bash "$IMAGE" -lc \
  "python /engine/tools/offload/p5_torch_foreign_ptr.py --out-dir /engine/$OUT_REL $ARGS_Q"
