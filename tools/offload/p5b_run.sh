#!/usr/bin/env bash
# P5b -- torch over a hipHostGetDevicePointer address, IN THE SERVE IMAGE, under graph capture.
#
# Runs tools/offload/p5b_torch_host_arena.py inside this repo's serve image with ROCm device
# passthrough, mounting THIS WORKTREE (never $PWD of the shared tree -- source-isolation rule).
#
#   ./tools/offload/p5b_run.sh                  # all default legs (both cards)
#   ./tools/offload/p5b_run.sh --selftest       # no GPU: arg + JSON-shape validation, in-image
#   MINISGL_IMAGE=minisgl-rdna4:lean ./tools/offload/p5b_run.sh --legs expandable_card1
#
# Notes bound by CLAUDE.md and the P5b brief:
#   * ROCR_VISIBLE_DEVICES=0,1 and HIP_VISIBLE_DEVICES deliberately UNSET -- ROCm device 2 is the
#     Ryzen iGPU (47 GB of GTT) and must never enter enumeration. The python script re-forces this
#     before torch is importable, and the legs select cards by TORCH INDEX inside that set.
#   * The GPU lease is WAIVED for this development work by explicit user instruction. Legs run
#     strictly serially and the parent takes an advisory flock; never start a second GPU job
#     alongside this one.
#   * PYTORCH_{CUDA,HIP}_ALLOC_CONF are NOT set here: the python parent composes them per leg,
#     because coexistence with the compose default (expandable_segments:True) is the question.
#   * --shm-size / --ipc host matter: the probe pins a ~700 MiB host arena per leg.
#   * The shared vllm-gfx1201 Triton cache is deliberately NOT mounted. P5b compiles nothing (plain
#     ATen ops), so there is no autotune to save, and mounting the shared production cache
#     read-write from a throwaway probe is the corruption risk CLAUDE.md warns about.
#   * /sys is visible read-only in the container, which is what lets the placement arm read the
#     per-card amdgpu mem_info_vram_used / mem_info_gtt_used counters. If it is not, the probe
#     falls back to hipMemGetInfo + MemAvailable + physics and SAYS SO in the artifact.
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

echo "[p5b_run] image=$IMAGE ($IMAGE_ID)"
echo "[p5b_run] worktree=$REPO  ->  /engine   git=$GIT_SHA dirty_files=$GIT_DIRTY"
echo "[p5b_run] out=$REPO/$OUT_REL"

# Host-side box state BEFORE the run perturbs it. The in-container /proc view of memory is the
# host's, but rocm-smi's per-card view is worth capturing from here as well -- and this is the
# record that both cards were idle when the run started, since no arbiter is enforcing it.
{
  echo "image=$IMAGE id=$IMAGE_ID"
  echo "worktree=$REPO git=$GIT_SHA dirty_files=$GIT_DIRTY"
  date -Is
  rocm-smi --showuse --showmeminfo vram 2>&1 || true
  free -g 2>&1 || true
  grep -E '^(pswpin|pswpout|pgmajfault) ' /proc/vmstat 2>&1 || true
  for c in /sys/class/drm/card*/device; do
    [[ -r "$c/mem_info_vram_used" ]] || continue
    echo "$(basename "$(readlink -f "$c")") vram_used=$(cat "$c/mem_info_vram_used" 2>/dev/null) gtt_used=$(cat "$c/mem_info_gtt_used" 2>/dev/null)"
  done
} | tee "$REPO/$OUT_REL/p5b_host_box_state.txt" | sed 's/^/[p5b_run] /'

# Args are forwarded through a properly quoted string: `$*` would split on spaces and silently
# mangle anything like --legs "a, b".
ARGS_Q=""
if [[ $# -gt 0 ]]; then ARGS_Q="$(printf '%q ' "$@")"; fi

exec docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e ROCR_VISIBLE_DEVICES=0,1 \
  -e MINISGL_P5B_ROCR_DEVICES=0,1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/tmp/triton-p5b \
  -e P5B_IMAGE="$IMAGE" -e P5B_IMAGE_ID="$IMAGE_ID" \
  -e P5B_GIT_SHA="$GIT_SHA" -e P5B_GIT_DIRTY="$GIT_DIRTY" \
  -e P5B_CHOWN_UID="$(id -u)" -e P5B_CHOWN_GID="$(id -g)" \
  -v "$REPO":/engine \
  --entrypoint bash "$IMAGE" -lc \
  "python /engine/tools/offload/p5b_torch_host_arena.py --out-dir /engine/$OUT_REL --scratch-dir /engine/tools/offload/_build $ARGS_Q"
