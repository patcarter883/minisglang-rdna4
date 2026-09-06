#!/usr/bin/env bash
# Drive tools/offload/hc_fusion_ab.py in the serve image: OLD hyper-connection code vs NEW, one boot.
#
# Same operating point as run_hc_capture_prize.sh (tools/serve.sh's `qwen4exp` arm at MEM_RATIO 0.90)
# so the numbers land on the same footing as docs/measurements/QWEN4EXP_ENDGAME.md's.
#
# GPU lease is WAIVED for this task. ONE invocation takes BOTH cards; never run two.
set -euo pipefail

REPO="${REPO:-/home/pat/code/minisgl-rdna4-hcfuse}"
IMAGE="${IMAGE:-minisgl-rdna4:m1b-20260903}"
BASE="${BASE:-HEAD}"
OUTDIR="$REPO/docs/measurements/HC_FUSION_2026-09-05"
OUT="${OUT:-/engine/docs/measurements/HC_FUSION_2026-09-05/hc_fusion_ab.json}"

mkdir -p "$OUTDIR"

# The BASELINE LEG's source, dumped from git HOST-SIDE. The container mounts the worktree without
# its `.git` (it is a linked worktree; the real gitdir lives under the parent repo), so the harness
# cannot run `git show` itself. Dumping the blobs here keeps the baseline the OLD CODE rather than a
# re-implementation of it, and records the sha it came from.
python3 - "$REPO" "$BASE" "$OUTDIR/baseline_source.json" <<'PY'
import json, subprocess, sys
repo, base, out = sys.argv[1:4]
sha = subprocess.check_output(["git", "-C", repo, "rev-parse", base], text=True).strip()
paths = ["python/minisgl/layers/hyperconnection.py", "python/minisgl/layers/norm.py"]
files = {p: subprocess.check_output(["git", "-C", repo, "show", f"{sha}:{p}"], text=True)
         for p in paths}
json.dump({"base_ref": base, "base_sha": sha, "files": files}, open(out, "w"), indent=1)
print(f"[baseline] {base} = {sha}  ({', '.join(paths)})")
PY

docker run --rm -i \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e ROCR_VISIBLE_DEVICES="${ROCR:-0,1}" \
  -v "$REPO":/engine \
  -v /home/pat/.cache/hf-q4e:/model:ro \
  -v /home/pat/.cache/hf-ple:/ple:ro \
  -e HF_HUB_OFFLINE=1 \
  -e MINISGL_PLE_META_FILES=/ple/model-bf16-00010.safetensors \
  -e MINISGL_WEIGHT_ARENA_CHUNK_MIB="${CHUNK_MIB:-750}" \
  -e MINISGL_WEIGHT_ARENA_FLOOR_GIB="${FLOOR_GIB:-9}" \
  -e OUT="$OUT" \
  --entrypoint bash "$IMAGE" -s -- "$@" <<'INNER'
set -euo pipefail
export MINISGL_PLE_FILES="$(ls /ple/model-plefp8-*.safetensors | paste -sd:)"
export PYTHONPATH=/engine/python:/opt/kernels
exec python /engine/tools/offload/hc_fusion_ab.py --json "$OUT" "$@"
INNER
