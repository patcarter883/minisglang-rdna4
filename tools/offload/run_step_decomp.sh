#!/usr/bin/env bash
# Drive tools/offload/step_decomp.py in the serve image at the COMMITTED 48-layer operating point.
#
# DIFFERENCES FROM run_prof_decode_attrib.sh, each of them load-bearing:
#  * REPO defaults to the ramperf worktree, not -offload. Mounting a tree another agent is editing
#    is the source-isolation failure this repo's CLAUDE.md calls the equal of an unleased GPU.
#  * CHUNK_MIB defaults to 1372, not 750. The e4m3 scale policy changed the per-layer ROW SET
#    (block scales + a per-channel f32 global joined the fp4 rows), so 750 MiB is no longer a whole
#    multiple of a layer and the arena abandons the remainder of every chunk. 1372 = 18 chunks at
#    99.5% fill with 0 torch fallbacks on this checkpoint.
#  * The freshly built e4m3 fp8_wmma torch-ext is mounted at /kern-ext and put FIRST on PYTHONPATH,
#    ahead of /opt/kernels, whose fp8_wmma predates the e4m3 WLoad policy. step_decomp.py prints the
#    sha256 of the .so it actually imported; a stale kernel silently voids every number.
#  * MINISGL_HOSTPROF is set to a number so large the periodic log NEVER fires, which is the point:
#    the probe reads and diffs `Scheduler._hp` itself, and a `_hp_tick` reset mid-leg would zero the
#    accumulators under it.
#
# The GPU lease is WAIVED for this task. ONE invocation takes BOTH cards; never run two.
set -euo pipefail

REPO="${REPO:-/home/pat/code/minisgl-rdna4-ramperf}"
KERN_EXT="${KERN_EXT:?set KERN_EXT to the built fp8_wmma torch-ext dir}"
IMAGE="${IMAGE:-minisgl-rdna4:m1b-20260903}"
OUT="${OUT:?set OUT to the json path inside /engine}"

docker run --rm -i \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e ROCR_VISIBLE_DEVICES="${ROCR:-0,1}" \
  -v "$REPO":/engine \
  -v "$KERN_EXT":/kern-ext:ro \
  -v "${OUTDIR:?set OUTDIR to a writable host dir for the json}":/out \
  -v /home/pat/.cache/hf-q4e:/model:ro \
  -v /home/pat/.cache/hf-ple:/ple:ro \
  -e HF_HUB_OFFLINE=1 \
  -e MINISGL_PLE_META_FILES=/ple/model-bf16-00010.safetensors \
  -e MINISGL_WEIGHT_ARENA_CHUNK_MIB="${CHUNK_MIB:-1372}" \
  -e MINISGL_WEIGHT_ARENA_FLOOR_GIB="${FLOOR_GIB:-12}" \
  -e MINISGL_HOSTPROF="${HOSTPROF:-1000000000}" \
  -e KERNELS_REF="${KERNELS_REF:-}" \
  -e OUT="$OUT" \
  --entrypoint bash "$IMAGE" -s -- "$@" <<'INNER'
set -euo pipefail
export MINISGL_PLE_FILES="$(ls /ple/model-plefp8-*.safetensors | paste -sd:)"
export PYTHONPATH=/kern-ext:/engine/python:/opt/kernels
python - <<'PY'
import hashlib, os, fp8_wmma
p = os.path.dirname(fp8_wmma.__file__)
assert p.startswith("/kern-ext"), f"fp8_wmma resolved to {p}, NOT the fresh e4m3 build"
so = [f for f in os.listdir(p) if f.startswith("fp8_wmma_C") and f.endswith(".so")][0]
print(">> fp8_wmma", os.path.join(p, so))
print(">> sha256  ", hashlib.sha256(open(os.path.join(p, so), "rb").read()).hexdigest())
PY
exec python /engine/tools/offload/step_decomp.py --json "$OUT" "$@"
INNER
