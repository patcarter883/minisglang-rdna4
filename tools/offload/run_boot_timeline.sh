#!/usr/bin/env bash
# Boot attribution for the offloaded qwen4_exp path: drive tests/qwen4exp_offload_serve_test.py in
# the serve image with `weights/boot_timeline.py` recording, and keep the JSON.
#
# DIFFERENCES FROM tools/run_offload_serve.sh, each load-bearing:
#  * REPO defaults to the ramperf worktree (source isolation; -offload has another agent in it).
#  * MINISGL_BOOT_TIMELINE_JSON is exported, so every rank writes its own rank-suffixed report into
#    a host-visible directory instead of only into the log.
#  * THE GPU LEASE IS NOT WAIVED. Call this THROUGH `gpu-lease -n 1 --` (or `-n 2` for --tp 2) and
#    forward the arbiter's composed pair; this script does NOT lease for you and does NOT hardcode a
#    card. Passing ROCR/HIP by hand is how a card-1 assignment turns into "No HIP GPUs are available".
#  * Everything is wrapped in `timeout`, because a boot that wedges holds a lease and silently
#    contaminates every concurrent CPU-side measurement on the box.
#  * KERN_EXT (the freshly built e4m3 `fp8_wmma` torch-ext) is mounted at /kbuild and put on
#    PYTHONPATH AHEAD OF /opt/kernels, whose baked fp8_wmma predates the e4m3 WLoad policy and
#    carries ZERO `E4m3GroupScaleGlobal` symbols. Without this the boot's post_load hands the
#    kernels an e4m3 scale and dies with `scales must be fp16` — and the provenance assert below
#    is what turns "wrong kernel" from a silent void into a first-second failure. /opt/kernels
#    STAYS on the path: dropping it turns this repo's `__init__.py`-less HIP dirs into empty
#    namespace packages.
set -euo pipefail

REPO="${REPO:-/home/pat/code/minisgl-rdna4-ramperf}"
KERN_EXT="${KERN_EXT:?set KERN_EXT to the built e4m3 fp8_wmma torch-ext dir (the one CONTAINING fp8_wmma/)}"
IMAGE="${IMAGE:-minisgl-rdna4:m1b-20260903}"
TEST="${TEST:-/engine/tests/qwen4exp_offload_serve_test.py}"
OUTDIR="${OUTDIR:?set OUTDIR to a writable host dir for the boot-timeline json}"
TAG="${TAG:?set TAG, e.g. L8_dev2}"
RUN_TIMEOUT="${RUN_TIMEOUT:-3600}"

mkdir -p "$OUTDIR"

# REAP ON TIMEOUT. `timeout` kills the docker CLI, NOT the container it started — a run that hangs
# would otherwise keep both cards and 54 GiB of pinned RAM after this script has exited, which is
# precisely how the previous attempt's numbers were contaminated (an orphan spinning at 197% CPU for
# ten hours sat under every measurement taken that afternoon). Naming the container is what makes
# the cleanup possible at all.
CNAME="boot-timeline-${TAG}-$$"
cleanup() { docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

timeout --signal=KILL "$RUN_TIMEOUT" docker run --rm -i --name "$CNAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}" \
  -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}" \
  -v "$REPO":/engine \
  -v "$KERN_EXT":/kbuild:ro \
  -v "$OUTDIR":/out \
  -v /home/pat/.cache/hf-q4e:/model:ro \
  -v /home/pat/.cache/hf-ple:/ple:ro \
  -e HF_HUB_OFFLINE=1 \
  -e MINISGL_PLE_META_FILES=/ple/model-bf16-00010.safetensors \
  -e MINISGL_WEIGHT_ARENA_CHUNK_MIB="${CHUNK_MIB:-1372}" \
  -e MINISGL_WEIGHT_ARENA_FLOOR_GIB="${FLOOR_GIB:-12}" \
  -e MINISGL_BOOT_TIMELINE_JSON="/out/${TAG}.boot.json" \
  -e TEST="$TEST" \
  --entrypoint bash "$IMAGE" -s -- "$@" <<'INNER'
set -euo pipefail
export MINISGL_PLE_FILES="$(ls /ple/model-plefp8-*.safetensors | paste -sd:)"
export PYTHONPATH=/engine/python:/kbuild:/opt/kernels
python - <<'PY'
import hashlib, os, fp8_wmma
p = os.path.dirname(fp8_wmma.__file__)
assert p.startswith("/kbuild"), f"fp8_wmma resolved to {p}, NOT the fresh e4m3 build"
so = [f for f in os.listdir(p) if f.startswith("fp8_wmma_C") and f.endswith(".so")][0]
print(">> fp8_wmma", os.path.join(p, so), flush=True)
print(">> sha256  ", hashlib.sha256(open(os.path.join(p, so), "rb").read()).hexdigest(), flush=True)
PY
exec python "$TEST" "$@"
INNER
