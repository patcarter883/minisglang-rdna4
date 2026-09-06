#!/usr/bin/env bash
# Drive tests/qwen4exp_qsa_longctx_test.py in the serve image, TP=2, on BOTH leased cards.
#
# WHY THIS IS NOT tools/run_offload_serve.sh with different flags. Three things this run needs that
# that script cannot give it, each of which silently produces a wrong number rather than an error:
#
#  * TWO fresh kernel packages on the path AHEAD of /opt/kernels, not one. The image
#    (minisgl-rdna4:m1b-20260903) bakes kernels at 169be57, which predates BOTH the NVFP4 e4m3 scale
#    policy `fp8_wmma` needs for this checkpoint AND the `qsa_index` package that does not exist
#    there at all. A missing qsa_index does not crash: `build_qsa_runtime` catches the ImportError,
#    logs a warning, and the full-attention layers run DENSE — i.e. the run would measure the very
#    path this feature replaces, and every "sparse" number in it would be a lie. The harness gates
#    on `ctx.qsa is not None` for exactly that reason, and the provenance block below gates on the
#    .so PATH so a stale /opt/kernels copy cannot answer the import either.
#  * /opt/kernels STAYS on PYTHONPATH, appended last. Dropping it turns this repo's __init__.py-less
#    HIP directories into empty namespace packages that import clean with zero attributes.
#  * THE LEASE IS NOT WAIVED and is not taken here. Call this THROUGH `gpu-lease -n 2 --` and it
#    forwards the arbiter's composed pair verbatim. TP=2 genuinely needs both cards; `-n` is HOW
#    MANY, not which, and hand-setting HIP/ROCR is how a card-1 assignment becomes "No HIP GPUs".
#
# The container is NAMED and reaped on exit: a 48-layer TP=2 boot pins ~48 GiB of host RAM and both
# cards, and an orphan holding those poisons every subsequent measurement on this box.
set -euo pipefail

REPO="${REPO:-/home/pat/code/minisgl-rdna4-qsa}"
KERN="${KERN:-/home/pat/code/rdna4-hip-kernels-qsa}"
IMAGE="${IMAGE:-minisgl-rdna4:m1b-20260903}"
OUTDIR="${OUTDIR:?set OUTDIR to a writable host dir}"
TAG="${TAG:?set TAG}"
RUN_TIMEOUT="${RUN_TIMEOUT:-7200}"

mkdir -p "$OUTDIR"
CNAME="qsa-longctx-${TAG}-$$"
cleanup() { docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

timeout --signal=KILL "$RUN_TIMEOUT" docker run --rm -i --name "$CNAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}" \
  -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}" \
  -v "$REPO":/engine \
  -v "$KERN":/kern:ro \
  -v "$OUTDIR":/out \
  -v /home/pat/.cache/hf-q4e:/model:ro \
  -v /home/pat/.cache/hf-ple:/ple:ro \
  -e HF_HUB_OFFLINE=1 \
  -e MINISGL_PLE_META_FILES=/ple/model-bf16-00010.safetensors \
  -e MINISGL_WEIGHT_ARENA_CHUNK_MIB="${CHUNK_MIB:-1372}" \
  -e MINISGL_WEIGHT_ARENA_FLOOR_GIB="${FLOOR_GIB:-6}" \
  `# DEVICE-SYNCED per-forward timing, ON by default HERE and off everywhere else. STEP_LOG's` \
  `# default window closes when the host finished ENQUEUING, so a captured decode step — one` \
  `# hipGraphLaunch that returns immediately — logs its launch latency and not its forward. This` \
  `# script exists to produce an eager-vs-captured ms/forward, and that comparison is meaningless` \
  `# without the bracket: MEASURED, 0.54 ms/step captured against 60.8 eager, a 112x that is` \
  `# entirely the instrument. The harness GATES on it for any --graph-bs > 0 leg and records it in` \
  `# the artifact. STEP_LOG_SYNC=0 to take the host-enqueue number deliberately.` \
  -e MINISGL_STEP_LOG_SYNC="${STEP_LOG_SYNC:-1}" \
  `# DEBUG-ONLY passthrough. AMD_SERIALIZE_KERNEL=3 makes every dispatch synchronous so a` \
  `# HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION names the kernel that faulted instead of` \
  `# aborting the queue some launches later. Unset (empty) on every measurement run: it` \
  `# serialises the whole forward and the timings it produces are not comparable.` \
  -e AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-}" \
  -e AMD_LOG_LEVEL="${AMD_LOG_LEVEL:-}" \
  --entrypoint bash "$IMAGE" -s -- "$@" <<'INNER'
set -euo pipefail
export MINISGL_PLE_FILES="$(ls /ple/model-plefp8-*.safetensors | paste -sd:)"
export PYTHONPATH=/engine/python:/kern/fp8_wmma/torch-ext:/kern/qsa_index/torch-ext:/opt/kernels
python - <<'PY'
import hashlib, os
import fp8_wmma, qsa_index
for mod, want in ((fp8_wmma, "/kern/fp8_wmma"), (qsa_index, "/kern/qsa_index")):
    p = os.path.dirname(mod.__file__)
    assert p.startswith(want), f"{mod.__name__} resolved to {p}, NOT the fresh build under {want}"
    so = [f for f in os.listdir(p) if f.endswith(".so")][0]
    h = hashlib.sha256(open(os.path.join(p, so), "rb").read()).hexdigest()
    print(f">> {mod.__name__}: {os.path.join(p, so)}\n>> sha256 {h}", flush=True)
PY
exec python /engine/tests/qwen4exp_qsa_longctx_test.py "$@"
INNER
