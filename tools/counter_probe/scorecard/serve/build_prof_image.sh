#!/usr/bin/env bash
# Rebuild the rocprofv3-capable serve image FROM WHATEVER `lean` IS NOW.
#
# WHY THIS EXISTS. minisgl-rdna4:lean-prof was built once, by hand, and then went stale. The engine
# source is hot-mounted from the worktree (today's), but /opt/kernels is BAKED (that image's), so a
# stale prof image serves today's engine against three-day-old kernels — and the engine dies at boot:
#
#   RuntimeError: tail_hip_C::store_kv() expected at most 7 argument(s) but received 9 argument(s)
#
# i.e. the kernel ABI moved (per-tensor -> per-head fp8 KV scales) and the baked package did not.
# That is the "a .so built in one image will NOT load in another" rule from CLAUDE.md, in its slower
# form: it does not fail at build, it fails at boot, on the run you were about to measure.
#
# The escape hatches are both worse than fixing the image:
#   MINISGL_KV_FP8=0    does NOT help — the CALL SIGNATURE changed, not just the dtype policy.
#   MINISGL_TAIL_HIP=0  boots, but swaps the HIP tail for a torch fallback and so changes the very
#                       kernel mix being timed.
#
# WHAT `lean-prof` ACTUALLY IS. Two things, and MISSING EITHER ONE fails at boot, not at build:
#
#   1. A symlink. torch vendors its own librocprofiler-sdk.so, which registers AFTER rocprofv3 has
#      closed its configuration window, and the process then core-dumps in rocprofiler_configure.
#      Pointing torch's copy at the ROCm one makes them the same library.
#
#   2. LD_LIBRARY_PATH=/opt/rocprof-deps. Without it the rocprofv3 LD_PRELOAD cannot resolve libdw,
#      and since the preload applies to EVERY exec in the container the first casualty is /usr/bin/env:
#        /usr/bin/env: error while loading shared libraries: libdw.so.1: cannot open shared object file
#      The library is present in both images — it is only unreachable — so this reads like a missing
#      package and is not one.
#
# Nothing else about the image differs, so the profiled and unprofiled legs can and should run on
# THIS image, which makes them matched by construction rather than by hope.
#
#   bash tools/counter_probe/scorecard/serve/build_prof_image.sh [BASE_TAG] [OUT_TAG]
# CPU only. No GPU, no lease.
set -euo pipefail
BASE=${1:-minisgl-rdna4:lean}
OUT=${2:-minisgl-rdna4:lean-prof-today}
LIB=/opt/venv/lib/python3.12/site-packages/torch/lib/librocprofiler-sdk.so

echo "building $OUT from $BASE"
docker build -t "$OUT" -f - . <<DOCKERFILE
FROM $BASE
RUN set -e; \
    if [ ! -e "$LIB.vendored.bak" ]; then mv "$LIB" "$LIB.vendored.bak"; fi; \
    ln -sf /opt/rocm/lib/librocprofiler-sdk.so.1 "$LIB"
ENV LD_LIBRARY_PATH=/opt/rocprof-deps:\${LD_LIBRARY_PATH}
DOCKERFILE

echo "verifying the symlink, and that rocprofv3 can actually LAUNCH something under its preload"
echo "(a --version check would pass even with the libdw path broken — it is the exec that fails):"
docker run --rm --entrypoint bash "$OUT" -lc "ls -la $LIB && rocprofv3 --kernel-trace -- /usr/bin/env true && echo 'preloaded exec OK'"
echo "OK: $OUT"
