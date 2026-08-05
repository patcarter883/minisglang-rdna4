#!/usr/bin/env bash
# car_vec_serve_ab.sh — end-to-end serve A/B for the vectorized custom_ar P2P collectives.
#
# The ONLY difference between the two arms is which custom_ar_C*.so is mounted at
# /opt/kernels/custom_ar. Everything else — image, worktree, model, prompts, seeds, engine env — is
# byte-identical, so a difference in the result is attributable to the kernel and nothing else.
#
#   BASE  = the image's own custom_ar (scalar peer read: one element per thread per iteration)
#   VEC   = the candidate (16-byte dwordx4 peer read)
#
# Provenance is ASSERTED, not assumed: each arm md5s the .so it is about to serve with and the two
# md5s must DIFFER, otherwise the "A/B" is new-vs-itself and the script fails before booting anything.
#
# The kernel is bit-exact (tools/car_ar_vec_bench.py proves vec == scalar == RCCL bit for bit), so the
# quality gate is the strongest one available: at temperature 0 the generated TEXT must be
# CHARACTER-IDENTICAL. It is gated PER PHASE, because only one of the two phases can carry that gate:
#
#   canvas  — HARD GATE. Byte-identical or the change was not lossless. (Measured: identical.)
#   AR      — REPORTED, NOT GATED. The autoregressive serve is not reproducible run to run past ~32
#             tokens, so an AR text diff proves nothing on its own. Run --control to get the floor:
#             two BASE runs against each other. Measured floor was 10 differing lines base-vs-base,
#             against 15 for base-vs-vec and 12-13 for vec-vs-a-third-base-run — i.e. the candidate
#             sits inside the noise, not outside it. Judge the AR phase on tok/s (the 44.2 guard).
#
# Both phases of tools/diffusiongemma_generate.sh run in each arm:
#   canvas  — DiffusionGemma block diffusion, the payload that motivated this (93 all-reduces/step
#             of [256, 2816] bf16 = 1.44 MB each)
#   AR      — the gemma-4 autoregressive guard, whose all-reduces are [1, 2816] = 5.6 KB. It exists to
#             prove the change does not regress the small-tensor case the kernel was tuned for.
#
# Usage (from the ENGINE worktree, holding a 2-card lease):
#   gpu-lease -n 2 --timeout 3600 -- bash tools/car_vec_serve_ab.sh
set -uo pipefail

ENGINE="${ENGINE:-$(cd "$(dirname "$0")/.." && pwd)}"
KERNELS_VEC="${KERNELS_VEC:-/home/pat/code/rdna4-hip-kernels-carfp8/custom_ar/torch-ext/custom_ar}"
IMAGE="${MINISGL_IMAGE:-minisgl-rdna4:gemma4}"
OUT="${OUT:-$ENGINE/_car_vec_ab}"
mkdir -p "$OUT"

SO="$(ls "$KERNELS_VEC"/custom_ar_C*.so 2>/dev/null | head -1)"
[ -n "$SO" ] || { echo "!! no built custom_ar .so at $KERNELS_VEC"; exit 1; }

docker_run() {  # $1 = extra mount args (may be empty), $2.. = command
  local mounts="$1"; shift
  # shellcheck disable=SC2086
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 \
    -v "$ENGINE":/engine \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    $mounts \
    --entrypoint bash "$IMAGE" -lc "$*"
}

MOUNT_VEC="-v $KERNELS_VEC:/opt/kernels/custom_ar:ro"

# ---- provenance: the two arms MUST be different binaries ------------------------------------------
md5_base="$(docker_run "" 'md5sum /opt/kernels/custom_ar/custom_ar_C*.so' | awk '{print $1}')"
md5_vec="$(docker_run "$MOUNT_VEC" 'md5sum /opt/kernels/custom_ar/custom_ar_C*.so' | awk '{print $1}')"
echo "== provenance  BASE=$md5_base  VEC=$md5_vec"
if [ "$md5_base" = "$md5_vec" ]; then
  echo "!! ABORT: both arms would serve the SAME custom_ar .so — this A/B measures nothing."
  exit 1
fi

run_arm() {  # $1 = arm name, $2 = extra mounts
  local arm="$1" mounts="$2"
  echo "== arm $arm =================================================================="
  docker_run "$mounts" \
    "md5sum /opt/kernels/custom_ar/custom_ar_C*.so && bash /engine/tools/diffusiongemma_generate.sh" \
    >"$OUT/$arm.log" 2>&1
  cp -f "$ENGINE/_diffusiongemma_results.txt" "$OUT/$arm.results.txt" 2>/dev/null || true
  echo "-- $arm timings:"
  grep -E "^TIMING:|^md5|forwards per emitted token|blocks=" "$OUT/$arm.log" || true
}

run_arm base ""
run_arm vec "$MOUNT_VEC"

# ---- verdict ---------------------------------------------------------------------------------------
echo
echo "================================ VERDICT ================================"
python3 - "$OUT/base.results.txt" "$OUT/vec.results.txt" <<'PY'
import re, sys

def load(path):
    try:
        return open(path, errors="replace").read()
    except OSError:
        return ""

base, vec = load(sys.argv[1]), load(sys.argv[2])
if not base or not vec:
    print("!! missing results file — one of the arms never produced a completion"); sys.exit(1)

# 1. TEXT, split by phase. The canvas phase is the HARD gate (see the header); the AR phase is only
#    reported, because it is not reproducible run to run even against itself.
def split_phases(s):
    out, cur = {}, None
    for line in s.splitlines():
        if "BLOCK DIFFUSION" in line:
            cur = "canvas"; out[cur] = []
        elif "AUTOREGRESSIVE GUARD" in line:
            cur = "ar"; out[cur] = []
        elif cur is not None and not line.startswith(("TIMING:", "[canvas]")):
            out[cur].append(line)      # [canvas] log lines race between TP ranks; not output text
    return out

pb, pv = split_phases(base), split_phases(vec)
canvas_same = pb.get("canvas") == pv.get("canvas")
ar_diff = sum(1 for x, y in zip(pb.get("ar", []), pv.get("ar", [])) if x != y)
print(f"CANVAS TEXT IDENTICAL: {canvas_same}    <- the gate")
print(f"AR TEXT differing lines: {ar_diff}      <- compare against the base-vs-base control floor")
if not canvas_same:
    for i, (x, y) in enumerate(zip(pb.get("canvas", []), pv.get("canvas", []))):
        if x != y:
            print(f"  !! GATE FAILED, canvas line {i}:\n    base: {x[:160]}\n    vec : {y[:160]}")
            break

# 2. THROUGHPUT, per phase, paired by prompt order.
def timings(s):
    phases, cur = {}, None
    for line in s.splitlines():
        if "BLOCK DIFFUSION" in line:
            cur = "canvas"
        elif "AUTOREGRESSIVE GUARD" in line:
            cur = "ar"
        m = re.match(r"TIMING: (\d+) tokens in ([\d.]+)s = ([\d.]+) tok/s", line)
        if m and cur:
            phases.setdefault(cur, []).append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return phases

tb, tv = timings(base), timings(vec)
for phase, label in (("canvas", "block diffusion (1.44 MB all-reduce)"),
                     ("ar", "autoregressive guard (5.6 KB all-reduce)")):
    b, v = tb.get(phase, []), tv.get(phase, [])
    if not b or not v:
        print(f"\n{label}: MISSING (base={len(b)} vec={len(v)} completions)")
        continue
    print(f"\n{label}:")
    print(f"  {'prompt':>6s} {'tok':>5s} {'base tok/s':>11s} {'vec tok/s':>10s} {'delta':>9s}")
    for i, (x, y) in enumerate(zip(b, v)):
        d = (y[2] / x[2] - 1) * 100 if x[2] else 0.0
        print(f"  {i:6d} {x[0]:5d} {x[2]:11.1f} {y[2]:10.1f} {d:+8.1f}%")
    # Aggregate over all prompts: total tokens / total wall, which is the number that survives
    # per-request TTFT noise.
    agg = lambda t: sum(r[0] for r in t) / sum(r[1] for r in t)
    ab, av = agg(b), agg(v)
    print(f"  {'ALL':>6s} {sum(r[0] for r in b):5d} {ab:11.1f} {av:10.1f} {(av/ab-1)*100:+8.1f}%")
PY
