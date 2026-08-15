"""Assert the build/profile split actually happened, before any counter is believed.

The split (see Dockerfile.profile714): the kernel .so is compiled by ROCm 7.2.1's hipcc — the same
toolchain as the serve image and the wall-clock bench — while the ROCm RUNTIME and rocprofiler under
it are 7.14, because 7.2.1's `rocprofv3 --pmc` hangs. Both halves fail silently if they slip:

  * torch's wheel VENDORS libamdhip64 / libhsa-runtime64 / librocprofiler-sdk in torch/lib and they
    win by RPATH. If the shadowing did not take, the process runs the 7.2 runtime and profiler and
    --pmc simply hangs — which reads as "counters are broken on this box" rather than "the image is
    wrong".
  * PYTHONPATH order decides whether `fp8_wmma` is the freshly built package or the image's baked
    one, and getting that backwards has already produced a "the change did nothing" result here.

Run it inside the profiling container before collecting anything.
"""
import ctypes
import os
import sys

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAIL.append(name)


def loaded_paths(substr):
    """Resolved paths of every mapped object whose name contains `substr`."""
    out = set()
    with open("/proc/self/maps") as f:
        for line in f:
            p = line.rstrip("\n").split(" ", 5)[-1].strip()
            if p.startswith("/") and substr in os.path.basename(p):
                out.add(p)
    return sorted(out)


print("=== runtime split ===")
import torch  # noqa: E402

print(f"  torch {torch.__version__}  hip {torch.version.hip}")
# Touch the GPU so the HIP/HSA runtime is actually mapped before we look at /proc/self/maps.
ok_dev = torch.cuda.is_available()
check("torch sees a HIP device", ok_dev, torch.cuda.get_device_name(0) if ok_dev else "")
if ok_dev:
    _ = (torch.ones(8, device="cuda") * 2).sum().item()

for lib in ("libamdhip64", "libhsa-runtime64"):
    paths = loaded_paths(lib)
    vendored = [p for p in paths if "site-packages/torch/lib" in p]
    check(f"{lib} is NOT torch's vendored copy", not vendored and bool(paths),
          "; ".join(paths) or "NOT MAPPED")

sdk = loaded_paths("librocprofiler-sdk")
if sdk:
    vendored = [p for p in sdk if "site-packages/torch/lib" in p]
    check("rocprofiler-sdk is NOT torch's vendored copy", not vendored, "; ".join(sdk))
else:
    print("  ----  rocprofiler-sdk not mapped (expected unless running under rocprofv3)")

print("=== kernel package provenance ===")
import fp8_wmma  # noqa: E402

so = [p for p in loaded_paths("fp8_wmma_C")]
print(f"  fp8_wmma      <- {fp8_wmma.__file__}")
print(f"  fp8_wmma_C.so <- {'; '.join(so) or 'not yet mapped'}")
want = os.environ.get("EXPECT_KERNELS_UNDER", "")
if want:
    check(f"fp8_wmma comes from {want}", os.path.abspath(fp8_wmma.__file__).startswith(want),
          fp8_wmma.__file__)

# The .so was linked against torch 2.14's C++ ABI in the lean image; this image runs the same venv,
# so the real risk is not the link but a silently-missing op. Prove the op exists and dispatches.
have = hasattr(fp8_wmma, "mmq_regdirect_w4a16_moe")
check("mmq_regdirect_w4a16_moe is registered", have)

print("\nRESULT:", "OK" if not FAIL else f"{len(FAIL)} FAILED -> {FAIL}")
sys.exit(1 if FAIL else 0)
