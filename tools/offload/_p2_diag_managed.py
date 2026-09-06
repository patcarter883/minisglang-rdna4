#!/usr/bin/env python3
"""Is there ANY route to a mixed device/host single contiguous VA on this box?

P1 and P2 both establish that hipMemCreate(location.type=hipMemLocationTypeHost) is silently
ignored on gfx1201 / ROCm 7.2.1 here: it returns hipSuccess, hipMemMap/hipMemSetAccess succeed,
the VA works — but the pages consume VRAM and read at full HBM speed. That kills the plan's §2
placement-only architecture through the VMM API.

The only remaining candidate for a SINGLE CONTIGUOUS VA whose sub-ranges live on different media
is managed memory + per-range hipMemAdvise(SetPreferredLocation = CPU). This tests it:
  * does hipMallocManaged succeed, and where do the pages land (per-BDF sysfs vram/gtt)?
  * does hipMemAdvise(hipMemAdviseSetPreferredLocation, hipCpuDeviceId) on the SECOND HALF move
    those pages to host RAM?
  * does a kernel read of the advised half run at PCIe speed (~28 GB/s) or HBM speed (~690)?
A ratio near 1.0 means no media separation and the mechanism is unavailable too.
"""
import ctypes, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BDF = "0000:03:00.0"


def sysfs(bdf):
    out = {}
    for k in ("mem_info_vram_used", "mem_info_gtt_used"):
        p = f"/sys/bus/pci/devices/{bdf}/{k}"
        try:
            out[k] = int(open(p).read().strip())
        except Exception as e:
            out[k] = f"<{e}>"
    return out


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        d[k] = int(v.strip().split()[0])
    return d


def main():
    import torch
    lib = ctypes.CDLL("libamdhip64.so")
    for n, a in {
        "hipMallocManaged": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint],
        "hipMemAdvise": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int],
        "hipMemPrefetchAsync": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p],
        "hipDeviceSynchronize": [],
        "hipFree": [ctypes.c_void_p],
    }.items():
        f = getattr(lib, n, None)
        if f is None:
            print(json.dumps({"missing_symbol": n})); return 2
        f.argtypes = a; f.restype = ctypes.c_int

    torch.zeros(1, device="cuda")  # bring HIP up first
    N = 512 << 20
    HALF = N // 2
    HIP_CPU_DEVICE_ID = -1
    ADVISE_SET_PREFERRED = 3   # hipMemAdviseSetPreferredLocation
    ADVISE_SET_ACCESSED_BY = 5  # hipMemAdviseSetAccessedBy
    GLOBAL = 1                  # hipMemAttachGlobal

    out = {"bdf": BDF, "bytes": N}
    out["sysfs_before"] = sysfs(BDF)
    out["memavailable_before_kb"] = meminfo()["MemAvailable"]

    p = ctypes.c_void_p()
    rc = lib.hipMallocManaged(ctypes.byref(p), N, GLOBAL)
    out["hipMallocManaged_rc"] = rc
    if rc:
        print(json.dumps(out, indent=1)); return 2
    base = int(p.value)
    out["base"] = hex(base)

    out["advise_second_half_rc"] = lib.hipMemAdvise(ctypes.c_void_p(base + HALF), HALF,
                                                    ADVISE_SET_PREFERRED, HIP_CPU_DEVICE_ID)
    out["advise_accessed_by_rc"] = lib.hipMemAdvise(ctypes.c_void_p(base + HALF), HALF,
                                                    ADVISE_SET_ACCESSED_BY, 0)
    out["prefetch_first_half_rc"] = lib.hipMemPrefetchAsync(ctypes.c_void_p(base), HALF, 0, None)
    out["prefetch_second_half_cpu_rc"] = lib.hipMemPrefetchAsync(
        ctypes.c_void_p(base + HALF), HALF, HIP_CPU_DEVICE_ID, None)
    lib.hipDeviceSynchronize()

    out["sysfs_after"] = sysfs(BDF)
    out["memavailable_after_kb"] = meminfo()["MemAvailable"]
    out["vram_delta"] = (out["sysfs_after"]["mem_info_vram_used"]
                         - out["sysfs_before"]["mem_info_vram_used"]) \
        if isinstance(out["sysfs_after"]["mem_info_vram_used"], int) else None
    out["gtt_delta"] = (out["sysfs_after"]["mem_info_gtt_used"]
                        - out["sysfs_before"]["mem_info_gtt_used"]) \
        if isinstance(out["sysfs_after"]["mem_info_gtt_used"], int) else None
    out["memavail_delta_kb"] = out["memavailable_after_kb"] - out["memavailable_before_kb"]

    # kernel-read bandwidth of each half, via a torch tensor bound to the managed VA
    def band(off, label):
        n_el = HALF // 4
        t = torch.empty(0, dtype=torch.float32, device="cuda")
        # bind via from_blob-equivalent: use a uint8 view through cupy-free ctypes is awkward;
        # use torch's UntypedStorage from a raw pointer instead.
        st = torch.cuda.caching_allocator_alloc  # unused; kept explicit that we do NOT alloc
        arr = torch.frombuffer if False else None
        # DLPack-free: torch.as_strided on a tensor made from a raw pointer
        ten = _tensor_from_ptr(torch, base + off, n_el)
        for _ in range(2):
            ten.sum()
        torch.cuda.synchronize()
        best = None
        for _ in range(5):
            t0 = time.perf_counter()
            ten.sum()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            gb = HALF / dt / 1e9
            best = gb if best is None else max(best, gb)
        return round(best, 1)

    out["read_gbps_device_half"] = band(0, "device")
    out["read_gbps_cpu_advised_half"] = band(HALF, "cpu_advised")
    out["ratio_device_over_cpu"] = round(
        out["read_gbps_device_half"] / max(1e-9, out["read_gbps_cpu_advised_half"]), 3)
    out["verdict"] = ("MEDIA SEPARATION PRESENT" if out["ratio_device_over_cpu"] > 3
                      else "NO MEDIA SEPARATION — managed+advise does not give a host tier either")
    lib.hipFree(ctypes.c_void_p(base))
    print(json.dumps(out, indent=1))
    return 0


def _tensor_from_ptr(torch, ptr, n_el):
    import p2_mixed_media_moe as P2
    return P2.tensor_at(torch, ptr, (n_el,), torch.float32, 0)


if __name__ == "__main__":
    sys.exit(main())
