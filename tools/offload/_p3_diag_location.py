#!/usr/bin/env python3
"""Does ANY hipMemCreate location value actually give HOST memory on this box?

The ceiling diagnostic showed location.type=hipMemLocationTypeHost(2) driving
mem_info_vram_used from 0.75 to 15.75 GiB in lockstep with the allocation while
MemAvailable and GTT never moved, and stopping at 15.5 GiB == the card's VRAM.  That says
the location field is being IGNORED and the allocation is device memory.

Test every documented location type, with a device allocation as the control, and use
three independent discriminators rather than a return code:
  1. sysfs mem_info_vram_used / mem_info_gtt_used delta
  2. /proc/meminfo MemAvailable delta
  3. FIRST-TOUCH BANDWIDTH -- HBM is ~400-600 GB/s here, PCIe H2D is 28 GB/s.  Three
     orders of magnitude apart; no ambiguity.
Also asks the driver what it thinks it allocated, via
hipMemGetAllocationPropertiesFromHandle.

driver_types.h (ROCm 7.2.4): Invalid/None=0, Device=1, Host=2, HostNuma=3,
HostNumaCurrent=4.  hipMemLocation is {int type; int id;}.
"""
import ctypes, json, os, subprocess, sys, time

os.environ["ROCR_VISIBLE_DEVICES"] = "0,1"
os.environ.pop("HIP_VISIBLE_DEVICES", None)

MIB = 1 << 20
GIB = 1 << 30
D2H, H2D = 2, 1
LOC_NAMES = {0: "Invalid/None", 1: "Device", 2: "Host", 3: "HostNuma", 4: "HostNumaCurrent"}


class Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class Flags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort)]


class Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleType", ctypes.c_int),
                ("location", Loc), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", Flags)]


class Desc(ctypes.Structure):
    _fields_ = [("location", Loc), ("flags", ctypes.c_int)]


def L():
    lib = ctypes.CDLL("libamdhip64.so")
    sigs = {
        "hipMemAddressReserve": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                 ctypes.c_size_t, ctypes.c_void_p, ctypes.c_ulonglong],
        "hipMemAddressFree": [ctypes.c_void_p, ctypes.c_size_t],
        "hipMemCreate": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                         ctypes.POINTER(Prop), ctypes.c_ulonglong],
        "hipMemMap": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                      ctypes.c_void_p, ctypes.c_ulonglong],
        "hipMemSetAccess": [ctypes.c_void_p, ctypes.c_size_t,
                            ctypes.POINTER(Desc), ctypes.c_size_t],
        "hipMemsetD32": [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t],
        "hipMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
        "hipMemUnmap": [ctypes.c_void_p, ctypes.c_size_t],
        "hipMemRelease": [ctypes.c_void_p],
        "hipMemGetAllocationGranularity": [ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(Prop), ctypes.c_int],
    }
    for n, a in sigs.items():
        getattr(lib, n).argtypes = a
        getattr(lib, n).restype = ctypes.c_int
    try:
        lib.hipMemGetAllocationPropertiesFromHandle.argtypes = [
            ctypes.POINTER(Prop), ctypes.c_void_p]
        lib.hipMemGetAllocationPropertiesFromHandle.restype = ctypes.c_int
        have_props = True
    except AttributeError:
        have_props = False
    lib.hipSetDevice(0)
    return lib, have_props


def mem():
    o = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        if k in ("MemAvailable", "MemFree"):
            o[k] = int(v.split()[0]) * 1024
    return o


def gpu():
    b = "/sys/class/drm/card1/device/"
    o = {}
    for f in ("mem_info_vram_used", "mem_info_gtt_used"):
        try:
            o[f] = int(open(b + f).read().strip())
        except Exception:
            o[f] = None
    return o


def snap():
    return {**mem(), **gpu()}


def one(loctype, locid, size):
    lib, have_props = L()
    out = {"location_type": loctype, "location_name": LOC_NAMES.get(loctype),
           "location_id": locid, "size": size}
    p = Prop()
    ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
    p.type = 1                      # hipMemAllocationTypePinned
    p.location.type = loctype
    p.location.id = locid

    gran = ctypes.c_size_t(0)
    out["rc_granularity"] = lib.hipMemGetAllocationGranularity(ctypes.byref(gran),
                                                               ctypes.byref(p), 0)
    out["granularity"] = gran.value

    s0 = snap()
    va = ctypes.c_void_p()
    out["rc_reserve"] = lib.hipMemAddressReserve(ctypes.byref(va), size, 2 * MIB, None, 0)
    h = ctypes.c_void_p()
    t = time.perf_counter()
    out["rc_create"] = lib.hipMemCreate(ctypes.byref(h), size, ctypes.byref(p), 0)
    out["t_create_s"] = time.perf_counter() - t
    if out["rc_create"] != 0:
        out["verdict"] = "hipMemCreate REFUSED"
        print(json.dumps(out)); return

    if have_props:
        q = Prop()
        ctypes.memset(ctypes.byref(q), 0, ctypes.sizeof(q))
        rc = lib.hipMemGetAllocationPropertiesFromHandle(ctypes.byref(q), h)
        out["props_readback"] = {"rc": rc, "type": q.type,
                                 "location_type": q.location.type,
                                 "location_name": LOC_NAMES.get(q.location.type),
                                 "location_id": q.location.id}

    d = Desc()
    ctypes.memset(ctypes.byref(d), 0, ctypes.sizeof(d))
    d.location.type = 1
    d.location.id = 0
    d.flags = 3
    out["rc_map"] = lib.hipMemMap(va, size, 0, h, 0)
    out["rc_sa"] = lib.hipMemSetAccess(va, size, ctypes.byref(d), 1)
    if out["rc_map"] or out["rc_sa"]:
        out["verdict"] = "map/setaccess REFUSED"
        lib.hipMemRelease(h)
        print(json.dumps(out)); return
    time.sleep(0.4)
    s1 = snap()

    # warm the kernel path, then time a full-range device write
    lib.hipMemsetD32(va, 0x1, 4096)
    lib.hipDeviceSynchronize()
    t = time.perf_counter()
    out["rc_touch"] = lib.hipMemsetD32(va, 0x5A5A1234, size // 4)
    out["rc_sync"] = lib.hipDeviceSynchronize()
    out["t_touch_s"] = time.perf_counter() - t
    out["touch_gb_s"] = size / out["t_touch_s"] / 1e9 if out["t_touch_s"] > 0 else None
    time.sleep(0.6)
    s2 = snap()

    out["delta_gib"] = {
        "map":   {k: round((s1[k] - s0[k]) / GIB, 3) for k in s0},
        "touch": {k: round((s2[k] - s1[k]) / GIB, 3) for k in s0},
        "total": {k: round((s2[k] - s0[k]) / GIB, 3) for k in s0},
    }
    # discriminator: which pool absorbed the allocation?
    dv = (s2["mem_info_vram_used"] or 0) - (s0["mem_info_vram_used"] or 0)
    dg = (s2["mem_info_gtt_used"] or 0) - (s0["mem_info_gtt_used"] or 0)
    dh = s0["MemAvailable"] - s2["MemAvailable"]
    frac = lambda x: round(x / size, 3)
    out["absorbed_fraction"] = {"vram": frac(dv), "gtt": frac(dg), "host_memavailable": frac(dh)}
    bw = out["touch_gb_s"] or 0
    out["bandwidth_class"] = ("HBM(device)" if bw > 100 else
                              "PCIe(host)" if bw < 60 else "ambiguous")
    out["verdict"] = ("DEVICE VRAM" if frac(dv) > 0.8 else
                      "HOST RAM" if frac(dh) > 0.8 else
                      "GTT" if frac(dg) > 0.8 else "UNACCOUNTED")

    lib.hipMemUnmap(va, size)
    lib.hipMemRelease(h)
    lib.hipMemAddressFree(va, size)
    print(json.dumps(out))


if __name__ == "__main__":
    if len(sys.argv) == 4:
        one(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
        sys.exit(0)
    SIZE = 2 * GIB
    res = []
    for lt, lid in [(1, 0), (2, 0), (3, 0), (4, 0), (2, -1)]:
        pr = subprocess.run([sys.executable, __file__, str(lt), str(lid), str(SIZE)],
                            capture_output=True, text=True, timeout=300)
        lines = [json.loads(l) for l in pr.stdout.splitlines() if l.startswith("{")]
        r = lines[-1] if lines else {"error": "no output"}
        r["proc_rc"] = pr.returncode
        r["stderr"] = pr.stderr.strip()[-300:]
        res.append(r)
        print(f"loc={lt}({LOC_NAMES.get(lt)}) id={lid}: rc_create={r.get('rc_create')} "
              f"verdict={r.get('verdict')} bw={r.get('touch_gb_s')} "
              f"absorbed={r.get('absorbed_fraction')} props={r.get('props_readback')}",
              file=sys.stderr)
    print(json.dumps(res, indent=2))
