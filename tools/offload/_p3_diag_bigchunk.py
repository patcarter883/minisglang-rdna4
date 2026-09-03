#!/usr/bin/env python3
"""Diagnostic for the P3 ratebench abort: at what host-located hipMemCreate chunk size
does the DEVICE access to the mapped range start faulting?

Each size runs in its OWN process, because a GPU memory access fault is an
unrecoverable SIGABRT -- that is precisely why it destroyed the P3 run.

Usage:  _p3_diag_bigchunk.py            -> parent, sweeps sizes, prints JSON
        _p3_diag_bigchunk.py <bytes> <touch_bytes>  -> child, one size
"""
import ctypes, json, os, subprocess, sys, time

os.environ["ROCR_VISIBLE_DEVICES"] = "0,1"
os.environ.pop("HIP_VISIBLE_DEVICES", None)

MIB = 1 << 20
GIB = 1 << 30
LIB = "libamdhip64.so"


class Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class Flags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class Prop(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleType", ctypes.c_int),
                ("location", Loc), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", Flags)]


class Desc(ctypes.Structure):
    _fields_ = [("location", Loc), ("flags", ctypes.c_int)]


def child(size, touch):
    lib = ctypes.CDLL(LIB)
    lib.hipMemAddressReserve.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                         ctypes.c_size_t, ctypes.c_void_p, ctypes.c_ulonglong]
    lib.hipMemCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                 ctypes.POINTER(Prop), ctypes.c_ulonglong]
    lib.hipMemMap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                              ctypes.c_void_p, ctypes.c_ulonglong]
    lib.hipMemSetAccess.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.POINTER(Desc), ctypes.c_size_t]
    lib.hipMemsetD32.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.hipMemUnmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.hipMemRelease.argtypes = [ctypes.c_void_p]

    out = {"size": size, "touch": touch}
    lib.hipSetDevice(0)
    p = Prop()
    ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
    p.type = 1
    p.location.type = 2      # HOST
    d = Desc()
    ctypes.memset(ctypes.byref(d), 0, ctypes.sizeof(d))
    d.location.type = 1      # DEVICE
    d.location.id = 0
    d.flags = 3

    va = ctypes.c_void_p()
    out["rc_reserve"] = lib.hipMemAddressReserve(ctypes.byref(va), size, 2 * MIB, None, 0)
    out["va"] = hex(va.value or 0)
    if out["rc_reserve"] != 0:
        print(json.dumps(out)); return 0
    h = ctypes.c_void_p()
    t0 = time.perf_counter()
    out["rc_create"] = lib.hipMemCreate(ctypes.byref(h), size, ctypes.byref(p), 0)
    out["t_create_s"] = time.perf_counter() - t0
    out["handle"] = hex(h.value or 0)
    if out["rc_create"] != 0:
        print(json.dumps(out)); return 0
    out["rc_map"] = lib.hipMemMap(va, size, 0, h, 0)
    out["rc_setaccess"] = lib.hipMemSetAccess(va, size, ctypes.byref(d), 1)
    if out["rc_map"] or out["rc_setaccess"]:
        print(json.dumps(out)); return 0

    sys.stdout.write(json.dumps({**out, "stage": "about_to_touch"}) + "\n")
    sys.stdout.flush()
    t0 = time.perf_counter()
    out["rc_memset"] = lib.hipMemsetD32(va, 0x5A5A1234, touch // 4)
    out["rc_sync"] = lib.hipDeviceSynchronize()
    out["t_touch_s"] = time.perf_counter() - t0
    # read back at head, middle-of-touched, and end-of-touched
    buf = (ctypes.c_uint32 * 4)()
    checks = []
    for off in (0, 0x2000, touch // 2 & ~4095, max(0, touch - 16)):
        rc = lib.hipMemcpy(ctypes.cast(buf, ctypes.c_void_p),
                           ctypes.c_void_p(va.value + off), 16, 2)
        checks.append({"off": off, "rc": rc, "got": hex(buf[0]) if rc == 0 else None,
                       "ok": rc == 0 and buf[0] == 0x5A5A1234})
    out["readback"] = checks
    out["all_ok"] = all(c["ok"] for c in checks)
    lib.hipMemUnmap(va, size)
    lib.hipMemRelease(h)
    print(json.dumps(out))
    return 0


def main():
    sizes = [(512 * MIB, 512 * MIB), (1 * GIB, 1 * GIB), (2 * GIB, 2 * GIB),
             (3 * GIB, 3 * GIB), (4 * GIB, 4 * GIB),
             (4 * GIB, 1 * MIB),      # 4 GiB mapped, tiny touch: is it map or memset?
             (4 * GIB - 4096, 4 * GIB - 4096),
             (8 * GIB, 1 * MIB)]
    res = []
    for size, touch in sizes:
        pr = subprocess.run([sys.executable, __file__, str(size), str(touch)],
                            capture_output=True, text=True, timeout=300)
        lines = [l for l in pr.stdout.strip().splitlines() if l.startswith("{")]
        rec = {"size": size, "touch": touch, "rc": pr.returncode,
               "stdout": lines, "stderr": pr.stderr.strip()[-600:]}
        try:
            rec["last"] = json.loads(lines[-1]) if lines else None
        except Exception:
            rec["last"] = None
        res.append(rec)
        print(f"size={size/GIB:.3f} GiB touch={touch/GIB:.3f} GiB rc={pr.returncode} "
              f"last={rec['last']}", file=sys.stderr)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        sys.exit(child(int(sys.argv[1]), int(sys.argv[2])))
    main()
