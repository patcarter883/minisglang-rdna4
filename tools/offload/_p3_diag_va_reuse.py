#!/usr/bin/env python3
"""Two questions the P3 abort raised.

A) VA REUSE ACROSS RESERVATIONS.  The ratebench crashed on its SECOND arm.  Each arm
   does hipMemAddressReserve -> map/touch/unmap/release -> hipMemAddressFree.  If the
   second reserve hands back the SAME virtual range, we are re-mapping a VA this box is
   documented to serve stale page-table state for.  Test it explicitly.

B) ARE THESE PAGES ACTUALLY HOST-LOCATED?  hipMemsetD32 over a 4 GiB host-located range
   completed in 7.6 ms == 565 GB/s, which is HBM speed and roughly 20x the box's measured
   28 GB/s PCIe H2D.  Either the memset is not doing what it says, or location.type=HOST
   is being silently ignored.  Measure MemAvailable and sysfs VRAM around it.

Each scenario runs in its own process: a GPU memory fault is an unrecoverable SIGABRT.
"""
import ctypes, json, os, subprocess, sys, time

os.environ["ROCR_VISIBLE_DEVICES"] = "0,1"
os.environ.pop("HIP_VISIBLE_DEVICES", None)

MIB = 1 << 20
GIB = 1 << 30


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


def L():
    lib = ctypes.CDLL("libamdhip64.so")
    lib.hipMemAddressReserve.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                                         ctypes.c_size_t, ctypes.c_void_p, ctypes.c_ulonglong]
    lib.hipMemAddressFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
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
    lib.hipHostMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    lib.hipSetDevice(0)
    return lib


def hp():
    p = Prop()
    ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
    p.type = 1
    p.location.type = 2
    return p


def dsc():
    d = Desc()
    ctypes.memset(ctypes.byref(d), 0, ctypes.sizeof(d))
    d.location.type = 1
    d.location.id = 0
    d.flags = 3
    return d


def memavail():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def vram_used():
    # card1 == 0000:03:00.0 == HIP dev 0 (established by the P3 run)
    try:
        return int(open("/sys/class/drm/card1/device/mem_info_vram_used").read().strip())
    except Exception:
        return None


def emit(o):
    sys.stdout.write(json.dumps(o) + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------
def scen_va_reuse():
    """Exactly the ratebench shape: two arms, each reserve->use->free."""
    lib = L()
    out = {"scenario": "va_reuse", "arms": []}
    for arm, (chunk, nch) in enumerate([(512 * MIB, 8), (4 * GIB, 1)]):
        reserve = chunk * nch * 6
        va = ctypes.c_void_p()
        rc = lib.hipMemAddressReserve(ctypes.byref(va), reserve, 2 * MIB, None, 0)
        rec = {"arm": arm, "chunk": chunk, "reserve": reserve, "rc_reserve": rc,
               "va": hex(va.value or 0), "reps": []}
        out["arms"].append(rec)
        emit({**out, "stage": f"arm{arm}_reserved"})
        if rc:
            break
        for rep in range(6):
            base = va.value + rep * chunk * nch
            hs = []
            r = {"rep": rep, "base": hex(base)}
            for i in range(nch):
                addr = ctypes.c_void_p(base + i * chunk)
                h = ctypes.c_void_p()
                r[f"rc_create{i}"] = lib.hipMemCreate(ctypes.byref(h), chunk, ctypes.byref(hp()), 0)
                r[f"rc_map{i}"] = lib.hipMemMap(addr, chunk, 0, h, 0)
                r[f"rc_sa{i}"] = lib.hipMemSetAccess(addr, chunk, ctypes.byref(dsc()), 1)
                hs.append((addr, h))
            rec["reps"].append(r)
            emit({**out, "stage": f"arm{arm}_rep{rep}_about_to_touch"})
            for i, (addr, h) in enumerate(hs):
                r[f"rc_set{i}"] = lib.hipMemsetD32(addr, 0x5A5A1234, chunk // 4)
            r["rc_sync"] = lib.hipDeviceSynchronize()
            for addr, h in hs:
                lib.hipMemUnmap(addr, chunk)
                lib.hipMemRelease(h)
        lib.hipMemAddressFree(va, reserve)
    out["completed"] = True
    emit(out)


def scen_attribution():
    """Where does a host-located commit LAND -- host RAM or VRAM?  And is the
    astonishing memset rate real work or a no-op?"""
    lib = L()
    SZ = 4 * GIB
    out = {"scenario": "attribution", "size": SZ}
    out["ma0"], out["vr0"] = memavail(), vram_used()

    va = ctypes.c_void_p()
    lib.hipMemAddressReserve(ctypes.byref(va), SZ, 2 * MIB, None, 0)
    h = ctypes.c_void_p()
    t = time.perf_counter()
    out["rc_create"] = lib.hipMemCreate(ctypes.byref(h), SZ, ctypes.byref(hp()), 0)
    out["t_create_s"] = time.perf_counter() - t
    time.sleep(0.5)
    out["ma_after_create"], out["vr_after_create"] = memavail(), vram_used()

    out["rc_map"] = lib.hipMemMap(va, SZ, 0, h, 0)
    out["rc_sa"] = lib.hipMemSetAccess(va, SZ, ctypes.byref(dsc()), 1)
    time.sleep(0.5)
    out["ma_after_map"], out["vr_after_map"] = memavail(), vram_used()

    emit({**out, "stage": "about_to_touch"})
    t = time.perf_counter()
    out["rc_set"] = lib.hipMemsetD32(va, 0x5A5A1234, SZ // 4)
    out["rc_sync"] = lib.hipDeviceSynchronize()
    out["t_touch_s"] = time.perf_counter() - t
    out["touch_gb_s"] = SZ / out["t_touch_s"] / 1e9
    time.sleep(1.0)
    out["ma_after_touch"], out["vr_after_touch"] = memavail(), vram_used()

    # Second memset with a DIFFERENT word, then read back: proves the write lands.
    t = time.perf_counter()
    lib.hipMemsetD32(va, 0x0BADF00D, SZ // 4)
    lib.hipDeviceSynchronize()
    out["t_touch2_s"] = time.perf_counter() - t
    buf = (ctypes.c_uint32 * 4)()
    ck = []
    for off in (0, SZ // 2, SZ - 16):
        rc = lib.hipMemcpy(ctypes.cast(buf, ctypes.c_void_p),
                           ctypes.c_void_p(va.value + off), 16, 2)
        ck.append({"off": off, "rc": rc, "got": hex(buf[0]) if rc == 0 else None})
    out["readback_second_word"] = ck
    out["second_word_ok"] = all(c["rc"] == 0 and c["got"] == "0xbadf00d" for c in ck)

    # Reference: hipHostMalloc of the same size, CPU-memset, for the delta shape.
    p = ctypes.c_void_p()
    t = time.perf_counter()
    out["rc_hostmalloc"] = lib.hipHostMalloc(ctypes.byref(p), SZ, 0)
    out["t_hostmalloc_s"] = time.perf_counter() - t
    time.sleep(0.5)
    out["ma_after_hostmalloc"], out["vr_after_hostmalloc"] = memavail(), vram_used()
    t = time.perf_counter()
    ctypes.memset(p, 0x5A, SZ)
    out["t_cpu_memset_s"] = time.perf_counter() - t
    out["cpu_memset_gb_s"] = SZ / out["t_cpu_memset_s"] / 1e9
    time.sleep(0.5)
    out["ma_after_cpu_touch"], out["vr_after_cpu_touch"] = memavail(), vram_used()
    lib.hipHostFree(p)

    lib.hipMemUnmap(va, SZ)
    lib.hipMemRelease(h)
    lib.hipMemAddressFree(va, SZ)
    time.sleep(0.5)
    out["ma_final"], out["vr_final"] = memavail(), vram_used()
    out["completed"] = True
    emit(out)


SCEN = {"va_reuse": scen_va_reuse, "attribution": scen_attribution}

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in SCEN:
        SCEN[sys.argv[1]]()
        sys.exit(0)
    res = []
    for name in SCEN:
        pr = subprocess.run([sys.executable, __file__, name],
                            capture_output=True, text=True, timeout=600)
        lines = [json.loads(l) for l in pr.stdout.splitlines() if l.startswith("{")]
        res.append({"scenario": name, "rc": pr.returncode,
                    "last": lines[-1] if lines else None,
                    "trace_tail": lines[-3:],
                    "stderr": pr.stderr.strip()[-800:]})
        print(f"--- {name}: rc={pr.returncode} completed={(lines[-1] if lines else {}).get('completed')}",
              file=sys.stderr)
        print(f"    stderr: {pr.stderr.strip()[-300:]}", file=sys.stderr)
    print(json.dumps(res, indent=2))
