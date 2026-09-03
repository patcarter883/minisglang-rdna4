#!/usr/bin/env python3
"""THE decisive question for P3.

The attribution diagnostic showed a 4 GiB host-located hipMemCreate+Map+SetAccess+device
memset move MemAvailable by ~40 MB (a hipHostMalloc of the same size moved it by 4.31 GB),
yet a read-back of the written word succeeded at offset 0, 2 GiB and 4 GiB-16.

Those two facts are only compatible if the 4 GiB VA range is backed by far fewer physical
pages than it claims -- i.e. ALIASING.  A read-back cannot see that when every page was
written the SAME word, which is what the first diagnostic did.

So: write a DISTINCT word per page across the range, then read them all back.  If page i
returns page j's word, the range aliases and any capacity number taken from this API is
fiction.

Also records where the commit lands (VRAM vs GTT vs MemAvailable) and whether
hipMemRelease actually returns the resource.
"""
import ctypes, json, os, subprocess, sys, time

os.environ["ROCR_VISIBLE_DEVICES"] = "0,1"
os.environ.pop("HIP_VISIBLE_DEVICES", None)

MIB = 1 << 20
GIB = 1 << 30
PAGE = 4096
H2D, D2H = 1, 2


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
    for n, a in {
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
        "hipMemGetInfo": [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)],
    }.items():
        getattr(lib, n).argtypes = a
        getattr(lib, n).restype = ctypes.c_int
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


def mem():
    mi = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":", 1)
        if k in ("MemAvailable", "MemFree"):
            mi[k] = int(v.split()[0]) * 1024
    return mi


def gpu():
    base = "/sys/class/drm/card1/device/"
    o = {}
    for f in ("mem_info_vram_used", "mem_info_gtt_used", "mem_info_vis_vram_used"):
        try:
            o[f] = int(open(base + f).read().strip())
        except Exception:
            o[f] = None
    return o


def snap(tag):
    return {"tag": tag, **mem(), **gpu()}


def emit(o):
    sys.stdout.write(json.dumps(o) + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------- alias test
def scen_alias():
    """Distinct word per page over a 4 GiB host-located range."""
    lib = L()
    SZ = 4 * GIB
    NPROBE = 4096                       # sampled pages, spread evenly over 4 GiB
    stride = (SZ // NPROBE) & ~(PAGE - 1)
    out = {"scenario": "alias", "size": SZ, "n_probe_pages": NPROBE, "stride": stride,
           "snaps": [snap("t0")]}

    va = ctypes.c_void_p()
    out["rc_reserve"] = lib.hipMemAddressReserve(ctypes.byref(va), SZ, 2 * MIB, None, 0)
    h = ctypes.c_void_p()
    out["rc_create"] = lib.hipMemCreate(ctypes.byref(h), SZ, ctypes.byref(hp()), 0)
    out["rc_map"] = lib.hipMemMap(va, SZ, 0, h, 0)
    out["rc_sa"] = lib.hipMemSetAccess(va, SZ, ctypes.byref(dsc()), 1)
    if any(out[k] for k in ("rc_reserve", "rc_create", "rc_map", "rc_sa")):
        emit(out); return
    out["snaps"].append(snap("after_map"))

    emit({**out, "stage": "about_to_write_distinct_words"})
    w = ctypes.c_uint32(0)
    t0 = time.perf_counter()
    for i in range(NPROBE):
        w.value = (0xC0DE0000 | i) & 0xFFFFFFFF
        rc = lib.hipMemcpy(ctypes.c_void_p(va.value + i * stride),
                           ctypes.byref(w), 4, H2D)
        if rc:
            out["write_fail"] = {"i": i, "rc": rc}
            break
    out["t_write_s"] = time.perf_counter() - t0
    out["snaps"].append(snap("after_distinct_writes"))

    buf = ctypes.c_uint32(0)
    bad = []
    for i in range(NPROBE):
        want = (0xC0DE0000 | i) & 0xFFFFFFFF
        rc = lib.hipMemcpy(ctypes.byref(buf), ctypes.c_void_p(va.value + i * stride), 4, D2H)
        got = buf.value
        if rc or got != want:
            alias = (got & 0xFFFF) if (got & 0xFFFF0000) == 0xC0DE0000 else None
            bad.append({"page": i, "off": i * stride, "rc": rc,
                        "expected": hex(want), "got": hex(got),
                        "looks_like_page": alias})
    out["mismatches"] = len(bad)
    out["mismatch_sample"] = bad[:24]
    out["ALIASES"] = len(bad) > 0
    # how many DISTINCT words survived?  n distinct == n real physical pages (lower bound)
    seen = set()
    for i in range(NPROBE):
        lib.hipMemcpy(ctypes.byref(buf), ctypes.c_void_p(va.value + i * stride), 4, D2H)
        seen.add(buf.value)
    out["distinct_words_readable"] = len(seen)
    out["distinct_words_expected"] = NPROBE

    lib.hipMemUnmap(va, SZ)
    lib.hipMemRelease(h)
    lib.hipMemAddressFree(va, SZ)
    out["snaps"].append(snap("after_release"))
    out["completed"] = True
    emit(out)


# ------------------------------------------------- release / capacity ceiling
def scen_ceiling():
    """Does hipMemRelease actually give the resource back, and what is the real
    ceiling on LIVE host-located handles?  Two loops:
      A) create/map/touch/unmap/release the same 512 MiB over and over (should never OOM
         if release works),
      B) grow a live arena 512 MiB at a time until hipMemCreate or hipMemMap fails.
    Every return code is CHECKED -- the P3 abort happened because a rc=2 was ignored and
    the next memset hit an unmapped VA, which is a hard SIGABRT, not an exception."""
    lib = L()
    CH = 512 * MIB
    out = {"scenario": "ceiling", "chunk": CH, "snaps": [snap("t0")]}

    # --- A: churn -----------------------------------------------------------
    churn = []
    va = ctypes.c_void_p()
    lib.hipMemAddressReserve(ctypes.byref(va), 64 * CH, 2 * MIB, None, 0)
    for it in range(48):                       # 48 * 512 MiB = 24 GiB cumulative
        addr = ctypes.c_void_p(va.value + (it % 64) * CH)
        h = ctypes.c_void_p()
        rc_c = lib.hipMemCreate(ctypes.byref(h), CH, ctypes.byref(hp()), 0)
        if rc_c:
            churn.append({"iter": it, "rc_create": rc_c, "cumulative_gib": it * CH / GIB})
            break
        rc_m = lib.hipMemMap(addr, CH, 0, h, 0)
        rc_s = lib.hipMemSetAccess(addr, CH, ctypes.byref(dsc()), 1)
        if rc_m or rc_s:
            churn.append({"iter": it, "rc_map": rc_m, "rc_sa": rc_s})
            lib.hipMemRelease(h)
            break
        lib.hipMemsetD32(addr, 0xAA550000 | it, CH // 4)
        lib.hipDeviceSynchronize()
        lib.hipMemUnmap(addr, CH)
        lib.hipMemRelease(h)
        if it % 8 == 0:
            out["snaps"].append(snap(f"churn{it}"))
    out["churn_iters_ok"] = 48 - len(churn)
    out["churn_stop"] = churn
    lib.hipMemAddressFree(va, 64 * CH)
    out["snaps"].append(snap("after_churn"))
    emit({**out, "stage": "churn_done"})

    # --- B: live growth -----------------------------------------------------
    N = 200
    va2 = ctypes.c_void_p()
    lib.hipMemAddressReserve(ctypes.byref(va2), N * CH, 2 * MIB, None, 0)
    live = []
    stop = None
    for i in range(N):
        addr = ctypes.c_void_p(va2.value + i * CH)
        h = ctypes.c_void_p()
        rc_c = lib.hipMemCreate(ctypes.byref(h), CH, ctypes.byref(hp()), 0)
        if rc_c:
            stop = {"i": i, "stage": "hipMemCreate", "rc": rc_c}
            break
        rc_m = lib.hipMemMap(addr, CH, 0, h, 0)
        if rc_m:
            stop = {"i": i, "stage": "hipMemMap", "rc": rc_m}
            lib.hipMemRelease(h)
            break
        rc_s = lib.hipMemSetAccess(addr, CH, ctypes.byref(dsc()), 1)
        if rc_s:
            stop = {"i": i, "stage": "hipMemSetAccess", "rc": rc_s}
            lib.hipMemUnmap(addr, CH)
            lib.hipMemRelease(h)
            break
        rc_t = lib.hipMemsetD32(addr, 0xA5A50000 | i, CH // 4)
        rc_y = lib.hipDeviceSynchronize()
        if rc_t or rc_y:
            stop = {"i": i, "stage": "touch", "rc_memset": rc_t, "rc_sync": rc_y}
            break
        live.append((addr, h))
        if i % 8 == 0:
            out["snaps"].append(snap(f"grow{i}"))
        if mem()["MemAvailable"] < 8 * GIB:
            stop = {"i": i, "stage": "mem_available_floor"}
            break
    out["live_chunks"] = len(live)
    out["live_gib"] = len(live) * CH / GIB
    out["grow_stop"] = stop or {"stage": "reached_N", "i": N}
    out["snaps"].append(snap("at_peak"))

    # resweep: each chunk carries its own word, so an alias shows up here
    buf = ctypes.c_uint32(0)
    fails = []
    for i, (addr, h) in enumerate(live):
        want = (0xA5A50000 | i) & 0xFFFFFFFF
        for off in (0, CH // 2, CH - 4):
            rc = lib.hipMemcpy(ctypes.byref(buf), ctypes.c_void_p(addr.value + off), 4, D2H)
            if rc or buf.value != want:
                fails.append({"chunk": i, "off": off, "rc": rc,
                              "expected": hex(want), "got": hex(buf.value)})
    out["resweep_failures"] = len(fails)
    out["resweep_sample"] = fails[:16]

    for addr, h in live:
        lib.hipMemUnmap(addr, CH)
        lib.hipMemRelease(h)
    lib.hipMemAddressFree(va2, N * CH)
    time.sleep(2)
    out["snaps"].append(snap("after_release"))
    out["completed"] = True
    emit(out)


SCEN = {"alias": scen_alias, "ceiling": scen_ceiling}

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in SCEN:
        SCEN[sys.argv[1]]()
        sys.exit(0)
    res = []
    for name in SCEN:
        pr = subprocess.run([sys.executable, __file__, name],
                            capture_output=True, text=True, timeout=1200)
        lines = [json.loads(l) for l in pr.stdout.splitlines() if l.startswith("{")]
        res.append({"scenario": name, "rc": pr.returncode,
                    "last": lines[-1] if lines else None,
                    "stderr": pr.stderr.strip()[-800:]})
        last = lines[-1] if lines else {}
        print(f"--- {name}: rc={pr.returncode} completed={last.get('completed')}", file=sys.stderr)
        for k in ("ALIASES", "mismatches", "distinct_words_readable", "churn_iters_ok",
                  "churn_stop", "live_gib", "grow_stop", "resweep_failures"):
            if k in last:
                print(f"    {k} = {last[k]}", file=sys.stderr)
        if pr.stderr.strip():
            print(f"    stderr: {pr.stderr.strip()[-300:]}", file=sys.stderr)
    print(json.dumps(res, indent=2))
