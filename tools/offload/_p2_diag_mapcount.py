#!/usr/bin/env python3
"""Diagnostic for the P2 hipMemSetAccess(hipError 1) abort.

Question: is there a per-process / per-reservation LIMIT on the number of distinct
hipMemCreate+hipMemMap+hipMemSetAccess chunks, and does it depend on the media
(device vs host) or on the total bytes?

Reserves one VA range and maps fixed-size chunks into it one at a time, reporting the
index and cumulative bytes at which the FIRST failure occurs, and WHICH call failed.
"""
import ctypes, os, sys, json

HIP_SO = "libamdhip64.so"
_PINNED = 1
_LOC_DEV, _LOC_HOST = 1, 2
_ACCESS_RW = 3
_GRAN_MINIMUM = 0


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort)]


class _MemAllocationProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleType", ctypes.c_int),
                ("location", _MemLocation), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", _AllocFlags)]


class _MemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _MemLocation), ("flags", ctypes.c_int)]


lib = ctypes.CDLL(HIP_SO)
for n, a in {
    "hipMemGetAllocationGranularity": [ctypes.POINTER(ctypes.c_size_t),
                                       ctypes.POINTER(_MemAllocationProp), ctypes.c_int],
    "hipMemAddressReserve": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t,
                             ctypes.c_void_p, ctypes.c_ulonglong],
    "hipMemCreate": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
                     ctypes.POINTER(_MemAllocationProp), ctypes.c_ulonglong],
    "hipMemMap": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
                  ctypes.c_ulonglong],
    "hipMemUnmap": [ctypes.c_void_p, ctypes.c_size_t],
    "hipMemSetAccess": [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_MemAccessDesc),
                        ctypes.c_size_t],
    "hipMemRelease": [ctypes.c_void_p],
}.items():
    f = getattr(lib, n); f.argtypes = a; f.restype = ctypes.c_int


def prop(loc, i):
    p = _MemAllocationProp()
    ctypes.memset(ctypes.byref(p), 0, ctypes.sizeof(p))
    p.type = _PINNED
    p.location.type = loc
    p.location.id = i
    return p


def trial(name, chunk, n, media, card=0, coalesce_access=False, reserve_extra=0):
    """media: 'dev' | 'host' | 'alt'"""
    total = chunk * n + reserve_extra
    va = ctypes.c_void_p()
    rc = lib.hipMemAddressReserve(ctypes.byref(va), total, 2 << 20, None, 0)
    if rc != 0:
        return {"trial": name, "reserve_rc": rc, "ok": False}
    base = int(va.value)
    desc = _MemAccessDesc()
    desc.location.type = _LOC_DEV
    desc.location.id = card
    desc.flags = _ACCESS_RW
    handles, mapped = [], []
    fail = None
    for i in range(n):
        loc = {"dev": _LOC_DEV, "host": _LOC_HOST}.get(media) or (
            _LOC_HOST if i % 2 else _LOC_DEV)
        h = ctypes.c_void_p()
        rc = lib.hipMemCreate(ctypes.byref(h), chunk,
                              ctypes.byref(prop(loc, card if loc == _LOC_DEV else 0)), 0)
        if rc != 0:
            fail = {"call": "hipMemCreate", "i": i, "rc": rc, "loc": loc}
            break
        handles.append(h)
        ptr = base + i * chunk
        rc = lib.hipMemMap(ctypes.c_void_p(ptr), chunk, 0, h, 0)
        if rc != 0:
            fail = {"call": "hipMemMap", "i": i, "rc": rc, "loc": loc}
            break
        mapped.append((ptr, chunk))
        if not coalesce_access:
            rc = lib.hipMemSetAccess(ctypes.c_void_p(ptr), chunk, ctypes.byref(desc), 1)
            if rc != 0:
                fail = {"call": "hipMemSetAccess", "i": i, "rc": rc, "loc": loc}
                break
    if coalesce_access and fail is None:
        rc = lib.hipMemSetAccess(ctypes.c_void_p(base), chunk * n, ctypes.byref(desc), 1)
        if rc != 0:
            fail = {"call": "hipMemSetAccess(coalesced)", "i": n, "rc": rc}
    res = {"trial": name, "chunk": chunk, "n": n, "media": media,
           "coalesce_access": coalesce_access,
           "mapped_ok": len(mapped), "bytes_mapped": len(mapped) * chunk,
           "fail": fail, "ok": fail is None}
    for p, s in mapped:
        lib.hipMemUnmap(ctypes.c_void_p(p), s)
    for h in handles:
        lib.hipMemRelease(h)
    lib.hipMemAddressFree = getattr(lib, "hipMemAddressFree", None)
    return res


def main():
    out = {"granularity_dev": None, "granularity_host": None, "trials": []}
    g = ctypes.c_size_t(0)
    lib.hipMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(prop(_LOC_DEV, 0)),
                                       _GRAN_MINIMUM)
    out["granularity_dev"] = g.value
    lib.hipMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(prop(_LOC_HOST, 0)),
                                       _GRAN_MINIMUM)
    out["granularity_host"] = g.value

    MB = 1 << 20
    trials = [
        ("dev_1MiB_x512", MB, 512, "dev", False),
        ("host_1MiB_x512", MB, 512, "host", False),
        ("alt_1MiB_x512", MB, 512, "alt", False),
        ("dev_4p5MiB_x256", 4718592, 256, "dev", False),
        ("dev_4KiB_x4096", 4096, 4096, "dev", False),
        ("alt_1MiB_x512_coalesced_access", MB, 512, "alt", True),
    ]
    for name, chunk, n, media, coal in trials:
        r = trial(name, chunk, n, media, coalesce_access=coal)
        out["trials"].append(r)
        print(json.dumps(r), flush=True)
    print(json.dumps(out["trials"], indent=1))


if __name__ == "__main__":
    main()
