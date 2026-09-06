#!/usr/bin/env python3
"""Pin down the hipMemSetAccess(hipError 1) rule for HOST-located VMM mappings.

Sweeps (VA alignment within the reservation) x (mapping size) for both media and reports
which combinations hipMemCreate / hipMemMap / hipMemSetAccess accept.
"""
import ctypes, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_mixed_media_moe as P2

_LOC_DEV, _LOC_HOST = 1, 2
_ACCESS_RW = 3
_GRAN_MINIMUM, _GRAN_RECOMMENDED = 0, 1

hip = P2.Hip()


def gran(loc, flag):
    g = ctypes.c_size_t(0)
    rc = hip.lib.hipMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(hip.prop(loc, 0)),
                                                flag)
    return {"rc": rc, "value": g.value}


def one(base, offset, size, loc):
    desc = P2._MemAccessDesc()
    desc.location.type = _LOC_DEV
    desc.location.id = 0
    desc.flags = _ACCESS_RW
    h = ctypes.c_void_p()
    rc_c = hip.lib.hipMemCreate(ctypes.byref(h), size, ctypes.byref(hip.prop(loc, 0)), 0)
    if rc_c:
        return {"create": rc_c, "map": None, "setaccess": None}
    ptr = base + offset
    rc_m = hip.lib.hipMemMap(ctypes.c_void_p(ptr), size, 0, h, 0)
    rc_s = None
    if rc_m == 0:
        rc_s = hip.lib.hipMemSetAccess(ctypes.c_void_p(ptr), size, ctypes.byref(desc), 1)
        hip.lib.hipMemUnmap(ctypes.c_void_p(ptr), size)
    hip.lib.hipMemRelease(h)
    return {"create": rc_c, "map": rc_m, "setaccess": rc_s}


def main():
    out = {"granularity": {
        "dev_min": gran(_LOC_DEV, _GRAN_MINIMUM), "dev_rec": gran(_LOC_DEV, _GRAN_RECOMMENDED),
        "host_min": gran(_LOC_HOST, _GRAN_MINIMUM), "host_rec": gran(_LOC_HOST, _GRAN_RECOMMENDED),
    }, "sweep": []}
    print(json.dumps(out["granularity"]), flush=True)

    SPAN = 1 << 30
    va = ctypes.c_void_p()
    hip.ck(hip.lib.hipMemAddressReserve(ctypes.byref(va), SPAN, 2 << 20, None, 0), "reserve")
    base = int(va.value)
    out["base_align"] = base & ((2 << 20) - 1)
    K = 1 << 10
    M = 1 << 20
    offsets = [0, 4 * K, 64 * K, 256 * K, 512 * K, 1 * M, 1536 * K, 2 * M, 3 * M, 4 * M]
    sizes = [4 * K, 64 * K, 256 * K, 1 * M, 1536 * K, 2 * M, 3 * M, 7680 * K]
    for loc, lname in ((_LOC_DEV, "device"), (_LOC_HOST, "host")):
        for off in offsets:
            for sz in sizes:
                r = one(base, off, sz, loc)
                r.update({"media": lname, "offset": off, "size": sz,
                          "va_align_pow2": (base + off) & -(base + off) if (base + off) else 0})
                out["sweep"].append(r)
                if r["setaccess"] not in (0, None) or r["map"] not in (0, None) or r["create"]:
                    print("FAIL", json.dumps({k: r[k] for k in
                                              ("media", "offset", "size", "create", "map",
                                               "setaccess")}), flush=True)
    ok_host = [r for r in out["sweep"] if r["media"] == "host" and r["setaccess"] == 0]
    bad_host = [r for r in out["sweep"] if r["media"] == "host" and r["setaccess"] not in (0, None)]
    ok_dev = [r for r in out["sweep"] if r["media"] == "device" and r["setaccess"] == 0]
    bad_dev = [r for r in out["sweep"] if r["media"] == "device" and r["setaccess"] not in (0, None)]
    print(json.dumps({
        "host_ok": len(ok_host), "host_bad": len(bad_host),
        "dev_ok": len(ok_dev), "dev_bad": len(bad_dev),
        "host_ok_examples": [(r["offset"], r["size"]) for r in ok_host][:20],
        "host_bad_examples": [(r["offset"], r["size"]) for r in bad_host][:20],
        "dev_bad_examples": [(r["offset"], r["size"]) for r in bad_dev][:20],
    }, indent=1), flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build",
                           "p2_diag_align.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
