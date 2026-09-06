#!/usr/bin/env python3
"""Find a hipMemSetAccess call pattern that is 100% reliable for P2's mixed-media layout.

Established by _p2_diag_flake.py:
  * a run of ADJACENT, EQUAL-SIZED mappings + per-chunk hipMemSetAccess never fails (0/3072)
  * as soon as adjacent mappings have DIFFERENT sizes, per-chunk hipMemSetAccess returns
    hipErrorInvalidValue ~33-50% of the time, on BOTH media, and a retry at the same VA
    also fails.

Candidate fixes, each stressed over P2's real layout (E=512, random placement, 4 components):
  F1  per-chunk SetAccess                                   (the current, broken pattern)
  F2  ONE SetAccess per component region, after all its runs are mapped
  F3  ONE SetAccess over the whole reservation, after everything is mapped
  F4  per-chunk SetAccess, but every chunk is exactly one expert row (uniform within a
      component) with a 2 MiB gap between component regions
Each is repeated so a ~33% per-chunk flake cannot pass by luck.
"""
import ctypes, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_mixed_media_moe as P2

_LOC_DEV, _LOC_HOST = 1, 2
_ACCESS_RW = 3
hip = P2.Hip()
desc = P2._MemAccessDesc()
desc.location.type = _LOC_DEV
desc.location.id = 0
desc.flags = _ACCESS_RW
GRAN = 4096
MB = 1 << 20


def build(gap=0):
    comps = P2.build_components(2048, 768, 128)
    E = 512
    mask = P2.build_placement(E, 10, 25, "random", random.Random(0))
    off = 0
    for c in comps:
        c.offset = off
        off += c.row_bytes * E
        off = ((off + gap + (2 << 20) - 1) // (2 << 20)) * (2 << 20) if gap else \
            ((off + GRAN - 1) // GRAN) * GRAN
    total = ((off + (2 << 20) - 1) // (2 << 20)) * (2 << 20)
    return comps, E, mask, total


def setaccess(ptr, size):
    return hip.lib.hipMemSetAccess(ctypes.c_void_p(ptr), size, ctypes.byref(desc), 1)


def run(mode, rep):
    gap = 2 << 20 if mode == "F4" else 0
    comps, E, mask, total = build(gap)
    va = ctypes.c_void_p()
    hip.ck(hip.lib.hipMemAddressReserve(ctypes.byref(va), total, 2 << 20, None, 0), "reserve")
    base = int(va.value)
    handles, mapped, fails = [], [], []
    rs = P2.runs(mask)
    chunks = 0
    for c in comps:
        cbase = base + c.offset
        plan = ([(i, 1, mask[i]) for i in range(E)] if mode == "F4" else rs)
        for start, count, is_host in plan:
            loc = _LOC_HOST if is_host else _LOC_DEV
            ptr = cbase + start * c.row_bytes
            size = count * c.row_bytes
            h = ctypes.c_void_p()
            rc = hip.lib.hipMemCreate(ctypes.byref(h), size, ctypes.byref(hip.prop(loc, 0)), 0)
            if rc:
                fails.append({"call": "create", "rc": rc, "comp": c.name}); continue
            handles.append(h)
            rc = hip.lib.hipMemMap(ctypes.c_void_p(ptr), size, 0, h, 0)
            if rc:
                fails.append({"call": "map", "rc": rc, "comp": c.name}); continue
            mapped.append((ptr, size))
            chunks += 1
            if mode in ("F1", "F4"):
                rc = setaccess(ptr, size)
                if rc:
                    fails.append({"call": "setaccess", "rc": rc, "comp": c.name,
                                  "va": hex(ptr), "size": size})
        if mode == "F2":
            rc = setaccess(cbase, c.row_bytes * E)
            if rc:
                fails.append({"call": "setaccess(component)", "rc": rc, "comp": c.name,
                              "va": hex(cbase), "size": c.row_bytes * E})
    if mode == "F3":
        span = comps[-1].offset + comps[-1].row_bytes * E
        rc = setaccess(base, span)
        if rc:
            fails.append({"call": "setaccess(whole)", "rc": rc, "va": hex(base), "size": span})
    res = {"mode": mode, "rep": rep, "chunks": chunks, "n_fail": len(fails),
           "first_fail": fails[0] if fails else None}
    for p, s in mapped:
        hip.lib.hipMemUnmap(ctypes.c_void_p(p), s)
    for h in handles:
        hip.lib.hipMemRelease(h)
    hip.lib.hipMemAddressFree(ctypes.c_void_p(base), total)
    return res


def main():
    REPS = int(os.environ.get("P2_DIAG_REPS", "4"))
    summary = {}
    for mode in ("F1", "F2", "F3", "F4"):
        rs = []
        for rep in range(REPS):
            try:
                r = run(mode, rep)
            except Exception as e:
                r = {"mode": mode, "rep": rep, "exception": repr(e), "n_fail": -1}
            rs.append(r)
            print(json.dumps(r), flush=True)
        summary[mode] = {"reps": REPS, "total_fail": sum(max(0, r["n_fail"]) for r in rs),
                         "clean_reps": sum(1 for r in rs if r.get("n_fail") == 0)}
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
