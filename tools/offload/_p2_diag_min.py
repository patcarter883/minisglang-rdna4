#!/usr/bin/env python3
"""Minimal ablation for the hipMemSetAccess(hipError 1) failure.

Failing case (from the exact-layout replay):
    reserve 1782579200 @2MiB
    chunk A: DEVICE, ptr=base+0,       size=1572864   -> create/map/setaccess all OK
    chunk B: HOST,   ptr=base+1572864, size=7864320   -> create OK, map OK, setaccess = 1

Each scenario is run in a FRESH reservation; every variable is ablated one at a time.
"""
import ctypes, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_mixed_media_moe as P2

_LOC_DEV, _LOC_HOST = 1, 2
_ACCESS_RW = 3
hip = P2.Hip()


def scenario(name, total, chunks, access_dev_id=0):
    """chunks: list of (offset, size, loc)"""
    va = ctypes.c_void_p()
    rc = hip.lib.hipMemAddressReserve(ctypes.byref(va), total, 2 << 20, None, 0)
    if rc:
        return {"scenario": name, "reserve_rc": rc}
    base = int(va.value)
    desc = P2._MemAccessDesc()
    desc.location.type = _LOC_DEV
    desc.location.id = access_dev_id
    desc.flags = _ACCESS_RW
    handles, mapped, steps = [], [], []
    for i, (off, size, loc) in enumerate(chunks):
        h = ctypes.c_void_p()
        rc_c = hip.lib.hipMemCreate(ctypes.byref(h), size, ctypes.byref(hip.prop(loc, 0)), 0)
        st = {"i": i, "loc": "host" if loc == _LOC_HOST else "dev", "off": off, "size": size,
              "create": rc_c}
        if rc_c == 0:
            handles.append(h)
            ptr = base + off
            st["map"] = hip.lib.hipMemMap(ctypes.c_void_p(ptr), size, 0, h, 0)
            if st["map"] == 0:
                mapped.append((ptr, size))
                st["setaccess"] = hip.lib.hipMemSetAccess(ctypes.c_void_p(ptr), size,
                                                          ctypes.byref(desc), 1)
        steps.append(st)
    ok = all(s.get("create") == 0 and s.get("map") == 0 and s.get("setaccess") == 0
             for s in steps)
    for s in steps:
        va_abs = base + s["off"]
        s["va"] = hex(va_abs)
        s["va_align"] = va_abs & -va_abs
    for p, s in mapped:
        hip.lib.hipMemUnmap(ctypes.c_void_p(p), s)
    for h in handles:
        hip.lib.hipMemRelease(h)
    hip.lib.hipMemAddressFree(ctypes.c_void_p(base), total)
    return {"scenario": name, "ok": ok, "base": hex(base),
            "base_align": base & -base, "steps": steps}


D, H = _LOC_DEV, _LOC_HOST
A_SZ, B_SZ = 1572864, 7864320
TOTAL = 1782579200
GB = 1 << 30
MB = 1 << 20


def main():
    scen = [
        # 1. the exact minimal failing pair
        ("exact_pair", TOTAL, [(0, A_SZ, D), (A_SZ, B_SZ, H)]),
        # 2. same pair, small reservation -> is the 1.78 GB reservation the variable?
        ("exact_pair_small_reservation", GB, [(0, A_SZ, D), (A_SZ, B_SZ, H)]),
        # 3. host chunk ALONE at the same VA/size
        ("host_alone_same_va", TOTAL, [(A_SZ, B_SZ, H)]),
        # 4. host FIRST, then device -> is the order the variable?
        ("host_first", TOTAL, [(A_SZ, B_SZ, H), (0, A_SZ, D)]),
        # 5. non-adjacent: leave a gap between the device and host chunks
        ("gap_between", TOTAL, [(0, A_SZ, D), (A_SZ + 2 * MB, B_SZ, H)]),
        # 6. two DEVICE chunks, adjacent, same sizes -> is 'host' the variable?
        ("dev_dev_adjacent", TOTAL, [(0, A_SZ, D), (A_SZ, B_SZ, D)]),
        # 7. two HOST chunks, adjacent
        ("host_host_adjacent", TOTAL, [(0, A_SZ, H), (A_SZ, B_SZ, H)]),
        # 8. device then host, both 1 MiB, adjacent (the alt_1MiB case that PASSED)
        ("dev_host_adjacent_1MiB", TOTAL, [(0, MB, D), (MB, MB, H)]),
        # 9. device then host adjacent, both 1.5 MiB
        ("dev_host_adjacent_1p5MiB", TOTAL, [(0, A_SZ, D), (A_SZ, A_SZ, H)]),
        # 10. device 1 MiB then host 7.5 MiB adjacent
        ("dev1M_host7p5M_adjacent", TOTAL, [(0, MB, D), (MB, B_SZ, H)]),
        # 11. device 1.5 MiB then host 1 MiB adjacent
        ("dev1p5M_host1M_adjacent", TOTAL, [(0, A_SZ, D), (A_SZ, MB, H)]),
        # 12. host at a 2 MiB-aligned VA after a device chunk
        ("dev_then_host_2MiB_aligned", TOTAL, [(0, 2 * MB, D), (2 * MB, B_SZ, H)]),
    ]
    results = []
    REPEATS = int(os.environ.get("P2_DIAG_REPEATS", "3"))
    for rep in range(REPEATS):
        for name, total, chunks in scen:
            r = scenario(f"{name}#{rep}", total, chunks)
            results.append(r)
            print(json.dumps(r), flush=True)
    print("\nSUMMARY")
    for r in results:
        hoststeps = [s for s in r.get("steps", []) if s["loc"] == "host"]
        print(f"  {'PASS' if r.get('ok') else 'FAIL'}  {r['scenario']:34s} base={r.get('base')} "
              + " ".join(f"host@{s['va']}(align={s['va_align']},sz={s['size']})->sa={s.get('setaccess')}"
                         for s in hoststeps))


if __name__ == "__main__":
    main()
