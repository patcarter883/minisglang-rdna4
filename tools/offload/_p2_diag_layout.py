#!/usr/bin/env python3
"""Replay P2's EXACT arena mapping with per-chunk logging, using P2's own layout code.

Answers: which chunk index / component / media / size / VA does hipMemSetAccess first reject,
and is the failure a function of chunk COUNT, of the VA, or of the media transition.
"""
import ctypes, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_mixed_media_moe as P2

HIP_SO = "libamdhip64.so"
_PINNED = 1
_LOC_DEV, _LOC_HOST = 1, 2
_ACCESS_RW = 3


def main():
    hip = P2.Hip()
    gran = hip.granularity(_LOC_DEV, 0)
    comps = P2.build_components(2048, 768, 128)
    E = 512
    rng = random.Random(0)
    mask = P2.build_placement(E, 10, 25, "random", rng)
    layout = P2.plan_layout(comps, E, gran, 268435456)
    total = layout["total_bytes"]
    rs = P2.runs(mask)
    print(json.dumps({"gran": gran, "total_bytes": total, "n_runs": len(rs),
                      "n_chunks_expected": len(rs) * len(comps) + 2,
                      "row_bytes": {c.name: c.row_bytes for c in comps},
                      "offsets": {c.name: c.offset for c in comps}}), flush=True)

    va = ctypes.c_void_p()
    hip.ck(hip.lib.hipMemAddressReserve(ctypes.byref(va), total, 2 << 20, None, 0), "reserve")
    base = int(va.value)
    desc = P2._MemAccessDesc()
    desc.location.type = _LOC_DEV
    desc.location.id = 0
    desc.flags = _ACCESS_RW

    handles, mapped = [], []
    n = 0
    dev_bytes = host_bytes = 0
    fail = None
    plan = []
    for c in comps:
        c.base = base + c.offset
        for start, count, is_host in rs:
            plan.append((c, start, count, is_host))
    for name, off in layout["scratch_offsets"].items():
        pass

    for c, start, count, is_host in plan:
        loc = _LOC_HOST if is_host else _LOC_DEV
        ptr = c.base + start * c.row_bytes
        size = count * c.row_bytes
        h = ctypes.c_void_p()
        rc = hip.lib.hipMemCreate(ctypes.byref(h), size,
                                  ctypes.byref(hip.prop(loc, 0 if is_host else 0)), 0)
        if rc:
            fail = {"call": "hipMemCreate", "rc": rc}
        else:
            handles.append(h)
            rc = hip.lib.hipMemMap(ctypes.c_void_p(ptr), size, 0, h, 0)
            if rc:
                fail = {"call": "hipMemMap", "rc": rc}
            else:
                mapped.append((ptr, size))
                rc = hip.lib.hipMemSetAccess(ctypes.c_void_p(ptr), size, ctypes.byref(desc), 1)
                if rc:
                    fail = {"call": "hipMemSetAccess", "rc": rc}
        if fail:
            fail.update({"chunk_index": n, "component": c.name, "is_host": is_host,
                         "expert_start": start, "count": count, "size": size,
                         "ptr_hex": hex(ptr), "ptr_align": ptr & 0xFFFFF,
                         "dev_bytes_so_far": dev_bytes, "host_bytes_so_far": host_bytes,
                         "chunks_ok": n})
            break
        if is_host:
            host_bytes += size
        else:
            dev_bytes += size
        n += 1
    print(json.dumps({"chunks_ok": n, "dev_bytes": dev_bytes, "host_bytes": host_bytes,
                      "fail": fail}, indent=1), flush=True)

    # If it failed, probe the immediate neighbourhood: was it the SIZE, the VA, or exhaustion?
    if fail and fail.get("call") == "hipMemSetAccess":
        # try SetAccess again on the same range (idempotency / transient?)
        p, s = mapped[-1]
        rc2 = hip.lib.hipMemSetAccess(ctypes.c_void_p(p), s, ctypes.byref(desc), 1)
        # try a smaller sub-range
        rc3 = hip.lib.hipMemSetAccess(ctypes.c_void_p(p), gran, ctypes.byref(desc), 1)
        print(json.dumps({"retry_same_range_rc": rc2, "retry_one_gran_rc": rc3}), flush=True)

    for p, s in mapped:
        hip.lib.hipMemUnmap(ctypes.c_void_p(p), s)
    for h in handles:
        hip.lib.hipMemRelease(h)


if __name__ == "__main__":
    main()
