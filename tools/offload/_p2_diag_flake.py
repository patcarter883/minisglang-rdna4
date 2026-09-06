#!/usr/bin/env python3
"""Quantify the hipMemSetAccess(hipError 1) flake on gfx1201 / ROCm 7.2.

Three independent stressors, each reporting a failure RATE and WHICH call failed:
  A. one reservation, N adjacent uniform DEVICE chunks
  B. one reservation, N adjacent uniform HOST chunks
  C. one reservation, N adjacent chunks alternating media at P2's real row sizes
  D. the minimal (dev 1.5 MiB @0, host 7.5 MiB @1.5 MiB) pair, repeated in fresh reservations

Everything is raw ctypes; no torch.
"""
import ctypes, json, os, sys, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_mixed_media_moe as P2

_LOC_DEV, _LOC_HOST = 1, 2
_ACCESS_RW = 3
hip = P2.Hip()
desc = P2._MemAccessDesc()
desc.location.type = _LOC_DEV
desc.location.id = 0
desc.flags = _ACCESS_RW


class Res:
    def __init__(self, total):
        va = ctypes.c_void_p()
        hip.ck(hip.lib.hipMemAddressReserve(ctypes.byref(va), total, 2 << 20, None, 0), "reserve")
        self.base = int(va.value)
        self.total = total
        self.handles, self.mapped = [], []

    def chunk(self, off, size, loc):
        h = ctypes.c_void_p()
        rc = hip.lib.hipMemCreate(ctypes.byref(h), size, ctypes.byref(hip.prop(loc, 0)), 0)
        if rc:
            return ("create", rc)
        self.handles.append(h)
        ptr = self.base + off
        rc = hip.lib.hipMemMap(ctypes.c_void_p(ptr), size, 0, h, 0)
        if rc:
            return ("map", rc)
        self.mapped.append((ptr, size))
        rc = hip.lib.hipMemSetAccess(ctypes.c_void_p(ptr), size, ctypes.byref(desc), 1)
        if rc:
            return ("setaccess", rc)
        return None

    def free(self, address_free=True):
        for p, s in self.mapped:
            hip.lib.hipMemUnmap(ctypes.c_void_p(p), s)
        for h in self.handles:
            hip.lib.hipMemRelease(h)
        if address_free:
            hip.lib.hipMemAddressFree(ctypes.c_void_p(self.base), self.total)


def stress(name, sizes_media, address_free=True):
    total = sum(s for s, _ in sizes_media)
    total = ((total + (2 << 20) - 1) // (2 << 20)) * (2 << 20)
    r = Res(total)
    fails = []
    off = 0
    for i, (sz, loc) in enumerate(sizes_media):
        f = r.chunk(off, sz, loc)
        if f:
            fails.append({"i": i, "call": f[0], "rc": f[1], "size": sz,
                          "media": "host" if loc == _LOC_HOST else "dev",
                          "va": hex(r.base + off)})
        off += sz
    r.free(address_free)
    return {"stress": name, "n": len(sizes_media), "n_fail": len(fails),
            "rate": round(len(fails) / max(1, len(sizes_media)), 4),
            "first_fail": fails[0] if fails else None,
            "fail_calls": dict(collections.Counter(f["call"] for f in fails)),
            "fail_media": dict(collections.Counter(f["media"] for f in fails))}


D, H = _LOC_DEV, _LOC_HOST
MB = 1 << 20
W13, W2 = 1572864, 786432


def main():
    N = int(os.environ.get("P2_DIAG_N", "256"))
    out = []
    for trial in range(int(os.environ.get("P2_DIAG_TRIALS", "3"))):
        out.append(stress(f"A_dev_uniform1p5M#{trial}", [(W13, D)] * N))
        out.append(stress(f"B_host_uniform1p5M#{trial}", [(W13, H)] * N))
        out.append(stress(f"C_alt_w13rows#{trial}",
                          [(W13, H if i % 2 else D) for i in range(N)]))
        out.append(stress(f"D_alt_mixedsize#{trial}",
                          [((W13 if i % 3 else W2), H if i % 2 else D) for i in range(N)]))
        for r in out[-4:]:
            print(json.dumps(r), flush=True)

    # E: the minimal pair, many fresh reservations in one process
    pair_fail = 0
    PAIRS = int(os.environ.get("P2_DIAG_PAIRS", "100"))
    firsts = collections.Counter()
    for i in range(PAIRS):
        r = Res(1782579200)
        f1 = r.chunk(0, W13, D)
        f2 = r.chunk(W13, 5 * W13, H)
        if f1 or f2:
            pair_fail += 1
            firsts[(f1 or f2)[0] + "/" + ("dev" if f1 else "host")] += 1
        r.free()
    print(json.dumps({"stress": "E_minimal_pair_fresh_reservations", "n": PAIRS,
                      "n_fail": pair_fail, "rate": round(pair_fail / PAIRS, 4),
                      "fail_kinds": dict(firsts)}), flush=True)

    tot_n = sum(r["n"] for r in out)
    tot_f = sum(r["n_fail"] for r in out)
    print(json.dumps({"AGGREGATE_chunks": tot_n, "AGGREGATE_fails": tot_f,
                      "AGGREGATE_rate": round(tot_f / max(1, tot_n), 5)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
