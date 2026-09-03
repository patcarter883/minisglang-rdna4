#!/usr/bin/env python3
"""Scoping test for the P5 COLLATERAL finding.

P5 measured that under `expandable_segments:True` a tensor allocated from torch's DEFAULT caching
allocator immediately after `torch.cuda.empty_cache()` can read back as ZEROS (its `fill_` never
lands), at a VA torch had just unmapped and re-mapped. That was observed with a user `MemPool` and
a custom allocator live in the same process.

The scoping question this answers: does the defect need any of that, or does it reproduce in a
PLAIN torch process? Only the second reading makes it a serve-wide hazard
(`engine/graph.py:314` calls `empty_cache()`, and the compose default sets expandable_segments).

NO custom allocator, NO MemPool, NO VMM reservation here -- just torch. Verification is host-side
(D2H into a buffer allocated once, CPU compare), so the checker itself cannot be the thing that
breaks.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import sys
import time

import torch

hipMemcpyDeviceToHost = 2


def main() -> int:
    dev = 0
    n_elem = int(os.environ.get("DIAG_ELEMS", str(4 << 20)))  # 16 MiB of float32
    reps = int(os.environ.get("DIAG_REPS", "12"))
    torch.cuda.set_device(dev)
    keep = [torch.zeros(1024, device=f"cuda:{dev}")]
    torch.cuda.synchronize(dev)

    lib = ctypes.CDLL(ctypes.util.find_library("amdhip64") or "libamdhip64.so")
    lib.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    nb = n_elem * 4
    buf = (ctypes.c_ubyte * nb)()
    addr = ctypes.addressof(buf)
    hostview = torch.frombuffer(memoryview(buf), dtype=torch.uint8)

    def host_is_fill(ptr: int, value: float) -> dict:
        rc = lib.hipMemcpy(ctypes.c_void_p(addr), ctypes.c_void_p(ptr), ctypes.c_size_t(nb),
                           hipMemcpyDeviceToHost)
        if rc != 0:
            return {"ok": False, "hipMemcpy_rc": rc}
        expect = torch.empty(n_elem, dtype=torch.float32).fill_(value).view(torch.uint8)
        ok = bool(torch.equal(hostview, expect))
        d = {"ok": ok}
        if not ok:
            asf = hostview.view(torch.float32)
            d.update({"n_bad_elems": int((asf != value).sum().item()),
                      "first_value_seen": float(asf[0].item()), "expected": value})
        return d

    segs = torch.cuda.memory_snapshot()
    flags = [s.get("is_expandable") for s in segs if "is_expandable" in s]
    out = {
        "schema": "p5-diag-pure-torch/1",
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "NO custom allocator, NO MemPool, NO VMM reservation. Plain torch only.",
        "env": {k: os.environ.get(k) for k in
                ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "ROCR_VISIBLE_DEVICES")},
        "torch": {"version": torch.__version__, "hip": getattr(torch.version, "hip", None),
                  "allocator_backend": torch.cuda.get_allocator_backend(),
                  "expandable_effective": (any(flags) if flags else None)},
        "device": {"name": torch.cuda.get_device_properties(dev).name,
                   "elems": n_elem, "bytes": nb},
    }

    reps_out = []
    for i in range(reps):
        v = float(i + 1)
        # a block torch will cache, then release on empty_cache()
        tmp = torch.empty(n_elem, dtype=torch.float32, device=f"cuda:{dev}").fill_(v)
        ptr_tmp = tmp.data_ptr()
        del tmp
        torch.cuda.empty_cache()          # <- engine/graph.py:314 does exactly this
        t = torch.empty(n_elem, dtype=torch.float32, device=f"cuda:{dev}").fill_(v)
        torch.cuda.synchronize(dev)
        chk = host_is_fill(t.data_ptr(), v)
        reps_out.append({"rep": i, "value": v, "ptr_before": ptr_tmp, "ptr_after": t.data_ptr(),
                         "va_reused": t.data_ptr() == ptr_tmp, "host_check": chk})
        del t

    bad = [r for r in reps_out if not r["host_check"]["ok"]]
    out["reps"] = reps_out
    out["n_reps"] = len(reps_out)
    out["n_reps_corrupt"] = len(bad)
    out["va_reuse_rate"] = sum(1 for r in reps_out if r["va_reused"]) / max(1, len(reps_out))
    out["verdict"] = (
        "REPRODUCES in plain torch -- serve-wide hazard" if bad
        else "does NOT reproduce without a user MemPool present")
    os.makedirs(os.path.dirname(os.environ["DIAG_OUT"]), exist_ok=True)
    with open(os.environ["DIAG_OUT"], "w") as fh:
        json.dump(out, fh, indent=2)
    try:
        os.chown(os.environ["DIAG_OUT"], int(os.environ.get("P5_CHOWN_UID", "0")),
                 int(os.environ.get("P5_CHOWN_GID", "0")))
    except Exception:
        pass
    print(json.dumps({"verdict": out["verdict"], "n_reps_corrupt": out["n_reps_corrupt"],
                      "n_reps": out["n_reps"], "va_reuse_rate": out["va_reuse_rate"],
                      "expandable": out["torch"]["expandable_effective"],
                      "first_bad": (bad[0] if bad else None)}, indent=2))
    keep.clear()
    return 0


if __name__ == "__main__":
    sys.exit(main())
