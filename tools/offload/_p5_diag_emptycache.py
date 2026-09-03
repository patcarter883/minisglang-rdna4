#!/usr/bin/env python3
"""P5 DIAGNOSTIC -- who actually gets corrupted around torch.cuda.empty_cache()?

The P5 `empty_cache_live` arm failed in BOTH expandable_segments:True legs (and passed in both
plain legs) with a perfectly ALTERNATING per-rep pattern. That arm's check is

    torch.equal(arena, torch.full_like(arena, v))

which compares TWO device tensors: our foreign-VA arena, and a fresh tensor allocated from torch's
OWN (expandable-segment) caching allocator. A mismatch is therefore ambiguous, and the two readings
have opposite consequences:

  (A) OUR arena lost its contents  -> the offload arena is unsafe under the compose default, P5 red.
  (B) TORCH's comparison tensor is wrong -> our arena is fine, and torch's expandable_segments
      allocator is returning corrupt memory after empty_cache() on this driver. That is a
      SERVE-WIDE correctness bug, not a P5 result. (It is also exactly the shape of the known
      hipMemUnmap->hipMemMap-at-a-used-VA defect on this box: expandable segments unmap physical
      handles on empty_cache() and re-map them at the SAME VA on the next allocation.)

This diagnostic removes the ambiguity: every verification is a ctypes hipMemcpy D2H into a host
buffer allocated ONCE before the loop, compared on the CPU. No device allocation happens inside the
checked region unless the arm is explicitly probing one, and each candidate (arena vs a plain torch
tensor) is verified INDEPENDENTLY against a host-side expectation.

Arms per configuration:
  seq_arena_noec   : fill/verify the arena in a loop with NO empty_cache  (is empty_cache needed?)
  seq_arena_ec     : fill / empty_cache / verify / write / verify         (host-verified)
  seq_torchtensor  : the same for a PLAIN torch tensor from the default allocator (host-verified)
  torch_equal_arm  : reproduces P5's original device-vs-device comparison AND host-verifies BOTH
                     operands in the same rep -- this is the discriminator.

Run via _p5_diag_run.sh (in the serve image, worktree mounted).
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import p5_torch_foreign_ptr as p5  # noqa: E402  (sets ROCR_VISIBLE_DEVICES on import)

OUT_DIR = os.environ.get(
    "DIAG_OUT",
    "/engine/docs/measurements/WEIGHT_OFFLOAD_2026-09-02/p5_diagnostics",
)


def build_arena(hip, dev: int, backing: str, size_want: int):
    prop = hip.prop(backing, dev)
    gran = max(hip.granularity(prop, p5.hipMemAllocationGranularityMinimum), 4096)
    size = p5._round_up(size_want, gran)
    va = ctypes.c_void_p()
    hip.check(
        hip.lib.hipMemAddressReserve(
            ctypes.byref(va), ctypes.c_size_t(size), ctypes.c_size_t(gran), None,
            ctypes.c_ulonglong(0)),
        "hipMemAddressReserve")
    handle = ctypes.c_void_p()
    hip.check(
        hip.lib.hipMemCreate(ctypes.byref(handle), ctypes.c_size_t(size), ctypes.byref(prop),
                             ctypes.c_ulonglong(0)),
        "hipMemCreate")
    hip.check(hip.lib.hipMemMap(va, ctypes.c_size_t(size), ctypes.c_size_t(0), handle,
                                ctypes.c_ulonglong(0)), "hipMemMap")
    desc = p5.HipMemAccessDesc()
    desc.location.type = p5.hipMemLocationTypeDevice
    desc.location.id = dev
    desc.flags = p5.hipMemAccessFlagsProtReadWrite
    hip.check(hip.lib.hipMemSetAccess(va, ctypes.c_size_t(size), ctypes.byref(desc),
                                      ctypes.c_size_t(1)), "hipMemSetAccess")
    p5._KEEP.extend([va, handle, prop, desc])
    return int(va.value), int(size), gran


class HostCheck:
    """One host buffer, allocated ONCE. d2h(ptr,nbytes) -> (n_mismatch, first_bad_index,
    first_bad_value) against a scalar float32 expectation. No device allocation, no torch."""

    def __init__(self, hip, nbytes: int):
        self.hip = hip
        self.nbytes = nbytes
        self.buf = (ctypes.c_float * (nbytes // 4))()
        self.addr = ctypes.addressof(self.buf)

    def d2h(self, dptr: int, nbytes: int) -> None:
        rc = self.hip.lib.hipMemcpy(ctypes.c_void_p(self.addr), ctypes.c_void_p(dptr),
                                    ctypes.c_size_t(nbytes), p5.hipMemcpyDeviceToHost)
        self.hip.check(rc, "hipMemcpy D2H")

    def verify_scalar(self, dptr: int, n_elem: int, expect: float) -> dict:
        nb = n_elem * 4
        self.d2h(dptr, nb)
        bad = 0
        first_i = None
        first_v = None
        b = self.buf
        for i in range(n_elem):
            if b[i] != expect:
                bad += 1
                if first_i is None:
                    first_i, first_v = i, float(b[i])
        return {"n_elem": n_elem, "expect": expect, "n_mismatch": bad,
                "first_bad_index": first_i, "first_bad_value": first_v,
                "ok": bad == 0}


def main() -> int:
    import torch

    dev = 0
    backing = os.environ.get("DIAG_BACKING", "host")
    slot_bytes = int(os.environ.get("DIAG_SLOT_BYTES", str(16 << 20)))
    n_slots = int(os.environ.get("DIAG_SLOTS", "13"))
    reps = int(os.environ.get("DIAG_REPS", "8"))
    # verify only a prefix on the host: a full 16 MiB python-loop compare would dominate runtime and
    # the corruption we are chasing was whole-tensor, so a 64 Ki-element prefix is ample.
    n_check = int(os.environ.get("DIAG_CHECK_ELEMS", str(1 << 16)))

    torch.cuda.set_device(dev)
    p5._KEEP.append(torch.zeros(1024, device=f"cuda:{dev}"))
    torch.cuda.synchronize(dev)

    hip = p5.Hip()
    p5._HIP = hip
    p5._KEEP.append(hip)
    hip.set_device(dev)

    out = {
        "schema": "p5-diag-emptycache/1",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {"backing": backing, "slot_bytes": slot_bytes, "n_slots": n_slots,
                   "reps": reps, "check_elems": n_check, "device": dev},
        "env": {k: os.environ.get(k) for k in
                ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF",
                 "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")},
        "torch": {"version": torch.__version__, "hip": getattr(torch.version, "hip", None)},
    }
    try:
        out["torch"]["allocator_backend"] = torch.cuda.get_allocator_backend()
    except Exception:
        pass
    segs = torch.cuda.memory_snapshot()
    flags = [s.get("is_expandable") for s in segs if "is_expandable" in s]
    out["torch"]["expandable_effective"] = (any(flags) if flags else None)
    props = torch.cuda.get_device_properties(dev)
    out["device"] = {"name": props.name, "pci": hip.pci_bus_id(dev),
                     "physical_card": (os.environ.get("ROCR_VISIBLE_DEVICES") or "").split(",")[dev]}

    base, size, gran = build_arena(hip, dev, backing, 3 * (n_slots + 2) * slot_bytes)
    p5._ARENA.update({"base": base, "size": size, "cursor": 0, "device": dev, "fallbacks": 0})
    out["arena"] = {"base": base, "size": size, "granularity": gran, "backing": backing}

    alloc_cb, free_cb = p5._install_callbacks()
    allocator = torch._C._cuda_customAllocator(
        ctypes.cast(alloc_cb, ctypes.c_void_p).value,
        ctypes.cast(free_cb, ctypes.c_void_p).value)
    p5._KEEP.append(allocator)
    pool = torch.cuda.MemPool(allocator)
    p5._KEEP.append(pool)

    n_elem = slot_bytes // 4
    devstr = f"cuda:{dev}"
    slots = []
    for _ in range(n_slots):
        with torch.cuda.use_mem_pool(pool):
            t = torch.empty(n_elem, dtype=torch.float32, device=devstr)
        p5._KEEP.append(t)
        slots.append(t)
    arena = slots[0]
    out["arena"]["first_ptr"] = arena.data_ptr()
    out["arena"]["first_ptr_is_base"] = arena.data_ptr() == base
    out["arena"]["fallbacks"] = p5._ARENA["fallbacks"]

    hc = HostCheck(hip, n_check * 4)

    # ---- ARM 1: is empty_cache() even required to see it? ---------------------------------------
    a1 = []
    for i in range(reps):
        v = float(i + 1)
        arena.fill_(v)
        torch.cuda.synchronize(dev)
        a1.append({"rep": i, "after_fill": hc.verify_scalar(arena.data_ptr(), n_check, v)})
    out["seq_arena_noec"] = {
        "question": "does the arena hold its contents across fills with NO empty_cache?",
        "reps": a1,
        "all_ok": all(r["after_fill"]["ok"] for r in a1),
    }

    # ---- ARM 2: the arena across empty_cache(), HOST-verified ------------------------------------
    a2 = []
    ptr0 = arena.data_ptr()
    for i in range(reps):
        pre = float((i % 97) + 1)     # same value schedule as the P5 arm
        post = float((i % 89) + 3)
        arena.fill_(pre)
        torch.cuda.synchronize(dev)
        before = hc.verify_scalar(arena.data_ptr(), n_check, pre)
        n_free0 = len(p5._FREE_EVENTS)
        t0 = time.perf_counter_ns()
        torch.cuda.empty_cache()
        t1 = time.perf_counter_ns()
        after = hc.verify_scalar(arena.data_ptr(), n_check, pre)
        arena.fill_(post)
        torch.cuda.synchronize(dev)
        written = hc.verify_scalar(arena.data_ptr(), n_check, post)
        a2.append({
            "rep": i, "pre": pre, "post": post,
            "ptr_stable": arena.data_ptr() == ptr0,
            "free_cbs": len(p5._FREE_EVENTS) - n_free0,
            "empty_cache_us": (t1 - t0) / 1e3,
            "arena_ok_before_empty_cache": before,
            "arena_ok_after_empty_cache": after,
            "arena_writable_after_empty_cache": written,
        })
    out["seq_arena_ec"] = {
        "question": "HOST-VERIFIED: does OUR arena survive empty_cache()?",
        "reps": a2,
        "all_survived": all(r["arena_ok_after_empty_cache"]["ok"] for r in a2),
        "all_writable": all(r["arena_writable_after_empty_cache"]["ok"] for r in a2),
        "ptr_stable": all(r["ptr_stable"] for r in a2),
        "free_cb_fired": any(r["free_cbs"] for r in a2),
    }

    # ---- ARM 3: a PLAIN torch tensor (default allocator) across empty_cache, host-verified -------
    plain = torch.empty(n_elem, dtype=torch.float32, device=devstr)  # NOT in our pool
    p5._KEEP.append(plain)
    a3 = []
    pptr0 = plain.data_ptr()
    for i in range(reps):
        pre = float((i % 97) + 1)
        post = float((i % 89) + 3)
        plain.fill_(pre)
        torch.cuda.synchronize(dev)
        before = hc.verify_scalar(plain.data_ptr(), n_check, pre)
        torch.cuda.empty_cache()
        after = hc.verify_scalar(plain.data_ptr(), n_check, pre)
        plain.fill_(post)
        torch.cuda.synchronize(dev)
        written = hc.verify_scalar(plain.data_ptr(), n_check, post)
        a3.append({"rep": i, "pre": pre, "post": post,
                   "ptr_stable": plain.data_ptr() == pptr0,
                   "ok_before": before, "ok_after": after, "writable_after": written})
    out["seq_torchtensor"] = {
        "question": "HOST-VERIFIED: does a PLAIN torch tensor survive empty_cache() on this box?",
        "reps": a3,
        "all_survived": all(r["ok_after"]["ok"] for r in a3),
        "all_writable": all(r["writable_after"]["ok"] for r in a3),
        "ptr_stable": all(r["ptr_stable"] for r in a3),
    }

    # ---- ARM 4: THE DISCRIMINATOR --------------------------------------------------------------
    # Reproduce P5's device-vs-device comparison and host-verify BOTH operands in the same rep.
    a4 = []
    for i in range(reps):
        pre = float((i % 97) + 1)
        post = float((i % 89) + 3)
        arena.fill_(pre)
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        ref = torch.full_like(arena, pre)          # fresh alloc from torch's OWN allocator
        torch.cuda.synchronize(dev)
        equal_says = bool(torch.equal(arena, ref))
        arena_host = hc.verify_scalar(arena.data_ptr(), n_check, pre)
        ref_host = hc.verify_scalar(ref.data_ptr(), n_check, pre)
        arena.fill_(post)
        torch.cuda.synchronize(dev)
        ref2 = torch.full_like(arena, post)
        torch.cuda.synchronize(dev)
        equal_says2 = bool(torch.equal(arena, ref2))
        arena_host2 = hc.verify_scalar(arena.data_ptr(), n_check, post)
        ref2_host = hc.verify_scalar(ref2.data_ptr(), n_check, post)
        a4.append({
            "rep": i, "pre": pre, "post": post,
            "ref_ptr": ref.data_ptr(), "ref2_ptr": ref2.data_ptr(),
            "torch_equal_pre": equal_says, "arena_host_pre": arena_host, "ref_host_pre": ref_host,
            "torch_equal_post": equal_says2, "arena_host_post": arena_host2,
            "ref_host_post": ref2_host,
        })
        del ref, ref2
    n_arena_bad = sum(1 for r in a4
                      if not (r["arena_host_pre"]["ok"] and r["arena_host_post"]["ok"]))
    n_ref_bad = sum(1 for r in a4
                    if not (r["ref_host_pre"]["ok"] and r["ref_host_post"]["ok"]))
    n_equal_false = sum(1 for r in a4 if not (r["torch_equal_pre"] and r["torch_equal_post"]))
    out["torch_equal_arm"] = {
        "question": "when torch.equal(arena, full_like) is FALSE, which operand is actually wrong?",
        "reps": a4,
        "n_reps": len(a4),
        "n_reps_torch_equal_false": n_equal_false,
        "n_reps_arena_host_wrong": n_arena_bad,
        "n_reps_ref_tensor_host_wrong": n_ref_bad,
        "verdict": (
            "arena_corrupted" if n_arena_bad else
            "ref_tensor_corrupted" if n_ref_bad else
            "both_operands_fine" if n_equal_false else "no_failure_reproduced"
        ),
    }

    out["conclusion"] = {
        "arena_survives_empty_cache_host_verified": out["seq_arena_ec"]["all_survived"]
        and out["seq_arena_ec"]["all_writable"],
        "plain_torch_tensor_survives_empty_cache": out["seq_torchtensor"]["all_survived"]
        and out["seq_torchtensor"]["all_writable"],
        "discriminator": out["torch_equal_arm"]["verdict"],
    }
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{backing}_{'exp' if out['torch'].get('expandable_effective') else 'plain'}"
    path = os.path.join(OUT_DIR, f"p5_diag_emptycache_{tag}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
    try:
        os.chown(path, int(os.environ.get("P5_CHOWN_UID", "0")),
                 int(os.environ.get("P5_CHOWN_GID", "0")))
    except Exception:
        pass
    print(json.dumps({"wrote": path, "conclusion": out["conclusion"],
                      "seq_arena_noec_all_ok": out["seq_arena_noec"]["all_ok"],
                      "expandable_effective": out["torch"]["expandable_effective"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
