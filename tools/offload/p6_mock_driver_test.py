#!/usr/bin/env python3
"""Offline verification of P6's measurement logic against a MOCK HIP driver.

No GPU is touched.  A simulated VMM implements two drivers:

  * `conformant` -- hipMemMap always binds the requested handle.
  * `stale`      -- the bug this box has: a VA is permanently bound to the FIRST
                    handle ever mapped there; every call still returns hipSuccess.

The point is that `--selftest` only proves the JSON shape.  It cannot prove the
probe would REACH the right verdict, which is the thing that matters for a
tripwire: a probe that can only ever say BROKEN is not a tripwire, and a probe
that says BROKEN because its own shader arm is dead is worse than no probe.

    python3 tools/offload/p6_mock_driver_test.py
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import p6_vmm_remap_conformance as p6  # noqa: E402

HIP_SUCCESS = 0


def _addr_of(x):
    """Address behind a ctypes byref()/pointer/instance."""
    if isinstance(x, ctypes._Pointer):
        return ctypes.cast(x, ctypes.c_void_p).value
    if hasattr(x, "_obj"):
        return ctypes.addressof(x._obj)
    if isinstance(x, ctypes.c_void_p):
        return x.value
    if isinstance(x, int):
        return x
    return ctypes.addressof(x)


class MockLib:
    """Just enough of libamdhip64 to drive VariantProbe end to end."""

    def __init__(self, mode: str, size_hint: int = 1 << 20):
        assert mode in ("conformant", "stale")
        self.mode = mode
        self.pages: dict[int, bytearray] = {}
        self.next_handle = 1
        self.va_next = 0x7000_0000_0000
        self.reservations: list[tuple[int, int]] = []
        self.mapped: dict[int, tuple[int, int]] = {}     # va_base -> (handle, size)
        self.first_bound: dict[int, int] = {}            # va_base -> handle
        self.malloc: dict[int, bytearray] = {}           # base -> buffer
        self.calls: dict[str, int] = {}

    # -- helpers ---------------------------------------------------------
    def _tick(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def _resolve(self, addr: int):
        for base, (h, sz) in self.mapped.items():
            if base <= addr < base + sz:
                eff = self.first_bound[base] if self.mode == "stale" else h
                return self.pages[eff], addr - base
        for base, buf in self.malloc.items():
            if base <= addr < base + len(buf):
                return buf, addr - base
        raise AssertionError(f"unmapped access at 0x{addr:x}")

    # -- HIP surface -----------------------------------------------------
    def hipGetErrorString(self, rc):
        return b"mock"

    def hipDeviceSynchronize(self):
        self._tick("sync")
        return HIP_SUCCESS

    def hipMemGetAllocationGranularity(self, out, prop, kind):
        out._obj.value = 4096
        return HIP_SUCCESS

    def hipMemCreate(self, out, size, prop, flags):
        h = self.next_handle
        self.next_handle += 1
        self.pages[h] = bytearray(size)
        out._obj.value = h
        self._tick("hipMemCreate")
        return HIP_SUCCESS

    def hipMemRelease(self, h):
        return HIP_SUCCESS

    def hipMemAddressReserve(self, out, size, align, addr, flags):
        base = (self.va_next + align - 1) & ~(align - 1)
        self.va_next = base + size + (1 << 20)
        self.reservations.append((base, size))
        out._obj.value = base
        return HIP_SUCCESS

    def hipMemAddressFree(self, va, size):
        return HIP_SUCCESS

    def hipMemMap(self, va, size, off, h, flags):
        base = _addr_of(va)
        assert base not in self.mapped, f"double map at 0x{base:x}"
        handle = _addr_of(h)
        self.mapped[base] = (handle, size)
        self.first_bound.setdefault(base, handle)
        self._tick("hipMemMap")
        return HIP_SUCCESS

    def hipMemUnmap(self, va, size):
        self.mapped.pop(_addr_of(va), None)
        self._tick("hipMemUnmap")
        return HIP_SUCCESS

    def hipMemSetAccess(self, va, size, desc, count):
        return HIP_SUCCESS

    def hipMemsetD32(self, dst, val, n):
        buf, off = self._resolve(_addr_of(dst))
        word = (val & 0xFFFFFFFF).to_bytes(4, "little")
        buf[off:off + 4 * n] = word * n
        self._tick("hipMemsetD32")
        return HIP_SUCCESS

    def hipMemcpy(self, dst, src, n, kind):
        if kind == p6.hipMemcpyDeviceToHost:
            buf, off = self._resolve(_addr_of(src))
            ctypes.memmove(_addr_of(dst), bytes(buf[off:off + n]), n)
        else:
            buf, off = self._resolve(_addr_of(dst))
            buf[off:off + n] = ctypes.string_at(_addr_of(src), n)
        self._tick("hipMemcpy")
        return HIP_SUCCESS

    def hipMalloc(self, out, size):
        base = self.va_next
        self.va_next += size + 4096
        self.malloc[base] = bytearray(size)
        out._obj.value = base
        return HIP_SUCCESS

    def hipFree(self, p):
        return HIP_SUCCESS


class MockHip(p6.Hip):
    def __init__(self, mode):
        self.libpath = f"<mock:{mode}>"
        self.resolved = self.libpath
        self.lib = MockLib(mode)
        self.mode = mode


def run(mode, size=1 << 20, nh=4, reps=5, warmup=1, flush=True):
    hip = MockHip(mode)
    flusher = p6.CacheFlusher(hip, (4 << 20) if flush else 0)
    return hip, p6.run_variant(hip, 0, "device", size, nh, reps, warmup, None, flusher)


def expect(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        return 1
    print(f"  ok: {msg}")
    return 0


def main() -> int:
    bad = 0
    print("mock driver = conformant (hipMemMap binds what you asked for)")
    hip, v = run("conformant")
    bad += expect(v["complete"], "variant completes")
    bad += expect(v["control_ok"] is True, "cross-VA control passes")
    bad += expect(v["minimal_remap"]["copy_conformant"], "minimal_remap 100% correct")
    bad += expect(v["minimal_remap"]["stale_pattern"] == "conformant",
                  "stale_pattern == conformant")
    bad += expect(v["parked_read"]["copy_conformant"], "parked_read 100% correct")
    bad += expect(v["parked_write"]["copy_conformant"], "parked_write lands on the "
                                                        "requested handle")
    bad += expect(v["read_conformant"] and v["write_conformant"],
                  "variant verdict CONFORMANT")
    bad += expect(v["all_calls_returned_hipSuccess"], "no non-zero HIP rc")
    n_obs = 4 * 5
    bad += expect(v["minimal_remap"]["reads_taken_after_cache_flush"] == n_obs,
                  f"all {n_obs} minimal reads taken after a cache flush")
    bad += expect(v["timing_us"]["full_cycle_map_setaccess_unmap"]["n"] == 2 * 4 * 5,
                  "timing has 2 x handles x reps samples, warm-up discarded")
    bad += expect(hip.lib.calls["hipMemMap"] > v["timing_us"]["map"]["n"],
                  "warm-up maps happened but were NOT timed")

    print("\nmock driver = stale (the bug this box has)")
    hip, v = run("stale")
    bad += expect(v["complete"], "variant still completes (every rc is hipSuccess)")
    bad += expect(v["control_ok"] is True,
                  "cross-VA control PASSES -- fresh maps are fine, so a stale read "
                  "downstream is attributable to the remap")
    bad += expect(not v["minimal_remap"]["copy_conformant"], "minimal_remap not conformant")
    bad += expect(v["minimal_remap"]["single_page_hypothesis"],
                  "minimal_remap score == handles^-1 is classified as TOTALLY broken, "
                  "not as 'works 25% of the time'")
    bad += expect("always_serves_a_single_physical_page" in v["minimal_remap"]["stale_pattern"],
                  f"stale_pattern classified: {v['minimal_remap']['stale_pattern'][:70]}...")
    bad += expect(v["parked_read"]["served_handle_sequence"].count("park/handle0")
                  == len(v["parked_read"]["observations"]),
                  "every parked read is attributed to handle 0's page via the "
                  "fingerprint ledger")
    bad += expect(all(o["landed_copy"] == [0] for o in v["parked_write"]["observations"]),
                  "every write lands on handle 0's physical page")
    bad += expect(not v["read_conformant"] and not v["write_conformant"],
                  "variant verdict BROKEN in BOTH directions")
    bad += expect(v["all_calls_returned_hipSuccess"],
                  "and every HIP call still returned hipSuccess (the silent class)")
    bad += expect(v["minimal_only_verdict"] is not None,
                  "minimal_only_verdict is populated for the upstream repro")

    print("\ncontrol failure must SUPPRESS the parked verdict")
    hip = MockHip("stale")
    # Make even a fresh map return the wrong page: hipMemMap becomes a no-op that
    # always hands back handle 1.  Now 'stale' is indistinguishable from 'hipMemMap
    # never worked', and the probe must refuse rather than report.
    real_map = hip.lib.hipMemMap

    def broken_map(va, size, off, h, flags):
        rc = real_map(va, size, off, h, flags)
        hip.lib.first_bound[_addr_of(va)] = 1
        return rc

    hip.lib.hipMemMap = broken_map
    flusher = p6.CacheFlusher(hip, 4 << 20)
    v = p6.run_variant(hip, 0, "device", 1 << 20, 4, 5, 1, None, flusher)
    bad += expect(v["control_ok"] is False, "control fails")
    bad += expect(not v["complete"], "variant is marked INCOMPLETE")
    bad += expect("parked_read" not in v, "no parked_read conformance number is reported")
    bad += expect(v.get("read_conformant") is None,
                  "no read_conformant verdict is reported")
    bad += expect(v["minimal_only_verdict"] is not None,
                  "but the standalone minimal_remap repro SURVIVES the failure")
    bad += expect(v["timing_us"]["unmap"] is not None,
                  "and the remap-cycle timing survives too")

    print("\npartial remap (only the first granule moves) must NOT read as conformant")
    hip = MockHip("conformant")
    real_res = hip.lib._resolve

    def partial_resolve(addr):
        buf, off = real_res(addr)
        # everything past the first 4096 B granule is served by handle 1's page
        for base, (h, sz) in hip.lib.mapped.items():
            if base <= addr < base + sz and (addr - base) >= 4096:
                return hip.lib.pages[hip.lib.first_bound[base]], addr - base
        return buf, off

    hip.lib._resolve = partial_resolve
    flusher = p6.CacheFlusher(hip, 4 << 20)
    v = p6.run_variant(hip, 0, "device", 1 << 20, 4, 5, 1, None, flusher)
    mr = v.get("minimal_remap") or {}
    bad += expect(not mr.get("copy_conformant", True),
                  "a range that is only partly remapped is NOT scored conformant")
    bad += expect(mr.get("reads_with_offset_disagreement", 0) > 0,
                  "offset disagreement is detected (dword 0 alone would have missed it)")

    print()
    print("MOCK DRIVER TEST: " + ("PASS" if bad == 0 else f"FAIL ({bad} checks)"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
