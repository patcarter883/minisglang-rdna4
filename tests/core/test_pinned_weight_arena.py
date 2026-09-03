"""`PinnedWeightArena` lifecycle, against an injected fake HIP. NO GPU, NO torch.

The fake is not a mock that records calls — it is real memory with real pointers, so the bump
arithmetic, the pointer translation, the fingerprint self-test and the rollback paths all execute
for real. Two *deliberately broken* fakes are the point of the file:

* `AliasingHip` — every chunk hands back the same physical pages, the P6 failure shape
  (`hipMemUnmap`->`hipMemMap` served the stale page, `nonzero_hip_return_codes = []`);
* `AmnesiacHip` — every call returns success and stores nothing, the P1/P2/P3 failure shape
  (`location.type = Host` echoed back verbatim for pages that were VRAM).

Both return success from every entry point. If the arena passed either one, its self-test would be
asserting on return codes instead of on data — which is precisely what Phase 0 proved you cannot do
on this box.

The GPU-requiring test at the bottom is marked `@pytest.mark.gpu`; run it serially, alone.
"""

from __future__ import annotations

import ctypes
import struct

import pytest
from minisgl.weights import hipmem
from minisgl.weights import host_capacity as hc
from minisgl.weights import pinned_arena as pa
from minisgl.weights.chunk_plan import MIB, RegionRequest
from minisgl.weights.host_capacity import HostArenaCapacityError
from minisgl.weights.pinned_arena import (
    ArenaPhase,
    ArenaSelfTestError,
    ArenaStateError,
    PinnedWeightArena,
    chunk_fingerprint,
    decode_fingerprint,
    verify_offsets,
)

CHUNK = 2 * MIB  # the smallest legal chunk, so a whole arena is a few MiB of test RAM

# Real `hipHostMalloc` returns PAGE-aligned memory. A bare `(c_ubyte * n)()` is aligned only to its
# element type (1 B), so an unpadded fake would be a *less* capable mapping than the thing it stands
# in for — and the arena's structural check on chunk-base alignment would fire on the fake alone.
_FAKE_ALIGN = 4096


def _aligned_buffer(nbytes: int):
    """A `_FAKE_ALIGN`-aligned span of real memory, plus the array that owns it."""
    buf = (ctypes.c_ubyte * (nbytes + _FAKE_ALIGN))()
    raw = ctypes.addressof(buf)
    return buf, (raw + _FAKE_ALIGN - 1) & ~(_FAKE_ALIGN - 1)


# =================================================================================================
# fakes
# =================================================================================================
class FakeHip:
    """Honest fake: real ctypes buffers, unified VA (host_ptr == device_ptr, as measured on this
    box by P1 and P5b)."""

    def __init__(self) -> None:
        self._buffers: dict[int, object] = {}
        self.n_host_alloc = 0
        self.n_host_free = 0
        self.n_sync = 0
        self.device = -1

    # identity
    def set_device(self, dev: int) -> int:
        self.device = int(dev)
        return self.device

    def current_device(self) -> int:
        return self.device

    def pci_bus_id(self, dev: int) -> str:
        return f"0000:0{3 + int(dev) * 4}:00.0"

    def device_name(self, dev: int) -> str:
        return "FakeRDNA4"

    def free_vram(self) -> int:
        return 16 << 30

    def sync(self) -> None:
        self.n_sync += 1

    # the mechanism
    def host_alloc(self, nbytes: int, flags: int = 0):
        self.n_host_alloc += 1
        buf, addr = _aligned_buffer(nbytes)
        self._buffers[addr] = buf  # the backing array is kept alive under the ALIGNED key
        return addr, self._device_ptr_for(addr)

    def _device_ptr_for(self, host_addr: int) -> int:
        return host_addr

    def host_free(self, host_ptr: int) -> None:
        self.n_host_free += 1
        self._buffers.pop(int(host_ptr), None)

    def dev_alloc(self, nbytes: int) -> int:  # pragma: no cover - only the torch fallback uses it
        buf, addr = _aligned_buffer(nbytes)
        self._buffers[addr] = buf
        return addr

    # data movement
    def memcpy(self, dst: int, src: int, nbytes: int, kind: int) -> None:
        ctypes.memmove(int(dst), int(src), int(nbytes))

    def memset_d32(self, dptr: int, word: int, n_words: int) -> None:
        ctypes.memmove(int(dptr), struct.pack("<I", int(word) & 0xFFFFFFFF) * int(n_words),
                       int(n_words) * 4)

    def read_u32(self, dptr: int) -> int:
        return int(ctypes.c_uint32.from_address(int(dptr)).value)

    @property
    def live_allocations(self) -> int:
        return len(self._buffers)


class AliasingHip(FakeHip):
    """Every chunk after the first is a second mapping of chunk 0's pages. All calls succeed."""

    def host_alloc(self, nbytes: int, flags: int = 0):
        self.n_host_alloc += 1
        if not self._buffers:
            buf, self._first = _aligned_buffer(nbytes)
            self._buffers[self._first] = buf
        return self._first, self._first


class AmnesiacHip(FakeHip):
    """Writes are discarded; reads return zeros. Every call returns success."""

    def memset_d32(self, dptr: int, word: int, n_words: int) -> None:
        return

    def memcpy(self, dst: int, src: int, nbytes: int, kind: int) -> None:
        return

    def read_u32(self, dptr: int) -> int:
        return 0


@pytest.fixture(autouse=True)
def _unfreeze_hipmem():
    """`hipmem.freeze()` is process-global (rule R1 is a process-lifetime property), so a test that
    freezes must not leak that into the next one."""
    yield
    hipmem._FROZEN = False
    hipmem._FREEZE_REASON = ""


@pytest.fixture
def no_capacity_limit(monkeypatch):
    """A huge MemAvailable so capacity is not what these tests are measuring.

    Both bindings must be patched: `check_capacity` resolves `mem_available_bytes` in
    `host_capacity`'s globals, while the per-chunk loop in `attach()` uses the name imported into
    `pinned_arena`. Patching one and not the other silently leaves the real /proc in the loop.
    """
    monkeypatch.setattr(hc, "mem_available_bytes", lambda: 1 << 50)
    monkeypatch.setattr(pa, "mem_available_bytes", lambda: 1 << 50)
    return None


def _arena(hip=None, *, chunks_worth: int = 3, **kw) -> tuple[PinnedWeightArena, list]:
    hip = FakeHip() if hip is None else hip
    a = PinnedWeightArena(0, hip=hip, chunk_bytes=CHUNK, floor_bytes=0, **kw)
    reqs = [RegionRequest(f"r{i}", CHUNK // 2 - 4096) for i in range(chunks_worth * 2)]
    return a, reqs


# =================================================================================================
# reserve: pure, and fails BEFORE anything is pinned
# =================================================================================================
class TestReserve:
    def test_reserve_pins_nothing(self, no_capacity_limit):
        hip = FakeHip()
        a, reqs = _arena(hip)
        plan = a.reserve(reqs)
        assert hip.n_host_alloc == 0          # the whole point of the reserve/attach split
        assert a.phase is ArenaPhase.RESERVED
        assert plan.n_chunks == 3

    def test_capacity_failure_raises_before_any_pinning(self, monkeypatch):
        monkeypatch.setattr(hc, "mem_available_bytes", lambda: 1 << 30)
        hip = FakeHip()
        a = PinnedWeightArena(0, hip=hip, chunk_bytes=CHUNK, floor_bytes=64 << 30)
        with pytest.raises(HostArenaCapacityError):
            a.reserve([RegionRequest("r", 1024)])
        assert hip.n_host_alloc == 0
        assert a.phase is ArenaPhase.NEW

    def test_capacity_charges_every_local_rank(self, monkeypatch):
        monkeypatch.setattr(hc, "mem_available_bytes", lambda: 3 * CHUNK)
        one = PinnedWeightArena(0, hip=FakeHip(), chunk_bytes=CHUNK, floor_bytes=0, local_ranks=1)
        one.reserve([RegionRequest("r", CHUNK - 4096)])  # 1 chunk x 1 rank fits
        two = PinnedWeightArena(0, hip=FakeHip(), chunk_bytes=CHUNK, floor_bytes=0, local_ranks=4)
        with pytest.raises(HostArenaCapacityError):
            two.reserve([RegionRequest("r", CHUNK - 4096)])  # 1 chunk x 4 ranks does not

    def test_reserve_twice_is_refused(self, no_capacity_limit):
        a, reqs = _arena()
        a.reserve(reqs)
        with pytest.raises(ArenaStateError):
            a.reserve(reqs)

    def test_attach_before_reserve_is_refused(self):
        a = PinnedWeightArena(0, hip=FakeHip(), chunk_bytes=CHUNK, floor_bytes=0)
        with pytest.raises(ArenaStateError):
            a.attach()


# =================================================================================================
# attach: pins, first-touches, self-tests
# =================================================================================================
class TestAttach:
    def test_attaches_and_self_tests(self, no_capacity_limit):
        hip = FakeHip()
        a, reqs = _arena(hip)
        a.reserve(reqs)
        a.attach()
        assert a.phase is ArenaPhase.ATTACHED
        assert hip.n_host_alloc == 3
        assert a.pinned_bytes == 3 * CHUNK
        assert a._selftest is not None and a._selftest.passed
        assert all(c.same_va for c in a.chunks)
        a.close()
        assert hip.live_allocations == 0

    def test_selftest_can_run_without_first_touch(self, no_capacity_limit):
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach(first_touch=False)
        assert a._selftest.passed
        # only the probe offsets were stamped, not the whole chunk
        assert all(c.touch_seconds == 0.0 for c in a.chunks)

    def test_aliased_chunks_are_caught_and_identified(self, no_capacity_limit):
        """The P6 shape: every call succeeds, the pages are wrong."""
        hip = AliasingHip()
        a, reqs = _arena(hip)
        a.reserve(reqs)
        with pytest.raises(ArenaSelfTestError) as exc:
            a.attach()
        msg = str(exc.value)
        assert "aliasing" in msg
        assert "looks_like" in msg  # names the chunk whose pages we actually got
        assert a.chunks == []       # rolled back
        assert hip.live_allocations == 0

    def test_pages_that_store_nothing_are_caught(self, no_capacity_limit):
        """The P1/P2/P3 shape: hipSuccess everywhere, data never lands."""
        hip = AmnesiacHip()
        a, reqs = _arena(hip)
        a.reserve(reqs)
        with pytest.raises(ArenaSelfTestError):
            a.attach()
        assert hip.live_allocations == 0

    def test_mid_attach_capacity_drop_rolls_back(self, monkeypatch):
        """The box moved under us — P3b's rank 1 stopped at 28.0 of 34.0 GiB exactly here."""
        monkeypatch.setattr(hc, "mem_available_bytes", lambda: 1 << 50)  # reserve()/attach() gate
        seq = iter([1 << 50, 0])                                        # the per-chunk loop
        monkeypatch.setattr(pa, "mem_available_bytes", lambda: next(seq, 0))
        hip = FakeHip()
        a, reqs = _arena(hip)
        a.reserve(reqs)
        with pytest.raises(HostArenaCapacityError) as exc:
            a.attach()
        assert "pinning stopped at chunk" in str(exc.value)
        assert hip.live_allocations == 0  # nothing left pinned on a box we just failed to fit in

    def test_allocation_failure_rolls_back(self, no_capacity_limit):
        class FailsOnThird(FakeHip):
            def host_alloc(self, nbytes, flags=0):
                if self.n_host_alloc >= 2:
                    raise hipmem.HipError("hipHostMalloc failed: rc=2 (out of memory)")
                return super().host_alloc(nbytes, flags)

        hip = FailsOnThird()
        a, reqs = _arena(hip)
        a.reserve(reqs)
        with pytest.raises(hipmem.HipError):
            a.attach()
        assert hip.live_allocations == 0


# =================================================================================================
# carve
# =================================================================================================
class TestCarve:
    def _attached(self, **kw):
        a, reqs = _arena(**kw)
        a.reserve(reqs)
        a.attach()
        return a, reqs

    def test_carved_offsets_match_the_plan(self, no_capacity_limit):
        a, reqs = self._attached()
        for r in reqs:
            a.allocate(r.name, r.nbytes)
        a.verify_matches_plan()
        planned = a.plan.by_name()
        for name, region in a.regions.items():
            assert (region.chunk_index, region.offset) == (
                planned[name].chunk_index,
                planned[name].offset,
            )
        a.close()

    def test_pointers_land_inside_their_chunk(self, no_capacity_limit):
        a, reqs = self._attached()
        for r in reqs:
            region = a.allocate(r.name, r.nbytes)
            chunk = a.chunks[region.chunk_index]
            assert chunk.device_ptr <= region.device_ptr
            assert region.device_ptr + region.nbytes <= chunk.device_ptr + chunk.nbytes
        a.close()

    def test_host_writes_are_visible_through_the_device_pointer(self, no_capacity_limit):
        a, _ = self._attached()
        r = a.allocate("w", 4096)
        buf = r.host_buffer()
        buf[0], buf[1], buf[2], buf[3] = 0xEF, 0xBE, 0xAD, 0xDE
        assert a._hip.read_u32(r.device_ptr) == 0xDEADBEEF
        a.close()

    def test_duplicate_carve_is_refused(self, no_capacity_limit):
        a, _ = self._attached()
        a.allocate("w", 1024)
        with pytest.raises(ArenaStateError):
            a.allocate("w", 1024)
        a.close()

    def test_out_of_order_carve_is_detected(self, no_capacity_limit):
        """Detected AT THE CARVE, not deferred to the optional `verify_matches_plan()`.

        Carving in reverse order drifts every offset. `allocate()` now compares against the plan on
        the spot, so the failure names the region that drifted instead of waiting for a call no
        production caller makes.
        """
        a, reqs = self._attached()
        with pytest.raises(ArenaStateError) as exc:
            for r in reversed(reqs):
                a.allocate(r.name, r.nbytes)
        assert reqs[-1].name in str(exc.value)
        a.close()

    def test_missing_carve_is_detected(self, no_capacity_limit):
        a, reqs = self._attached()
        for r in reqs[:-1]:
            a.allocate(r.name, r.nbytes)
        with pytest.raises(ArenaStateError) as exc:
            a.verify_matches_plan()
        assert "never carved" in str(exc.value)
        a.close()

    def test_allocate_raw_returns_none_instead_of_raising(self, no_capacity_limit):
        a, reqs = self._attached()
        for r in reqs:
            a.allocate(r.name, r.nbytes)
        # An exception inside the torch ctypes callback becomes a NULL return and a segfault, so
        # exhaustion must be a None, not a raise.
        assert a.allocate_raw(CHUNK // 2) is None
        a.close()

    def test_allocate_raw_survives_populate(self, no_capacity_limit):
        a, _ = self._attached()
        a.mark_populated()
        assert a.allocate_raw(4096) is not None  # capture warmup allocates after the weights are in
        a.close()

    def test_unknown_region_raises_keyerror(self, no_capacity_limit):
        a, _ = self._attached()
        with pytest.raises(KeyError):
            a.region("nope")
        a.close()


# =================================================================================================
# lock-down
# =================================================================================================
class TestLockDown:
    def test_selftest_is_locked_out_after_populate(self, no_capacity_limit):
        """It writes fingerprints — running it post-populate would overwrite weights (plan A1.4)."""
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        a.mark_populated()
        with pytest.raises(ArenaStateError):
            a.selftest_light()
        a.close()

    def test_freeze_blocks_further_pinning(self, no_capacity_limit):
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        a.freeze()
        assert hipmem.is_frozen()
        assert a.phase is ArenaPhase.POPULATED
        # The fake HIP bypasses the module-level gate (it is not the real binding), so assert the
        # gate that the real binding consults on every mapping entry point.
        with pytest.raises(hipmem.HipFrozenError) as exc:
            hipmem._check_not_frozen("hipHostMalloc")
        assert "R1" in str(exc.value)
        # close() after freeze is REFUSED unless forced: captured graphs hold these device pointers.
        with pytest.raises(ArenaStateError) as close_exc:
            a.close()
        assert "force=True" in str(close_exc.value)
        a.close(force=True)  # the explicit teardown reopens the window and must still work

    def test_teardown_window_restores_the_freeze(self, no_capacity_limit):
        hipmem.freeze("test")
        with hipmem.teardown_window("test"):
            hipmem._check_not_frozen("hipHostFree")  # allowed inside
        assert hipmem.is_frozen()                     # and closed again after

    def test_close_is_idempotent(self, no_capacity_limit):
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        a.close()
        a.close()
        assert a.phase is ArenaPhase.CLOSED

    def test_stats_and_summary_are_populated(self, no_capacity_limit):
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        a.allocate("w", 4096)
        s = a.stats()
        assert s["n_chunks"] == 3
        assert s["pinned_bytes"] == 3 * CHUNK
        assert s["plan_digest"] and s["plan_digest"] in a.summary()
        assert s["selftest"].startswith("selftest PASS")
        assert s["torch_hipmalloc_fallbacks"] == 0
        a.close()


# =================================================================================================
# fingerprints
# =================================================================================================
class TestFingerprints:
    def test_unique_per_rank_device_and_chunk(self):
        seen = {
            chunk_fingerprint(r, d, i)
            for r in range(4)
            for d in range(2)
            for i in range(64)
        }
        assert len(seen) == 4 * 2 * 64

    def test_round_trip_identifies_the_owner(self):
        w = chunk_fingerprint(1, 0, 7)
        assert decode_fingerprint(w) == {"rank": 1, "device": 0, "chunk": 7}

    def test_foreign_words_decode_to_none(self):
        assert decode_fingerprint(0) is None
        assert decode_fingerprint(0xDEADBEEF) is None

    def test_verify_offsets_cover_head_interior_and_tail(self):
        offs = verify_offsets(CHUNK)
        assert offs[0] == 0
        assert offs[-1] == CHUNK - 4
        assert len(offs) >= 8  # head + tail alone cannot see a wrong page in the middle
        assert all(0 <= o <= CHUNK - 4 for o in offs)


# =================================================================================================
# GPU — run serially, alone, never concurrently with another GPU job
# =================================================================================================
@pytest.mark.gpu
class TestRealHip:
    def test_small_arena_round_trip(self):
        try:
            hip = hipmem.get_hip()
        except hipmem.HipError as exc:
            pytest.skip(f"no HIP runtime: {exc}")
        a = PinnedWeightArena(0, hip=hip, chunk_bytes=2 * MIB, floor_bytes=0, label="gputest")
        try:
            a.reserve([RegionRequest("r0", 1 << 20), RegionRequest("r1", 1 << 20)])
            a.attach()
            assert a._selftest.passed, a._selftest.failures
            r = a.allocate("r0", 1 << 20)
            buf = r.host_buffer()
            buf[0], buf[1], buf[2], buf[3] = 0xEF, 0xBE, 0xAD, 0xDE
            assert hip.read_u32(r.device_ptr) == 0xDEADBEEF
        finally:
            a.close()


# =================================================================================================
# GRAPH CAPTURE / TP=2 regressions
# =================================================================================================
class TestTpRankSkew:
    """`attach()` must not re-charge a peer rank's ALREADY-PINNED bytes.

    The ranks are separate processes and never attach simultaneously (pinning 34 GiB takes ~7 s).
    Charging `local_ranks` in attach()'s re-check therefore double-counts whichever peer got there
    first, against a MemAvailable reading that has already had that peer's bytes subtracted. Rank 0
    boots, rank 1 aborts, and the first collective hangs — a TP desync manufactured by the very gate
    whose docstring says it exists to prevent one.
    """

    def test_attach_survives_a_peer_that_already_pinned(self, monkeypatch):
        hip = FakeHip()
        # 3 chunks/rank at TP=2. Pre-pin MemAvailable clears the 2-rank charge with room to spare.
        pre_pin = 9 * CHUNK
        monkeypatch.setattr(hc, "mem_available_bytes", lambda: pre_pin)
        monkeypatch.setattr(pa, "mem_available_bytes", lambda: pre_pin)
        a, reqs = _arena(hip, local_ranks=2)
        a.reserve(reqs)  # 2-rank charge: 6 chunks <= 9 -> passes on BOTH ranks

        # ...now the peer pins its 3 chunks. MemAvailable drops by exactly that much, and this rank
        # only now reaches attach(). 2-rank charge would be 6 > 6 available - floor -> false abort.
        after_peer = 6 * CHUNK
        monkeypatch.setattr(hc, "mem_available_bytes", lambda: after_peer)
        monkeypatch.setattr(pa, "mem_available_bytes", lambda: after_peer)
        a.attach()

        assert a.phase is ArenaPhase.ATTACHED
        assert hip.n_host_alloc == 3
        # reserve()'s multi-rank verdict is still the reported one; the weaker attach-time
        # single-rank verdict is kept separately rather than overwriting it.
        assert a.capacity.local_ranks == 2
        assert a.capacity_at_attach.local_ranks == 1
        a.close()

    def test_attach_still_aborts_when_this_rank_alone_cannot_fit(self, monkeypatch):
        hip = FakeHip()
        monkeypatch.setattr(hc, "mem_available_bytes", lambda: 1 << 40)
        monkeypatch.setattr(pa, "mem_available_bytes", lambda: 1 << 40)
        a, reqs = _arena(hip, local_ranks=2)
        a.reserve(reqs)
        monkeypatch.setattr(hc, "mem_available_bytes", lambda: CHUNK)  # one chunk, need three
        monkeypatch.setattr(pa, "mem_available_bytes", lambda: CHUNK)
        with pytest.raises(HostArenaCapacityError):
            a.attach()
        assert hip.n_host_alloc == 0  # aborted before a single byte was pinned


class TestCarveDigest:
    """`ChunkPlan.digest()` cannot see a rank divergence on the path that ships.

    `bake.py` reserves the host tier as anonymous headroom — `reserve([], extra_bytes=...)` — so
    `placements` is empty and the plan digest collapses to a hash of the chunk COUNT. The layout
    that matters is produced later by the order of `allocate_raw` calls, which is exactly what can
    differ between ranks.
    """

    def _headroom_arena(self, hip):
        a = PinnedWeightArena(0, hip=hip, chunk_bytes=CHUNK, floor_bytes=0)
        a.reserve([], extra_bytes=3 * CHUNK)
        a.attach()
        return a

    def test_plan_digest_is_blind_but_carve_digest_is_not(self, no_capacity_limit):
        a, b = self._headroom_arena(FakeHip()), self._headroom_arena(FakeHip())
        sizes = [64 * 1024, 128 * 1024, 32 * 1024]
        for n in sizes:
            a.allocate_raw(n)
        for n in reversed(sizes):  # the peer rank enumerated in a different order
            b.allocate_raw(n)

        # The advertised cross-rank equality proof reports the two ranks as identical...
        assert a.plan.digest() == b.plan.digest()
        # ...while their actual layouts differ, and the carve digest says so.
        assert a.carve_digest() != b.carve_digest()
        assert a.stats()["carve_digest"] == a.carve_digest()
        assert a.carve_digest() in a.summary()
        a.close()
        b.close()

    def test_carve_digest_matches_for_identical_carves(self, no_capacity_limit):
        a, b = self._headroom_arena(FakeHip()), self._headroom_arena(FakeHip())
        for n in (4096, 8192, 4096):
            a.allocate_raw(n)
            b.allocate_raw(n)
        assert a.carve_digest() == b.carve_digest()
        a.close()
        b.close()

    def test_verify_matches_plan_passes_vacuously_on_the_headroom_shape(self, no_capacity_limit):
        """Documents WHY carve_digest exists: the plan-vs-carve check has nothing to compare."""
        a = self._headroom_arena(FakeHip())
        a.allocate_raw(4096)
        a.verify_matches_plan()  # no named regions planned -> passes with zero coverage
        assert a.plan.placements == ()
        a.close()


class TestOwnsPointer:
    """The only residency test that is trustworthy on this box.

    `hipPointerGetAttributes` is wrong in BOTH directions here (Phase 0: "Host" for VRAM; P5b:
    "Device" for the real host arena), so the torch pool proves a tensor came from the arena by
    arithmetic against the pointers `hipHostGetDevicePointer` actually returned.
    """

    def test_inside_and_outside(self, no_capacity_limit):
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        r = a.allocate(reqs[0].name, reqs[0].nbytes)
        assert a.owns_pointer(r.device_ptr, r.nbytes)
        assert a.owns_pointer(r.host_ptr, r.nbytes)
        assert not a.owns_pointer(r.device_ptr - (1 << 30))
        a.close()

    def test_a_span_that_runs_off_the_end_of_its_chunk_is_not_owned(self, no_capacity_limit):
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        c = a.chunks[0]
        assert a.owns_pointer(c.device_ptr + c.nbytes - 8, 8)
        assert not a.owns_pointer(c.device_ptr + c.nbytes - 8, 9)  # straddles into chunk 1
        a.close()


class TestCaptureSafety:
    def test_close_after_freeze_is_refused_without_force(self, no_capacity_limit):
        """Captured HIP graphs hold these device pointers verbatim in their recorded kernel args."""
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        a.freeze()
        with pytest.raises(ArenaStateError) as exc:
            a.close()
        assert "force=True" in str(exc.value)
        assert a.chunks  # nothing was unmapped
        a.close(force=True)
        assert a.phase is ArenaPhase.CLOSED

    def test_overlapping_teardown_windows_do_not_lose_the_freeze(self):
        """Rule R1's latch must survive two windows interleaving.

        The old save/restore form lost it permanently: A saves True, B saves False, A restores True,
        B restores False -> nothing ever raises again. `ArenaMemPool._fallback` opens a window from
        inside torch's allocator callback on an arbitrary thread, so the overlap is reachable at
        runtime, not just in tests.
        """
        hipmem.freeze("test")
        outer = hipmem.teardown_window("outer")
        inner = hipmem.teardown_window("inner")
        outer.__enter__()
        inner.__enter__()
        assert not hipmem.is_frozen()          # open while either window is up
        outer.__exit__(None, None, None)       # exits out of order, as two threads would
        assert not hipmem.is_frozen()          # inner still holds it open
        inner.__exit__(None, None, None)
        assert hipmem.is_frozen()              # and the latch is BACK
        assert hipmem.teardown_depth() == 0


# =================================================================================================
# SILENT WRONG NUMBERS — the chunk table itself
#
# The fingerprint self-test is a SAMPLE: ten 4-byte probes per chunk, i.e. 40 B of 2 GiB. Two ways
# the driver can hand back a table that produces plausible weights and no crash are decidable
# EXACTLY from the pointers, and both are checked by `_verify_chunk_structure`.
# =================================================================================================
class MisalignedHip(FakeHip):
    """Bases are page-aligned + 64 B. Every HIP call succeeds; every offset the bump allocator
    computes is still `ALIGN`-aligned; every REGION POINTER is not.

    This is the shape that a sampled data check provably cannot see — the pages store exactly what
    was written to them, so all ten fingerprint probes come back correct. What is wrong is where the
    regions land: a `float4` / WMMA load off a packed-int4 stack at `base + offset` with
    `base % 512 == 64` reads wrong-but-well-formed bytes.
    """

    def host_alloc(self, nbytes: int, flags: int = 0):
        self.n_host_alloc += 1
        buf = (ctypes.c_ubyte * (nbytes + _FAKE_ALIGN + 64))()
        raw = ctypes.addressof(buf)
        addr = (((raw + _FAKE_ALIGN - 1) & ~(_FAKE_ALIGN - 1))) + 64
        self._buffers[addr] = buf
        return addr, addr


class PartialOverlapHip(FakeHip):
    """One big span; each chunk is a window into it, advancing by only half a chunk.

    So chunks i and i+1 share half their address space. Two regions carved into them ARE the same
    bytes — on a quantized checkpoint, one stack silently overwriting another. Every call returns
    success, and (unlike `AliasingHip`) the chunks are not identical, so the failure is a *partial*
    one of the kind sampling is not guaranteed to hit.
    """

    def __init__(self, chunk_bytes: int, n_chunks: int) -> None:
        super().__init__()
        span = chunk_bytes * (n_chunks + 1)
        self._backing, self._base = _aligned_buffer(span)
        self._buffers[self._base] = self._backing
        self._stride = chunk_bytes // 2
        self._n = 0

    def host_alloc(self, nbytes: int, flags: int = 0):
        self.n_host_alloc += 1
        addr = self._base + self._n * self._stride
        self._n += 1
        return addr, addr

    def host_free(self, host_ptr: int) -> None:
        self.n_host_free += 1
        self._buffers.pop(int(host_ptr), None)


class TestChunkTableIsStructurallySound:
    def test_a_misaligned_chunk_base_is_refused(self, no_capacity_limit):
        """`BumpAllocator` guarantees aligned OFFSETS; the kernel gets `base + offset`.

        Nothing checked the base, so the alignment contract was a statement about offset space only.
        The data self-test passes here — the pages are perfectly good pages — which is precisely why
        this needs its own exact check.
        """
        hip = MisalignedHip()
        a, reqs = _arena(hip)
        a.reserve(reqs)
        with pytest.raises(ArenaSelfTestError) as exc:
            a.attach()
        msg = str(exc.value)
        assert "not 512 B aligned" in msg
        assert "structurally impossible" in msg
        assert hip.live_allocations == 0            # rolled back, nothing left pinned
        assert a.phase is ArenaPhase.CLOSED

    def test_the_data_selftest_alone_would_have_passed_the_misaligned_arena(self, no_capacity_limit):
        """Proves the previous test is not just re-detecting what the fingerprints already catch."""
        hip = MisalignedHip()
        a, reqs = _arena(hip)
        a.reserve(reqs)
        with pytest.raises(ArenaSelfTestError):
            a.attach()
        # `attach()` runs the data check FIRST and only then the structural one, so a recorded PASS
        # here means the fingerprints were entirely happy with these pages.
        assert a._selftest is not None and a._selftest.passed

    def test_partially_overlapping_chunks_are_refused(self, no_capacity_limit):
        """A half-chunk overlap, caught by the EXACT check with the sampled one switched off.

        (With the fingerprint sweep on, this particular shape is caught there too — an overlap
        between two equal-size chunks always covers one of them at head or tail, so a probe lands in
        it. That is worth stating rather than hiding: the structural check's unique value is the
        misalignment case above, plus catching overlap deterministically and with `selftest=0`.)
        """
        hip = PartialOverlapHip(CHUNK, 3)
        a, reqs = _arena(hip)
        a.reserve(reqs)
        with pytest.raises(ArenaSelfTestError) as exc:
            a.attach(selftest=False, first_touch=False)
        msg = str(exc.value)
        assert "OVERLAP" in msg
        assert "same bytes" in msg
        assert hip.live_allocations == 0

    def test_structure_is_checked_even_with_the_data_selftest_switched_off(self, no_capacity_limit):
        """`selftest=False` is a measurement convenience (`MINISGL_WEIGHT_ARENA_SELFTEST=0`).

        It is exactly the configuration in which nothing else is watching, so the exact check must
        not be gated behind the sampled one.
        """
        hip = AliasingHip()
        a, reqs = _arena(hip)
        a.reserve(reqs)
        with pytest.raises(ArenaSelfTestError) as exc:
            a.attach(selftest=False, first_touch=False)
        assert "OVERLAP" in str(exc.value)
        assert hip.live_allocations == 0


# =================================================================================================
# SILENT WRONG NUMBERS — evidence that is not evidence
# =================================================================================================
class TestSelfTestIsNeverVacuous:
    def test_zero_chunks_probed_is_not_a_pass(self, no_capacity_limit):
        """`passed = not failures` reports PASS when nothing was probed.

        `_rollback()` leaves exactly that state (chunks emptied, plan intact), and the boot banner
        would print `selftest PASS chunks=0` — a green light asserting nothing, which is the same
        species of evidence as the `hipSuccess` return codes Phase 0 spent six probes learning not
        to trust.
        """
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        a.chunks = []                      # the post-rollback shape, reached without a rollback
        res = a.selftest_light()
        assert not res.passed
        assert res.vacuous
        assert res.chunks_checked == 0 and res.expected_chunks == 3
        assert "VACUOUS" in res.summary()
        assert "0/3" in res.summary()

    def test_a_partially_attached_arena_is_not_a_pass(self, no_capacity_limit):
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        a.chunks = a.chunks[:2]            # 2 of the 3 the plan reserved
        res = a.selftest_light()
        assert not res.passed
        assert not res.vacuous             # it DID probe pages; it just did not probe all of them
        assert any("plan reserved" in str(f.get("note", "")) for f in res.failures)

    def test_rollback_leaves_the_arena_closed_not_attached(self, no_capacity_limit):
        """A caught attach failure must not leave an object that passes every phase gate while
        owning no memory."""
        hip = AmnesiacHip()
        a, reqs = _arena(hip)
        a.reserve(reqs)
        with pytest.raises(ArenaSelfTestError):
            a.attach()
        assert a.phase is ArenaPhase.CLOSED
        with pytest.raises(ArenaStateError):
            a.selftest_light()
        with pytest.raises(ArenaStateError):
            a.allocate("w", 4096)


# =================================================================================================
# SILENT WRONG NUMBERS — a fallback to VRAM makes the capacity plan a fiction
# =================================================================================================
class TestFallbacksAreFatalAtPopulate:
    def test_mark_populated_refuses_when_a_carve_fell_back_to_hipmalloc(self, no_capacity_limit):
        """`__init__` documents `torch_fallbacks` as "MUST be 0 after populate" and nothing enforced
        it: `ArenaMemPool.assert_clean()` has no caller, and `StageARuntime.freeze()` reaches
        `mark_populated()` without touching the pool. A fallback is a `hipMalloc` — bytes budgeted
        as host-resident that actually landed in VRAM — so the capacity verdict, the device-tier
        fraction, and the KV pool sized inside the `old_free - new_free` window are all computed
        from a number that is not true."""
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        a.torch_fallbacks = 1              # what ArenaMemPool._fallback does
        a.headroom_denied_bytes = 3 << 20
        a.headroom_denied_max = 3 << 20
        with pytest.raises(ArenaStateError) as exc:
            a.mark_populated()
        msg = str(exc.value)
        assert "hipMalloc" in msg and "VRAM" in msg
        assert "refused" in msg            # names the shortfall that caused it
        assert a.phase is ArenaPhase.ATTACHED   # refused, not half-transitioned
        a.mark_populated(allow_fallbacks=True)  # the deliberate measurement escape hatch
        assert a.phase is ArenaPhase.POPULATED
        a.close()

    def test_a_clean_arena_still_populates(self, no_capacity_limit):
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        a.mark_populated()
        assert a.phase is ArenaPhase.POPULATED
        a.close()


# =================================================================================================
# SILENT WRONG NUMBERS — contracts the C ABI relies on
# =================================================================================================
class TestAllocateRawContract:
    def test_a_region_larger_than_a_chunk_returns_none_and_does_not_raise(self, no_capacity_limit):
        """`allocate_raw` is called from a ctypes callback and its docstring promises "it never
        raises". `BumpAllocator.try_allocate` RAISES `RegionTooLargeError` for
        `nbytes > chunk_bytes` — a fused w13 stack is exactly that shape — so the contract was held
        only because one caller happened to catch `BaseException`."""
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        assert a.allocate_raw(CHUNK * 4) is None
        assert a.headroom_denied_max == CHUNK * 4   # counted, not silently dropped
        a.close()

    def test_verify_matches_plan_does_not_exempt_a_granule_merely_named_torch(
        self, no_capacity_limit
    ):
        """The exemption for MemPool carves was a `name.startswith("torch:")` test. An exemption a
        string can spoof is not an exemption, and the thing being skipped is the check that stops
        one granule's bytes landing on another's."""
        a, reqs = _arena()
        a.reserve(reqs)
        a.attach()
        for r in reqs:
            a.allocate(r.name, r.nbytes)
        a.allocate("torch:impostor", 4096)          # a NAMED carve, never in the plan
        with pytest.raises(ArenaStateError) as exc:
            a.verify_matches_plan()
        assert "torch:impostor" in str(exc.value)
        assert "carved but not in the plan" in str(exc.value)
        a.close()

    def test_a_real_raw_carve_is_still_exempt(self, no_capacity_limit):
        a, reqs = _arena()
        a.reserve(reqs[:2])                         # only 2 planned; the rest is headroom
        a.attach()
        for r in reqs[:2]:
            a.allocate(r.name, r.nbytes)
        assert a.allocate_raw(4096) is not None
        a.verify_matches_plan()                     # must not complain about the raw carve
        a.close()


# =================================================================================================
# SILENT WRONG NUMBERS — a per-request alignment must survive into the carve
# =================================================================================================
class TestPlannedAlignmentSurvivesTheCarve:
    def test_carve_reproduces_a_4096_aligned_plan_without_being_told(self, no_capacity_limit):
        """`RegionRequest.align` was an input the plan consumed and forgot. `allocate(name, nbytes)`
        defaults `align` to the arena-wide 512, so a 4096-aligned plan was carved at 512-aligned
        offsets: every region after the first shifts, no pointer leaves its chunk, nothing faults,
        and each granule reads its neighbour's bytes."""
        hip = FakeHip()
        a = PinnedWeightArena(0, hip=hip, chunk_bytes=CHUNK, floor_bytes=0)
        reqs = [
            RegionRequest("w13", 1000, align=4096),
            RegionRequest("w13_scale", 1000, align=4096),
            RegionRequest("w13_zero", 1000, align=4096),
        ]
        plan = a.reserve(reqs)
        assert [p.offset for p in plan.placements] == [0, 4096, 8192]
        assert all(p.align == 4096 for p in plan.placements)
        a.attach()
        for r in reqs:
            region = a.allocate(r.name, r.nbytes)   # NOTE: align deliberately not passed
            assert region.offset == plan.by_name()[r.name].offset
            assert region.device_ptr % 4096 == 0
        a.verify_matches_plan()
        a.close()

    def test_an_explicitly_wrong_align_is_refused_at_the_carve(self, no_capacity_limit):
        a = PinnedWeightArena(0, hip=FakeHip(), chunk_bytes=CHUNK, floor_bytes=0)
        reqs = [RegionRequest("a", 1000, align=4096), RegionRequest("b", 1000, align=4096)]
        a.reserve(reqs)
        a.attach()
        a.allocate("a", 1000)
        with pytest.raises(ArenaStateError) as exc:
            a.allocate("b", 1000, align=512)        # 512 -> offset 1024, plan says 4096
        assert "drifted from the plan" in str(exc.value)
        a.close()
