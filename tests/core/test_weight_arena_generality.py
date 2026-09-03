"""Regression tests for the GENERALITY defects found in the pinned weight arena.

Each class here pins down one thing that used to require a human — an operator setting an env var, a
deployment shape the gate could not see, a knob that was read and then dropped — because the standing
rule for this feature is that a NEW model, quant format, TP size or deployment shape must have to
implement *nothing*.

NO GPU, NO torch. Everything below runs against pure arithmetic or an injected fake HIP.
"""

from __future__ import annotations

import ctypes
import struct

import pytest
from minisgl.weights import chunk_plan as cp
from minisgl.weights import host_capacity as hc
from minisgl.weights import pinned_arena as pa
from minisgl.weights.chunk_plan import GIB, MIB, RegionRequest, RegionTooLargeError
from minisgl.weights.config import ArenaSettings, create_pinned_weight_arena
from minisgl.weights.pinned_arena import PinnedWeightArena, chunk_fingerprint, decode_fingerprint

CHUNK = 2 * MIB
_FAKE_ALIGN = 4096


def _aligned_buffer(nbytes: int):
    """A `_FAKE_ALIGN`-aligned span of real memory, plus the array that owns it. The arena's
    structural check refuses an unaligned chunk base, and a real `hipHostMalloc` is page-aligned."""
    buf = (ctypes.c_ubyte * (nbytes + _FAKE_ALIGN))()
    raw = ctypes.addressof(buf)
    return buf, (raw + _FAKE_ALIGN - 1) & ~(_FAKE_ALIGN - 1)


# =================================================================================================
# fake HIP — real ctypes buffers, real pointers (same shape as test_pinned_weight_arena.py's)
# =================================================================================================
class FakeHip:
    def __init__(self, start_device: int = -1) -> None:
        self._buffers: dict[int, object] = {}
        self.device = start_device
        self.device_history: list[int] = []
        self.n_host_alloc = 0
        self.n_host_free = 0

    def set_device(self, dev: int) -> int:
        self.device = int(dev)
        self.device_history.append(self.device)
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
        pass

    def host_alloc(self, nbytes: int, flags: int = 0):
        self.n_host_alloc += 1
        buf, addr = _aligned_buffer(nbytes)
        self._buffers[addr] = buf
        return addr, addr

    def host_free(self, host_ptr: int) -> None:
        self.n_host_free += 1
        self._buffers.pop(int(host_ptr), None)

    def dev_alloc(self, nbytes: int) -> int:  # pragma: no cover - unused here
        raise AssertionError("the arena must not hipMalloc")

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


@pytest.fixture(autouse=True)
def _unfreeze_hipmem():
    from minisgl.weights import hipmem

    yield
    hipmem._FROZEN = False
    hipmem._FREEZE_REASON = ""


@pytest.fixture
def no_capacity_limit(monkeypatch):
    monkeypatch.setattr(hc, "mem_available_bytes", lambda: 1 << 50)
    monkeypatch.setattr(pa, "mem_available_bytes", lambda: 1 << 50)
    return None


# =================================================================================================
# 1. the capacity gate must see the DEPLOYMENT, not just the host
# =================================================================================================
class TestCgroupAwareCapacity:
    """`/proc/meminfo` is not namespaced. This repo's Dockerfile IS the serve image, so the gate has
    to read the cgroup or a `--memory`-limited container passes on the HOST's free memory and is
    OOM-killed mid-pin with no message from this module at all."""

    def test_v2_max_means_unlimited(self):
        assert hc.parse_cgroup_available("max\n", "1234\n", "") is None

    def test_v1_sentinel_means_unlimited(self):
        # cgroup v1 spells "no limit" as a huge number, not a word. Treating it as a real limit
        # would make every v1 host look like it had ~9 EiB free, which is the same blindness.
        assert hc.parse_cgroup_available("9223372036854771712\n", "0\n", "") is None

    def test_limit_minus_current_plus_reclaimable(self):
        got = hc.parse_cgroup_available(
            f"{32 * GIB}\n", f"{20 * GIB}\n", f"anon {8 * GIB}\ninactive_file {4 * GIB}\n"
        )
        assert got == 32 * GIB - 20 * GIB + 4 * GIB

    def test_v1_stat_key_is_accepted_too(self):
        got = hc.parse_cgroup_available(
            f"{32 * GIB}\n", f"{20 * GIB}\n", f"total_inactive_file {2 * GIB}\n"
        )
        assert got == 14 * GIB

    def test_unreadable_files_are_unlimited_not_zero(self):
        # None must not read as "0 bytes available" — that would fail every boot on a host with no
        # cgroup at all, which is the opposite failure.
        assert hc.parse_cgroup_available(None, None, None) is None

    def test_garbage_is_unlimited_not_a_crash(self):
        assert hc.parse_cgroup_available("not-a-number\n", "\n", "junk\n") is None

    def test_mem_available_takes_the_smaller_of_host_and_cgroup(self, monkeypatch):
        monkeypatch.setattr(hc, "read_meminfo", lambda: {"MemAvailable": 80 * GIB})
        monkeypatch.setattr(hc, "cgroup_available_bytes", lambda: 12 * GIB)
        assert hc.mem_available_bytes() == 12 * GIB

    def test_unlimited_cgroup_changes_nothing(self, monkeypatch):
        monkeypatch.setattr(hc, "read_meminfo", lambda: {"MemAvailable": 80 * GIB})
        monkeypatch.setattr(hc, "cgroup_available_bytes", lambda: None)
        assert hc.mem_available_bytes() == 80 * GIB

    def test_unreadable_meminfo_still_fails_closed(self, monkeypatch):
        monkeypatch.setattr(hc, "read_meminfo", lambda: {})
        monkeypatch.setattr(hc, "cgroup_available_bytes", lambda: 40 * GIB)
        assert hc.mem_available_bytes() == 0

    def test_a_containerised_boot_that_would_have_passed_now_aborts(self, monkeypatch):
        """The whole point, end to end: 34 GiB/rank inside a 32 GiB container."""
        monkeypatch.setattr(hc, "read_meminfo", lambda: {"MemAvailable": 80 * GIB})
        monkeypatch.setattr(hc, "cgroup_available_bytes", lambda: 20 * GIB)
        with pytest.raises(hc.HostArenaCapacityError):
            hc.check_capacity(34 * GIB, 1, 12 * GIB)


# =================================================================================================
# 2. anonymous headroom must not assume perfect packing
# =================================================================================================
class TestHeadroomSizing:
    """`ceil(bytes / chunk)` is the intuitive reservation and it is short by one abandoned tail per
    chunk, because next-fit may never straddle. The rows that do not fit come back as `hipMalloc`
    VRAM, which makes the capacity plan a fiction."""

    def test_ceil_is_the_no_bound_behaviour(self):
        assert cp.headroom_chunks(5 * CHUNK, CHUNK) == 5
        assert cp.headroom_chunks(5 * CHUNK + 1, CHUNK) == 6
        assert cp.headroom_chunks(0, CHUNK) == 0

    def test_a_bound_reserves_for_the_abandoned_tail(self):
        # 34 GiB of <=400 MiB rows: 17 chunks is the naive answer and it does not fit.
        naive = cp.headroom_chunks(34 * GIB, 2 * GIB)
        bounded = cp.headroom_chunks(34 * GIB, 2 * GIB, 400 * MIB)
        assert naive == 17
        assert bounded > naive

    def test_a_bound_bigger_than_a_chunk_is_refused_with_the_fix(self):
        with pytest.raises(RegionTooLargeError) as exc:
            cp.headroom_chunks(1 * GIB, CHUNK, CHUNK)
        assert "straddle" in str(exc.value)

    def test_uniform_rows_really_do_exhaust_the_naive_reservation(self):
        """Not a proof by arithmetic — actually carve the rows and watch it run out.

        Rows of 3/8 of a chunk tile twice per chunk (6/8 used, 2/8 abandoned), so a reservation of
        ceil(total/chunk) chunks cannot hold them.
        """
        row = (CHUNK * 3) // 8
        n_rows = 16
        total = row * n_rows
        naive = cp.plan_regions([], CHUNK, extra_bytes=total)
        bump = cp.BumpAllocator(CHUNK, n_chunks=naive.n_chunks)
        placed = sum(1 for _ in range(n_rows) if bump.try_allocate(row) is not None)
        assert placed < n_rows                      # the defect, reproduced

        bounded = cp.plan_regions([], CHUNK, extra_bytes=total, extra_max_region_bytes=row)
        bump2 = cp.BumpAllocator(CHUNK, n_chunks=bounded.n_chunks)
        placed2 = sum(1 for _ in range(n_rows) if bump2.try_allocate(row) is not None)
        assert placed2 == n_rows                    # the fix

    def test_unbounded_headroom_carries_a_loud_advisory(self):
        plan = cp.plan_regions([], CHUNK, extra_bytes=8 * CHUNK)
        msg = plan.headroom_advisory()
        assert msg and "hipMalloc" in msg and "extra_max_region_bytes" in msg

    def test_bounded_headroom_has_no_advisory(self):
        plan = cp.plan_regions([], CHUNK, extra_bytes=8 * CHUNK, extra_max_region_bytes=CHUNK // 4)
        assert plan.headroom_advisory() is None

    def test_named_regions_alone_never_advise(self):
        plan = cp.plan_regions([RegionRequest("a", 4096)], CHUNK)
        assert plan.headroom_advisory() is None

    def test_the_plan_records_what_it_was_asked_for(self):
        plan = cp.plan_regions([], CHUNK, extra_bytes=3 * CHUNK, extra_max_region_bytes=1024)
        assert plan.extra_bytes == 3 * CHUNK
        assert plan.extra_max_region_bytes == 1024

    def test_refused_carves_are_diagnosed_not_just_counted(self, no_capacity_limit):
        """`torch_fallbacks` says a hipMalloc happened; it cannot say WHY. The arena records the
        shortfall and the largest refusal, which together name the bound that was missing."""
        a = PinnedWeightArena(0, hip=FakeHip(), chunk_bytes=CHUNK, floor_bytes=0)
        a.reserve([], extra_bytes=CHUNK)
        a.attach()
        assert a.allocate_raw(CHUNK // 2) is not None
        assert a.allocate_raw(CHUNK) is None                 # never straddles -> refused
        assert a.headroom_denied_bytes == CHUNK
        assert a.headroom_denied_max == CHUNK
        assert a.stats()["headroom_denied_bytes"] == CHUNK
        a.close()


# =================================================================================================
# 3. a new checkpoint must not need an operator
# =================================================================================================
class TestChunkSizeIsNotACheckpointProperty:
    def test_an_oversized_region_grows_the_chunk_instead_of_raising(self, no_capacity_limit):
        big = CHUNK + 4096
        a = PinnedWeightArena(0, hip=FakeHip(), chunk_bytes=CHUNK, floor_bytes=0)
        plan = a.reserve([RegionRequest("huge", big)])
        assert a.chunk_bytes >= big
        assert plan.chunk_bytes == a.chunk_bytes
        assert a.chunk_growth_advisory and "grown" in a.chunk_growth_advisory

    def test_growth_is_a_pure_function_of_the_regions_so_ranks_agree(self, no_capacity_limit):
        reqs = [RegionRequest("a", CHUNK + 4096), RegionRequest("b", 8192)]
        sizes = []
        for rank in (0, 1):
            a = PinnedWeightArena(0, hip=FakeHip(), rank=rank, chunk_bytes=CHUNK, floor_bytes=0)
            plan = a.reserve(reqs)
            sizes.append((a.chunk_bytes, plan.digest()))
        assert sizes[0] == sizes[1]

    def test_the_headroom_bound_also_forces_growth(self, no_capacity_limit):
        a = PinnedWeightArena(0, hip=FakeHip(), chunk_bytes=CHUNK, floor_bytes=0)
        a.reserve([], extra_bytes=8 * CHUNK, extra_max_region_bytes=CHUNK + 4096)
        assert a.chunk_bytes > CHUNK

    def test_growth_still_refuses_past_the_unmeasured_cap(self, no_capacity_limit):
        a = PinnedWeightArena(0, hip=FakeHip(), chunk_bytes=CHUNK, floor_bytes=0)
        with pytest.raises(RegionTooLargeError):
            a.reserve([RegionRequest("absurd", 5 * GIB)])

    def test_a_normal_plan_does_not_grow(self, no_capacity_limit):
        a = PinnedWeightArena(0, hip=FakeHip(), chunk_bytes=CHUNK, floor_bytes=0)
        a.reserve([RegionRequest("a", CHUNK // 2)])
        assert a.chunk_bytes == CHUNK
        assert a.chunk_growth_advisory is None


# =================================================================================================
# 4. resolved settings must reach the function that consumes them
# =================================================================================================
class TestSettingsPlumbing:
    def test_factory_propagates_selftest_and_first_touch(self):
        s = ArenaSettings(chunk_bytes=CHUNK, floor_bytes=0, selftest=False, first_touch=False)
        a = create_pinned_weight_arena(0, settings=s)
        assert a.selftest_default is False
        assert a.first_touch_default is False

    def test_attach_honours_the_constructor_defaults(self, no_capacity_limit):
        hip = FakeHip()
        a = PinnedWeightArena(
            0, hip=hip, chunk_bytes=CHUNK, floor_bytes=0,
            selftest_default=False, first_touch_default=False,
        )
        a.reserve([RegionRequest("a", 4096)])
        a.attach()
        assert a.selftest_result() is None if hasattr(a, "selftest_result") else True
        assert a.stats()["selftest"] is None      # the knob actually reached the work
        a.close()

    def test_an_explicit_argument_still_wins(self, no_capacity_limit):
        a = PinnedWeightArena(
            0, hip=FakeHip(), chunk_bytes=CHUNK, floor_bytes=0, selftest_default=False,
        )
        a.reserve([RegionRequest("a", 4096)])
        a.attach(selftest=True)
        assert a.stats()["selftest"] is not None
        a.close()


# =================================================================================================
# 5. the self-test's encoding must not silently outgrow a bigger TP
# =================================================================================================
class TestFingerprintFieldWidths:
    def test_round_trips_inside_the_fields(self):
        w = chunk_fingerprint(15, 15, 0xFFFF)
        assert decode_fingerprint(w) == {"rank": 15, "device": 15, "chunk": 0xFFFF}

    def test_rank_16_raises_instead_of_aliasing_rank_0(self):
        # Masked, rank 16 and rank 0 stamp the same word and the cross-process aliasing check
        # becomes unfalsifiable — a self-test that cannot fail.
        with pytest.raises(ValueError) as exc:
            chunk_fingerprint(16, 0, 0)
        assert "fingerprint" in str(exc.value)
        assert chunk_fingerprint(15, 0, 0) != chunk_fingerprint(0, 0, 0)

    def test_device_and_chunk_fields_are_checked_too(self):
        with pytest.raises(ValueError):
            chunk_fingerprint(0, 16, 0)
        with pytest.raises(ValueError):
            chunk_fingerprint(0, 0, 1 << 16)


# =================================================================================================
# 6. attach() owns the process-global device, and never leaks pinned pages
# =================================================================================================
class TestAttachSideEffects:
    def test_the_ambient_hip_device_is_restored(self, no_capacity_limit):
        hip = FakeHip(start_device=0)
        a = PinnedWeightArena(1, hip=hip, chunk_bytes=CHUNK, floor_bytes=0)
        a.reserve([RegionRequest("a", 4096)])
        a.attach()
        # torch caches the current device; leaving the driver on card 1 while torch believes it is
        # on card 0 launches kernels on the wrong card, and the two cards differ by 2x of PCIe.
        assert hip.current_device() == 0
        assert 1 in hip.device_history        # it really did bind its own card while pinning
        a.close()

    def test_a_swap_thrash_abort_does_not_leak_pinned_chunks(self, no_capacity_limit, monkeypatch):
        """`SwapTripwire.check` raising out of the pinning loop used to leave every chunk pinned —
        a failed boot holding tens of GiB of unevictable host RAM, which is the box-destroying
        outcome the rollback exists to prevent."""

        class Tripwire:
            def __init__(self, *a, **kw):
                self.n = 0

            def check(self, context: str = "") -> None:
                self.n += 1
                if self.n > 2:
                    raise hc.HostArenaSwapThrashError("synthetic thrash")

        monkeypatch.setattr(pa, "SwapTripwire", Tripwire)
        hip = FakeHip(start_device=0)
        a = PinnedWeightArena(0, hip=hip, chunk_bytes=CHUNK, floor_bytes=0)
        a.reserve([RegionRequest(f"r{i}", CHUNK - 4096) for i in range(6)])
        with pytest.raises(hc.HostArenaSwapThrashError):
            a.attach()
        assert hip.live_allocations == 0
        assert a.chunks == []
