"""The capacity gate — the LOUD, EARLY failure. NO GPU, NO torch.

Phase 0's binding constraint is capacity, not bandwidth: on an idle box with no engine loaded, P3b
pinned 34.0 GiB for one rank but only 62.0 GiB across two, with 114,813 pages swapped. These tests
pin down the three rules that make the failure survivable:

1. charge EVERY local rank (a per-rank check passes twice and the box still dies);
2. keep a floor (pinned pages are unevictable);
3. pass or raise — NEVER auto-shrink, because `MemAvailable` is timing-dependent and a rank that
   quietly reserves less desyncs the TP group.
"""

from __future__ import annotations

import pytest
from minisgl.weights import host_capacity as hc
from minisgl.weights.chunk_plan import GIB
from minisgl.weights.host_capacity import (
    HostArenaCapacityError,
    HostArenaSwapThrashError,
    SwapTripwire,
    check_capacity,
    evaluate_capacity,
    parse_kv_kb,
)

MEMINFO_SAMPLE = """MemTotal:       96329536 kB
MemFree:         2097152 kB
MemAvailable:   73400320 kB
Buffers:            1024 kB
Cached:         20971520 kB
SwapTotal:      10485760 kB
SwapFree:       10485760 kB
Shmem:            524288 kB
"""


class TestParsing:
    def test_parses_kb_into_bytes(self):
        got = parse_kv_kb(MEMINFO_SAMPLE, ("MemAvailable", "SwapTotal"))
        assert got["MemAvailable"] == 73400320 * 1024
        assert got["SwapTotal"] == 10485760 * 1024

    def test_missing_keys_are_none_not_zero(self):
        # None means "unknown"; 0 would mean "measured as empty", and the two must not be conflated
        # by a gate that fails closed on 0.
        assert parse_kv_kb("MemTotal: 4 kB\n", ("MemAvailable",))["MemAvailable"] is None

    def test_junk_lines_do_not_raise(self):
        assert parse_kv_kb("MemAvailable: banana\nnot a line\n", ("MemAvailable",))[
            "MemAvailable"
        ] is None


class TestEvaluateCapacity:
    def test_fits_with_headroom(self):
        v = evaluate_capacity(20 * GIB, 1, 80 * GIB, 12 * GIB)
        assert v.fits
        assert v.headroom_bytes == 48 * GIB
        assert v.shortfall_bytes == 0

    def test_charges_every_local_rank(self):
        # 34 GiB/rank fits once and not twice — exactly P3b's result on this box's ~92 GiB.
        assert evaluate_capacity(34 * GIB, 1, 70 * GIB, 12 * GIB).fits
        v = evaluate_capacity(34 * GIB, 2, 70 * GIB, 12 * GIB)
        assert not v.fits
        assert v.needed_total_bytes == 68 * GIB

    def test_floor_is_enforced(self):
        # Fits on raw bytes, fails on the floor: the difference between a slow box and a dead one.
        assert evaluate_capacity(70 * GIB, 1, 80 * GIB, 0).fits
        assert not evaluate_capacity(70 * GIB, 1, 80 * GIB, 12 * GIB).fits

    def test_unreadable_meminfo_fails_closed(self):
        v = evaluate_capacity(1 * GIB, 1, 0, 12 * GIB)
        assert not v.fits
        assert "blind" in v.reason

    def test_advises_when_above_anything_ever_demonstrated(self):
        v = evaluate_capacity(34 * GIB, 2, 1024 * GIB, 12 * GIB)
        assert v.fits  # MemAvailable says yes...
        assert any("demonstrated" in a for a in v.advisories)  # ...but nothing here ever has

    def test_failure_message_carries_numbers_and_quantified_fixes(self):
        msg = evaluate_capacity(34 * GIB, 2, 40 * GIB, 12 * GIB).failure_message()
        assert "68.00 GiB" in msg              # what it wanted
        assert "40.00 GiB" in msg              # what there was
        assert "f=0.25" in msg                 # the PHASE0 §3.4 device-tier table, quantified
        assert "NOT a fix: reserving less" in msg  # the TP-desync trap, called out by name
        assert "114,813" in msg                # the measured precedent

    def test_rejects_nonsense_inputs(self):
        with pytest.raises(ValueError):
            evaluate_capacity(1, 0, 1, 1)
        with pytest.raises(ValueError):
            evaluate_capacity(-1, 1, 1, 1)


class TestCheckCapacity:
    def test_raises_by_default(self, monkeypatch):
        monkeypatch.setattr(hc, "mem_available_bytes", lambda: 10 * GIB)
        with pytest.raises(HostArenaCapacityError):
            check_capacity(60 * GIB, 1, 12 * GIB)

    def test_can_report_instead_of_raising(self, monkeypatch):
        monkeypatch.setattr(hc, "mem_available_bytes", lambda: 10 * GIB)
        v = check_capacity(60 * GIB, 1, 12 * GIB, raise_on_fail=False)
        assert not v.fits

    def test_never_shrinks_the_request(self, monkeypatch):
        """The verdict reports the request unchanged — there is no code path that reduces it."""
        monkeypatch.setattr(hc, "mem_available_bytes", lambda: 10 * GIB)
        v = check_capacity(60 * GIB, 1, 12 * GIB, raise_on_fail=False)
        assert v.needed_per_rank_bytes == 60 * GIB


class TestSwapTripwire:
    def test_disabled_when_the_box_has_no_swap(self, monkeypatch):
        monkeypatch.setattr(hc, "read_meminfo", lambda: {"SwapTotal": 0})
        monkeypatch.setattr(hc, "read_pswpout_pages", lambda: 0)
        t = SwapTripwire(threshold_pages=1)
        assert not t.enabled
        t.check()  # must not raise

    def test_fires_once_the_threshold_is_crossed(self, monkeypatch):
        monkeypatch.setattr(hc, "read_meminfo", lambda: {"SwapTotal": 10 * GIB})
        pages = {"n": 1000}
        monkeypatch.setattr(hc, "read_pswpout_pages", lambda: pages["n"])
        t = SwapTripwire(threshold_pages=100)
        t.check()                       # baseline, no delta
        pages["n"] = 1050
        t.check()                       # under threshold
        pages["n"] = 1100
        with pytest.raises(HostArenaSwapThrashError) as exc:
            t.check("at chunk 7/17")
        assert "chunk 7/17" in str(exc.value)

    def test_counter_going_backwards_is_clamped(self, monkeypatch):
        monkeypatch.setattr(hc, "read_meminfo", lambda: {"SwapTotal": 10 * GIB})
        pages = {"n": 1000}
        monkeypatch.setattr(hc, "read_pswpout_pages", lambda: pages["n"])
        t = SwapTripwire(threshold_pages=10)
        pages["n"] = 0
        assert t.delta_pages() == 0
        t.check()


class TestSwapTripwireScalesWithThePin:
    """A FLAT page count is a boot failure at the sizes this feature exists for.

    64 MiB is 0.2 % of a 30 GiB pin, i.e. below the ambient swap traffic of any box that swaps at
    all — every 48-layer boot died there. But raising the constant would make the tripwire blinder
    the bigger the arena gets, and a 64 MiB pin that drives 64 MiB of eviction is genuinely
    pathological. Hence a rate, floored at the old constant.
    """

    def _wire(self, monkeypatch, pages):
        monkeypatch.setattr(hc, "read_meminfo", lambda: {"SwapTotal": 10 * GIB})
        monkeypatch.setattr(hc, "read_pswpout_pages", lambda: pages["n"])

    def test_a_small_pin_keeps_the_old_flat_floor(self, monkeypatch):
        self._wire(monkeypatch, {"n": 0})
        assert SwapTripwire(pin_bytes=64 << 20).threshold_pages == hc.DEFAULT_SWAP_TRIPWIRE_PAGES

    def test_a_large_pin_scales_the_threshold(self, monkeypatch):
        self._wire(monkeypatch, {"n": 0})
        t = SwapTripwire(pin_bytes=30 * GIB)
        # 1 % of 30 GiB at 4 KiB pages, and still under the 114,813 pages P3b measured AT the
        # ceiling — so the fault P3b hit is caught while ambient traffic is not.
        assert t.threshold_pages == int(30 * GIB / 4096 * hc.DEFAULT_SWAP_TRIPWIRE_FRACTION)
        assert t.threshold_pages < 114_813

    def test_the_scaled_threshold_really_gates(self, monkeypatch):
        pages = {"n": 0}
        self._wire(monkeypatch, pages)
        t = SwapTripwire(pin_bytes=30 * GIB)
        pages["n"] = hc.DEFAULT_SWAP_TRIPWIRE_PAGES * 2  # would have fired on the flat constant
        t.check(available=0, floor=1)
        pages["n"] = t.threshold_pages
        with pytest.raises(HostArenaSwapThrashError):
            t.check(available=0, floor=1)


class TestSwapTripwireArmsOnHeadroom:
    """`pswpout` is box-wide and carries no attribution, so scarcity is the discriminator.

    Measured 2026-09-03 on this box: pinning 10 GiB with 71 GiB free coincided with 432,173 pages of
    swap-out (16 % of the pin) that our pinning provably did not cause — a count-only tripwire
    refuses every large arena there. P3b's real thrash, by contrast, happened with rank 1 AT its
    floor, which this arming condition still catches.
    """

    def _wire(self, monkeypatch, n):
        monkeypatch.setattr(hc, "read_meminfo", lambda: {"SwapTotal": 10 * GIB})
        monkeypatch.setattr(hc, "read_pswpout_pages", lambda: n)

    def test_disarmed_when_memory_is_plentiful(self, monkeypatch):
        self._wire(monkeypatch, 0)
        t = SwapTripwire(threshold_pages=10)
        self._wire(monkeypatch, 1_000_000)
        t.check("plenty of room", available=60 * GIB, floor=12 * GIB)  # must not raise
        assert t.disarmed_checks == 1

    def test_armed_when_close_to_the_floor(self, monkeypatch):
        self._wire(monkeypatch, 0)
        t = SwapTripwire(threshold_pages=10)
        self._wire(monkeypatch, 1_000_000)
        with pytest.raises(HostArenaSwapThrashError) as exc:
            t.check("near the floor", available=20 * GIB, floor=12 * GIB)
        assert "near the floor" in str(exc.value)

    def test_no_headroom_reading_means_permanently_armed(self, monkeypatch):
        # A caller with no MemAvailable sample gets the old, unconditional behaviour rather than a
        # silently disabled gate.
        self._wire(monkeypatch, 0)
        t = SwapTripwire(threshold_pages=10)
        self._wire(monkeypatch, 1_000_000)
        with pytest.raises(HostArenaSwapThrashError):
            t.check()


class TestSwapTripwireCountsOnlyTheArmedWindow:
    """Arming must gate WHICH swap is counted, not only WHEN the comparison runs.

    THE BUG THIS PINS, measured 2026-09-04 on the real path: qwen4_exp 48 layers at TP=2 pinning
    27.10 GiB/rank aborted at chunk 33/37 with "21,660,130 pages (82.63 GiB) were swapped out while
    pinning". A 27.10 GiB pin cannot evict 82.63 GiB — the count was box-wide `pswpout` accumulated
    across the ~10 minutes the tripwire spent DISARMED with tens of GiB free, and it was billed in
    full the instant `MemAvailable` first fell inside `2x floor`. That is exactly the ambient
    co-tenant traffic `TestSwapTripwireArmsOnHeadroom` exists to ignore, arriving one chunk late.

    The tell that it was an artifact and not a real refusal: the same plan, same box, same binary
    booted and generated coherent text with a LOWER floor — which adds no capacity at all, it just
    moves the arming threshold down far enough that the pin finishes before the window opens.
    """

    def _wire(self, monkeypatch, n):
        monkeypatch.setattr(hc, "read_meminfo", lambda: {"SwapTotal": 10 * GIB})
        monkeypatch.setattr(hc, "read_pswpout_pages", lambda: n)

    def test_swap_from_the_disarmed_window_is_not_billed_on_arming(self, monkeypatch):
        self._wire(monkeypatch, 0)
        t = SwapTripwire(threshold_pages=10)
        # A long, comfortable stretch of the pin during which a co-tenant evicts 1M pages.
        self._wire(monkeypatch, 1_000_000)
        t.check("chunk 5/37", available=60 * GIB, floor=12 * GIB)
        t.check("chunk 20/37", available=40 * GIB, floor=12 * GIB)
        assert t.disarmed_checks == 2
        # Now memory gets scarce. Nothing new has been evicted since, so nothing is our fault.
        t.check("chunk 33/37", available=20 * GIB, floor=12 * GIB)  # must not raise
        assert t.armed_baseline_taken

    def test_swap_DURING_the_armed_window_still_aborts(self, monkeypatch):
        # The re-baseline must not blind the gate: real thrash after arming still fires.
        self._wire(monkeypatch, 0)
        t = SwapTripwire(threshold_pages=10)
        self._wire(monkeypatch, 1_000_000)
        t.check("chunk 5/37", available=60 * GIB, floor=12 * GIB)
        t.check("chunk 33/37", available=20 * GIB, floor=12 * GIB)  # arms, re-baselines at 1M
        self._wire(monkeypatch, 1_000_050)  # 50 more pages, threshold is 10
        with pytest.raises(HostArenaSwapThrashError) as exc:
            t.check("chunk 34/37", available=18 * GIB, floor=12 * GIB)
        assert "50 pages" in str(exc.value)

    def test_rebaseline_happens_once_so_oscillation_cannot_blind_it(self, monkeypatch):
        # MemAvailable flaps across the arming threshold as chunks are pinned and page cache is
        # reclaimed. Re-baselining on every re-arm would reset the count forever.
        self._wire(monkeypatch, 0)
        t = SwapTripwire(threshold_pages=10)
        self._wire(monkeypatch, 1_000_000)
        t.check("disarmed", available=60 * GIB, floor=12 * GIB)
        t.check("armed", available=20 * GIB, floor=12 * GIB)  # re-baselines at 1M
        t.check("disarmed again", available=60 * GIB, floor=12 * GIB)
        self._wire(monkeypatch, 1_000_099)
        with pytest.raises(HostArenaSwapThrashError):
            t.check("armed again", available=20 * GIB, floor=12 * GIB)

    def test_always_armed_caller_keeps_the_construction_baseline(self, monkeypatch):
        # No `available` ever passed => armed from the first check => no disarmed window => the
        # construction baseline stands, byte-for-byte the pre-fix behaviour.
        self._wire(monkeypatch, 0)
        t = SwapTripwire(threshold_pages=10)
        self._wire(monkeypatch, 500)
        with pytest.raises(HostArenaSwapThrashError):
            t.check("no headroom reading")
        assert not t.armed_baseline_taken


class TestRankSkewCaveat:
    """A multi-rank shortfall and a benign boot skew produce IDENTICAL arithmetic.

    `MemAvailable` already has every byte a peer rank ALREADY pinned subtracted from it. So once a
    peer starts pinning, charging `local_ranks` double-counts that peer and the verdict reads
    "does not fit" for a plan that fits perfectly well. The numbers cannot distinguish the two
    cases, so the message must, or the operator shrinks a plan that was never too big — and a
    shrink on one rank only is the TP desync this whole file exists to prevent.
    """

    def test_caveat_appears_when_only_the_multi_rank_total_fails(self):
        # 34 GiB/rank, TP=2, floor 12. Peer already pinned its 34, so MemAvailable is down to 56:
        # one rank fits (34 + 12 = 46 <= 56), two do not (68 + 12 = 80 > 56).
        v = evaluate_capacity(34 * GIB, 2, 56 * GIB, 12 * GIB)
        assert not v.fits
        assert v.single_rank_fits
        assert "RANK SKEW CAVEAT" in v.failure_message()

    def test_no_caveat_when_this_rank_alone_cannot_fit(self):
        v = evaluate_capacity(34 * GIB, 2, 20 * GIB, 12 * GIB)
        assert not v.fits
        assert not v.single_rank_fits
        assert "RANK SKEW CAVEAT" not in v.failure_message()

    def test_no_caveat_at_tp1(self):
        v = evaluate_capacity(34 * GIB, 1, 40 * GIB, 12 * GIB)
        assert not v.fits
        assert "RANK SKEW CAVEAT" not in v.failure_message()
