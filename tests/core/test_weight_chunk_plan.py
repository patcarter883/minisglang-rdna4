"""Layout arithmetic for the pinned weight arena. NO GPU, NO torch — pure integers.

The invariant these tests exist for: **a region may never straddle a chunk boundary.** Chunks are
independent `hipHostMalloc` mappings whose device pointers are not contiguous, so a straddling
region reads the right bytes for its first half and an unrelated chunk's bytes for its second —
plausible weights, no crash, no return code.
"""

from __future__ import annotations

import pytest
from minisgl.weights.chunk_plan import (
    ALIGN,
    CHUNK_GRANULE,
    GIB,
    MIB,
    ArenaExhaustedError,
    ArenaLayoutError,
    BumpAllocator,
    RegionRequest,
    RegionTooLargeError,
    plan_regions,
    round_up,
    suggest_chunk_bytes,
)

CHUNK = 2 * MIB  # smallest legal chunk, so the tests are fast and boundary cases are reachable


def _reqs(*sizes: int) -> list[RegionRequest]:
    return [RegionRequest(f"r{i}", n) for i, n in enumerate(sizes)]


class TestBumpAllocator:
    def test_alignment_is_respected(self):
        b = BumpAllocator(CHUNK)
        b.allocate(1, name="a")
        p = b.allocate(1, name="b")
        assert p.offset % ALIGN == 0
        assert p.offset == ALIGN

    def test_never_straddles_a_chunk(self):
        b = BumpAllocator(CHUNK)
        b.allocate(CHUNK - 4096, name="big")
        p = b.allocate(8192, name="wont_fit_in_the_tail")
        assert p.chunk_index == 1 and p.offset == 0
        assert p.end <= CHUNK

    def test_abandoned_tail_is_accounted_not_hidden(self):
        b = BumpAllocator(CHUNK)
        b.allocate(CHUNK - 4096, name="big")
        b.allocate(8192, name="next")
        assert b.abandoned_bytes == 4096
        assert b.consumed_bytes == (CHUNK - 4096) + 8192
        # every reserved byte is accounted for exactly once
        assert b.consumed_bytes + b.abandoned_bytes + b.free_bytes == b.reserved_bytes

    def test_exact_fit_stays_in_one_chunk(self):
        b = BumpAllocator(CHUNK)
        p = b.allocate(CHUNK, name="exact")
        assert (p.chunk_index, p.offset) == (0, 0)
        assert b.n_chunks == 1

    def test_zero_byte_region_does_not_move_the_cursor(self):
        # A component that is legitimately empty must not shift every later offset, or two ranks
        # that disagree about whether it exists disagree about the whole layout.
        b = BumpAllocator(CHUNK)
        b.allocate(64, name="a")
        before = b.consumed_bytes
        p = b.allocate(0, name="empty")
        assert p.nbytes == 0
        assert b.consumed_bytes == before
        assert b.allocate(64, name="c").offset == ALIGN

    def test_region_larger_than_a_chunk_raises_with_the_fix(self):
        b = BumpAllocator(CHUNK)
        with pytest.raises(RegionTooLargeError) as exc:
            b.allocate(CHUNK + 1, name="huge")
        msg = str(exc.value)
        assert "straddle" in msg
        assert "MINISGL_WEIGHT_ARENA_CHUNK_MIB" in msg  # actionable, not just a complaint

    def test_bounded_allocator_raises_when_out_of_chunks(self):
        b = BumpAllocator(CHUNK, n_chunks=1)
        b.allocate(CHUNK - 1024, name="a")
        assert b.try_allocate(4096, name="b") is None
        with pytest.raises(ArenaExhaustedError) as exc:
            b.allocate(4096, name="b")
        # free_bytes alone would be misleading here; the message must quote the placeable run
        assert "largest placeable run" in str(exc.value)

    def test_growable_allocator_grows(self):
        b = BumpAllocator(CHUNK, n_chunks=None)
        for i in range(5):
            b.allocate(CHUNK // 2 + 1024, name=f"r{i}")
        assert b.n_chunks == 5  # each region is > half a chunk, so one per chunk

    def test_zero_chunk_allocator_places_nothing(self):
        b = BumpAllocator(CHUNK, n_chunks=0)
        assert b.largest_free_run == 0
        assert b.try_allocate(1, name="a") is None
        assert b.try_allocate(0, name="empty") is None

    def test_rejects_illegal_chunk_sizes(self):
        with pytest.raises(ValueError):
            BumpAllocator(CHUNK - 1)
        with pytest.raises(ValueError):
            BumpAllocator(1024)


class TestPlanRegions:
    def test_no_placement_straddles_and_all_are_inside_a_chunk(self):
        plan = plan_regions(_reqs(*(700 * 1024 for _ in range(20))), CHUNK)
        for p in plan.placements:
            assert 0 <= p.offset and p.end <= CHUNK
            assert 0 <= p.chunk_index < plan.n_chunks

    def test_placements_within_a_chunk_do_not_overlap(self):
        plan = plan_regions(_reqs(*(97 * 1024 + i for i in range(40))), CHUNK)
        per_chunk: dict[int, list] = {}
        for p in plan.placements:
            per_chunk.setdefault(p.chunk_index, []).append(p)
        for placements in per_chunk.values():
            placements.sort(key=lambda p: p.offset)
            for a, b in zip(placements, placements[1:]):
                assert a.end <= b.offset

    def test_accounting_adds_up(self):
        plan = plan_regions(_reqs(*(300 * 1024 for _ in range(30))), CHUNK)
        assert plan.reserved_bytes == plan.n_chunks * CHUNK
        assert plan.payload_bytes == 30 * 300 * 1024
        assert plan.padding_bytes >= 0
        assert plan.overhead_bytes == plan.reserved_bytes - plan.payload_bytes
        assert 0.0 < plan.fill_frac <= 1.0

    def test_headroom_is_whole_chunks_and_is_not_counted_as_waste(self):
        plan = plan_regions(_reqs(1024), CHUNK, extra_bytes=CHUNK + 1)
        assert plan.extra_chunks == 2
        assert plan.n_chunks == 3
        assert plan.headroom_bytes == 2 * CHUNK
        # the headroom chunks are untouched, not abandoned
        assert plan.abandoned_bytes == 0

    def test_digest_is_stable_and_order_sensitive(self):
        a = plan_regions(_reqs(1024, 2048, 4096), CHUNK)
        b = plan_regions(_reqs(1024, 2048, 4096), CHUNK)
        assert a.digest() == b.digest()
        reordered = [RegionRequest("r2", 4096), RegionRequest("r1", 2048), RegionRequest("r0", 1024)]
        # Order-sensitivity is the POINT: two TP ranks that enumerate regions differently must get
        # visibly different digests rather than silently different offsets.
        assert plan_regions(reordered, CHUNK).digest() != a.digest()

    def test_digest_changes_when_chunk_size_changes(self):
        a = plan_regions(_reqs(1024), CHUNK)
        b = plan_regions(_reqs(1024), 2 * CHUNK)
        assert a.digest() != b.digest()

    def test_duplicate_names_raise(self):
        with pytest.raises(ArenaLayoutError) as exc:
            plan_regions([RegionRequest("dup", 16), RegionRequest("dup", 32)], CHUNK)
        assert "dup" in str(exc.value)

    def test_by_name_round_trip(self):
        plan = plan_regions(_reqs(1024, 2048), CHUNK)
        assert set(plan.by_name()) == {"r0", "r1"}
        assert plan.by_name()["r1"].nbytes == 2048

    def test_empty_plan_reserves_nothing(self):
        # The "everything fits in VRAM" no-op path must cost zero bytes, not one whole chunk.
        empty = plan_regions([], CHUNK)
        assert empty.n_chunks == 0
        assert empty.reserved_bytes == 0
        assert empty.abandoned_bytes == 0

    def test_describe_mentions_the_digest(self):
        plan = plan_regions(_reqs(1024), CHUNK)
        assert plan.digest() in plan.describe()


class TestSuggestChunkBytes:
    def test_returns_the_preferred_size_when_it_is_big_enough(self):
        assert suggest_chunk_bytes(_reqs(1024), preferred=CHUNK) == CHUNK

    def test_grows_to_fit_the_largest_region_and_stays_granular(self):
        got = suggest_chunk_bytes(_reqs(CHUNK + 1), preferred=CHUNK)
        assert got >= CHUNK + 1
        assert got % CHUNK_GRANULE == 0

    def test_refuses_to_silently_exceed_the_measured_envelope(self):
        with pytest.raises(RegionTooLargeError) as exc:
            suggest_chunk_bytes(_reqs(5 * GIB), preferred=CHUNK)
        assert "2 GiB" in str(exc.value)  # cites what was actually measured on this box

    def test_empty_request_list_is_legal(self):
        assert suggest_chunk_bytes([], preferred=CHUNK) == CHUNK


def test_round_up():
    assert round_up(0, 512) == 0
    assert round_up(1, 512) == 512
    assert round_up(512, 512) == 512
    with pytest.raises(ValueError):
        round_up(1, 0)


def test_region_request_validation():
    with pytest.raises(ValueError):
        RegionRequest("bad", -1)
    with pytest.raises(ValueError):
        RegionRequest("bad", 16, align=3)  # not a power of two


class TestAllZeroBytePlan:
    """A plan whose regions are ALL zero-byte must still reserve the chunk they point into.

    Reachable at TP=2 without any degenerate checkpoint: the zero-byte region exists precisely so
    two ranks that disagree about whether a component is present still agree on every subsequent
    offset, so a rank whose shard is empty (an expert-parallel split that gives it no experts)
    produces exactly this plan while its peer produces a populated one. Leaving every cursor at 0
    used to report "no chunks used" while every Placement still named chunk 0 — the arena then
    pinned nothing and the first allocate() for a region the plan said was placed raised
    ArenaExhaustedError.
    """

    def test_reserves_the_chunk_the_placements_point_into(self):
        plan = plan_regions([RegionRequest("a", 0), RegionRequest("b", 0)], CHUNK)
        assert plan.n_chunks == 1
        assert all(p.chunk_index < plan.n_chunks for p in plan.placements)
        assert plan.payload_bytes == 0

    def test_the_bounded_allocator_can_actually_place_them(self):
        plan = plan_regions([RegionRequest("a", 0)], CHUNK)
        b = BumpAllocator(CHUNK, n_chunks=plan.n_chunks)
        assert b.allocate(0, name="a").chunk_index == 0  # no ArenaExhaustedError

    def test_a_genuinely_empty_plan_still_reserves_nothing(self):
        assert plan_regions([], CHUNK).n_chunks == 0
