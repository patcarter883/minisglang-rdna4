"""Correctness gates for the per-expert VRAM residency cache.

THE TEST THAT MATTERS IS T2, THE INVARIANT. A cache that reports a great hit rate while pointing an
expert at another expert's bytes produces plausible wrong numbers with no error anywhere — the
failure mode this whole feature could plausibly ship with. So T2 does not check bookkeeping; it
checks the actual SLAB CONTENTS against the actual HOST CONTENTS for every expert the table claims
is resident, after a stream of promotions and evictions has churned every slot several times over.

CPU-only. The fence discipline is a no-op without CUDA streams, but the table and slab mutations are
identical, so every invariant below is exercised.

    python3 -m pytest tests/expert_cache_test.py -q -o addopts=""
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from minisgl.weights.expert_cache import ExpertCacheError, ExpertResidencyCache, _SLRU  # noqa: E402

E = 32          # experts per layer
LAYERS = 4
ROW = 8         # bytes per expert row (tiny; the invariant does not care about size)
CPU = torch.device("cpu")


def _build(slots: int):
    """A cache whose host tensors are UNIQUELY VALUED PER EXPERT, so a mis-pointed slot is provable.

    Every expert's row is filled with its own global key. If the cache ever points expert e at
    another expert's bytes, the row simply will not equal e's key — no tolerance, no ambiguity.
    """
    cache = ExpertResidencyCache(num_experts=E, expert_bytes=ROW * 4,
                                 budget_bytes=slots * ROW * 4, device=CPU)
    hosts = {}
    for lid in range(LAYERS):
        # DISTINCT VALUES PER PLANE as well as per expert: if the two planes were ever crossed —
        # gate_up read from the down slab or vice versa — the +500 offset makes it provable.
        w13 = torch.arange(E, dtype=torch.float32).view(E, 1).repeat(1, ROW) + lid * 1000
        s13 = torch.arange(E, dtype=torch.float32).view(E, 1).repeat(1, 2) + lid * 1000
        w2 = w13 + 500
        s2 = s13 + 500
        cache.register_layer(lid, (w13, s13, None), (w2, s2, None))
        hosts[lid] = {"gate_up": (w13, s13), "down": (w2, s2)}
    return cache, hosts


def _check_invariant(cache, hosts):
    """For EVERY expert the table claims resident, the slab row must equal the host row."""
    bad = []
    for lid in range(LAYERS):
        tbl = cache._layers[lid]["slot_of"]
        for e in range(E):
            slot = int(tbl[e])
            if slot < 0:
                continue
            for plane in ("gate_up", "down"):
                hw, hs = hosts[lid][plane]
                if not torch.equal(cache._slabs[f"{plane}_w"][slot], hw[e]):
                    bad.append((lid, e, slot, f"{plane}.w"))
                if not torch.equal(cache._slabs[f"{plane}_s"][slot], hs[e]):
                    bad.append((lid, e, slot, f"{plane}.s"))
    return bad


def test_t0_budget_below_one_expert_is_a_refusal():
    """A budget that holds no whole expert must RAISE, not silently produce a zero-slot cache that
    reports 0% hit rate and looks like 'the cache does not help'."""
    with pytest.raises(ExpertCacheError):
        ExpertResidencyCache(num_experts=E, expert_bytes=4096, budget_bytes=100, device=CPU)


def test_t1_unpopulated_cache_is_todays_behaviour():
    """Every table starts at -1: an unpopulated cache reads everything from the host base, which is
    exactly the shipped path. A cache that starts in any other state is a wrong answer at step 0."""
    cache, _ = _build(8)
    for lid in range(LAYERS):
        assert int(cache._layers[lid]["slot_of"].min()) == -1
        assert int(cache._layers[lid]["slot_of"].max()) == -1


def test_t2_invariant_holds_through_churn():
    """THE GATE. Drive far more distinct experts than slots so eviction runs constantly, then prove
    every 'resident' claim against the real bytes."""
    cache, hosts = _build(slots=16)          # 16 slots vs 128 (layer, expert) pairs
    rng = random.Random(20260908)
    for _ in range(4000):
        lid = rng.randrange(LAYERS)
        ids = rng.sample(range(E), 10)
        cache.observe(lid, ids)
    assert cache.stats["evictions"] > 100, "test did not actually churn the slots"
    bad = _check_invariant(cache, hosts)
    assert not bad, f"resident experts pointing at the WRONG bytes: {bad[:8]}"


def test_t3_no_slot_is_double_claimed():
    """Two experts mapped to one slot means one of them is reading the other's weights."""
    cache, _ = _build(slots=12)
    rng = random.Random(7)
    for _ in range(2000):
        cache.observe(rng.randrange(LAYERS), rng.sample(range(E), 10))
    seen = {}
    for lid in range(LAYERS):
        tbl = cache._layers[lid]["slot_of"]
        for e in range(E):
            slot = int(tbl[e])
            if slot < 0:
                continue
            assert slot not in seen, f"slot {slot} claimed by {seen[slot]} AND {(lid, e)}"
            seen[slot] = (lid, e)
    assert len(seen) <= cache.slots


def test_t4_residency_never_exceeds_capacity():
    cache, _ = _build(slots=10)
    rng = random.Random(11)
    for _ in range(1500):
        cache.observe(rng.randrange(LAYERS), rng.sample(range(E), 10))
        resident = sum(int((cache._layers[l]["slot_of"] >= 0).sum()) for l in range(LAYERS))
        assert resident <= cache.slots, f"{resident} resident against {cache.slots} slots"


def test_t5_locality_is_exploited_and_uniform_is_not():
    """The NULL test, mirroring the oracle's: on a HOT working set the cache must hit; on uniform
    routing over a set far larger than the cache it must NOT manufacture hits. A cache that 'wins'
    on uniform references is measuring its own bookkeeping."""
    hot_cache, _ = _build(slots=16)
    rng = random.Random(3)
    hot = list(range(12))                     # working set smaller than the cache
    for _ in range(2000):
        hot_cache.observe(0, rng.sample(hot, 10))
    tot = hot_cache.stats["hits"] + hot_cache.stats["misses"]
    h_hot = hot_cache.stats["hits"] / tot
    assert h_hot > 0.85, f"cache failed to exploit a working set that FITS: h={h_hot:.3f}"

    uni_cache, _ = _build(slots=16)
    for _ in range(2000):
        lid = rng.randrange(LAYERS)
        uni_cache.observe(lid, rng.sample(range(E), 10))
    tot = uni_cache.stats["hits"] + uni_cache.stats["misses"]
    h_uni = uni_cache.stats["hits"] / tot
    coverage = 16 / (LAYERS * E)
    assert h_uni < coverage + 0.10, (
        f"uniform routing produced h={h_uni:.3f} against coverage {coverage:.3f} — the cache is "
        f"reporting hits it cannot physically have")


def test_t6_slru_probation_absorbs_a_one_touch_sweep():
    """The reason SLRU beats LRU here: a prefill-style sweep must not evict the decode working set.
    Establish a protected set, sweep once through many cold experts, and the protected set survives.
    """
    pol = _SLRU(cap=20, protected_frac=0.8)
    for k in range(10):                       # touch twice -> protected
        pol.admit(k); pol.touch(k)
    for k in range(100, 160):                 # a one-touch cold sweep
        pol.admit(k)
    survived = sum(1 for k in range(10) if k in pol)
    assert survived == 10, f"the one-touch sweep evicted {10 - survived} protected entries"


def test_t7_take_victim_frees_capacity_without_admitting():
    """`admit()` couples "make room" to "insert this key". The manager needs them SEPARATE: slots
    come back through a scheduler round trip, so capacity must be freed ahead of demand.

    This is the primitive whose absence froze replacement — with only `admit`, a miss on a full
    cache evicted a victim and then dropped its own key, so 240,000 references produced 8 evictions.
    """
    pol = _SLRU(cap=4, protected_frac=0.5)
    for k in range(4):
        pol.admit(k)
    assert len(pol) == 4
    v = pol.take_victim()
    assert v is not None and v not in pol, "take_victim did not remove the victim"
    assert len(pol) == 3, "take_victim inserted something"
    for _ in range(3):
        pol.take_victim()
    assert len(pol) == 0 and pol.take_victim() is None, "take_victim must be safe when empty"


def test_t8_replacement_keeps_running_on_a_full_cache():
    """THE REGRESSION. Drive far more distinct experts than slots and assert the cache keeps
    PLACING, not just evicting. The stall this guards produced promotions ~= slots (fill once, then
    frozen) while references kept arriving."""
    cache, hosts = _build(slots=16)
    rng = random.Random(99)
    for _ in range(400):                       # fill, then churn well past capacity
        cache.observe(rng.randrange(LAYERS), rng.sample(range(E), 10))
    assert cache.stats["promotions"] > cache.slots * 3, (
        f"replacement stalled: {cache.stats['promotions']} promotions against {cache.slots} slots — "
        f"the cache filled once and stopped placing")
    assert not _check_invariant(cache, hosts), "replacement broke the resident-bytes invariant"


def test_t9_outstanding_copies_are_bounded():
    """THE PCIe GUARD. Unbounded, the manager queues the whole cold fill at once — measured 4,418
    in-flight copies = 6.2 GB on the same link the forward streams its own experts over, which
    stalled the scheduler and jammed the cache at free=0. The ceiling must hold even when the
    scheduler never publishes."""
    cache, _ = _build(slots=64)
    cache._thread = object()            # threaded bookkeeping without a real manager
    cache.device = torch.device("cpu")  # keeps _promote on the non-CUDA path
    try:
        cache._max_inflight = 4
        cache._inflight = [("x", i, None) for i in range(4)]   # already at the ceiling
        before = cache.stats["promotions"]
        for _ in range(50):
            cache.observe(0, [1, 2, 3])
            while cache._q:
                lid, ids = cache._q.popleft()
                cache._observe_now(lid, ids)
        assert cache.stats.get("throttled", 0) > 0, "the ceiling never engaged"
        assert cache.stats["promotions"] == before, "promoted past the in-flight ceiling"
    finally:
        cache._thread = None


def test_t10_a_one_touch_sweep_costs_no_transfers():
    """SECOND-REFERENCE ADMISSION. Every promotion is a copy over a link the forward has already
    saturated, so a one-touch expert must cost ZERO bandwidth rather than displacing a resident one.

    This is the admission-side analogue of what SLRU probation does on the eviction side, and it is
    built on measurement, not intuition: raising the reclaim rate 41x (low_water 25 -> 1024) moved
    the hit rate DOWN 0.6314 -> 0.5961 and TPOT UP 50.26 -> 53.14 ms. Indiscriminate promotion
    loses.
    """
    cache, _ = _build(slots=32)
    assert cache._admit_second_ref, "the filter under test is off"
    for e in range(E):                       # a single sweep, every expert touched ONCE
        cache.observe(0, [e])
    assert cache.stats["promotions"] == 0, (
        f"a one-touch sweep spent {cache.stats['promotions']} transfers")
    assert cache.stats["admit_deferred"] == E


def test_t11_a_recurring_expert_is_admitted_on_its_second_reference():
    """The other half: the filter must not starve the working set it exists to protect."""
    cache, hosts = _build(slots=32)
    cache.observe(0, [3])
    assert cache.stats["promotions"] == 0
    cache.observe(0, [3])                    # second sighting -> earns its transfer
    assert cache.stats["promotions"] == 1
    assert int(cache._layers[0]["slot_of"][3]) >= 0
    assert not _check_invariant(cache, hosts)


def test_t12_the_candidate_window_is_bounded():
    """A candidate that never returns must age out — no unbounded ghost table on a long serve."""
    cache, _ = _build(slots=16)
    cache._candidate_cap = 64
    for k in range(500):
        cache.observe(k % LAYERS, [k % E])
    assert len(cache._candidates) <= cache._candidate_cap + 1
