"""Unit tests for the ghost prefix oracle (kvcache/ghost_cache.py).

Pure CPU, no torch device, no model. The oracle is measure-only, but a measurement that is
silently wrong is worse than no measurement — it would send the host-tier work down the wrong
branch — so its arithmetic is pinned here.
"""

from __future__ import annotations

import importlib.util
import pathlib

# Loaded by PATH, not by package import, on purpose: `minisgl.kvcache.__init__` pulls in torch,
# which needs the ROCm container, whereas ghost_cache.py is deliberately dependency-free. Loading
# the file directly keeps this test runnable on the bare host — which is the whole value of a
# pure-Python probe. If this import ever breaks, ghost_cache.py has grown a dependency it should
# not have.
_SRC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "python"
    / "minisgl"
    / "kvcache"
    / "ghost_cache.py"
)
_spec = importlib.util.spec_from_file_location("_ghost_cache", _SRC)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
GhostPrefixOracle = _mod.GhostPrefixOracle

PS = 16


def _prompt(seed: int, pages: int) -> list[int]:
    return [seed * 100_000 + i for i in range(pages * PS)]


def test_first_sighting_has_no_potential():
    o = GhostPrefixOracle(page_size=PS)
    o.observe(_prompt(1, 4), actual_len=0, kv_len=0)
    assert o.prompt_tokens == 4 * PS
    assert o.potential_tokens == 0
    assert o.gap_evict_tokens == 0


def test_repeat_prompt_is_fully_recoverable():
    o = GhostPrefixOracle(page_size=PS)
    p = _prompt(1, 4)
    o.observe(p, actual_len=0, kv_len=0)
    o.observe(p, actual_len=0, kv_len=0)  # cache missed entirely the second time
    # The whole 4-page prefix was seen before, so an ideal tier could have served it.
    assert o.potential_tokens == 4 * PS
    assert o.gap_evict_tokens == 4 * PS


def test_shared_prefix_then_divergence():
    o = GhostPrefixOracle(page_size=PS)
    base = _prompt(1, 6)
    o.observe(base, actual_len=0, kv_len=0)
    diverged = base[: 3 * PS] + _prompt(2, 3)
    o.observe(diverged, actual_len=0, kv_len=0)
    # Only the shared 3-page head is recoverable, not the divergent tail.
    assert o.potential_tokens == 3 * PS


def test_gap_decomposition_separates_rec_cap_from_eviction():
    """The load-bearing case: the whole point is telling these two apart."""
    o = GhostPrefixOracle(page_size=PS)
    p = _prompt(1, 8)
    o.observe(p, actual_len=0, kv_len=0)
    # Second sighting: KV resident for 6 pages, but the snapshot cap trimmed the match to 2.
    o.observe(p, actual_len=2 * PS, kv_len=6 * PS)
    assert o.gap_rec_tokens == 4 * PS  # KV was there, snapshot was not -> raise the cap
    assert o.gap_evict_tokens == 2 * PS  # never resident at all -> needs a host KV tier


def test_potential_is_never_below_what_actually_happened():
    """`potential` is an upper bound; a bookkeeping slip that puts it under `actual` would show
    up as a negative gap, so clamp and assert it."""
    o = GhostPrefixOracle(page_size=PS)
    # Never-before-seen prompt that nonetheless reports a real match (chunked continuation).
    o.observe(_prompt(7, 5), actual_len=5 * PS, kv_len=5 * PS)
    assert o.potential_tokens >= o.actual_tokens
    assert o.gap_rec_tokens == 0
    assert o.gap_evict_tokens == 0


def test_lru_capacity_is_enforced():
    o = GhostPrefixOracle(page_size=PS, capacity=8)
    for s in range(20):
        o.observe(_prompt(s, 2), actual_len=0, kv_len=0)
    assert len(o._seen) <= 8


def test_deepest_hit_wins_when_an_interior_page_was_dropped():
    """Hashes are chained, so a deeper surviving entry still implies the full shared prefix; the
    oracle must take the deepest hit, not the first miss."""
    o = GhostPrefixOracle(page_size=PS, capacity=64)
    p = _prompt(3, 5)
    o.observe(p, actual_len=0, kv_len=0)
    # Drop one interior page-prefix hash by hand, simulating LRU pressure.
    keys = list(o._seen.keys())
    del o._seen[keys[2]]
    o.observe(p, actual_len=0, kv_len=0)
    assert o.potential_tokens == 5 * PS


def test_persistence_round_trip_preserves_counters_and_lru(tmp_path):
    """The LRU must survive, not just the counters — see the comment in ghost_cache.save."""
    o = GhostPrefixOracle(page_size=PS)
    p = _prompt(11, 5)
    o.observe(p, actual_len=0, kv_len=0)
    o.observe(p, actual_len=1 * PS, kv_len=3 * PS)
    path = tmp_path / "g.bin"
    o.save(path)

    o2 = GhostPrefixOracle(page_size=PS)
    assert o2.load(path) is True
    assert (o2.requests, o2.prompt_tokens) == (o.requests, o.prompt_tokens)
    assert (o2.actual_tokens, o2.kv_tokens, o2.potential_tokens) == (
        o.actual_tokens,
        o.kv_tokens,
        o.potential_tokens,
    )
    # The load-bearing part: a prompt seen BEFORE the restart is still recognised after it.
    before = o2.potential_tokens
    o2.observe(p, actual_len=0, kv_len=0)
    assert o2.potential_tokens - before == 5 * PS


def test_restart_without_persistence_would_understate_reuse():
    """Pins the bias that persistence exists to remove: a cold oracle scores a repeat as novel."""
    p = _prompt(12, 5)
    cold = GhostPrefixOracle(page_size=PS)
    cold.observe(p, actual_len=0, kv_len=0)
    assert cold.potential_tokens == 0  # would read as "no reuse available" -> wrong call


def test_load_rejects_a_different_page_size(tmp_path):
    o = GhostPrefixOracle(page_size=PS)
    o.observe(_prompt(13, 3), actual_len=0, kv_len=0)
    path = tmp_path / "g.bin"
    o.save(path)
    other = GhostPrefixOracle(page_size=PS * 2)
    assert other.load(path) is False
    assert other.requests == 0  # fresh, not a silent mix of two page geometries


def test_load_of_missing_or_truncated_file_is_a_fresh_start(tmp_path):
    o = GhostPrefixOracle(page_size=PS)
    assert o.load(tmp_path / "nope.bin") is False
    o.observe(_prompt(14, 4), actual_len=0, kv_len=0)
    path = tmp_path / "g.bin"
    o.save(path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 9])  # torn tail, e.g. SIGKILL mid-write
    o2 = GhostPrefixOracle(page_size=PS)
    assert o2.load(path) is False
    assert o2.requests == 0


def test_load_keeps_the_newest_entries_when_capacity_shrank(tmp_path):
    big = GhostPrefixOracle(page_size=PS, capacity=64)
    for s in range(6):
        big.observe(_prompt(20 + s, 2), actual_len=0, kv_len=0)
    path = tmp_path / "g.bin"
    big.save(path)
    small = GhostPrefixOracle(page_size=PS, capacity=4)
    assert small.load(path) is True
    assert len(small._seen) == 4
    # The most recent prompt must still be recognised; the oldest need not be.
    before = small.potential_tokens
    small.observe(_prompt(25, 2), actual_len=0, kv_len=0)
    assert small.potential_tokens - before == 2 * PS


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    o = GhostPrefixOracle(page_size=PS)
    o.observe(_prompt(30, 3), actual_len=0, kv_len=0)
    path = tmp_path / "sub" / "g.bin"
    o.save(path)
    o.save(path)  # overwrite an existing file
    assert path.exists()
    assert [f.name for f in path.parent.iterdir()] == ["g.bin"]


def test_sub_page_prompt_is_counted_but_never_recoverable():
    o = GhostPrefixOracle(page_size=PS)
    short = [1, 2, 3]
    o.observe(short, actual_len=0, kv_len=0)
    o.observe(short, actual_len=0, kv_len=0)
    assert o.prompt_tokens == 6
    assert o.potential_tokens == 0  # below one page: the radix could never have matched it
