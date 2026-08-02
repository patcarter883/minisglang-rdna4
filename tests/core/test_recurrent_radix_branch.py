"""Recurrent radix: a request that DIVERGES inside a shared prefix must still reuse it.

Regression test for the behaviour that made the recurrent radix an exact-replay-only cache.
`attach_rec_state` only ever marks the node at a committed sequence's own END boundary. When a
second request shares a long prefix but diverges before that end, `_tree_walk` splits the node and
`split_at` leaves the new shallow parent with `rec_state=None`; `match_prefix` then walks up, finds
no snapshot, hits root, and returns cached_len 0 -> the whole shared prefix is re-prefilled from
zero recurrent state even though its KV is resident.

Measured on a live 35B GDN/MHA hybrid before the fix: identical prompt replayed = 100% hit
(TTFT 4.71s -> 0.05s), same 21,670-token prefix with a different tail = 0% hit, TTFT unchanged.

These tests are CPU-only: they exercise tree/bookkeeping logic, not kernels.
"""

from __future__ import annotations

import pytest
import torch

import minisgl.core as core
from minisgl.kvcache.radix_cache import RadixPrefixCache

PAGE = 16


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    core.set_global_ctx(core.Context(page_size=PAGE))
    yield
    core._GLOBAL_CTX = old_ctx


def _cache() -> RadixPrefixCache:
    return RadixPrefixCache(device=torch.device("cpu"), recurrent=True)


def _ids(vals) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.int32)


def _seq(n: int, salt: int = 0) -> torch.Tensor:
    return _ids([salt * 1_000_000 + i for i in range(n)])


def _indices(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.int32)


def test_exact_replay_hits_without_the_fix_too():
    """Baseline: the end boundary is a resume point, so an identical prompt fully hits."""
    c = _cache()
    ids = _seq(8 * PAGE)
    res = c.insert_prefix(ids, _indices(len(ids)))
    c.attach_rec_state(res.handle, rec_state="state@128")

    m = c.match_prefix(ids).cuda_handle
    assert m.cached_len == 8 * PAGE
    assert m.rec_state == "state@128"


def test_divergent_tail_without_interior_snapshot_gets_nothing():
    """The pre-fix behaviour, pinned so a regression is loud: diverge before the committed end and
    the entire shared prefix is discarded."""
    c = _cache()
    base = _seq(8 * PAGE)
    res = c.insert_prefix(base, _indices(len(base)))
    c.attach_rec_state(res.handle, rec_state="state@128")

    # Shares the first 6 pages, then diverges.
    other = torch.cat([base[: 6 * PAGE], _seq(2 * PAGE, salt=9)])
    m = c.match_prefix(other).cuda_handle
    assert m.cached_len == 0, "expected the pre-fix full-recompute fallback"
    assert m.rec_state is None


def test_interior_resume_point_makes_a_divergent_tail_reuse_the_prefix():
    """The fix: marking an interior chunk boundary lets a divergent request resume there."""
    c = _cache()
    base = _seq(8 * PAGE)
    res = c.insert_prefix(base, _indices(len(base)))
    ok = c.attach_rec_state_at(res.handle, 4 * PAGE, rec_state="state@64")
    assert ok
    c.attach_rec_state(res.handle, rec_state="state@128")

    other = torch.cat([base[: 6 * PAGE], _seq(2 * PAGE, salt=9)])
    m = c.match_prefix(other).cuda_handle
    # Resumes at the interior boundary instead of recomputing from zero.
    assert m.cached_len == 4 * PAGE
    assert m.rec_state == "state@64"


def test_deepest_interior_point_below_the_divergence_wins():
    """With several resume points, the match caps at the deepest one at or below the divergence."""
    c = _cache()
    base = _seq(8 * PAGE)
    res = c.insert_prefix(base, _indices(len(base)))
    for b in (2 * PAGE, 4 * PAGE, 6 * PAGE):
        assert c.attach_rec_state_at(res.handle, b, rec_state=f"state@{b}")

    other = torch.cat([base[: 7 * PAGE], _seq(PAGE, salt=9)])
    m = c.match_prefix(other).cuda_handle
    assert m.cached_len == 6 * PAGE
    assert m.rec_state == "state@96"


def test_exact_replay_still_hits_fully_after_interior_marking():
    """Interior splits must not degrade the exact-replay path."""
    c = _cache()
    base = _seq(8 * PAGE)
    res = c.insert_prefix(base, _indices(len(base)))
    c.attach_rec_state_at(res.handle, 4 * PAGE, rec_state="state@64")
    c.attach_rec_state(res.handle, rec_state="state@128")

    m = c.match_prefix(base).cuda_handle
    assert m.cached_len == 8 * PAGE
    assert m.rec_state == "state@128"


def test_matched_indices_survive_the_split():
    """Splitting to plant a resume point must preserve the KV indices of the prefix."""
    c = _cache()
    base = _seq(8 * PAGE)
    res = c.insert_prefix(base, _indices(len(base)))
    c.attach_rec_state_at(res.handle, 4 * PAGE, rec_state="state@64")
    c.attach_rec_state(res.handle, rec_state="state@128")

    m = c.match_prefix(base).cuda_handle
    got = m.get_matched_indices()
    assert torch.equal(got, _indices(8 * PAGE)), "KV indices corrupted by the split"


def test_rejects_unaligned_and_out_of_range_boundaries():
    c = _cache()
    base = _seq(8 * PAGE)
    res = c.insert_prefix(base, _indices(len(base)))
    assert not c.attach_rec_state_at(res.handle, 4 * PAGE + 1, "x"), "unaligned must be refused"
    assert not c.attach_rec_state_at(res.handle, 0, "x")
    assert not c.attach_rec_state_at(res.handle, 8 * PAGE, "x"), "end boundary is attach_rec_state's"
    assert not c.attach_rec_state_at(res.handle, 99 * PAGE, "x")


def test_lru_cap_evicts_oldest_interior_snapshot():
    c = RadixPrefixCache(device=torch.device("cpu"), recurrent=True, max_rec_snapshots=2)
    base = _seq(8 * PAGE)
    res = c.insert_prefix(base, _indices(len(base)))
    c.attach_rec_state_at(res.handle, 2 * PAGE, "a")
    c.attach_rec_state_at(res.handle, 4 * PAGE, "b")
    c.attach_rec_state_at(res.handle, 6 * PAGE, "c")
    live = [n.rec_state for n in c._rec_nodes if n.rec_state is not None]
    assert len(live) == 2 and "a" not in live, f"cap not enforced: {live}"


def test_dense_cache_is_untouched():
    """A non-recurrent radix already partial-matches; the new path must be inert there."""
    c = RadixPrefixCache(device=torch.device("cpu"), recurrent=False)
    base = _seq(8 * PAGE)
    res = c.insert_prefix(base, _indices(len(base)))
    assert not c.attach_rec_state_at(res.handle, 4 * PAGE, "x")
    other = torch.cat([base[: 6 * PAGE], _seq(2 * PAGE, salt=9)])
    assert c.match_prefix(other).cuda_handle.cached_len == 6 * PAGE
