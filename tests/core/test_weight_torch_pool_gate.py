"""`ArenaMemPool`'s merge gate and its capture guard. NO GPU, NO torch.

torch is not importable on this host, and `ArenaMemPool.__init__` needs it (`_cuda_customAllocator`,
`MemPool`). So these tests build the object without running `__init__` and exercise the two pieces
that are pure Python: `assert_clean()` and `_capturing()`.

That is not a shortcut — `assert_clean()` IS the merge gate. Its whole job is to answer "did any
byte that the capacity plan booked against host RAM actually land in VRAM?", and the original
implementation could only see ONE of the three ways that happens:

* a counted `hipMalloc` fallback              -> was caught;
* an allocation served during graph capture   -> was not caught (and cannot be, mid-capture, by the
                                                 fallback path: hipMalloc is illegal during capture);
* the pool never routing at all               -> was not caught, and this is the quiet one. Every
  way the MemPool stops being consulted (installed on the wrong device at TP=2, a torch that changes
  `use_mem_pool` semantics, a caller that forgot `use()`) yields `alloc_events == 0` AND
  `torch_fallbacks == 0` — a perfectly clean ledger over a serve that put the entire host tier in
  16 GiB of VRAM. It surfaces much later as an unrelated OOM, on one rank only.
"""

from __future__ import annotations

import pytest
from minisgl.weights.torch_pool import ArenaMemPool, _capturing


class _FakeArena:
    """Just the attributes the gate reads."""

    def __init__(self, *, pinned_bytes: int = 1 << 30, fallbacks: int = 0, device_index: int = 0):
        self.pinned_bytes = pinned_bytes
        self.torch_fallbacks = fallbacks
        self.device_index = device_index


def _pool(**arena_kw) -> ArenaMemPool:
    """An `ArenaMemPool` with `__init__` bypassed — torch is not importable on this host."""
    p = ArenaMemPool.__new__(ArenaMemPool)
    p.arena = _FakeArena(**arena_kw)
    p.allow_fallback = True
    p.alloc_events = 0
    p.free_events = 0
    p.served_from_arena = 0
    p.served_bytes = 0
    p.alloc_during_capture = 0
    p._last_error = None
    return p


class TestAssertClean:
    def test_a_healthy_pool_passes(self):
        p = _pool()
        p.alloc_events = 8
        p.served_from_arena = 8
        p.served_bytes = 1 << 20
        p.assert_clean()

    def test_a_counted_fallback_fails(self):
        p = _pool(fallbacks=1)
        p.alloc_events = 8
        p.served_from_arena = 7
        with pytest.raises(RuntimeError) as exc:
            p.assert_clean()
        assert "escaped the arena" in str(exc.value)

    def test_the_pool_never_routing_is_caught(self):
        """The failure the original gate was blind to: zero fallbacks over a serve that used none
        of the arena, because the callback was never invoked at all."""
        p = _pool()  # 1 GiB pinned, nothing ever asked for
        with pytest.raises(RuntimeError) as exc:
            p.assert_clean()
        msg = str(exc.value)
        assert "NEVER fired" in msg
        assert "cuda:0" in msg  # names the device the pool should have been bound to

    def test_an_unpinned_arena_does_not_trip_the_never_fired_check(self):
        """The degenerate all-device plan pins nothing and allocates nothing. That is not a fault."""
        _pool(pinned_bytes=0).assert_clean()

    def test_an_allocation_during_graph_capture_fails_the_gate(self):
        p = _pool()
        p.alloc_events = 3
        p.served_from_arena = 3
        p.alloc_during_capture = 1
        with pytest.raises(RuntimeError) as exc:
            p.assert_clean()
        assert "graph capture" in str(exc.value)

    def test_expect_served_bytes_catches_a_partially_routed_pool(self):
        p = _pool()
        p.alloc_events = 4
        p.served_from_arena = 4
        p.served_bytes = 1 << 20
        p.assert_clean(expect_served_bytes=1 << 20)
        with pytest.raises(RuntimeError) as exc:
            p.assert_clean(expect_served_bytes=1 << 30)
        assert "host-resident" in str(exc.value)

    def test_a_swallowed_callback_error_fails_even_with_no_fallback(self):
        p = _pool()
        p.alloc_events = 2
        p.served_from_arena = 2
        p._last_error = "boom"
        with pytest.raises(RuntimeError):
            p.assert_clean()


class TestCapturingProbe:
    def test_is_false_and_never_raises_without_torch(self):
        """Runs inside the C ABI callback, where a raise becomes a NULL return and a segfault with
        no diagnostic. It must answer, never throw — including on a host with no torch at all."""
        assert _capturing() is False
