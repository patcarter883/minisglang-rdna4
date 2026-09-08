"""The WIRING, not the policy — does the cache bind to the tensors the kernel actually reads?

`expert_cache_test.py` proves the cache's own invariant against synthetic tensors. This file proves
the thing that file cannot: that the triple the cache copies is the SAME triple the forward hands
the kernel. That is the failure this feature could plausibly ship with — a cache with a great hit
rate pointing an expert at another expert's bytes produces plausible wrong numbers with no error.

    python3 -m pytest tests/expert_cache_wiring_test.py -q -o addopts=""
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.layers.moe import (  # noqa: E402
    MoEQuantMethod,
    _GroupedNvFp4Experts,
    _NvFp4MoEMethod,
    create_moe_quant_method,
)
from minisgl.quant.config import QuantConfig  # noqa: E402

E, N, K = 8, 128, 512


def _nvfp4_quant():
    return QuantConfig(method="compressed-tensors", bits=4, group_size=16, sym=True,
                       weight_type="float", ct_format="nvfp4-pack-quantized")


def _built():
    c = _GroupedNvFp4Experts(E, N, K, _nvfp4_quant())
    for t in (c.weight_packed, c.weight_scale, c.weight_global):
        t.copy_(torch.randint(0, 200, t.shape).to(t.dtype) if t.dtype != torch.float32
                else torch.randn(t.shape))
    c.post_load()
    return c


def test_the_declared_triple_is_what_the_forward_passes():
    """THE GATE. `apply()` builds its kernel arguments from `cache_plane_attrs` via `plane()`, so
    the two cannot drift. Pin that the declaration resolves to real post-load tensors, in the
    kernel's (w, scales, zeros) order, with E on dim 0 of each — the slab geometry depends on it."""
    c = _built()
    w, s, z = _NvFp4MoEMethod.plane(c)
    assert w is c._w_op and s is c._scales_op and z is c._global_op
    for name, t in (("w", w), ("scales", s), ("zeros/global", z)):
        assert t is not None, f"{name} resolved to None"
        assert t.shape[0] == E, f"{name} has {t.shape[0]} on dim 0, expected E={E}"


def test_the_global_rides_the_zeros_slot_as_int32_bits():
    """NVFP4's third tensor is NOT zero-points: it is the f32 per-output-channel global, BITCAST to
    int32 for a `const int*` slot. A cache that widened or value-converted it would zero every
    expert output. Pin the bitcast, because `.view()` and `.to()` are one character apart."""
    c = _built()
    z = _NvFp4MoEMethod.plane(c)[2]
    assert z.dtype == torch.int32
    assert z.view(torch.float32).abs().sum() > 0, "the global decoded to all zeros — value-converted?"


def test_a_method_with_no_declaration_is_cache_off_not_cache_wrong():
    """An unsupported format must read the host base, never get a guessed triple."""
    assert MoEQuantMethod.cache_plane_attrs == ()


def test_registering_the_real_container_matches_slab_geometry():
    """The slabs are `(slots, *t.shape[1:])`. Register a REAL post-load container and prove each
    slab row is byte-compatible with the host row it will hold — a mismatch here is the wrong-bytes
    bug, and it is invisible without comparing the actual shapes."""
    from minisgl.weights.expert_cache import ExpertResidencyCache

    c13, c2 = _built(), _built()
    row_bytes = sum(t.numel() // E * t.element_size() for t in _NvFp4MoEMethod.plane(c13))
    row_bytes += sum(t.numel() // E * t.element_size() for t in _NvFp4MoEMethod.plane(c2))
    cache = ExpertResidencyCache(num_experts=E, expert_bytes=row_bytes,
                                 budget_bytes=row_bytes * 4, device=torch.device("cpu"))
    cache.register_layer(0, _NvFp4MoEMethod.plane(c13), _NvFp4MoEMethod.plane(c2))
    for plane, container in (("gate_up", c13), ("down", c2)):
        for kind, host in zip(("w", "s", "z"), _NvFp4MoEMethod.plane(container)):
            slab = cache._slabs[f"{plane}_{kind}"]
            assert slab is not None, f"{plane}.{kind} slab missing for a tensor that exists"
            assert slab.shape[1:] == host.shape[1:], f"{plane}.{kind} row shape mismatch"
            assert slab.dtype == host.dtype, f"{plane}.{kind} dtype mismatch"
    # and the promotion actually reproduces the host bytes
    cache.observe(0, [1, 5])
    for e in (1, 5):
        slot = int(cache._layers[0]["slot_of"][e])
        assert slot >= 0
        for plane, container in (("gate_up", c13), ("down", c2)):
            for kind, host in zip(("w", "s", "z"), _NvFp4MoEMethod.plane(container)):
                assert torch.equal(cache._slabs[f"{plane}_{kind}"][slot], host[e]), (
                    f"expert {e} {plane}.{kind}: slab row != host row")


def test_the_quant_method_for_this_checkpoint_declares_a_triple():
    """qwen4_exp is NVFP4; if its method ever stopped declaring, the cache would silently go OFF
    for every layer and the serve would look merely slow."""
    m = create_moe_quant_method(_nvfp4_quant())
    assert isinstance(m, _NvFp4MoEMethod)
    assert len(m.cache_plane_attrs) == 3


def test_the_slot_map_op_actually_dispatches_through_the_seam():
    """THE ONE THAT WAS MISSING. `hasattr(fp8_wmma, "set_expert_slot_map")` passing proves nothing
    about the path the seam takes: `install()` calls `moe_interpose._set_expert_slot_map`, and that
    resolved the op as `torch.ops.fp8_wmma_C.set_expert_slot_map`. The op namespace is populated as
    a SIDE EFFECT of importing fp8_wmma, so on a rank that had not imported the package yet the
    lookup raised `'_OpNamespace' ... has no attribute 'set_expert_slot_map'` — and it raised on
    EVERY install, after the slab had been allocated and the ring subscribed. A live serve found it;
    nothing at import scope could.

    Calls the DISABLE form (all-None), which is the reachability case: it takes no tensor arguments,
    so it is also the form a backend dispatch key would make unreachable.
    """
    pytest.importorskip("fp8_wmma")
    from minisgl.weights import moe_interpose as mi

    mi._SET_MAP_OP = None                      # force the resolution path this test is about
    mi._set_expert_slot_map(None, None, None, None, None, None, None)


def test_observe_does_no_policy_work_once_a_manager_is_running():
    """THE POINT OF THE THREAD. `observe()` runs on the SCHEDULER thread (route_trace.drain() is
    called from begin_forward), so once a manager exists it must enqueue and return — no policy
    update, no allocation, no copy. Measured cost of getting this wrong: fwd_launch 5356 ms/step
    (92% of the step) against a 68.5 ms baseline."""
    from minisgl.weights.expert_cache import ExpertResidencyCache

    cache = ExpertResidencyCache(num_experts=8, expert_bytes=64, budget_bytes=640,
                                 device=torch.device("cpu"))
    w = torch.arange(8, dtype=torch.float32).view(8, 1).repeat(1, 4)
    cache.register_layer(0, (w, w, None), (w + 100, w + 100, None))
    cache._thread = object()          # pretend a manager is running
    try:
        cache.observe(0, [1, 2, 3])
        assert cache.stats["promotions"] == 0, "observe() promoted on the scheduler thread"
        assert cache.stats["hits"] == 0 and cache.stats["misses"] == 0, "observe() ran the policy"
        assert len(cache._q) == 1, "the reference was not queued"
    finally:
        cache._thread = None


def test_the_queue_drops_rather_than_blocks():
    """A full queue must lose references, never stall the scheduler. Dropping is the safe
    direction: a staler policy only ever under-reports residency, which is a host read."""
    from minisgl.weights.expert_cache import ExpertResidencyCache

    cache = ExpertResidencyCache(num_experts=8, expert_bytes=64, budget_bytes=640,
                                 device=torch.device("cpu"))
    cache._thread = object()
    try:
        for _ in range(cache._q.maxlen + 25):
            cache.observe(0, [1])
        assert len(cache._q) == cache._q.maxlen
        assert cache.stats["dropped_refs"] == 25
    finally:
        cache._thread = None


def test_apply_pending_is_a_noop_without_cuda():
    """The scheduler calls this EVERY step. On CPU (and with nothing pending) it must do nothing
    and cost nothing — a per-step hook that can raise is a per-step outage."""
    from minisgl.weights.expert_cache import ExpertResidencyCache

    cache = ExpertResidencyCache(num_experts=8, expert_bytes=64, budget_bytes=640,
                                 device=torch.device("cpu"))
    cache.apply_pending()
    cache.apply_pending()
