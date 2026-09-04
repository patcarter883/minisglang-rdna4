"""END-TO-END wiring of the CPU tier through a REAL `MoELayer`: bind, attach, forward.

TORCH REQUIRED, GPU NOT. Everything here runs on `device="cpu"` — which is not a compromise, it is
the point: a CPU-tier layer's weights are ordinary pageable host tensors and its compute is a host
call, so the entire path from `plan_three_tier` through `bind(StackKind.CPU)` to
`MoELayer.forward` returning a CPU-computed partial is exercisable with no card. The only thing
that genuinely needs a GPU is the D2H/H2D activation round trip, and that is exactly what is
carried as UNMEASURED in `cpu_tier.CpuTierPrior.handoff_us_bracket`.

Run inside the serve image:
    docker run --rm -v <worktree>:/engine --entrypoint bash minisgl-rdna4:<tag> -lc \\
      'cd /engine && PYTHONPATH=/engine/python:/opt/kernels python -m pytest \\
       tests/core/test_cpu_tier_seam.py -q'
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.layers.base import BaseOP  # noqa: E402
from minisgl.weights.cpu_tier import CpuTierMode, HandoffError, graph_segments  # noqa: E402
from minisgl.weights.cpu_worker import CpuMoEWorker  # noqa: E402
from minisgl.weights.moe_interpose import (  # noqa: E402
    InterpositionError,
    attach_seams,
    bind_plan,
    build_layer_weights,
    discover_moe_layers,
)
from minisgl.weights.placement import plan_three_tier  # noqa: E402
from minisgl.weights.stacks import StackKind, TorchStackAllocator  # noqa: E402


@pytest.fixture()
def tp1():
    """Single-rank TP with EP off — enough to construct a real `MoELayer` on CPU."""
    from minisgl.distributed import info

    old = (info._TP_INFO, info._DP_INFO, info._ENABLE_EP, info._EP_OVER_TP)
    info._TP_INFO = info.DistributedInfo(0, 1)
    info._DP_INFO = None
    info._ENABLE_EP = False
    info._EP_OVER_TP = False
    try:
        yield
    finally:
        info._TP_INFO, info._DP_INFO, info._ENABLE_EP, info._EP_OVER_TP = old


class _Block(BaseOP):
    def __init__(self, moe):
        self.mlp = moe

    def forward(self, *a, **kw):  # pragma: no cover
        raise RuntimeError


class _Model(BaseOP):
    def __init__(self, blocks):
        self.layers = blocks

    def forward(self, *a, **kw):  # pragma: no cover
        raise RuntimeError


def _moe(n=8, k=2, hidden=32, inter=16):
    from minisgl.layers.moe import MoELayer

    return MoELayer(num_experts=n, top_k=k, hidden_size=hidden, intermediate_size=inter)


def _model(nlayers=4, **kw):
    m = _Model([_Block(_moe(**kw)) for _ in range(nlayers)])
    for _, layer in discover_moe_layers(m):
        for attr in ("gate_up_proj", "down_proj"):
            t = getattr(layer, attr)
            setattr(layer, attr, torch.randn_like(t.float()).to(t.dtype))
    return m


class _CountingBackend:
    """Returns a distinguishable constant per layer so a mis-attached partial is visible."""

    name = "counting"

    def __init__(self, hidden: int) -> None:
        self.hidden = hidden
        self.calls: list[tuple] = []

    def compute(self, x, ids, weights):
        import numpy as np

        self.calls.append((tuple(np.asarray(ids).reshape(-1).tolist()), tuple(
            np.asarray(weights, dtype=np.float64).reshape(-1).tolist()
        )))
        n = np.asarray(x, dtype=np.float64).reshape(-1, self.hidden).shape[0]
        return np.full((n, self.hidden), float(len(self.calls)), dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestBindCpuTier:
    def test_a_cpu_layer_binds_to_PAGEABLE_cpu_tensors_with_no_arena(self, tp1):
        """The capacity claim, at the allocator: no `host_alloc` is injected and it still binds."""
        model = _model(3)
        seams = attach_seams(model)
        layers = build_layer_weights(seams)
        plan = plan_three_tier(layers, device_budget_bytes=0, num_cpu_layers=2)
        assert plan.num_cpu_layers == 2 and plan.num_host_layers == 1

        # No host arena at all. A HOST layer would raise `HostStackUnavailable` here; the CPU
        # layers must not, because they need no pinned pages.
        alloc = TorchStackAllocator(device="cpu", host_alloc=None)
        cpu_only = plan_three_tier(layers, device_budget_bytes=0, num_cpu_layers=3)
        out = bind_plan(
            seams, cpu_only, alloc, selftest=4, log=False,
            cpu_worker=CpuMoEWorker(_CountingBackend(32)),
        )
        assert out.cpu_layers == 3
        assert out.host_layers == 0
        assert out.cpu_resident_bytes > 0
        assert alloc.bytes_used(StackKind.HOST) == 0
        assert alloc.bytes_used(StackKind.CPU) == out.cpu_resident_bytes
        for s in seams:
            assert s.kind is StackKind.CPU
            w13 = getattr(s._layer, "gate_up_proj")
            assert w13.device.type == "cpu"
            assert not w13.is_pinned(), (
                "the CPU tier must be PAGEABLE: pinning it puts it straight back on the "
                "hipHostMalloc ceiling it exists to escape, and buys nothing — the consumer is "
                "AVX-512 cores issuing ordinary loads."
            )

    def test_the_bake_still_read_back_verifies_the_cpu_tier(self, tp1):
        """Same `_bake`, same bitwise read-back. A "it's just a torch copy" shortcut would drop it."""
        model = _model(2)
        seams = attach_seams(model)
        plan = plan_three_tier(
            build_layer_weights(seams), device_budget_bytes=0, num_cpu_layers=2
        )
        out = bind_plan(
            seams, plan, TorchStackAllocator(device="cpu"), selftest=8, log=False,
            cpu_worker=CpuMoEWorker(_CountingBackend(32)),
        )
        assert all(r.verified_components > 0 for r in out.reports)

    def test_the_values_survive_the_move(self, tp1):
        model = _model(1)
        (_, layer), = discover_moe_layers(model)
        before = layer.gate_up_proj.clone()
        seams = attach_seams(model)
        plan = plan_three_tier(
            build_layer_weights(seams), device_budget_bytes=0, num_cpu_layers=1
        )
        bind_plan(
            seams, plan, TorchStackAllocator(device="cpu"), selftest=8, log=False,
            cpu_worker=CpuMoEWorker(_CountingBackend(32)),
        )
        assert torch.equal(layer.gate_up_proj, before)

    def test_resolve_REFUSES_on_a_cpu_seam(self, tp1):
        """A GPU kernel must never be handed a pageable, unmapped host pointer."""
        model = _model(1)
        seams = attach_seams(model)
        plan = plan_three_tier(
            build_layer_weights(seams), device_budget_bytes=0, num_cpu_layers=1
        )
        bind_plan(
            seams, plan, TorchStackAllocator(device="cpu"), selftest=4, log=False,
            cpu_worker=CpuMoEWorker(_CountingBackend(32)),
        )
        s = seams[0]
        with pytest.raises(InterpositionError, match="CPU-COMPUTE tier"):
            s.resolve(s._layer.gate_up_proj, s._layer.down_proj)

    def test_a_worker_cannot_be_attached_to_a_device_or_host_seam(self, tp1):
        """A layer computed on the host AND streamed to the card is double-counted in the residual."""
        model = _model(1)
        seams = attach_seams(model)
        plan = plan_three_tier(
            build_layer_weights(seams), device_budget_bytes=1 << 40, num_cpu_layers=0
        )
        # freeze=False: `attach_cpu_worker` is a PLACEMENT entry point and rule R1 makes every one
        # of them raise after freeze(), so the "wrong tier" refusal is only reachable before it.
        bind_plan(seams, plan, None, selftest=4, log=False, freeze=False)
        assert seams[0].kind is StackKind.DEVICE
        with pytest.raises(InterpositionError, match="not CPU"):
            seams[0].attach_cpu_worker(object())

    def test_binding_a_cpu_plan_with_no_worker_is_a_BOOT_error(self, tp1):
        """Not a degraded mode. Without a worker the first token raises, so the bind must."""
        model = _model(1)
        seams = attach_seams(model)
        plan = plan_three_tier(
            build_layer_weights(seams), device_budget_bytes=0, num_cpu_layers=1
        )
        with pytest.raises(InterpositionError, match="no `cpu_worker` was passed"):
            bind_plan(seams, plan, TorchStackAllocator(device="cpu"), selftest=4, log=False)

    def test_a_cpu_seam_without_a_worker_refuses_rather_than_falling_through(self, tp1):
        """The seam-level guard, reached only by a caller that bypassed `bind_plan`."""
        model = _model(1)
        seams = attach_seams(model)
        plan = plan_three_tier(
            build_layer_weights(seams), device_budget_bytes=0, num_cpu_layers=1
        )
        seams[0].bind(StackKind.CPU, TorchStackAllocator(device="cpu"), selftest=4)
        assert not seams[0].computes_on_cpu
        with pytest.raises(InterpositionError, match="no worker attached"):
            seams[0].cpu_submit(torch.randn(1, 32), torch.rand(1, 2), torch.zeros(1, 2, dtype=torch.int32))

    def test_the_worker_is_attached_in_PLAN_order_with_a_packed_expert_offset(self, tp1):
        """Rank-identical offsets: derived from the plan walk, never from the caller's enumeration."""
        model = _model(4, n=8)
        seams = attach_seams(model)
        # A budget big enough that the non-CPU layer lands on DEVICE: this allocator has no host
        # arena, which is itself the point (a CPU layer needs none).
        plan = plan_three_tier(
            build_layer_weights(seams), device_budget_bytes=1 << 40, num_cpu_layers=3
        )
        worker = CpuMoEWorker(_CountingBackend(32))
        bind_plan(
            seams, plan, TorchStackAllocator(device="cpu"), selftest=4, log=False,
            cpu_worker=worker,
        )
        offs = [s._cpu_expert_offset for s in seams if s.kind is StackKind.CPU]
        assert offs == [0, 8, 16]
        assert all(s.cpu_worker is worker for s in seams if s.kind is StackKind.CPU)
        assert all(s.cpu_worker is None for s in seams if s.kind is not StackKind.CPU)


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestForwardThroughTheCpuTier:
    def _wire(self, nlayers=3, ncpu=2, hidden=32, k=2, n=8):
        model = _model(nlayers, n=n, k=k, hidden=hidden, inter=16)
        seams = attach_seams(model)
        plan = plan_three_tier(
            build_layer_weights(seams), device_budget_bytes=1 << 40, num_cpu_layers=ncpu
        )
        backend = _CountingBackend(hidden)
        worker = CpuMoEWorker(backend)
        bind_plan(
            seams, plan, TorchStackAllocator(device="cpu"), selftest=4, log=False,
            cpu_worker=worker,
        )
        worker.start()
        return model, seams, plan, worker, backend

    def test_forward_routes_a_cpu_layer_to_the_worker_and_never_to_the_kernel(self, tp1):
        model, seams, plan, worker, backend = self._wire()
        try:
            worker.begin_step(1)
            cpu_seam = next(s for s in seams if s.kind is StackKind.CPU)
            hs = torch.randn(2, 32)
            logits = torch.randn(2, 8)
            out = cpu_seam._layer.forward(hs, logits)
            worker.barrier("end of step")
        finally:
            worker.stop()
        assert out.shape == hs.shape
        assert out.dtype == hs.dtype
        assert torch.all(out == 1.0)  # the backend's first-call marker
        assert len(backend.calls) == 1
        ids, weights = backend.calls[0]
        assert len(ids) == 2 * 2  # (tokens x top_k), flattened
        assert sum(weights) == pytest.approx(2.0, abs=1e-5)  # renormalised per token

    def test_a_device_layer_in_the_same_model_still_takes_the_kernel_path(self, tp1):
        """The tiers coexist. A CPU seam must not change what a DEVICE seam does.

        Asserted at the SEAM rather than by running the device forward: the grouped-kernel path
        needs `core.get_global_ctx()` and a real HIP kernel, neither of which exists on a box with
        no card. What is checkable here is the branch `MoELayer.forward` takes — `computes_on_cpu`
        — and that `resolve()`, the two-pointer identity check the kernel path goes through, still
        returns the layer's own containers on a non-CPU seam.
        """
        model, seams, plan, worker, backend = self._wire(nlayers=3, ncpu=1)
        try:
            other = next(s for s in seams if s.kind is not StackKind.CPU)
            assert not other.computes_on_cpu
            assert other.cpu_worker is None
            w13, w2 = other._layer.gate_up_proj, other._layer.down_proj
            assert other.resolve(w13, w2) == (w13, w2)
        finally:
            worker.stop()
        assert backend.calls == []  # the worker was never asked

    def test_the_route_reaching_the_worker_is_the_layer_s_real_topk(self, tp1):
        model, seams, plan, worker, backend = self._wire(n=8, k=3)
        try:
            worker.begin_step(1)
            cpu_seam = next(s for s in seams if s.kind is StackKind.CPU)
            logits = torch.full((1, 8), -10.0)
            logits[0, 5] = 10.0
            logits[0, 2] = 9.0
            logits[0, 7] = 8.0
            cpu_seam._layer.forward(torch.randn(1, 32), logits)
            worker.barrier("end of step")
        finally:
            worker.stop()
        ids, weights = backend.calls[0]
        assert set(ids) == {5, 2, 7}
        assert weights[0] > weights[1] > weights[2]
        assert sum(weights) == pytest.approx(1.0, abs=1e-5)

    def test_two_cpu_layers_in_one_step_stay_in_order_and_the_barrier_closes_it(self, tp1):
        model, seams, plan, worker, backend = self._wire(nlayers=4, ncpu=2)
        try:
            worker.begin_step(1)
            hs = torch.randn(1, 32)
            outs = [
                s._layer.forward(hs, torch.randn(1, 8))
                for s in seams
                if s.kind is StackKind.CPU
            ]
            worker.barrier("end of step")
            worker.begin_step(2)  # only possible because nothing leaked across the boundary
        finally:
            worker.stop()
        assert torch.all(outs[0] == 1.0) and torch.all(outs[1] == 2.0)
        assert worker.layers_computed == 2

    def test_the_async_form_lets_gpu_work_happen_between_submit_and_join(self, tp1):
        """SPLIT mode's mechanism, on the seam API rather than on the ledger directly."""
        import time

        class _Slow(_CountingBackend):
            def compute(self, x, ids, weights):
                time.sleep(0.12)
                return super().compute(x, ids, weights)

        model = _model(1, n=8, k=2, hidden=32, inter=16)
        seams = attach_seams(model)
        plan = plan_three_tier(
            build_layer_weights(seams), device_budget_bytes=0, num_cpu_layers=1
        )
        worker = CpuMoEWorker(_Slow(32))
        bind_plan(
            seams, plan, TorchStackAllocator(device="cpu"), selftest=4, log=False,
            cpu_worker=worker,
        )
        worker.start()
        try:
            worker.begin_step(1)
            hs = torch.randn(1, 32)
            t0 = time.perf_counter()
            h = seams[0].cpu_submit(hs, torch.rand(1, 2), torch.zeros(1, 2, dtype=torch.int32))
            submit_s = time.perf_counter() - t0
            time.sleep(0.12)  # stand-in for the GPU-side partial of the SAME layer
            got = seams[0].cpu_join(h, like=hs)
            total_s = time.perf_counter() - t0
            worker.barrier("end of step")
        finally:
            worker.stop()
        assert submit_s < 0.05
        assert total_s < 0.22, "the CPU partial did not overlap the concurrent work"
        assert got.shape == hs.shape and got.device.type == "cpu"

    def test_a_worker_failure_propagates_out_of_forward(self, tp1):
        """A dead CPU tier must be a raised exception, never a zero partial in the residual."""

        class _Angry:
            name = "angry"

            def compute(self, x, ids, weights):
                raise RuntimeError("no such expert")

        model = _model(1)
        seams = attach_seams(model)
        plan = plan_three_tier(
            build_layer_weights(seams), device_budget_bytes=0, num_cpu_layers=1
        )
        worker = CpuMoEWorker(_Angry())
        bind_plan(
            seams, plan, TorchStackAllocator(device="cpu"), selftest=4, log=False,
            cpu_worker=worker,
        )
        worker.start()
        try:
            worker.begin_step(1)
            with pytest.raises(HandoffError, match="no such expert"):
                seams[0]._layer.forward(torch.randn(1, 32), torch.randn(1, 8))
        finally:
            worker.stop()


# ════════════════════════════════════════════════════════════════════════════════════════════════
class TestCaptureInteraction:
    """The conflict with the graph-capture work, stated as an assertion rather than a caveat."""

    def test_every_cpu_layer_cuts_the_graph_INSIDE_itself(self, tp1):
        """K CPU layers = K+1 device segments. Contiguity does not reduce it and never did.

        A CPU-tier layer is not "a layer that runs on the CPU": its attention, its norms, its
        router and its residual adds are all still GPU work, so the host call lands in the MIDDLE
        of the layer's device work rather than between two layers.
        """
        model = _model(6)
        seams = attach_seams(model)
        plan = plan_three_tier(
            build_layer_weights(seams), device_budget_bytes=0, num_cpu_layers=2
        )
        assert plan.cpu_layer_indices == (4, 5)
        assert plan.cpu_block_is_contiguous
        assert graph_segments(len(plan.placements), plan.cpu_layer_indices, CpuTierMode.BLOCK) == 3
        # ...and a SCATTERED set of the same size costs exactly the same.
        assert graph_segments(6, (1, 4), CpuTierMode.BLOCK) == 3

    def test_no_mode_with_a_cpu_layer_is_capturable_as_one_graph(self, tp1):
        """`engine/graph.py` captures the whole forward into ONE CUDAGraph per bs bucket."""
        assert CpuTierMode.OFF.is_capturable
        assert not CpuTierMode.BLOCK.is_capturable
        assert not CpuTierMode.SPLIT.is_capturable
        assert graph_segments(48, range(27, 48), CpuTierMode.SPLIT) == 0
        assert graph_segments(48, range(27, 48), CpuTierMode.BLOCK) == 22
