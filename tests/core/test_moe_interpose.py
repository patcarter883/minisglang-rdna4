"""The MoE weight-offload interposition seam.

NO GPU. Everything here runs on CPU tensors: the "host stack" is a plain CPU allocator injected
into `TorchStackAllocator`, which is exactly the shape the pinned arena plugs into. What is being
tested is the SUBSTITUTION and its refusals — that the right tensors move, that aliases survive the
move instead of being silently de-aliased, that a granule's scales travel with its weights, that a
seam bound to the wrong layer is caught loudly, and that nothing can be re-placed after freeze().

The one thing these tests cannot cover is graph capture at the served TP with the real forward; that
is an M1-D gate on a leased card, not a unit test. See the module's `not_done` notes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.weights import moe_interpose  # noqa: E402
from minisgl.layers.base import BaseOP, OPList  # noqa: E402
from minisgl.weights.granule import ExpertContainer, GranuleError  # noqa: E402
from minisgl.weights.moe_interpose import (  # noqa: E402
    InterpositionError,
    MoEWeightSeam,
    attach_seams,
    bind_plan,
    build_layer_weights,
    detach_seams,
    discover_moe_layers,
    ep_size_of,
    seam_summary,
)
from minisgl.weights.placement import plan_layer_granular  # noqa: E402
from minisgl.weights.stacks import (  # noqa: E402
    HostStackUnavailable,
    StackKind,
    TorchStackAllocator,
)

# ==================================================================================================
# Fixtures: a synthetic expert container with every awkward property the shipped formats have
# ==================================================================================================


class _FakeExperts(ExpertContainer):
    """Stands in for `_Grouped*Experts`: a packed weight, a per-group scale, and zero-points.

    Deliberately mirrors the shapes the real containers present after `post_load()` — expert-major
    dim 0, underscore-prefixed names (which `BaseOP.state_dict`/`post_load` skip and the granule
    walk deliberately does not), and an optional replicated zeros stack (symmetric compressed-tensors
    `_zeros_op` is E identical copies of `0x88`).
    """

    def __init__(self, n: int, out_f: int, in_f: int, *, replicated_zeros: bool = False):
        self._num_experts = n
        self._w_op = torch.randint(-(2**31), 2**31 - 1, (n, out_f, in_f // 8), dtype=torch.int32)
        self._scales_op = torch.randn(n, in_f // 32, out_f, dtype=torch.float16)
        if replicated_zeros:
            z = torch.empty((n, in_f // 32, out_f // 8), dtype=torch.int32)
            z.view(torch.uint8).fill_(0x88)
        else:
            z = torch.randint(-(2**31), 2**31 - 1, (n, in_f // 32, out_f // 8), dtype=torch.int32)
        self._zeros_op = z

    def forward(self, *a, **kw):  # pragma: no cover - storage container
        raise RuntimeError("storage container")


class _SharedZeroExperts(_FakeExperts):
    """A container that DECLARES its zeros expert-invariant.

    Declared, never detected: `granule` downgraded bitwise row-invariance to a hint
    (`content_invariant`) because it is a property of THIS rank's shard, and two TP ranks hold
    different shards. `_residency_shared` is the rank-safe form and the only thing that still lands
    in `spec.replicated`, so it is what exercises the seam's replicated-tensor path.
    """

    _residency_shared = ("_zeros_op",)

    def __init__(self, n: int, out_f: int, in_f: int):
        super().__init__(n, out_f, in_f, replicated_zeros=True)


class _AliasedExperts(ExpertContainer):
    """The `_GroupedFP8Experts` shape, with BOTH of its real alias kinds.

    * `_w_op = weight.view(uint8)`  — same bytes, same shape, DIFFERENT dtype.
    * `_scales_op = weight_scale.squeeze(-1)` — same bytes, same dtype, DIFFERENT shape
      (`post_load`'s `.squeeze(-1).contiguous().float()` is a no-op chain on an already-contiguous
      f32 `(E, N, 1)`, so the two names really are one storage).

    Both directions matter: a rebind that keeps the names aliased but hands every name the CANONICAL
    view silently changes what `_w_op.element_size()` and every kernel binding compute.
    """

    def __init__(self, n: int, out_f: int, in_f: int):
        self._num_experts = n
        self.weight = torch.randint(0, 255, (n, out_f, in_f), dtype=torch.uint8).view(torch.int8)
        self._w_op = self.weight.view(torch.uint8)
        self.weight_scale = torch.randn(n, out_f, 1, dtype=torch.float32)
        self._scales_op = self.weight_scale.squeeze(-1)

    def forward(self, *a, **kw):  # pragma: no cover
        raise RuntimeError("storage container")


class _TwinViewExperts(ExpertContainer):
    """Two attribute names that are SIBLING views of a base tensor the container does not hold.

    This is the shape that breaks a leak proof which watches only the CANONICAL name: neither
    sibling is the other's `_base`, so dropping the canonical does not drop the storage. Holding
    either one alive keeps the device original resident.
    """

    def __init__(self, n: int, out_f: int, in_f: int):
        self._num_experts = n
        base = torch.randint(0, 255, (n, out_f, in_f // 8), dtype=torch.uint8)
        self._w_op = base.view(torch.int32)  # canonical (first in __dict__ order)
        self._w_raw = base.view(torch.uint8)  # sibling view, same bytes, NOT the canonical's base
        self._scales_op = torch.randn(n, in_f // 32, out_f, dtype=torch.float16)

    def forward(self, *a, **kw):  # pragma: no cover - storage container
        raise RuntimeError("storage container")


class _ListHeldExperts(ExpertContainer):
    """A container holding one component inside a LIST attribute.

    `granule._iter_tensors` descends into list/tuple/dict members and emits `f"{name}[{i}]"`, and
    `granule._lookup` resolves that form — so the seam's rebind must too. Nothing shipped does this
    today; the walk supports it, which is exactly why the two sides must not drift.
    """

    def __init__(self, n: int, out_f: int, in_f: int):
        self._num_experts = n
        self._w_op = torch.randint(-(2**31), 2**31 - 1, (n, out_f, in_f // 8), dtype=torch.int32)
        self._scales = [torch.randn(n, in_f // 32, out_f, dtype=torch.float16)]

    def forward(self, *a, **kw):  # pragma: no cover - storage container
        raise RuntimeError("storage container")


class _RefusingExperts(_FakeExperts):
    def offload_refusal(self) -> "str | None":
        return "this format materialises the whole stack per forward"


class _FakeMoELayer:
    """Duck-typed stand-in for `MoELayer` where only the seam's contract matters."""

    _weight_offload = None

    def __init__(self, n: int, k: int, w13, w2):
        self.local_num_experts = n
        self.top_k = k
        self.gate_up_proj = w13
        self.down_proj = w2

    def expert_containers(self):
        return {"gate_up_proj": self.gate_up_proj, "down_proj": self.down_proj}


def _cpu_allocator():
    """A `TorchStackAllocator` whose HOST stack is ordinary CPU memory.

    This is the exact injection point the pinned arena uses: `host_alloc(shape, dtype) -> Tensor`.
    Recording the handed-out tensors lets a test prove a weight really came from the "arena" and is
    not still the tensor `load_state_dict` produced.
    """
    handed: list = []

    def host_alloc(shape, dtype):
        t = torch.empty(shape, dtype=dtype)
        handed.append(t)
        return t

    return TorchStackAllocator(device="cpu", host_alloc=host_alloc), handed


def _seam(layer, n=None):
    n = n if n is not None else layer.local_num_experts
    from minisgl.weights.granule import spec_for_container

    return MoEWeightSeam(
        "model.layers[0].mlp",
        layer,
        w13_spec=spec_for_container(layer.gate_up_proj, n),
        w2_spec=spec_for_container(layer.down_proj, n),
    )


# ==================================================================================================
# The hot path
# ==================================================================================================


class TestResolve:
    def test_device_placement_is_a_pure_identity(self):
        layer = _FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32))
        seam = _seam(layer)
        seam.bind(StackKind.DEVICE)
        w13, w2 = seam.resolve(layer.gate_up_proj, layer.down_proj)
        assert w13 is layer.gate_up_proj
        assert w2 is layer.down_proj
        assert seam.report.moved_bytes == 0

    def test_host_placement_resolves_to_the_rebound_containers(self):
        layer = _FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32))
        alloc, handed = _cpu_allocator()
        seam = _seam(layer)
        seam.bind(StackKind.HOST, alloc)
        w13, w2 = seam.resolve(layer.gate_up_proj, layer.down_proj)
        assert w13 is layer.gate_up_proj and w2 is layer.down_proj
        # Every component now points at arena storage, not at the load-time tensor.
        arena_ptrs = {t.untyped_storage().data_ptr() for t in handed}
        for c in seam.w13_spec.components:
            assert getattr(w13, c.name).untyped_storage().data_ptr() in arena_ptrs

    def test_a_seam_bound_to_the_wrong_layer_is_caught_loudly(self):
        """The MTP draft head builds its own MoELayer with identically-shaped containers, so a
        colliding structural path would otherwise produce plausible logits and no crash."""
        a = _FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32))
        b = _FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32))
        seam = _seam(a)
        seam.bind(StackKind.DEVICE)
        with pytest.raises(InterpositionError, match="bound to different containers"):
            seam.resolve(b.gate_up_proj, b.down_proj)
        with pytest.raises(InterpositionError):
            seam.resolve(a.gate_up_proj, b.down_proj)


# ==================================================================================================
# The substitution
# ==================================================================================================


class TestBind:
    def test_every_component_moves_and_stays_byte_identical(self):
        """A granule is one expert's slice of EVERY tensor. Moving weights without their scales
        dequantizes expert e against expert f's scale — right shapes, plausible text, no crash."""
        layer = _FakeMoELayer(16, 4, _FakeExperts(16, 64, 128), _FakeExperts(16, 128, 32))
        before = {
            attr: {n: t.clone() for n, t in c.per_expert_tensors().items()}
            for attr, c in layer.expert_containers().items()
        }
        alloc, _ = _cpu_allocator()
        seam = _seam(layer)
        rep = seam.bind(StackKind.HOST, alloc)
        assert rep.components == 6  # 3 components x 2 GEMMs
        for attr, container in layer.expert_containers().items():
            after = container.per_expert_tensors()
            assert set(after) == set(before[attr])
            for name, t in after.items():
                assert torch.equal(t, before[attr][name]), f"{attr}.{name} changed across the move"

    def test_moved_bytes_equals_the_containers_total(self):
        layer = _FakeMoELayer(16, 4, _FakeExperts(16, 64, 128), _FakeExperts(16, 128, 32))
        alloc, _ = _cpu_allocator()
        seam = _seam(layer)
        rep = seam.bind(StackKind.HOST, alloc)
        assert rep.moved_bytes == seam.w13_spec.total_bytes + seam.w2_spec.total_bytes
        assert alloc.bytes_used(StackKind.HOST) == rep.moved_bytes
        assert alloc.bytes_used(StackKind.DEVICE) == 0

    def test_aliases_survive_the_move_instead_of_being_de_aliased(self):
        """`_w_op = weight.view(uint8)` is one storage under two names. A per-name copy would
        de-alias them, after which `dequant()` and the kernel read different memory."""
        layer = _FakeMoELayer(4, 1, _AliasedExperts(4, 8, 16), _AliasedExperts(4, 16, 8))
        alloc, _ = _cpu_allocator()
        seam = _seam(layer)
        # The walk must have merged the two names into ONE component with an alias.
        names = {c.name: c.aliases for c in seam.w13_spec.components}
        assert any(a for a in names.values()), f"expected an alias, got {names}"
        seam.bind(StackKind.HOST, alloc)
        c = layer.gate_up_proj
        assert c.weight.untyped_storage().data_ptr() == c._w_op.untyped_storage().data_ptr()
        # And still genuinely aliased: a write through one name is visible through the other.
        c._w_op[0, 0, 0] = 0x5A
        assert int(c.weight.view(torch.uint8)[0, 0, 0]) == 0x5A

    def test_each_alias_keeps_its_OWN_dtype_and_shape(self):
        """REGRESSION. The rebind reinterpreted every alias with the CANONICAL component's dtype
        and shape, so `_w_op` came back as `f8_e4m3` instead of `uint8` and `_scales_op` came back
        as `(E, N, 1)` instead of `(E, N)`.

        The names stayed aliased, so the existing alias test passed and nothing raised — but a
        kernel binding handed `_w_op` computes `numel()`, `element_size()` and its dtype dispatch
        from the tensor it is given. Reading an fp8 tensor where a uint8 byte-buffer was promised is
        a wrong-bytes bug with no crash, which is the same silent class the aliasing rule exists to
        prevent, entered from the other side.
        """
        layer = _FakeMoELayer(4, 1, _AliasedExperts(4, 8, 16), _AliasedExperts(4, 16, 8))
        before = {
            n: (getattr(layer.gate_up_proj, n).dtype, tuple(getattr(layer.gate_up_proj, n).shape))
            for n in ("weight", "_w_op", "weight_scale", "_scales_op")
        }
        # The fixture must actually present both alias kinds, or this proves nothing.
        assert before["weight"][0] != before["_w_op"][0], "need a dtype-differing alias"
        assert before["weight_scale"][1] != before["_scales_op"][1], "need a shape-differing alias"
        alloc, _ = _cpu_allocator()
        _seam(layer).bind(StackKind.HOST, alloc)
        c = layer.gate_up_proj
        after = {
            n: (getattr(c, n).dtype, tuple(getattr(c, n).shape))
            for n in ("weight", "_w_op", "weight_scale", "_scales_op")
        }
        assert after == before, f"alias identity changed across the bake: {before} -> {after}"
        # ...and they are still one storage, not four copies.
        assert c.weight.data_ptr() == c._w_op.data_ptr()
        assert c.weight_scale.data_ptr() == c._scales_op.data_ptr()

    def test_declared_shared_zeros_are_capacity_but_not_traffic(self):
        """A DECLARED-shared tensor costs capacity but not per-routed-expert traffic.

        Declared, not detected: `granule` downgraded bitwise expert-invariance to a HINT
        (`content_invariant`) because whether THIS rank's shard happens to have identical rows is a
        property of the bytes, and the two TP ranks hold different shards. `_residency_shared` is
        the rank-safe form and the only thing that still lands in `spec.replicated`.
        """
        plain = _FakeExperts(16, 64, 128)
        repl = _SharedZeroExperts(16, 64, 128)
        s_plain = plain.granule_spec()
        s_repl = repl.granule_spec()
        assert s_repl.granule_bytes < s_plain.granule_bytes
        assert s_repl.replicated_bytes > 0
        assert s_repl.total_bytes == s_plain.total_bytes

    def test_detected_invariance_never_changes_what_moves(self):
        """A rank-local bitwise coincidence must not change the component set.

        If it did, the rank whose shard happened to be uniform would move a different set of
        tensors than its peer, with no collective anywhere to notice.
        """
        repl = _FakeExperts(16, 64, 128, replicated_zeros=True)
        spec = repl.granule_spec()
        assert "_zeros_op" in spec.content_invariant
        assert "_zeros_op" in [c.name for c in spec.components]
        assert spec.granule_bytes == _FakeExperts(16, 64, 128).granule_spec().granule_bytes

    def test_bare_tensor_container_rebinds_the_owning_attribute(self):
        """`_UnquantizedMoEMethod.create_experts` returns a raw `torch.empty(E, out, in)` with no
        `__dict__` to rebind. Plan §6.1 rule 6 proposed wrapping it in a new BaseOP subclass and
        editing the two apply/ep_local reads; rebinding the OWNING attribute needs neither."""
        layer = _FakeMoELayer(8, 2, torch.randn(8, 64, 32), torch.randn(8, 32, 32))
        # A CLONE, not a second reference: holding the source alive would (correctly) trip the
        # leaked-source check, which is what `test_a_leaked_alias_is_reported_by_name` covers.
        expect = layer.gate_up_proj.clone()
        original_ptr = layer.gate_up_proj.untyped_storage().data_ptr()
        alloc, handed = _cpu_allocator()
        seam = _seam(layer)
        seam.bind(StackKind.HOST, alloc)
        assert layer.gate_up_proj.untyped_storage().data_ptr() != original_ptr
        assert torch.equal(layer.gate_up_proj, expect)
        assert layer.gate_up_proj.untyped_storage().data_ptr() in {
            t.untyped_storage().data_ptr() for t in handed
        }
        # And the seam resolves to the NEW binding, not the stale one.
        assert seam.resolve(layer.gate_up_proj, layer.down_proj)[0] is layer.gate_up_proj

    def test_host_placement_without_an_arena_refuses(self):
        layer = _FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32))
        seam = _seam(layer)
        with pytest.raises(InterpositionError, match="needs a StackAllocator"):
            seam.bind(StackKind.HOST, None)
        seam2 = _seam(layer)
        with pytest.raises(HostStackUnavailable):
            seam2.bind(StackKind.HOST, TorchStackAllocator(device="cpu"))

    def test_a_refusing_container_is_not_offloaded(self):
        """Plan §6.1 rule 4: a container whose forward materialises the whole (E,N,K) stack would
        stream every byte every step regardless of routing. Refuse at boot, name the reason."""
        layer = _FakeMoELayer(8, 2, _RefusingExperts(8, 64, 128), _RefusingExperts(8, 128, 32))
        alloc, handed = _cpu_allocator()
        seam = _seam(layer)
        with pytest.raises(InterpositionError, match="cannot be offloaded"):
            seam.bind(StackKind.HOST, alloc)
        assert handed == [], "the refusal must fire BEFORE any host page is committed"

    def test_selftest_reads_back_real_bytes(self):
        layer = _FakeMoELayer(32, 4, _FakeExperts(32, 64, 128), _FakeExperts(32, 128, 32))
        alloc, _ = _cpu_allocator()
        seam = _seam(layer)
        rep = seam.bind(StackKind.HOST, alloc)
        # SELFTEST_SAMPLE is 64, which is >= this layer's 6 components, so ALL are verified.
        assert rep.components == 6
        assert rep.verified_components == 6
        assert rep.verified_bytes == rep.moved_bytes

    def test_selftest_can_be_narrowed(self):
        layer = _FakeMoELayer(32, 4, _FakeExperts(32, 64, 128), _FakeExperts(32, 128, 32))
        alloc, _ = _cpu_allocator()
        rep = _seam(layer).bind(StackKind.HOST, alloc, selftest=2)
        assert rep.components == 6 and rep.verified_components == 2

    def test_selftest_catches_a_corrupt_arena(self):
        """The whole point of A1.4: this driver has returned `hipSuccess` over wrong state four
        separate times in Phase 0. Assert on the DATA, never on a return code."""

        class _CorruptAllocator(TorchStackAllocator):
            def alloc_like(self, kind, t):
                out = super().alloc_like(kind, t)
                # Pretend the mapping is wrong: the copy will land, then a "stale page" wins.
                out.__class__ = _PoisonTensor
                return out

        class _PoisonTensor(torch.Tensor):
            def copy_(self, src, *a, **kw):  # type: ignore[override]
                torch.Tensor.copy_(self, src, *a, **kw)
                torch.Tensor.zero_(self)  # the bytes are silently not what was written
                return self

        layer = _FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32))
        alloc = _CorruptAllocator(device="cpu", host_alloc=lambda s, d: torch.empty(s, dtype=d))
        seam = _seam(layer)
        with pytest.raises(InterpositionError, match="read-back does not match the source"):
            seam.bind(StackKind.HOST, alloc)

    def test_selftest_is_BITWISE_not_value_equality(self):
        """REGRESSION, both directions. The read-back used `torch.equal`, which is VALUE equality
        over tensors that are bit patterns.

        FALSE PASS: an arena that stored `-0.0` where `+0.0` was written verified clean, even
        though the gate's entire job is to prove the bytes it wrote are the bytes that are there
        (Phase 0: four cases of the driver returning success over wrong state).
        """

        class _SignFlipAllocator(TorchStackAllocator):
            def alloc_like(self, kind, t):
                out = super().alloc_like(kind, t)
                if out.dtype.is_floating_point:
                    out.__class__ = _SignFlipTensor
                return out

        class _SignFlipTensor(torch.Tensor):
            def copy_(self, src, *a, **kw):  # type: ignore[override]
                torch.Tensor.copy_(self, src, *a, **kw)
                flat = torch.Tensor.reshape(self, -1)
                torch.Tensor.copy_(
                    flat, torch.where(flat == 0, torch.full_like(flat, -0.0), flat)
                )
                return self

        c = _FakeExperts(8, 64, 128)
        c._scales_op[3].zero_()  # one expert's scales are +0.0; the arena will store -0.0
        layer = _FakeMoELayer(8, 2, c, _FakeExperts(8, 128, 32))
        alloc = _SignFlipAllocator(device="cpu", host_alloc=lambda s, d: torch.empty(s, dtype=d))
        with pytest.raises(InterpositionError, match="read-back does not match the source"):
            _seam(layer).bind(StackKind.HOST, alloc)

    def test_a_NaN_in_the_weights_does_not_fail_a_CORRECT_bake(self):
        """REGRESSION, the other direction, and the worse one because it accuses the arena.

        `NaN != NaN` under value equality, so a byte-perfect copy of a stack containing a NaN
        failed with "the arena is wrong. Do not retry." That is reachable on a shipped checkpoint:
        `quant/mxfp4.convert_mxfp4_moe` reports `e8m0_nan_groups`, i.e. an MXFP4 checkpoint can
        carry NaN fp16 group scales straight into `_scales_op`.
        """
        c = _FakeExperts(8, 64, 128)
        c._scales_op[2, 0, 0] = float("nan")
        layer = _FakeMoELayer(8, 2, c, _FakeExperts(8, 128, 32))
        alloc, _ = _cpu_allocator()
        rep = _seam(layer).bind(StackKind.HOST, alloc)
        assert rep.verified_components == rep.components
        got = layer.gate_up_proj._scales_op
        assert bool(torch.isnan(got[2, 0, 0])), "the NaN must have travelled, bit for bit"

    def test_selftest_sample_is_rank_invariant_and_sorted(self):
        """Ranks run in lockstep: a rank-dependent sample makes a real failure look intermittent,
        because only whichever rank drew the bad index reports it."""
        from minisgl.weights.moe_interpose import select_selftest_indices

        a = select_selftest_indices(500, 64)
        b = select_selftest_indices(500, 64)
        assert a == b == sorted(a)
        assert len(a) == 64 and len(set(a)) == 64
        assert select_selftest_indices(6, 64) == list(range(6))  # k >= n -> verify everything
        assert select_selftest_indices(0, 64) == []

    def test_replicated_tensors_move_too(self):
        """They are excluded from `granule_bytes` (every expert reads the same row) but they are
        still resident bytes; leaving them behind makes arena occupancy disagree with the plan."""
        layer = _FakeMoELayer(
            16, 4,
            _SharedZeroExperts(16, 64, 128),
            _SharedZeroExperts(16, 128, 32),
        )
        seam = _seam(layer)
        assert seam.w13_spec.replicated, "the fixture must actually produce a replicated tensor"
        alloc, _ = _cpu_allocator()
        rep = seam.bind(StackKind.HOST, alloc)
        assert rep.moved_bytes == seam.w13_spec.total_bytes + seam.w2_spec.total_bytes
        assert rep.components == 6  # 2 components + 1 replicated, per GEMM
        for name, t in layer.gate_up_proj.per_expert_tensors().items():
            assert t.untyped_storage().data_ptr() == getattr(
                layer.gate_up_proj, name
            ).untyped_storage().data_ptr()

    def test_a_component_inside_a_list_attribute_is_really_rebound(self):
        """REGRESSION. `_assign` only recognised a subscript when the WHOLE dotted segment was
        `[i]`, but the walk emits `f"{name}[{i}]"` — so `_scales[0]` fell through to
        `setattr(container, "_scales[0]", value)`.

        That creates a junk attribute and leaves the real list element pointing at the device
        original: the container keeps reading the pre-offload tensor and the arena row is orphaned.
        No exception, correct numbers, and none of the offload the plan's capacity arithmetic
        assumed. `granule._lookup` is the walker every other consumer resolves with, so the rebind
        is now read back through it.
        """
        layer = _FakeMoELayer(
            8, 2, _ListHeldExperts(8, 64, 128), _ListHeldExperts(8, 128, 32)
        )
        names = [c.name for c in _seam(layer).w13_spec.components]
        assert "_scales[0]" in names, f"the fixture must exercise the subscript form: {names}"
        before = layer.gate_up_proj._scales[0].data_ptr()
        alloc, handed = _cpu_allocator()
        _seam(layer).bind(StackKind.HOST, alloc)
        c = layer.gate_up_proj
        assert not hasattr(c, "_scales[0]"), "a junk attribute was created instead of a rebind"
        assert c._scales[0].data_ptr() != before, "the list element still points at the original"
        assert any(h.data_ptr() == c._scales[0].data_ptr() for h in handed)

    def test_a_surviving_NON_CANONICAL_alias_is_still_caught(self):
        """REGRESSION. The leak proof watched only the canonical component's tensor object.

        An alias is a separate `torch.Tensor` over the same allocation. When the two names are
        SIBLING views of a common base (rather than view-of-canonical, where `_base` happens to
        keep the canonical alive), dropping the canonical frees nothing: the base — and the device
        VRAM the offload exists to reclaim — stays resident behind the surviving alias, and the
        canonical-only weakref reports clean. Every name is watched now.
        """
        layer = _FakeMoELayer(
            8, 2, _TwinViewExperts(8, 64, 128), _TwinViewExperts(8, 128, 32)
        )
        comp = {c.name: c.aliases for c in _seam(layer).w13_spec.components}
        assert comp.get("_w_op") == ("_w_raw",), f"fixture must produce the alias pair: {comp}"
        keep = layer.gate_up_proj._w_raw  # the NON-canonical name, held from outside
        alloc, _ = _cpu_allocator()
        with pytest.raises(InterpositionError, match=r"survived the rebind.*_w_raw"):
            _seam(layer).bind(StackKind.HOST, alloc)
        assert keep is not None

    def test_the_rebind_path_grammar_matches_granule_lookup(self):
        """`_assign` and `granule._lookup` must parse the same string the same way, or the seam
        rebinds a different object than every other consumer reads."""
        from minisgl.weights.granule import _lookup
        from minisgl.weights.moe_interpose import _assign

        class _Holder:
            pass

        inner = _Holder()
        inner.w = torch.zeros(2)
        outer = _Holder()
        outer.kids = [inner]
        outer.by_key = {"a": torch.zeros(2)}
        outer.leaf = torch.zeros(2)
        for path in ("leaf", "kids[0].w", "by_key[a]"):
            new = torch.ones(2)
            _assign(None, "unused", outer, path, new)
            assert _lookup(outer, path) is new, path

    def test_a_leaked_alias_is_reported_by_name(self):
        """Direct evidence, not an inference from allocator stats: a surviving source names the
        exact attribute still pointing at the device original."""
        layer = _FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32))
        alloc, _ = _cpu_allocator()
        seam = _seam(layer)
        keep = layer.gate_up_proj._w_op  # an outside reference the rebind cannot drop
        with pytest.raises(InterpositionError, match="survived the rebind"):
            seam.bind(StackKind.HOST, alloc)
        assert keep is not None


class TestFreeze:
    def test_double_bind_refuses(self):
        layer = _FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32))
        seam = _seam(layer)
        seam.bind(StackKind.DEVICE)
        with pytest.raises(InterpositionError, match="already bound"):
            seam.bind(StackKind.DEVICE)

    def test_nothing_may_be_placed_after_freeze(self):
        """Rule R1. Address stability after boot is what makes graph-capture legality vacuous
        rather than argued."""
        layer = _FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32))
        seam = _seam(layer)
        seam.bind(StackKind.DEVICE)
        seam.freeze()
        assert seam.frozen
        seam2 = _seam(layer)
        seam2._frozen = True
        with pytest.raises(InterpositionError, match="frozen"):
            seam2.bind(StackKind.DEVICE)

    def test_cannot_freeze_before_binding(self):
        layer = _FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32))
        with pytest.raises(InterpositionError, match="before it is bound"):
            _seam(layer).freeze()


class TestLayerWeightsAdapter:
    def test_granule_pair_is_summed_across_both_gemms(self):
        layer = _FakeMoELayer(16, 4, _FakeExperts(16, 64, 128), _FakeExperts(16, 128, 32))
        seam = _seam(layer)
        lw = seam.layer_weights()
        assert lw.granule_bytes == seam.w13_spec.granule_bytes + seam.w2_spec.granule_bytes
        assert lw.resident_bytes == seam.w13_spec.total_bytes + seam.w2_spec.total_bytes
        assert lw.num_experts == 16 and lw.top_k == 4
        assert lw.fingerprint

    def test_fingerprint_tracks_the_component_set(self):
        a = _seam(_FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32)))
        b = _seam(_FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32)))
        assert a.layer_weights().fingerprint == b.layer_weights().fingerprint
        c = _seam(_FakeMoELayer(8, 2, _AliasedExperts(8, 64, 128), _AliasedExperts(8, 128, 32)))
        assert c.layer_weights().fingerprint != a.layer_weights().fingerprint

    def test_priority_passes_through(self):
        seam = _seam(_FakeMoELayer(8, 2, _FakeExperts(8, 64, 128), _FakeExperts(8, 128, 32)))
        assert seam.layer_weights(priority=7).priority == 7
        assert build_layer_weights([seam], priority=[3])[0].priority == 3


# ==================================================================================================
# Discovery + the binder, against a REAL MoELayer
# ==================================================================================================


@pytest.fixture()
def tp1():
    """Single-rank TP with EP off — enough to construct a real `MoELayer` on CPU."""
    from minisgl.distributed import info

    old_tp, old_dp, old_ep, old_eot = (
        info._TP_INFO, info._DP_INFO, info._ENABLE_EP, info._EP_OVER_TP,
    )
    info._TP_INFO = info.DistributedInfo(0, 1)
    info._DP_INFO = None
    info._ENABLE_EP = False
    info._EP_OVER_TP = False
    yield
    info._TP_INFO, info._DP_INFO, info._ENABLE_EP, info._EP_OVER_TP = (
        old_tp, old_dp, old_ep, old_eot,
    )


def _real_moe(n=8, k=2, hidden=32, inter=16):
    from minisgl.layers.moe import MoELayer

    return MoELayer(num_experts=n, top_k=k, hidden_size=hidden, intermediate_size=inter)


def _quantize(model):
    """Swap the unquantized bare-tensor containers for multi-component ones.

    A `quant=None` MoELayer holds a raw `torch.empty(E, out, in)` — a single-component container.
    Substituting a `_FakeExperts` pair gives the end-to-end tests the shape every SHIPPED quantized
    format actually presents: packed weight + per-group scales + zero-points, all of which must
    travel together.
    """
    for _, layer in discover_moe_layers(model):
        n = layer.local_num_experts
        layer.gate_up_proj = _FakeExperts(n, 32, 32)
        layer.down_proj = _FakeExperts(n, 32, 32)


class _Block(BaseOP):
    def __init__(self, moe):
        self.mlp = moe

    def forward(self, *a, **kw):  # pragma: no cover
        raise RuntimeError


class _Model(BaseOP):
    def __init__(self, blocks, draft=None):
        self.layers = OPList(blocks)
        if draft is not None:
            self.draft_head = draft

    def forward(self, *a, **kw):  # pragma: no cover
        raise RuntimeError


class TestDiscovery:
    def test_paths_are_structural_and_include_the_draft_head(self, tp1):
        """A construction counter renumbers every layer once the MTP draft head builds its own
        MoELayer; the path must come from the module tree instead."""
        model = _Model([_Block(_real_moe()) for _ in range(3)], draft=_Block(_real_moe()))
        found = discover_moe_layers(model)
        paths = [p for p, _ in found]
        assert paths == [
            "layers.0.mlp",
            "layers.1.mlp",
            "layers.2.mlp",
            "draft_head.mlp",
        ]

    def test_paths_use_the_state_dict_grammar_not_the_object_graph(self, tp1):
        """REGRESSION. The walk used to number `OPList` members `op_list[i]`, mirroring the TENSOR
        walker. But the string that has to match is the PLAN's, and
        `weights/plan.py::moe_layer_shapes` builds `model.layers.{lid}.mlp.experts` — the checkpoint
        namespace, i.e. `BaseOP.state_dict`/`OPList.state_dict`'s prefix. Every family in this repo
        holds its decoder layers in an `OPList`, so the old grammar made `bind_plan`'s
        plan-vs-model comparison fail on every real model: weight offload could not boot at any TP,
        and no unit test caught it because they all planned from seam-derived paths.

        Asserted against `state_dict()` itself rather than against a literal, so the two grammars
        cannot drift apart again without this failing.
        """
        model = _Model([_Block(_real_moe()) for _ in range(3)], draft=_Block(_real_moe()))
        keys = list(model.state_dict())
        assert keys, "the unquantized MoELayer must publish its bare stacked tensors"
        for path, _ in discover_moe_layers(model):
            assert any(k.startswith(path + ".") for k in keys), (
                f"discovered layer path {path!r} is not a prefix of any state_dict key — the plan "
                f"resolver builds its paths in the state_dict/checkpoint grammar, so a path that "
                f"is not one can never match a plan. keys={keys!r}"
            )
        assert "layers.0.mlp.gate_up_proj" in keys

    def test_a_model_with_no_moe_yields_nothing(self, tp1):
        assert discover_moe_layers(_Model([])) == []

    def test_attach_installs_the_seam_on_every_layer(self, tp1):
        model = _Model([_Block(_real_moe()) for _ in range(2)])
        seams = attach_seams(model)
        assert len(seams) == 2
        for path, layer in discover_moe_layers(model):
            assert layer._weight_offload is not None
            assert layer._weight_offload.path == path
        detach_seams(seams)
        for _, layer in discover_moe_layers(model):
            assert layer._weight_offload is None

    def test_attach_refuses_meta_tensors(self, tp1):
        """The model is built on the meta device; a granule derived there has untrustworthy
        aliasing and expert-invariance because every meta tensor reports `data_ptr() == 0`."""
        model = _Model([_Block(_real_moe())])
        blk = model.layers.op_list[0]
        blk.mlp.gate_up_proj = blk.mlp.gate_up_proj.to("meta")
        blk.mlp.down_proj = blk.mlp.down_proj.to("meta")
        with pytest.raises((InterpositionError, GranuleError), match="META|meta"):
            attach_seams(model)


class TestBindPlan:
    def test_end_to_end_layer_granular_bind(self, tp1):
        model = _Model([_Block(_real_moe()) for _ in range(4)])
        _quantize(model)
        seams = attach_seams(model)
        layers = build_layer_weights(seams)
        one = layers[0].resident_bytes
        plan = plan_layer_granular(layers, device_budget_bytes=one * 2)
        alloc, handed = _cpu_allocator()
        out = bind_plan(seams, plan, alloc, log=False)
        assert out.device_layers == 2 and out.host_layers == 2
        assert out.moved_bytes == one * 2
        assert out.plan_digest == plan.digest()
        assert all(s.frozen for s in seams)
        # The device-resident layers were not touched at all: two host layers x two GEMMs x three
        # components each.
        assert len(handed) == 2 * 2 * 3
        assert not any(isinstance(s_, str) for s_ in ())
        dev_paths = [p.path for p in plan.placements if p.kind is StackKind.DEVICE]
        for s in seams:
            assert (s.report.moved_bytes == 0) == (s.path in dev_paths)
        assert "2 host" in seam_summary(seams)

    def test_all_host_plan_moves_everything(self, tp1):
        model = _Model([_Block(_real_moe()) for _ in range(3)])
        _quantize(model)
        seams = attach_seams(model)
        plan = plan_layer_granular(build_layer_weights(seams), device_budget_bytes=0)
        alloc, _ = _cpu_allocator()
        out = bind_plan(seams, plan, alloc, log=False)
        assert out.host_layers == 3 and out.device_layers == 0

    def test_no_op_plan_needs_no_allocator(self, tp1):
        model = _Model([_Block(_real_moe()) for _ in range(3)])
        seams = attach_seams(model)
        layers = build_layer_weights(seams)
        plan = plan_layer_granular(layers, device_budget_bytes=sum(l.resident_bytes for l in layers))
        assert plan.is_empty
        out = bind_plan(seams, plan, None, log=False)
        assert out.moved_bytes == 0 and out.host_layers == 0

    def test_a_layer_the_plan_excludes_by_policy_is_bound_device_not_refused(self, tp1):
        """REGRESSION. `bind_plan` used to demand plan-paths == seam-paths EXACTLY. The resolver
        deliberately excludes layers: `plan.OFFLOAD_MTP_HEAD = False` skips the MTP draft head, and
        the draft head's checkpoint namespace (`mtp.layers.0.mlp.experts`) is a string literal in
        the model that a module walk cannot reconstruct anyway. So set equality was unsatisfiable on
        every MTP-capable checkpoint, and — because an unplanned seam was also never bound —
        `freeze()` would then raise "cannot be frozen before it is bound".

        The excluded layer must be bound DEVICE (zero bytes moved) so `resolve()` still guards it:
        the draft head's containers are identically shaped to a backbone layer's, which is exactly
        the collision the identity check exists for.
        """
        model = _Model([_Block(_real_moe()) for _ in range(3)], draft=_Block(_real_moe()))
        _quantize(model)
        seams = attach_seams(model)
        by_path = {s.path: s for s in seams}
        assert "draft_head.mlp" in by_path
        backbone = [s for s in seams if s.path != "draft_head.mlp"]
        plan = plan_layer_granular(build_layer_weights(backbone), device_budget_bytes=0)
        alloc, _ = _cpu_allocator()

        out = bind_plan(seams, plan, alloc, log=False)

        assert out.host_layers == 3
        assert out.device_layers == 1
        assert any("draft_head.mlp" in n for n in out.notes)
        draft = by_path["draft_head.mlp"]
        assert draft.bound and draft.kind is StackKind.DEVICE
        assert draft.report.moved_bytes == 0
        assert all(s.frozen for s in seams)
        # ...and it is still guarded: the draft head's own containers resolve, a backbone layer's do not.
        draft.resolve(draft._layer.gate_up_proj, draft._layer.down_proj)
        other = by_path["layers.0.mlp"]
        with pytest.raises(InterpositionError, match="bound to different containers"):
            draft.resolve(other._layer.gate_up_proj, other._layer.down_proj)

    def test_a_planned_layer_the_model_does_not_have_is_still_fatal(self, tp1):
        """The other direction stays a hard failure: the plan spent budget on bytes that are not
        there, so the arena size and the KV pool are both about a different model."""
        model = _Model([_Block(_real_moe()) for _ in range(3)])
        seams = attach_seams(model)
        layers = list(build_layer_weights(seams))
        from dataclasses import replace

        layers.append(replace(layers[0], path="model.layers.99.mlp.experts"))
        plan = plan_layer_granular(layers, device_budget_bytes=0)
        with pytest.raises(InterpositionError, match="does not have"):
            bind_plan(seams, plan, None, log=False)


class TestCaptureFence:
    """`freeze()` is the last moment before graph capture at which a moved address can be checked.

    The whole capture argument is "addresses are constants by the time capture runs". Anything that
    rebinds a container between `bind()` and capture makes the captured graph hold a pointer the
    seam's ledger does not describe, and `resolve()` would only notice on the first forward — which
    under capture is AFTER the graph already baked the address in.
    """

    def test_freeze_catches_a_container_rebound_after_bind(self, tp1):
        model = _Model([_Block(_real_moe())])
        _quantize(model)
        seams = attach_seams(model)
        plan = plan_layer_granular(build_layer_weights(seams), device_budget_bytes=0)
        alloc, _ = _cpu_allocator()
        out = bind_plan(seams, plan, alloc, freeze=False, log=False)
        assert out.host_layers == 1
        layer = seams[0]._layer
        layer.gate_up_proj = _FakeExperts(layer.local_num_experts, 32, 32)  # a late repack
        with pytest.raises(InterpositionError, match="rebound between bind\\(\\) and freeze\\(\\)"):
            seams[0].freeze()

    def test_detach_after_freeze_is_refused(self, tp1):
        model = _Model([_Block(_real_moe())])
        seams = attach_seams(model)
        plan = plan_layer_granular(build_layer_weights(seams), device_budget_bytes=0)
        alloc, _ = _cpu_allocator()
        bind_plan(seams, plan, alloc, log=False)
        assert seams[0].frozen
        with pytest.raises(InterpositionError, match="frozen"):
            seams[0].detach()
        assert seams[0]._layer._weight_offload is seams[0]
        with pytest.raises(InterpositionError, match="frozen"):
            seams[0].attach()


class TestForwardSeamIsWired:
    """The seam must actually be consulted by `MoELayer.forward`, not merely exist."""

    def test_forward_passes_the_resolved_containers_to_the_method(self, tp1):
        layer = _real_moe(n=4, k=1, hidden=8, inter=8)
        seen = {}

        class _SpyMethod:
            supports_ep = False
            needs_precomputed_route = False
            supports_producer_actquant = False

            def apply(self, w13, w2, hidden_states, **kw):
                seen["w13"], seen["w2"] = w13, w2
                return torch.zeros_like(hidden_states)

        layer._moe_method = _SpyMethod()
        hs = torch.randn(2, 8)
        layer.forward(hs, torch.randn(2, 4))
        assert seen["w13"] is layer.gate_up_proj

        calls = []

        class _SpySeam:
            def resolve(self, w13, w2):
                calls.append((w13, w2))
                return w13, w2

        layer._weight_offload = _SpySeam()
        layer.forward(hs, torch.randn(2, 4))
        assert len(calls) == 1
        assert calls[0][0] is layer.gate_up_proj and calls[0][1] is layer.down_proj
        layer._weight_offload = None

    def test_forward_is_untouched_when_no_seam_is_attached(self, tp1):
        layer = _real_moe(n=4, k=1, hidden=8, inter=8)
        assert layer._weight_offload is None
        assert "_weight_offload" not in vars(layer), (
            "the seam must be a CLASS attribute so it stays out of the BaseOP state walk"
        )

    def test_seam_is_invisible_to_the_state_dict_walk(self, tp1):
        model = _Model([_Block(_real_moe())])
        seams = attach_seams(model)
        assert seams[0]._layer._weight_offload is seams[0]
        # `_`-prefixed names are skipped by BaseOP.state_dict, so attaching must not add keys.
        keys = set(model.state_dict())
        detach_seams(seams)
        assert set(model.state_dict()) == keys


# ==================================================================================================
# GENERALITY: what a NEW model or quant format has to implement (answer: nothing)
# ==================================================================================================


class _RenamedContainerLayer:
    """A layer whose two expert containers are NOT called `gate_up_proj` / `down_proj`.

    Nothing shipped renames them today. It is here because the seam's contract is "read the names
    off `expert_containers()`", and a module that also hardcodes the literals satisfies the contract
    on every existing model while silently violating it for the first one that does not — which is
    the whole failure mode the "a new format implements nothing" rule exists to prevent.
    """

    _weight_offload = None

    def __init__(self, n, k, up, down):
        self.local_num_experts = n
        self.top_k = k
        self.fused_up = up
        self.proj_down = down

    def expert_containers(self):
        return {"fused_up": self.fused_up, "proj_down": self.proj_down}


def _renamed_seam(layer):
    from minisgl.weights.granule import spec_for_container

    n = layer.local_num_experts
    return MoEWeightSeam(
        "model.layers[0].mlp",
        layer,
        w13_spec=spec_for_container(layer.fused_up, n),
        w2_spec=spec_for_container(layer.proj_down, n),
    )


class TestContainerNamesComeFromTheLayer:
    def test_bind_and_freeze_work_on_declared_names(self):
        """REGRESSION. `freeze()` read `self._layer.gate_up_proj` / `.down_proj` literally, so a
        layer that declares any other container names raised `AttributeError` at the freeze step —
        after the bake had already moved every byte onto the host stack."""
        layer = _RenamedContainerLayer(
            8, 2, _FakeExperts(8, 32, 64), _FakeExperts(8, 64, 32)
        )
        seam = _renamed_seam(layer)
        alloc, handed = _cpu_allocator()
        rep = seam.bind(StackKind.HOST, alloc)
        assert rep.moved_bytes > 0 and handed
        seam.freeze()  # used to raise AttributeError here
        assert seam.frozen
        # And the substitution really landed on the declared attributes.
        moved = {id(t) for t in handed}
        assert id(layer.fused_up._w_op) in moved and id(layer.proj_down._w_op) in moved

    def test_freeze_catches_a_rebind_through_the_declared_names(self):
        layer = _RenamedContainerLayer(
            8, 2, _FakeExperts(8, 32, 64), _FakeExperts(8, 64, 32)
        )
        seam = _renamed_seam(layer)
        seam.bind(StackKind.DEVICE)
        layer.proj_down = _FakeExperts(8, 64, 32)  # a late post_load / second bake
        with pytest.raises(InterpositionError, match="rebound between bind"):
            seam.freeze()

    def test_freeze_catches_a_layer_that_redeclared_its_container_set(self):
        layer = _RenamedContainerLayer(
            8, 2, _FakeExperts(8, 32, 64), _FakeExperts(8, 64, 32)
        )
        seam = _renamed_seam(layer)
        seam.bind(StackKind.DEVICE)
        layer.expert_containers = lambda: {"proj_down": layer.proj_down}
        with pytest.raises(InterpositionError, match="now declares containers"):
            seam.freeze()


class TestDetachRestoresThePristineLayer:
    def test_detach_removes_the_instance_attribute_entirely(self, tp1):
        """REGRESSION. `detach()` assigned `None` over the CLASS attribute, which creates a real
        entry in `vars(layer)`. `MoELayer` declares `_weight_offload` at class level precisely so a
        non-offloaded layer keeps nothing in the `__dict__` that `BaseOP.state_dict` /
        `load_state_dict` / `post_load` and `granule._iter_tensors` all walk — so the teardown path
        was silently undoing the property the declaration exists to guarantee."""
        layer = _real_moe(n=4, k=1, hidden=8, inter=8)
        assert "_weight_offload" not in vars(layer)
        seam = _seam(layer)
        seam.attach()
        assert "_weight_offload" in vars(layer)
        seam.detach()
        assert "_weight_offload" not in vars(layer), (
            "a detached layer must be byte-for-byte the shape an un-offloaded serve has"
        )
        assert layer._weight_offload is None  # still resolves, via the class attribute


class TestEPTrafficIsPerRank:
    """`num_experts` on the seam is the EP-LOCAL shard; `MoELayer.top_k` stays GLOBAL."""

    def _ep_layer(self, *, enable_ep, ep_size, local_n=8, global_k=4):
        layer = _FakeMoELayer(local_n, global_k, _FakeExperts(local_n, 32, 64),
                              _FakeExperts(local_n, 64, 32))
        layer.enable_ep = enable_ep
        layer.ep_size = ep_size
        return layer

    def test_seam_prices_only_this_ranks_routed_slots(self):
        """REGRESSION. `layer_weights()` passed the GLOBAL `top_k` next to the LOCAL expert count,
        so on an EP=2 rank it priced 2x the granules the rank actually reads — while
        `plan.size_planned_layers_from_model` corrected for EP, giving the two entry points
        different `LayerWeights`, different projections and different `OffloadPlan.digest()`es for
        one model on one rank."""
        seam = _seam(self._ep_layer(enable_ep=True, ep_size=2))
        assert seam.top_k == 4  # what the layer holds
        assert seam.top_k_local == 2  # what this rank computes
        assert seam.layer_weights().top_k == 2
        assert seam.layer_weights().active_bytes(1) == 2 * seam.w13_spec.granule_bytes + \
            2 * seam.w2_spec.granule_bytes

    def test_it_agrees_with_the_engine_planners_formula(self):
        """Both call `placement.ep_local_top_k`; assert they cannot drift apart again."""
        from minisgl.weights.placement import ep_local_top_k

        layer = self._ep_layer(enable_ep=True, ep_size=2)
        seam = _seam(layer)
        assert seam.top_k_local == ep_local_top_k(
            int(layer.top_k), ep_size_of(layer), int(layer.local_num_experts)
        )

    def test_ep_size_is_ignored_when_the_layer_is_not_actually_sharded(self):
        """The trap: `ep_size` is the PROCESS-wide value and stays set on a layer whose quant
        method vetoed EP (`supports_ep` is False for RXF and for unquantized experts) or which was
        built `force_no_ep=True` (the MTP draft head). Such a layer is fully REPLICATED, so reading
        `ep_size` without `enable_ep` would halve its traffic figure."""
        assert ep_size_of(self._ep_layer(enable_ep=False, ep_size=2)) == 1
        assert ep_size_of(self._ep_layer(enable_ep=True, ep_size=2)) == 2
        seam = _seam(self._ep_layer(enable_ep=False, ep_size=2))
        assert seam.top_k_local == seam.top_k == 4

    def test_a_layer_that_declares_nothing_is_ep1(self):
        """A duck-typed container-only double, and any future layer type that is never sharded,
        must need no declaration at all."""
        assert ep_size_of(_FakeMoELayer(8, 2, None, None)) == 1


class TestRefusingFormatsAreExcludedNotPlanned:
    def test_build_layer_weights_skips_a_container_that_refuses_host_residency(self, tp1):
        """REGRESSION, and a format-coverage gap. `build_layer_weights` never asked
        `offload_refusal`, so it planned ZAYA's fp8 experts under `MINISGL_ZAYA_OLDMOE` /
        `MINISGL_ZAYA_W8A16` (whose forwards read EVERY expert) onto HOST and then died in `bind()`.
        Weight offload could not boot at all on those knobs through this entry point, while
        `plan.size_planned_layers_from_model` — which has always filtered — handled them."""
        model = _Model([_Block(_real_moe()) for _ in range(3)])
        layers = [layer for _, layer in discover_moe_layers(model)]
        for layer in layers:
            n = layer.local_num_experts
            layer.gate_up_proj = _FakeExperts(n, 32, 32)
            layer.down_proj = _FakeExperts(n, 32, 32)
        # The middle layer's FORMAT declares it cannot be host-resident. Both containers of a
        # layer are the same class (`assert_granule_pair_consistent` enforces it), which is exactly
        # how the shipped case looks: the refusal is a property of the quant method, so it applies
        # to both GEMMs of the layer at once.
        n1 = layers[1].local_num_experts
        layers[1].gate_up_proj = _RefusingExperts(n1, 32, 32)
        layers[1].down_proj = _RefusingExperts(n1, 32, 32)

        seams = attach_seams(model)
        assert seams[1].offload_refusal() is not None
        assert [s.offload_refusal() for s in (seams[0], seams[2])] == [None, None]

        skipped: list[str] = []
        lw = build_layer_weights(seams, skipped=skipped)
        assert [x.path for x in lw] == [seams[0].path, seams[2].path]
        assert len(skipped) == 1 and "refuses host residency" in skipped[0]

    def test_the_refusing_layer_still_boots_as_device_resident(self, tp1):
        """Excluding is a correct COMPOSITION, not a silent drop: `bind_plan` binds every discovered
        seam the plan does not name to DEVICE (zero bytes) and names it in `notes` — the same
        treatment the policy-excluded MTP draft head gets. The all-host plan must therefore BOOT."""
        model = _Model([_Block(_real_moe()) for _ in range(3)])
        layers = [layer for _, layer in discover_moe_layers(model)]
        for layer in layers:
            n = layer.local_num_experts
            layer.gate_up_proj = _FakeExperts(n, 32, 32)
            layer.down_proj = _FakeExperts(n, 32, 32)
        n1 = layers[1].local_num_experts
        layers[1].gate_up_proj = _RefusingExperts(n1, 32, 32)
        layers[1].down_proj = _RefusingExperts(n1, 32, 32)

        seams = attach_seams(model)
        plan = plan_layer_granular(build_layer_weights(seams), device_budget_bytes=0)
        alloc, _ = _cpu_allocator()
        out = bind_plan(seams, plan, alloc, log=False)  # used to raise InterpositionError
        assert out.host_layers == 2 and out.device_layers == 1
        assert any("not in the plan" in n for n in out.notes)
        assert seams[1].kind is StackKind.DEVICE

    def test_binding_a_refusing_layer_to_host_directly_is_still_refused(self, tp1):
        """The planner filters; the bake still fences. A caller that hand-builds a plan naming a
        refusing layer must fail loudly before a page is committed, not stream the whole stack."""
        layer = _FakeMoELayer(8, 2, _RefusingExperts(8, 32, 64), _RefusingExperts(8, 64, 32))
        seam = _seam(layer)
        alloc, handed = _cpu_allocator()
        with pytest.raises(InterpositionError, match="cannot be offloaded"):
            seam.bind(StackKind.HOST, alloc)
        assert not handed, "no arena page may be committed for a refused layer"


# =================================================================================================
# REGRESSION (memory-accounting/boot lens, 2026-09-03): the read-back self-test's DEVICE scratch.
#
# `torch.equal` on CUDA is `self.eq(other).all()` — it materialises one bool PER BYTE on the device,
# through the ordinary caching allocator (the arena `MemPool` is only live inside
# `ArenaMemPool.use()`, which the bake has long exited). `_bake` runs it after `post_load()` and
# BEFORE the rebind drops a single device original, i.e. at the un-offloaded model's peak VRAM. A
# fused w13 weight stack is a ~1 GiB row on the target shape, so the un-sliced form asked a 16 GB
# card for a ~1 GiB scratch it does not have, up to `SELFTEST_SAMPLE` times, and failed as a bare
# CUDA OOM inside a self-test with nothing in the message pointing at the arena.
# =================================================================================================


class TestSelftestCompareIsBounded:
    def _spy(self, monkeypatch):
        seen = []
        real = torch.equal

        def spy(a, b):
            seen.append(a.numel())
            return real(a, b)

        monkeypatch.setattr(moe_interpose.torch, "equal", spy)
        return seen

    def test_no_single_comparison_exceeds_the_chunk(self, monkeypatch):
        n = 5 * moe_interpose.SELFTEST_COMPARE_CHUNK_BYTES + 12345
        a = torch.randint(0, 255, (n,), dtype=torch.uint8)
        seen = self._spy(monkeypatch)
        assert moe_interpose._bitwise_equal(a.clone(), a)
        assert seen, "the comparison must actually run"
        assert max(seen) <= moe_interpose.SELFTEST_COMPARE_CHUNK_BYTES
        assert sum(seen) == n, "every byte is still compared — the gate is not weakened"

    def test_a_flipped_bit_in_the_LAST_partial_slice_is_still_caught(self):
        """The slice loop's tail is where an off-by-one hides, and a missed tail turns the whole
        gate into a green light over an arena that is wrong at the end of every row."""
        n = 2 * moe_interpose.SELFTEST_COMPARE_CHUNK_BYTES + 7
        src = torch.zeros(n, dtype=torch.uint8)
        dst = src.clone()
        dst[-1] = 1
        assert not moe_interpose._bitwise_equal(dst, src)
        dst[-1] = 0
        dst[moe_interpose.SELFTEST_COMPARE_CHUNK_BYTES] = 1  # first byte of the second slice
        assert not moe_interpose._bitwise_equal(dst, src)

    def test_still_bytewise_so_NaN_passes_and_signed_zero_fails(self):
        """Unchanged by the slicing, and both directions matter: a shipped MXFP4 checkpoint can
        carry NaN fp16 group scales (`quant/mxfp4.convert_mxfp4_moe`), so a value comparison would
        accuse a correct arena; and `-0.0 == +0.0`, so it would also pass a flipped sign bit."""
        nan = torch.full((1024,), float("nan"), dtype=torch.float16)
        assert moe_interpose._bitwise_equal(nan.clone(), nan)
        pos = torch.zeros(1024, dtype=torch.float16)
        neg = pos.clone()
        neg[3] = -0.0
        assert not moe_interpose._bitwise_equal(neg, pos)

    def test_an_empty_row_compares_equal_and_allocates_nothing(self, monkeypatch):
        seen = self._spy(monkeypatch)
        empty = torch.zeros(0, dtype=torch.uint8)
        assert moe_interpose._bitwise_equal(empty, empty.clone())
        assert seen == []


# ==================================================================================================
# The seam PROOF — is the offload arm in the serving path?
# ==================================================================================================


class TestProveSeamResidency:
    """`prove_seam_residency` asks the LIVE MODEL what every other gate asks the arena or the plan.

    The gap it closes is not hypothetical for this repo: the PLE bring-up shipped a set of
    individually-green components whose seam to the engine did not exist, so the served path was 0%
    functional and every test passed. Each case below is one way that shape can recur in the offload
    path — and every one of them is silent without this check, because `MoELayer._weight_offload`
    defaults to None at the CLASS, so an unbound layer runs happily off its device containers while
    the plan, the arena reservation and the KV budget were all computed as though it had not.
    """

    def _bound_model(self, budget=0):
        model = _Model([_Block(_real_moe()) for _ in range(3)])
        _quantize(model)
        seams = attach_seams(model)
        plan = plan_layer_granular(build_layer_weights(seams), device_budget_bytes=budget)
        alloc, handed = _cpu_allocator()
        bind_plan(seams, plan, alloc, log=False)
        return model, seams, handed

    def test_a_fully_bound_model_proves_and_counts_what_it_walked(self, tp1):
        model, seams, handed = self._bound_model()
        owned = {t.data_ptr() for t in handed}
        proof = moe_interpose.prove_seam_residency(
            model, seams, owns_pointer=lambda p, n=0: p in owned
        )
        assert (proof.moe_layers, proof.host_layers, proof.device_layers) == (3, 3, 0)
        # 3 layers x 2 containers (w13, w2) x 3 components (packed weight, scales, zeros).
        assert proof.pointer_checked and proof.host_tensors == 18
        assert proof.host_bytes == sum(t.numel() * t.element_size() for t in handed)

    def test_the_proof_is_downgraded_not_faked_without_an_owns_pointer(self, tp1):
        """`pointer_checked=False` is the honest report of a walk that skipped its own evidence.
        A caller that gates on the object rather than on this flag gets a structural claim believing
        it has a residency one."""
        model, seams, _ = self._bound_model()
        assert not moe_interpose.prove_seam_residency(model, seams).pointer_checked

    def test_a_layer_whose_seam_was_detached_is_caught(self, tp1):
        """THE DISPATCH REGRESSION. A detached layer reads its device containers and produces
        perfectly finite output; nothing in the arena, the plan or the byte ledger can see it."""
        model, seams, _ = self._bound_model()
        for s in seams:
            s._frozen = False  # detach() correctly refuses a frozen seam; simulate the pre-freeze slip
        seams[1].detach()
        with pytest.raises(InterpositionError, match="carry no weight-offload seam"):
            moe_interpose.prove_seam_residency(model, seams)

    def test_a_seam_bound_to_a_layer_that_left_the_tree_is_caught(self, tp1):
        """The arena is populated, the ledger balances, and the layer the seam describes is not the
        one the engine will call forward() on."""
        model, seams, _ = self._bound_model()
        seams.append(
            moe_interpose.attach_seam("layers.99.mlp", _Block(_real_moe()).mlp)
        )
        with pytest.raises(InterpositionError, match="NOT reachable from the model"):
            moe_interpose.prove_seam_residency(model, seams)

    def test_a_host_tensor_outside_the_arena_is_caught(self, tp1):
        """The `hipMalloc` fallback, seen from the model side: the layer says HOST, the tensor its
        kernels will read is in VRAM. Double error against the KV budget — the bytes are resident
        AND `model_memory_correction` removes them as though they were not."""
        model, seams, _ = self._bound_model()
        with pytest.raises(InterpositionError, match="NOT inside the pinned arena"):
            moe_interpose.prove_seam_residency(model, seams, owns_pointer=lambda p, n=0: False)

    def test_a_device_tensor_inside_the_arena_is_caught(self, tp1):
        """The mirror case, and it is what makes the host claim non-vacuous: an `owns_pointer` that
        answers True for everything would otherwise 'prove' any model at all."""
        model, seams, _ = self._bound_model(budget=1 << 40)  # everything stays on the device
        assert all(s.kind is StackKind.DEVICE for s in seams)
        with pytest.raises(InterpositionError, match="lives INSIDE the host arena"):
            moe_interpose.prove_seam_residency(
                model, seams, owns_pointer=lambda p, n=0: True, require_host_layers=False
            )

    def test_an_all_device_bind_refuses_by_default_on_an_enabled_session(self, tp1):
        """`require_host_layers` exists because "the arena pinned and nothing is host-resident" is
        exactly what a vanished offload arm looks like, and it is otherwise indistinguishable from a
        correct all-device serve."""
        model, seams, _ = self._bound_model(budget=1 << 40)
        with pytest.raises(InterpositionError, match="not one MoE layer .* is host-resident"):
            moe_interpose.prove_seam_residency(model, seams)
        assert moe_interpose.prove_seam_residency(
            model, seams, require_host_layers=False
        ).device_layers == 3

    def test_resolve_publishes_the_ledger_line_and_assert_identity_does_not(self, tp1):
        """The ledger line's ONLY value is that it separates "the bake ran" from "a forward read the
        arena". The boot-time proof therefore has to use `assert_identity`, or it would emit the
        line itself and destroy the distinction."""
        from minisgl import _hip_engage

        model, seams, _ = self._bound_model()
        seam = seams[0]
        layer = dict(discover_moe_layers(model))[seam.path]
        pair = (layer.gate_up_proj, layer.down_proj)
        _hip_engage._seen.discard(seam._engage)
        seam.assert_identity(*pair)
        assert seam._engage not in _hip_engage._seen
        seam.resolve(*pair)
        assert seam._engage in _hip_engage._seen

    def test_live_tensors_walks_the_same_set_the_bake_moved(self, tp1):
        """One enumeration, two consumers. A proof that walked a different tensor set from the bake
        would be a proof about weights the forward does not read — vacuous in the direction that
        reads as a pass."""
        model, seams, handed = self._bound_model()
        walked = [t for s in seams for _n, t in s.live_tensors()]
        assert {t.data_ptr() for t in walked} == {t.data_ptr() for t in handed}
