"""COMPONENT-MAJOR placement arithmetic. Pure integers; no tensors, no device, no GPU.

THE BUG THIS FILE EXISTS TO PREVENT. `FrameLayout` (reused here as the granule DESCRIPTOR) is
frame-major: it interleaves a granule's components inside one frame. Every MoE kernel, by contrast,
does per-component implicit-contiguous pointer arithmetic — `w_base = w_rep + e*(N/16)*ktiles*32`,
`ws_e = w_scales + e*G*N`, `wq_e = w_fp8 + (long)e*N*K`. Pack the weights frame-major and every
component's row stride silently becomes `frame_bytes`; the kernel then reads a perfectly well-formed
wrong byte range for every expert but the zeroth. No fault, no assert, plausible text.

So the tests below assert the two layouts DISAGREE, not merely that the component-major one is
self-consistent — a guard that could pass while frame-major happened to coincide is not a guard.
"""

from __future__ import annotations

import pytest
import torch
from minisgl.weights.granule import (
    DEFAULT_ALIGN,
    ExpertComponent,
    GranuleSpec,
    derive_granule_spec,
    plan_component_major,
)


def _spec(n: int = 8) -> GranuleSpec:
    """A three-component int4 granule with DELIBERATELY unequal component sizes, so a frame-major
    stride can never coincide with a component-major one."""
    return GranuleSpec(
        kind="_FakeInt4Experts",
        num_experts=n,
        components=(
            ExpertComponent("_w_op", torch.int32, (64, 16), stacked_shape=(n, 64, 16)),
            ExpertComponent("_scales_op", torch.float16, (4, 64), stacked_shape=(n, 4, 64)),
            ExpertComponent("_zeros_op", torch.int32, (4, 8), stacked_shape=(n, 4, 8)),
        ),
    )


def test_row_stride_is_the_components_own_row_not_the_frame():
    spec = _spec()
    plan = plan_component_major(spec)
    frame_bytes = spec.layout.nbytes  # what a frame-major slab would use for EVERY component
    for comp in spec.components:
        assert plan.row_bytes(comp.name) == comp.nbytes
        for e in range(spec.num_granules - 1):
            step = plan.row_offset(comp.name, e + 1) - plan.row_offset(comp.name, e)
            assert step == comp.nbytes
            # THE GUARD: the frame-major stride is a different number for every component here.
            assert step != frame_bytes, (
                f"{comp.name}: component-major stride coincides with the frame stride, so this test "
                f"cannot distinguish the two layouts — pick component sizes that differ"
            )


def test_frame_major_would_place_expert_rows_elsewhere():
    """Concretely: expert 1's scales sit at a DIFFERENT byte under the two layouts."""
    spec = _spec()
    plan = plan_component_major(spec)
    layout = spec.layout
    frame_off = dict(zip([c.name for c in spec.components], layout.offsets))
    for e in (1, 2, 7):
        for comp in spec.components:
            frame_major = e * layout.nbytes + frame_off[comp.name]
            assert plan.row_offset(comp.name, e) != frame_major


def test_components_are_disjoint_aligned_and_ordered():
    spec = _spec()
    plan = plan_component_major(spec)
    prev_end = 0
    for c in plan.components:
        assert c.offset % DEFAULT_ALIGN == 0
        assert c.offset >= prev_end, "component slabs overlap"
        prev_end = c.end
    assert plan.nbytes >= prev_end
    assert plan.payload_bytes == spec.stacked_bytes
    # Padding is per-component (a start alignment), never per-row.
    assert 0 <= plan.pad_bytes < DEFAULT_ALIGN * len(plan.components)


def test_granule_offsets_are_not_contiguous():
    """A caller hoping for one memcpy per expert is asking for frame-major and must not get it."""
    spec = _spec()
    plan = plan_component_major(spec)
    offs = plan.granule_offsets(3)
    ordered = [offs[c.name] for c in spec.components]
    sizes = [c.nbytes for c in spec.components]
    assert ordered[1] != ordered[0] + sizes[0]
    assert ordered[2] != ordered[1] + sizes[1]


def test_every_expert_row_of_every_component_is_disjoint():
    spec = _spec(n=5)
    plan = plan_component_major(spec)
    spans = []
    for c in plan.components:
        for e in range(c.num_rows):
            o = plan.row_offset(c.name, e)
            spans.append((o, o + c.row_bytes, f"{c.name}[{e}]"))
    spans.sort()
    for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
        assert b0 >= a1, f"{an} [{a0},{a1}) overlaps {bn} [{b0},{b1})"


def test_plan_is_deterministic():
    """Two ranks must compute the same placement independently — no collective, no agreement step."""
    a = plan_component_major(_spec())
    b = plan_component_major(_spec())
    assert a == b


def test_align_must_be_a_power_of_two():
    for bad in (0, -256, 100):
        with pytest.raises(ValueError):
            plan_component_major(_spec(), align=bad)


def test_align_one_removes_all_padding():
    plan = plan_component_major(_spec(), align=1)
    assert plan.pad_bytes == 0
    assert plan.nbytes == plan.payload_bytes


def test_out_of_range_expert_and_unknown_component_raise():
    plan = plan_component_major(_spec(n=3))
    with pytest.raises(IndexError):
        plan.row_offset("_w_op", 3)
    with pytest.raises(KeyError):
        plan.row_offset("_not_a_component", 0)


def test_plan_matches_a_real_containers_tensor_offsets():
    """End-to-end: for a container whose components ARE contiguous stacks, the planned row offsets
    reproduce the real `t[e]` addresses once each component's slab base is subtracted."""

    class _Fake:
        def __init__(self, n):
            self._w_op = torch.zeros(n, 64, 16, dtype=torch.int32)
            self._scales_op = torch.zeros(n, 4, 64, dtype=torch.float16)
            self._w_op[:] = torch.arange(n).view(n, 1, 1)  # distinct rows: not expert-invariant
            self._scales_op[:] = torch.arange(n).view(n, 1, 1)

    n = 6
    c = _Fake(n)
    spec = derive_granule_spec(c, n)
    plan = plan_component_major(spec)
    stacks = spec.stacked_tensors(c)
    for comp in spec.components:
        base = stacks[comp.name].data_ptr()
        for e in range(n):
            real_rel = stacks[comp.name][e].data_ptr() - base
            planned_rel = plan.row_offset(comp.name, e) - plan.base_offset(comp.name)
            assert real_rel == planned_rel
