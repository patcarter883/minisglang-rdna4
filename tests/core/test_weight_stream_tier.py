"""The THIRD weight tier — aliasing, poisoning, the staged-layer fence and the refusals.

NO GPU. Everything is CPU tensors, because everything this file tests is bookkeeping: which tensor
object a container points at, which rows are NaN, which layer the buffers are holding, and which
configurations must raise instead of running. The one thing it cannot cover is that the MoE kernel
reads the rows this tier writes — that is `tests/qwen4exp_offload_serve_test.py --stream-layers` on
a leased card, and its `weight_offload.moe_resolve[*]` ledger line is what proves it.

WHY THE POISON TESTS ARE THE IMPORTANT ONES. Every stream layer aliases ONE buffer set, so a row
this tier fails to stage holds a DIFFERENT LAYER'S expert — real weights of the right shape and
magnitude. That produces fluent, plausible, wrong text and no error. `arm()`'s NaN fill and the
stale-row poison in `stage()` are what turn that into NaN logits, so a regression in either is
exactly the class of bug that ships silently.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.weights.granule import ExpertContainer  # noqa: E402
from minisgl.weights.stream_tier import (  # noqa: E402
    ExpertStreamTier,
    StreamTierError,
    layer_index_of_path,
)

E, N, K = 8, 16, 64


class _Experts(ExpertContainer):
    """`_GroupedNvFp4Experts` after `post_load`: packed int32 codes + a group-major fp16 scale."""

    def __init__(self, seed: int):
        g = torch.Generator().manual_seed(seed)
        self._num_experts = E
        self._w_op = torch.randint(
            -(2**31), 2**31 - 1, (E, N, K // 8), dtype=torch.int32, generator=g
        )
        self._scales_op = torch.randn(E, K // 16, N, dtype=torch.float16, generator=g)

    def forward(self, *a, **kw):  # pragma: no cover - storage container
        raise RuntimeError("storage container")


class _Layer:
    """Duck-typed `MoELayer`: the seam's contract plus the two attributes the route needs."""

    _weight_offload = None
    enable_ep = False

    def __init__(self, seed: int):
        self.local_num_experts = E
        self.num_experts = E
        self.top_k = 2
        self.renormalize = True
        self.gate_up_proj = _Experts(seed)
        self.down_proj = _Experts(seed + 1000)
        self.calls: list = []

    def expert_containers(self):
        return {"gate_up_proj": self.gate_up_proj, "down_proj": self.down_proj}

    def granule_specs(self, **kw):
        from minisgl.weights.granule import spec_for_container

        return {n: spec_for_container(c, E, **kw) for n, c in self.expert_containers().items()}

    def forward(self, hidden_states, router_logits=None, **kw):
        # Records the STATE OF THE BUFFERS at the moment a forward would read them, which is the
        # only place the tier's correctness is observable without a kernel.
        self.calls.append(
            (
                self.gate_up_proj._scales_op.isnan().all(dim=(1, 2)).tolist(),
                self.gate_up_proj._w_op.data_ptr(),
            )
        )
        return hidden_states


class _Source:
    """An `ExpertRowSource` that writes a value derived from (layer, expert), so a mis-staged row
    is identifiable rather than merely different."""

    def __init__(self):
        self.bytes_read = 0
        self.calls: list = []
        self.closed = False

    def gather(self, layer, expert_ids):
        ids = list(expert_ids)
        self.calls.append((layer, tuple(ids)))
        self.bytes_read += len(ids) * 4
        out = {}
        for attr, off in (("gate_up_proj", 0), ("down_proj", 500)):
            w = torch.stack(
                [torch.full((N, K // 8), layer * 100 + e + off, dtype=torch.int32) for e in ids]
            )
            s = torch.stack(
                [
                    torch.full((K // 16, N), float(layer * 100 + e + off), dtype=torch.float16)
                    for e in ids
                ]
            )
            out[attr] = {"_w_op": w, "_scales_op": s}
        return out

    def close(self):
        self.closed = True


def _tier(n_layers=3, **kw):
    src = _Source()
    tier = ExpertStreamTier(src, device=torch.device("cpu"), **kw)
    layers = []
    for lid in range(n_layers):
        op = _Layer(seed=lid)
        tier.adopt(f"model.layers.{lid}.mlp.experts", op, op.granule_specs())
        layers.append(op)
    return tier, src, layers


# ==================================================================================================


class TestPath:
    def test_layer_index_is_structural(self):
        assert layer_index_of_path("model.layers.31.mlp.experts") == 31
        assert layer_index_of_path("a.b.layers.0.c") == 0

    def test_a_path_with_no_index_raises_rather_than_defaulting(self):
        # Defaulting to 0 here would put every unnumbered MoE layer in one tier slot.
        with pytest.raises(StreamTierError):
            layer_index_of_path("model.mtp.mlp.experts")


class TestAliasing:
    def test_every_layer_ends_up_on_the_donor_s_allocation(self):
        tier, _, layers = _tier(4)
        assert tier.donor == 0
        for attr in ("gate_up_proj", "down_proj"):
            ptrs = {getattr(l, attr)._w_op.data_ptr() for l in layers}
            assert len(ptrs) == 1, f"{attr} is not aliased: {ptrs}"
            ptrs = {getattr(l, attr)._scales_op.data_ptr() for l in layers}
            assert len(ptrs) == 1

    def test_the_shared_set_is_ONE_layer_s_worth_of_bytes(self):
        tier, _, layers = _tier(6)
        one = sum(
            t.numel() * t.element_size()
            for c in layers[0].expert_containers().values()
            for t in (c._w_op, c._scales_op)
        )
        assert tier.stats()["stream_shared_bytes"] == one

    def test_a_shape_mismatch_is_refused_not_aliased(self):
        tier, _, _ = _tier(1)
        odd = _Layer(seed=9)
        odd.gate_up_proj._w_op = torch.zeros(E, N, K // 4, dtype=torch.int32)
        with pytest.raises(StreamTierError, match="donor"):
            tier.adopt("model.layers.9.mlp.experts", odd, odd.granule_specs())

    def test_adopting_the_same_layer_twice_is_refused(self):
        tier, _, layers = _tier(1)
        with pytest.raises(StreamTierError, match="twice"):
            tier.adopt("model.layers.0.mlp.experts", layers[0], layers[0].granule_specs())

    def test_ep_sharded_layer_is_refused(self):
        """The tier writes GLOBAL expert ids at the same index into a LOCAL shard. Silent."""
        src = _Source()
        tier = ExpertStreamTier(src, device=torch.device("cpu"))
        op = _Layer(seed=0)
        op.enable_ep = True
        with pytest.raises(StreamTierError, match="EP"):
            tier.adopt("model.layers.0.mlp.experts", op, op.granule_specs())


class TestPoison:
    def test_arm_poisons_every_row(self):
        tier, _, layers = _tier(2)
        assert not layers[0].gate_up_proj._scales_op.isnan().any()
        tier.arm()
        assert layers[0].gate_up_proj._scales_op.isnan().all()
        assert layers[1].down_proj._scales_op.isnan().all()

    def test_stage_before_arm_is_refused(self):
        tier, _, _ = _tier(2)
        with pytest.raises(StreamTierError, match="arm"):
            tier.stage(0, [1, 2])

    def test_only_the_staged_rows_are_live(self):
        tier, _, layers = _tier(2)
        tier.arm()
        tier.stage(1, [3, 5])
        nan_rows = layers[0].gate_up_proj._scales_op.isnan().all(dim=(1, 2))
        assert [i for i, v in enumerate(nan_rows.tolist()) if not v] == [3, 5]
        # And the live rows carry THIS layer's values, keyed (layer, expert).
        assert layers[0].gate_up_proj._w_op[3].unique().tolist() == [1 * 100 + 3]
        assert layers[0].down_proj._w_op[5].unique().tolist() == [1 * 100 + 5 + 500]

    def test_rows_the_next_layer_does_not_reuse_are_re_poisoned(self):
        """The whole point: layer 2 must not read layer 1's expert 3 out of the shared buffers."""
        tier, _, layers = _tier(3)
        tier.arm()
        tier.stage(1, [3, 5])
        tier.stage(2, [5, 6])
        nan_rows = layers[0].gate_up_proj._scales_op.isnan().all(dim=(1, 2)).tolist()
        assert [i for i, v in enumerate(nan_rows) if not v] == [5, 6]
        assert nan_rows[3] is True, "layer 1's expert 3 survived into layer 2's staging"
        assert layers[0].gate_up_proj._w_op[5].unique().tolist() == [2 * 100 + 5]

    def test_an_empty_route_is_refused(self):
        tier, _, _ = _tier(2)
        tier.arm()
        with pytest.raises(StreamTierError, match="empty route"):
            tier.stage(0, [])


class TestGatherBatching:
    def test_the_transient_is_bounded_by_gather_batch(self):
        """A prefill routes far more than `top_k` DISTINCT experts (the union over its tokens), and
        an unbounded stack of them OOM'd a real 48-layer boot. The batch is that bound."""
        tier, src, _ = _tier(2, gather_batch=3)
        tier.arm()
        tier.stage(0, list(range(8)))
        assert [len(ids) for _l, ids in src.calls] == [3, 3, 2]
        assert sorted(i for _l, ids in src.calls for i in ids) == list(range(8))

    def test_every_row_is_still_written_exactly_once(self):
        tier, _, layers = _tier(2, gather_batch=3)
        tier.arm()
        tier.stage(1, list(range(8)))
        assert not layers[0].gate_up_proj._scales_op.isnan().any()
        for e in range(8):
            assert layers[0].gate_up_proj._w_op[e].unique().tolist() == [100 + e]


class TestStagedLayerFence:
    def test_assert_staged_catches_the_wrong_layer(self):
        tier, _, _ = _tier(3)
        tier.arm()
        tier.stage(1, [0])
        tier.assert_staged(1)
        with pytest.raises(StreamTierError, match="hold layer 1 but layer 2"):
            tier.assert_staged(2)

    def test_a_source_failure_leaves_nothing_claimed_live(self):
        """A half-written buffer set that still claims a layer is the worst possible state."""

        class _Boom(_Source):
            def gather(self, layer, expert_ids):
                raise RuntimeError("disk")

        tier = ExpertStreamTier(_Boom(), device=torch.device("cpu"))
        op = _Layer(seed=0)
        tier.adopt("model.layers.0.mlp.experts", op, op.granule_specs())
        tier.arm()
        with pytest.raises(RuntimeError, match="disk"):
            tier.stage(0, [1])
        assert tier.staged_layer == -1
        with pytest.raises(StreamTierError):
            tier.assert_staged(0)


class TestHooks:
    def test_the_hook_stages_this_layer_before_its_forward_runs(self):
        tier, _, layers = _tier(3)
        tier.arm()
        tier.install_hooks()
        for lid, op in enumerate(layers):
            op.forward(torch.zeros(1, 4), router_logits=None, topk_ids=torch.tensor([[lid % E]]))
            live, _ptr = op.calls[-1]
            assert [i for i, v in enumerate(live) if not v] == [lid % E]
            assert tier.staged_layer == lid

    def test_install_hooks_before_arm_is_refused(self):
        tier, _, _ = _tier(2)
        with pytest.raises(StreamTierError, match="arm"):
            tier.install_hooks()

    def test_a_route_with_neither_logits_nor_ids_is_refused(self):
        tier, _, layers = _tier(1)
        tier.arm()
        with pytest.raises(StreamTierError, match="router_logits"):
            tier.route_ids(layers[0], None, None)


class TestStats:
    def test_counters_reflect_what_was_staged(self):
        tier, src, _ = _tier(2)
        tier.arm()
        tier.stage(0, [1, 2, 3])
        tier.stage(1, [4])
        st = tier.stats()
        assert st["stream_stages"] == 2
        assert st["stream_experts_staged"] == 4
        assert st["stream_max_experts_per_staging"] == 3
        assert st["stream_bytes_read"] == src.bytes_read
        assert st["stream_layers"] == [0, 1]

    def test_close_closes_the_source(self):
        tier, src, _ = _tier(1)
        tier.close()
        assert src.closed
