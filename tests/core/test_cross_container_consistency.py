"""One layer's two GEMMs must be READ the same way, and BAKED onto the same card.

Two silent-wrong-numbers holes that only differ in which axis they cross:

  * DECODE POLICY. `_GroupedCompressedTensorsExperts.post_load` samples w13's nibble histogram and
    w2's independently — two different tensors (w13 is `2*inter x hidden`, w2 is `hidden x inter`,
    and under TP they are split on different axes), so they really are two samples and can resolve
    opposite ways. One GEMM then dequantizes as `q + 8` and the other as two's-complement: every
    weight of that GEMM off by 8 quanta, right shapes, right dtypes, no kernel fault, plausible
    text. The comparison existed, but only inside `MoELayer.granule_specs()`, which is called by
    `weights/moe_interpose.attach_seams` and by nothing else — i.e. it ran ONLY on a serve that
    offloads weights. It is now in `MoELayer.post_load`, unconditionally.

  * BAKE DEVICE. `_validate` compared `dst.device.type` against `src.device.type` and stopped there,
    so a row carved from an arena pinned for `cuda:0` would accept a weight living on `cuda:1`. The
    arena's `hipHostGetDevicePointer` mapping is registered for ONE device; that copy is a peer
    access nothing established, on a box whose two cards are not even the same PCIe generation.

GPU-FREE: CPU tensors and fake containers throughout.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.weights.granule import (  # noqa: E402
    GranuleError,
    assert_decode_policy_agrees,
    decode_policy,
)


class _CtLike:
    """A container that declares the compressed-tensors decode decision, like the real one."""

    _granule_policy = ("_ct_sign.uint4b8",)

    def __init__(self, uint4b8: bool | None, margin: float = 0.5) -> None:
        if uint4b8 is not None:
            # A real `CtSignConvention` is a NamedTuple; only the dotted path matters here.
            self._ct_sign = type("Conv", (), {"uint4b8": uint4b8, "margin": margin})()


class _NoPolicy:
    """Every non-compressed-tensors container: declares nothing."""


class TestDecodePolicyReadout:
    def test_the_recorded_value_is_the_decision_not_the_whole_convention(self):
        """`margin`/`sampled_words` legitimately differ between w13 and w2 and between two ranks'
        shards. Recording them would make every pair look like a disagreement."""
        assert decode_policy(_CtLike(True, margin=0.31)) == (("_ct_sign.uint4b8", "True"),)
        assert decode_policy(_CtLike(True, margin=0.87)) == (("_ct_sign.uint4b8", "True"),)

    def test_a_container_with_no_policy_reads_empty(self):
        assert decode_policy(_NoPolicy()) == ()

    def test_an_undecided_container_reads_none_rather_than_raising(self):
        """`_granule_policy` is also read before `post_load` has run (a meta container)."""
        assert decode_policy(_CtLike(None)) == (("_ct_sign.uint4b8", "None"),)


class TestDecodePolicyAgreement:
    def test_agreeing_containers_pass(self):
        assert_decode_policy_agrees(
            {"gate_up_proj": decode_policy(_CtLike(True)), "down_proj": decode_policy(_CtLike(True))}
        )

    def test_no_policy_at_all_passes(self):
        assert_decode_policy_agrees(
            {"gate_up_proj": decode_policy(_NoPolicy()), "down_proj": decode_policy(_NoPolicy())}
        )

    def test_disagreement_raises_and_names_both_sides(self):
        """THE regression. w13 resolved uint4b8, w2 resolved two's-complement: every weight of the
        down projection off by 8 quanta, with nothing else in the descriptor moving."""
        with pytest.raises(GranuleError) as ei:
            assert_decode_policy_agrees(
                {
                    "gate_up_proj": decode_policy(_CtLike(True)),
                    "down_proj": decode_policy(_CtLike(False)),
                },
                where="MoELayer w13/w2",
            )
        msg = str(ei.value)
        assert "gate_up_proj" in msg and "down_proj" in msg
        assert "True" in msg and "False" in msg
        assert "MoELayer w13/w2" in msg
        # The remedy must be in the message: this is not something to average.
        assert "quantization_config" in msg

    def test_one_container_undecided_is_also_a_disagreement(self):
        """A `post_load` that ran on one container and not the other is the same hazard: the
        undecided one has not been transformed at all."""
        with pytest.raises(GranuleError):
            assert_decode_policy_agrees(
                {"a": decode_policy(_CtLike(True)), "b": decode_policy(_CtLike(None))}
            )


class TestMoELayerPostLoadRunsIt:
    """The check has to be on the path EVERY serve takes, not only an offloading one."""

    def test_post_load_refuses_a_mismatched_pair(self, monkeypatch):
        # `minisgl.layers.*` hard-requires the baked `tail_hip` .so, and a CPU-only test image's
        # kernels are older than the mounted python tree. The gate is exactly the opt-out for that.
        monkeypatch.setenv("MINISGL_TAIL_HIP", "0")
        moe = pytest.importorskip("minisgl.layers.moe")

        class _PairLayer(moe.MoELayer):
            # MoELayer.__init__ builds real quantized containers; the cross-check needs only the
            # two attributes and `expert_containers()`, both of which it inherits.
            def __init__(self, a, b):  # noqa: D107 - deliberately skips MoELayer.__init__
                self.gate_up_proj = a
                self.down_proj = b

        _PairLayer(_CtLike(True), _CtLike(True)).post_load()  # agreeing pair: no raise
        with pytest.raises(GranuleError, match="decode policy"):
            _PairLayer(_CtLike(True), _CtLike(False)).post_load()


class TestBakeRefusesACrossCardCopy:
    def test_validate_rejects_a_device_index_mismatch(self):
        """`dst.device.type != src.device.type` let `cuda:0 -> cuda:1` through: the arena's mapping
        is registered for one device, so that copy is a peer access nothing established, and on this
        box the two cards are not even the same PCIe generation."""
        mi = pytest.importorskip("minisgl.weights.moe_interpose")

        class _FakeTensor:
            def __init__(self, index):
                self.device = torch.device("cuda", index)
                self.dtype = torch.int32
                self.shape = (4, 8)

            def is_contiguous(self):
                return True

            def numel(self):
                return 32

            def element_size(self):
                return 4

        item = mi._BakeItem(
            name="layer.gate_up_proj._w_op",
            layer=None,
            owner_attr="gate_up_proj",
            container=None,
            attrs=(),
            src=_FakeTensor(0),
            dst=_FakeTensor(1),
        )
        with pytest.raises(mi.InterpositionError, match="arena is pinned and mapped for ONE device"):
            mi._validate(item)

    def test_validate_accepts_a_matching_index(self):
        mi = pytest.importorskip("minisgl.weights.moe_interpose")
        t = torch.zeros((4, 8), dtype=torch.int32)
        item = mi._BakeItem(
            name="cpu-bake",
            layer=None,
            owner_attr="gate_up_proj",
            container=None,
            attrs=(),
            src=t,
            dst=torch.zeros_like(t),
        )
        assert mi._validate(item) == t.numel() * t.element_size()
