"""The compressed-tensors int4 packing decision: decided ONCE, over the WHOLE stack.

The old detector sampled the leading 65536 int32 words of whatever tensor it was handed. On a
512-expert stack that is ~0.5% of the bytes, all of it from expert 0 — and, worse, it made the answer
a function of WHICH SLICE the caller held. A chunked per-expert-range repack (weight offload Stage B,
plan §5.2) would therefore be free to reach different answers for different chunks, XOR-corrupting
one contiguous block of experts and leaving the rest correct. That produces plausible text, no crash,
and nothing downstream can detect it.

GPU-FREE: CPU tensors throughout.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.quant.method import (  # noqa: E402
    _CT_SIGN_SAMPLE_BLOCKS,
    _CT_SIGN_SAMPLE_WORDS,
    CtSignConvention,
    _ct_sign_sample,
    apply_ct_sign,
    ct_packed_sign_convention,
)
from minisgl.layers.base import BaseOP  # noqa: E402
from minisgl.quant.method import (  # noqa: E402
    CtSignRankDivergence,
    collect_ct_sign_decisions,
    verify_ct_sign_across_ranks,
)


def _stack(e: int, words_per_expert: int, nibble: int) -> torch.Tensor:
    """An (E, words) int32 stack whose every nibble is `nibble`."""
    byte = (nibble << 4) | nibble
    t = torch.empty((e, words_per_expert), dtype=torch.int32)
    t.view(torch.uint8).fill_(byte)
    return t


class TestDecision:
    def test_uint4b8_stack_passes_through(self):
        conv = ct_packed_sign_convention(_stack(4, 4096, 8))
        assert conv.uint4b8

    def test_twos_complement_stack_is_flagged_for_xor(self):
        conv = ct_packed_sign_convention(_stack(4, 4096, 0))
        assert not conv.uint4b8

    def test_ambiguous_histogram_raises_instead_of_coin_flipping(self):
        # Exactly as many 8s as 0s. The old `counts[8] >= counts[0]` silently resolved this toward
        # pass-through; a wrong resolution XORs every nibble of the stack.
        half = _stack(2, 4096, 8)
        other = _stack(2, 4096, 0)
        t = torch.cat([half, other], dim=0)
        with pytest.raises(ValueError, match="cannot decide"):
            ct_packed_sign_convention(t, name="ambiguous")

    def test_refusal_message_carries_the_evidence(self):
        t = torch.cat([_stack(2, 4096, 8), _stack(2, 4096, 0)], dim=0)
        with pytest.raises(ValueError) as ei:
            ct_packed_sign_convention(t, name="w13")
        msg = str(ei.value)
        assert "w13" in msg and "stride" in msg and "quantization_config" in msg

    def test_empty_tensor_raises(self):
        with pytest.raises(ValueError, match="empty packed tensor"):
            ct_packed_sign_convention(torch.empty((0,), dtype=torch.int32))


class TestFullStackSampling:
    def test_the_sample_spans_every_expert(self):
        """THE regression this change exists for.

        Expert 0 is packed uint4b8; experts 1..15 are two's-complement. A prefix sample sees only
        expert 0 and answers "pass through", which then leaves 15/16 of the stack un-XORed.
        """
        words = _CT_SIGN_SAMPLE_WORDS // 4  # each expert alone is bigger than the sample budget
        stack = torch.cat(
            [_stack(1, words, 8)] + [_stack(15, words, 0)],
            dim=0,
        )
        assert not ct_packed_sign_convention(stack).uint4b8
        # ...and the prefix the old detector would have looked at says the opposite, which is what
        # makes this a silent corruption rather than a crash.
        assert ct_packed_sign_convention(stack.flatten()[:words].contiguous()).uint4b8

    def test_the_sampled_index_range_is_the_whole_tensor(self):
        """`stride * sampled_words >= numel` was a proxy, and a proxy that a block sample passes
        vacuously. Assert the thing itself: the first sampled row is 0 and the last is n-1."""
        raw = _stack(8, _CT_SIGN_SAMPLE_WORDS, 8).contiguous().view(torch.uint8).reshape(-1, 4)
        sample, blocks, stride = _ct_sign_sample(raw)
        n = raw.shape[0]
        assert blocks == _CT_SIGN_SAMPLE_BLOCKS and stride > 1
        block = _CT_SIGN_SAMPLE_WORDS // blocks
        starts = [(i * (n - block)) // (blocks - 1) for i in range(blocks)]
        assert starts[0] == 0 and starts[-1] + block == n
        assert sample.shape[0] == blocks * block

    def test_small_tensor_samples_everything_with_stride_one(self):
        conv = ct_packed_sign_convention(_stack(1, 64, 8))
        assert conv.stride == 1 and conv.sampled_words == 64 and conv.blocks == 1

    def test_the_sample_is_not_a_prefix_of_the_tensor(self):
        """REGRESSION: `raw[::n // 65536][:65536]` truncates, so `65536 < n < 131072` sampled a
        PREFIX of as little as 50% — the exact prefix bias the full-stack rewrite claims to remove.

        n = 98304 int32 words is a TP=2-sharded dense compressed-tensors linear (N=768, K=1024), a
        shipped shape, and it was covered 66.7%. Here the leading 40000 words are two's-complement
        and the remaining 58304 are uint4b8, so the prefix says XOR with a 22% margin (confidently
        wrong) and the whole tensor says pass through with an 18.6% margin. Confidently wrong is the
        bad case: it XORs every nibble of a stack that needed no XOR, with no error anywhere.
        """
        t = torch.cat([_stack(1, 40000, 0), _stack(1, 58304, 8)], dim=1)
        assert t.numel() == 98304
        assert ct_packed_sign_convention(t).uint4b8
        # ...and the prefix the truncating sampler read says the opposite.
        assert not ct_packed_sign_convention(t.flatten()[:_CT_SIGN_SAMPLE_WORDS].contiguous()).uint4b8

    def test_the_sample_is_not_one_packed_column(self):
        """REGRESSION, and the one that bites the shipped MoE shapes.

        A lattice stride of `n // 65536` came out an exact multiple of the packed row length on every
        real stack — w13 at E=128/N=1536/K=4096 gives stride 1536 = 3*(K/8), w2 gives 768 = 8*(K/8) —
        so a sample that spanned all 128 experts still read ONLY packed column 0: the same 8 input
        channels of the model, deciding the convention for the whole checkpoint.

        This stack reproduces the arithmetic exactly (Kp=16, n=2097152, n//65536 = 32 = 2*Kp) and
        makes column 0 atypical: it is all-zero nibbles while the other 15/16 of the stack is
        uint4b8. A column-0 lattice therefore votes two's-complement unanimously and XORs the whole
        stack; a block sample sees 15 uint4b8 columns for every aliased one and gets it right.
        """
        E, N, Kp = 16, 8192, 16
        t = _stack(E * N, Kp, 8).reshape(E, N, Kp).contiguous()
        t.view(torch.uint8).reshape(E, N, Kp, 4)[:, :, 0, :] = 0x00  # column 0 -> nibbles 0
        assert t.numel() // _CT_SIGN_SAMPLE_WORDS % Kp == 0, "the aliasing precondition"
        assert ct_packed_sign_convention(t).uint4b8
        # ...and the column-0 lattice the old sampler walked says the opposite, unanimously.
        assert not ct_packed_sign_convention(t[:, :, :1].contiguous()).uint4b8

    def test_decision_is_deterministic(self):
        # No RNG anywhere: every TP rank must reach the same answer bit-for-bit, or two ranks hold
        # differently-signed copies of the same weights.
        t = _stack(4, 8192, 0)
        assert ct_packed_sign_convention(t) == ct_packed_sign_convention(t)

    def test_result_is_slice_invariant_for_a_uniform_stack(self):
        # The property Stage B relies on when it CAN see the whole stack; when it cannot, it must
        # thread the CtSignConvention instead. Both halves of a uniform stack agree.
        t = _stack(8, 4096, 0)
        assert (
            ct_packed_sign_convention(t[:4]).uint4b8 == ct_packed_sign_convention(t[4:]).uint4b8
        )


class TestApply:
    def test_pass_through_is_the_same_object(self):
        t = _stack(2, 64, 8)
        assert apply_ct_sign(t, CtSignConvention(True, 1.0, 64, 1)) is t

    def test_xor_flips_every_nibbles_top_bit(self):
        t = _stack(2, 64, 0)
        out = apply_ct_sign(t, CtSignConvention(False, 1.0, 64, 1))
        assert torch.equal(out, _stack(2, 64, 8))

    def test_xor_is_an_involution(self):
        t = _stack(2, 64, 3)
        conv = CtSignConvention(False, 1.0, 64, 1)
        assert torch.equal(apply_ct_sign(apply_ct_sign(t, conv), conv), t)

    def test_output_is_contiguous_int32(self):
        out = apply_ct_sign(_stack(2, 64, 0), CtSignConvention(False, 1.0, 64, 1))
        assert out.dtype == torch.int32 and out.is_contiguous()

    def test_weight_and_zeros_share_one_convention(self):
        """A weight and its zero-point come from ONE quantizer.

        `scale*(W_u - Z_u) == scale*(q - zp)` holds only if both were moved into the same domain, so
        the transform is applied from a single decided `conv` rather than re-detected per tensor —
        a zero-point stack is tiny and its own histogram is not a reliable second vote.
        """
        conv = ct_packed_sign_convention(_stack(4, 4096, 0))
        w = apply_ct_sign(_stack(4, 4096, 0), conv)
        z = apply_ct_sign(_stack(4, 16, 0), conv)
        assert torch.equal(w, _stack(4, 4096, 8))
        assert torch.equal(z, _stack(4, 16, 8))


class TestCrossRankHazard:
    """Determinism of the FUNCTION is not agreement between RANKS, and the difference is a bug.

    `ct_packed_sign_convention`'s docstring used to argue "no RNG, so every TP rank agrees
    bit-for-bit". That is a non-sequitur: the ranks do not evaluate the function on the same input.
    `MoELayer.__init__` gives plain-TP rank r a `w13` of shape (E, 2*I/tp, H) and EP-over-TP rank r
    experts [r*E/ep, (r+1)*E/ep). Each rank runs `post_load` on its OWN container and decides
    independently, so on a mixed-packing stack the two ranks put the same logical weights in
    different sign domains -- and every symptom of that is plausible text.

    These tests pin the hazard rather than fix it: closing it needs a cross-rank compare on the CPU
    group at post_load, or a checkpoint-level decision threaded into every container (which is also
    what a chunked Stage-B repack needs). Neither is built, and until one is, this class is the
    record that the safety claim was not true.
    """

    def test_ep_shards_of_a_mixed_stack_decide_differently(self):
        """The EP shape: rank 0 owns experts 0..3, rank 1 owns 4..7, packed differently."""
        stack = torch.cat([_stack(4, 4096, 8), _stack(4, 4096, 0)], dim=0)
        rank0, rank1 = stack[:4], stack[4:]
        assert ct_packed_sign_convention(rank0).uint4b8
        assert not ct_packed_sign_convention(rank1).uint4b8
        # Same function, no RNG, opposite answers -- so "deterministic" proves nothing here. The
        # whole stack is the tie that `TestDecision` shows the detector refuses, but neither rank
        # ever sees the whole stack, so neither refusal fires.
        with pytest.raises(ValueError, match="cannot decide"):
            ct_packed_sign_convention(stack)

    def test_plain_tp_column_shards_decide_differently(self):
        """The plain-TP shape: the split is along the OUTPUT rows of every expert, not across
        experts, so a stack that is mixed per-output-row diverges the same way."""
        e, w = 4, 4096
        full = torch.cat([_stack(e, w, 8), _stack(e, w, 0)], dim=1)  # (E, 2w): left 8s, right 0s
        assert ct_packed_sign_convention(full[:, :w].contiguous()).uint4b8
        assert not ct_packed_sign_convention(full[:, w:].contiguous()).uint4b8

    def test_determinism_is_only_about_the_same_input(self):
        """What determinism DOES buy, stated so the two are not conflated again."""
        t = _stack(4, 8192, 0)
        assert ct_packed_sign_convention(t) == ct_packed_sign_convention(t.clone())


class TestContainerDtypeGenerality:
    """The decision and the transform must be properties of the BYTES, not of `int32`.

    Both functions used to hardcode a 32-bit container: the detector masked `& 0xFFFFFFFF` and
    unpacked `range(8)` nibbles per element, and `apply_ct_sign` reinterpreted its result as
    `torch.int32`. That is a dtype pinned into a shared core, which this repo's rules forbid — and
    it is not hypothetical: `_GroupedMxFp4Experts` and `_GroupedNvFp4Experts`
    in `layers/moe.py` already ship 4-bit weights in `uint8` containers, and a new int4 weight format
    is supposed to be a loader policy on the existing core rather than a new kernel or a new
    detector. Hand the old code one of those and it read nibble positions 2..7 of every element as
    zero, inflating `counts[0]` by 6/8 of the sample and deciding "two's-complement" with a large,
    entirely fake margin — then XORed a stack that needed no XOR.
    """

    @staticmethod
    def _u8_stack(rows: int, cols: int, nibble: int) -> torch.Tensor:
        byte = (nibble << 4) | nibble
        return torch.full((rows, cols), byte, dtype=torch.uint8)

    def test_uint8_packed_uint4b8_stack_is_not_misread_as_twos_complement(self):
        # THE REGRESSION. Every nibble is 8, so the honest answer is uint4b8 -> pass through. The
        # int32-hardcoded detector answered `uint4b8=False` here and XORed the whole stack.
        conv = ct_packed_sign_convention(self._u8_stack(8, 4096, 8), name="mxfp4-shaped")
        assert conv.uint4b8, "a uint8-packed q+8 stack must decide pass-through"

    def test_uint8_packed_twos_complement_stack_still_decides_xor(self):
        conv = ct_packed_sign_convention(self._u8_stack(8, 4096, 0), name="u8")
        assert not conv.uint4b8

    def test_uint8_ambiguous_stack_still_refuses(self):
        t = torch.cat([self._u8_stack(4, 4096, 8), self._u8_stack(4, 4096, 0)], dim=0)
        with pytest.raises(ValueError, match="cannot decide"):
            ct_packed_sign_convention(t, name="u8-tie")

    def test_int32_answers_are_unchanged_by_the_byte_wise_form(self):
        """The shipped path must be byte-identical: this is the only format in production."""
        assert ct_packed_sign_convention(_stack(4, 4096, 8)).uint4b8
        assert not ct_packed_sign_convention(_stack(4, 4096, 0)).uint4b8

    def test_apply_preserves_dtype_and_shape_for_uint8(self):
        conv = CtSignConvention(uint4b8=False, margin=1.0, sampled_words=1, stride=1)
        t = self._u8_stack(4, 64, 0)  # last dim divides by 4 -> the old code SILENTLY reshaped
        out = apply_ct_sign(t, conv)
        assert out.dtype == torch.uint8, "the flip must not reinterpret the container dtype"
        assert out.shape == t.shape
        assert torch.equal(out, torch.full_like(t, 0x88))

    def test_apply_is_an_involution_for_uint8(self):
        conv = CtSignConvention(uint4b8=False, margin=1.0, sampled_words=1, stride=1)
        t = torch.randint(0, 256, (4, 30), dtype=torch.uint8)  # last dim NOT divisible by 4
        assert torch.equal(apply_ct_sign(apply_ct_sign(t, conv), conv), t)

    def test_apply_preserves_dtype_and_shape_for_int32(self):
        conv = CtSignConvention(uint4b8=False, margin=1.0, sampled_words=1, stride=1)
        t = _stack(4, 64, 0)
        out = apply_ct_sign(t, conv)
        assert out.dtype == torch.int32 and out.shape == t.shape
        assert torch.equal(out.view(torch.uint8), torch.full_like(t.view(torch.uint8), 0x88))

    def test_sample_stays_word_aligned_across_widths(self):
        """`stride` is counted in packed WORDS, whatever their width — a byte stride that happened
        to be a multiple of the element size would only ever look at one byte lane."""
        wide = ct_packed_sign_convention(_stack(4, 4 * _CT_SIGN_SAMPLE_WORDS, 8))
        narrow = ct_packed_sign_convention(self._u8_stack(4, 4 * _CT_SIGN_SAMPLE_WORDS, 8))
        assert wide.stride > 1 and narrow.stride > 1
        assert wide.uint4b8 and narrow.uint4b8


class _Leaf(BaseOP):
    """A minimal real `BaseOP`. Real, not a stand-in with a `named_modules` method: `BaseOP` is not
    an `nn.Module`, and `collect_ct_sign_decisions` walks the repo's own op tree (`_iter_ops`). A
    duck-typed fake would let the collector pass this test and find nothing on a live model."""

    def __init__(self, uint4b8=None):
        if uint4b8 is not None:
            self._ct_sign = CtSignConvention(
                uint4b8=uint4b8, margin=0.1, sampled_words=1, stride=1, blocks=1
            )

    def forward(self, *a, **k):  # pragma: no cover - storage only
        raise RuntimeError


class _Tree(BaseOP):
    """`{"a.b": True}` -> a nested op tree whose `_iter_ops` paths are exactly those keys."""

    def __init__(self, decisions):
        for path, uint4b8 in decisions.items():
            node = self
            parts = path.split(".")
            for part in parts[:-1]:
                nxt = getattr(node, part, None)
                if not isinstance(nxt, BaseOP):
                    nxt = _Leaf()
                    setattr(node, part, nxt)
                node = nxt
            setattr(node, parts[-1], _Leaf(uint4b8))

    def forward(self, *a, **k):  # pragma: no cover
        raise RuntimeError


def _FakeModule(decisions):
    return _Tree(decisions)


class _FakeGroup:
    """A gloo group stand-in. `verify_ct_sign_across_ranks` calls exactly one collective, so the
    whole distributed dependency is that one function — patched, not mocked at the C level."""


class TestCrossRankVerification:
    """The CLOSE for `TestCrossRankHazard`. The detector cannot catch a mixed-packing stack alone:
    its tie-refusal is evaluated per shard, and a mixed stack is only a tie when you can see all of
    it, which no rank ever does. So the check has to be a collective."""

    def _patch(self, monkeypatch, per_rank):
        import torch.distributed as dist

        def _fake_all_gather(out_list, obj, group=None):
            for i, d in enumerate(per_rank):
                out_list[i] = d

        monkeypatch.setattr(dist, "all_gather_object", _fake_all_gather)

    def test_tp1_is_a_noop_and_needs_no_group(self):
        m = _FakeModule({"layers.0.mlp.experts.w13": True})
        assert verify_ct_sign_across_ranks(m, None, 1, 0) == {"layers.0.mlp.experts.w13": True}

    def test_a_model_with_no_ct_containers_collects_nothing(self):
        m = _FakeModule({"layers.0.mlp.experts.w13": None})
        assert verify_ct_sign_across_ranks(m, None, 2, 0) == {}

    def test_agreeing_ranks_pass(self, monkeypatch):
        agree = {"layers.0.mlp.experts.w13": True, "layers.0.mlp.experts.w2": True}
        self._patch(monkeypatch, [agree, agree])
        assert verify_ct_sign_across_ranks(_FakeModule(agree), _FakeGroup(), 2, 1) == agree

    def test_diverging_ranks_RAISE(self, monkeypatch):
        """The exact shape `TestCrossRankHazard::test_ep_shards_of_a_mixed_stack_decide_differently`
        produces: two ranks, same path, opposite answers, and previously nothing anywhere noticed."""
        r0 = {"layers.0.mlp.experts.w13": True}
        r1 = {"layers.0.mlp.experts.w13": False}
        self._patch(monkeypatch, [r0, r1])
        with pytest.raises(CtSignRankDivergence) as exc:
            verify_ct_sign_across_ranks(_FakeModule(r0), _FakeGroup(), 2, 0)
        assert "layers.0.mlp.experts.w13" in str(exc.value)
        assert "rank0=True" in str(exc.value) and "rank1=False" in str(exc.value)

    def test_a_path_on_only_one_rank_also_raises(self, monkeypatch):
        """Not a mixed checkpoint but a BUILD skew — one rank made a CT container where its peer
        made something else. Same silent-wrong-answer class, different fix, so it must not be
        waved through just because the shared keys agree."""
        r0 = {"layers.0.mlp.experts.w13": True, "layers.1.mlp.experts.w13": True}
        r1 = {"layers.0.mlp.experts.w13": True}
        self._patch(monkeypatch, [r0, r1])
        with pytest.raises(CtSignRankDivergence) as exc:
            verify_ct_sign_across_ranks(_FakeModule(r0), _FakeGroup(), 2, 0)
        assert "layers.1.mlp.experts.w13" in str(exc.value)

    def test_margin_and_sample_fields_are_NOT_compared(self, monkeypatch):
        """They are properties of the SHARD each rank sampled and legitimately differ. Comparing
        them would fail every healthy TP=2 boot on a perfectly uniform checkpoint."""
        import torch.distributed as dist

        captured = {}

        def _fake(out_list, obj, group=None):
            captured["obj"] = obj
            out_list[0] = out_list[1] = obj

        monkeypatch.setattr(dist, "all_gather_object", _fake)
        m = _FakeModule({"a": True})
        verify_ct_sign_across_ranks(m, _FakeGroup(), 2, 0)
        assert captured["obj"] == {"a": True}  # a dict of BOOLS, not of CtSignConvention
