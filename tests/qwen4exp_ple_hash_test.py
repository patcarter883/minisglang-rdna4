"""Tranche 1b — the qwen4_exp n-gram HASH.

This is the piece with no loud failure mode. Every wrong variant of it — wrong seed, wrong
multiplier, wrong shift, wrong EOS fill, heads assigned to the wrong n-gram order, an off-by-one in
the band offsets — still yields a legal row id inside the right head's band, so the model reads a
real 160-float embedding and only the QUALITY moves. So the tests here do not check that the code
runs; they check that it reproduces values the CHECKPOINT itself ships.

Three independent anchors, all CPU-only and all against `tests/fixtures/qwen4exp/ngram_meta.json`
(recorded from `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ 7b71922 — see the fixture's `_provenance`):

  1. `build_layer_multipliers(248320, 3, 0, seed=1234)` == the checkpoint's `layer_multipliers`.
     This is what pins the OTHERWISE UNDOCUMENTED seed: config.json does not set `seed`, so the
     whole hash rests on the architecture default being 1234, and this equality is the proof.
  2. `build_head_vocab_sizes(20_000_000, 16)` == the checkpoint's `ngram_heads_vocab_sizes`, and
     their prefix sum == `ngram_heads_offsets`.
  3. `Qwen4ExpNGramHasher` reproduces a torch transcription of
     `Qwen4ExpTextNGramEmbedding.forward` (batched, (B, L) layout) position for position.

Plus the properties that make the arithmetic safe at all: no int64 overflow, non-negative keys, and
EOS segmentation that does not leak across a boundary.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from minisgl.ple.hashing import (
    DEFAULT_NGRAM_SEED,
    Qwen4ExpNGramHasher,
    build_head_vocab_sizes,
    build_layer_multipliers,
    shift_right_ignore_eos,
)
from minisgl.weights.row_table import NgramHeads

META = json.load(
    open(os.path.join(os.path.dirname(__file__), "fixtures", "qwen4exp", "ngram_meta.json"))
)


def _torch():
    """`pytest.importorskip` is not enough here: on this box torch fails to import with an OSError
    (`libmpi_cxx.so.40`), not an ImportError, so the reference-parity tests would FAIL rather than
    skip when run outside the ROCm container. The hash itself is torch-free by design — these six
    tests are the only ones that need torch, and they must not turn a host run red."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - any import failure means "no torch here"
        pytest.skip(f"torch unavailable ({type(exc).__name__}: {exc}) — run in the ROCm container")
    return torch


def _hasher() -> Qwen4ExpNGramHasher:
    return Qwen4ExpNGramHasher(
        ngram_size=META["ngram_size"],
        heads_per_ngram=META["heads_per_ngram"],
        layer_multipliers=META["layer_multipliers"],
        eos_token_id=META["eos_token_id"],
    )


def _heads() -> NgramHeads:
    h = NgramHeads(
        offsets=np.array(META["ngram_heads_offsets"], dtype=np.int64),
        vocab_sizes=np.array(META["ngram_heads_vocab_sizes"], dtype=np.int64),
    )
    h.validate(META["rows_per_shard"] * META["n_shards"])
    return h


# -- anchor 1: the multipliers, hence the seed -------------------------------


def test_derived_multipliers_equal_the_checkpoint_tensor():
    got = build_layer_multipliers(
        META["vocab_size"], META["ngram_size"], META["ple_layer_index"], META["seed"]
    )
    assert got.tolist() == META["layer_multipliers"]


def test_the_seed_is_pinned_not_assumed():
    """config.json ships no `seed`, so 1234 is a DEFAULT the hash silently depends on. Any other
    seed must produce different multipliers — otherwise anchor 1 proves nothing."""
    assert DEFAULT_NGRAM_SEED == META["seed"]
    for other in (0, 1, 1233, 1235, 42):
        assert (
            build_layer_multipliers(META["vocab_size"], META["ngram_size"], 0, other).tolist()
            != META["layer_multipliers"]
        )


def test_multipliers_cannot_overflow_int64_for_any_legal_token():
    m = np.array(META["layer_multipliers"], dtype=object)  # exact python ints
    max_token = META["vocab_size"] - 1
    for mult in m:
        assert int(mult) % 2 == 1, "multipliers are 2*x+1 by construction"
        assert int(mult) * max_token < (1 << 63), (
            "multiplier * token must fit int64 — the reference's `multiplier_max` bound is what "
            "guarantees it, and numpy would wrap SILENTLY if it did not"
        )


def test_second_ple_layer_would_get_disjoint_multipliers():
    """`ple_layer_index` is mixed into the seed, so a checkpoint with two PLE layers does not reuse
    one layer's hash. Qwen3.8-Flash-Next has one, so this only guards the generalisation."""
    a = build_layer_multipliers(META["vocab_size"], META["ngram_size"], 0, META["seed"])
    b = build_layer_multipliers(META["vocab_size"], META["ngram_size"], 1, META["seed"])
    assert not np.array_equal(a, b)


# -- anchor 2: the head bands ------------------------------------------------


def test_derived_head_vocab_sizes_equal_the_checkpoint_tensor():
    got = build_head_vocab_sizes(
        META["ngram_vocab_size_base"], META["ngram_heads"], META["ple_layer_index"]
    )
    assert got.tolist() == META["ngram_heads_vocab_sizes"]


def test_offsets_are_the_prefix_sum_and_fit_the_table():
    v = np.array(META["ngram_heads_vocab_sizes"], dtype=np.int64)
    expect = np.concatenate([[0], np.cumsum(v)[:-1]])
    assert expect.tolist() == META["ngram_heads_offsets"]
    total = int(v.sum())
    table_rows = META["rows_per_shard"] * META["n_shards"]
    assert total <= table_rows
    # 90 padding rows at the top of the table are unaddressable — recorded so a future change to
    # `make_ngram_vocab_size_divisible_by` shows up here rather than as a silent re-addressing.
    assert table_rows - total == 90


def test_head_blocks_are_contiguous_per_ngram_order():
    """Heads 0..7 carry the 2-gram key, 8..15 the 3-gram key. If that ordering were interleaved
    instead, every head would read a real embedding from the wrong band."""
    h = _hasher()
    hist = np.array([5, 9, 13, 21, 34], dtype=np.int64)
    hashes = h.hashes(hist, 3)
    assert hashes.shape == (3, META["ngram_heads"])
    per = META["heads_per_ngram"]
    assert (hashes[:, :per] == hashes[:, :1]).all()
    assert (hashes[:, per:] == hashes[:, per : per + 1]).all()
    assert not (hashes[:, 0] == hashes[:, per]).all()


def test_row_ids_land_in_the_right_band():
    h, heads = _hasher(), _heads()
    hist = np.arange(1000, 1040, dtype=np.int64)
    rows = h.row_ids(heads, hist, 20)
    assert rows.shape == (20, 16)
    for head in range(16):
        lo = heads.offsets[head]
        hi = lo + heads.vocab_sizes[head]
        assert (rows[:, head] >= lo).all() and (rows[:, head] < hi).all()


# -- EOS segmentation --------------------------------------------------------


def test_shift_zero_is_identity():
    t = np.array([1, 2, 3], dtype=np.int64)
    assert shift_right_ignore_eos(t, 0, 99).tolist() == [1, 2, 3]


def test_shift_fills_with_eos_at_the_start():
    t = np.array([10, 11, 12, 13], dtype=np.int64)
    assert shift_right_ignore_eos(t, 1, 99).tolist() == [99, 10, 11, 12]
    assert shift_right_ignore_eos(t, 2, 99).tolist() == [99, 99, 10, 11]


def test_shift_does_not_reach_across_an_eos():
    # segment boundary: the EOS itself still belongs to the segment ENDING at it, and the token
    # after it starts a new segment whose first `shift` positions get the EOS fill.
    t = np.array([10, 11, 99, 20, 21, 22], dtype=np.int64)
    assert shift_right_ignore_eos(t, 1, 99).tolist() == [99, 10, 11, 99, 20, 21]
    assert shift_right_ignore_eos(t, 2, 99).tolist() == [99, 99, 10, 99, 99, 20]


def test_documents_do_not_leak_into_each_other():
    """The whole point of the EOS rule: the n-gram key of the first tokens of document B must not
    depend on document A's tail."""
    h = _hasher()
    eos = META["eos_token_id"]
    a1 = np.array([7, 8, 9, eos, 100, 101, 102], dtype=np.int64)
    a2 = np.array([70, 80, 90, eos, 100, 101, 102], dtype=np.int64)
    assert np.array_equal(h.hashes(a1, 3), h.hashes(a2, 3))


# -- anchor 3: parity with a transcription of the reference ------------------


def _reference_ngram_ids(input_ids, previous_context, meta):
    """torch transcription of `Qwen4ExpTextNGramEmbedding.forward`, (B, L) layout.

    Kept structurally identical to upstream (cummax, gather, torch.remainder, the per-n block loop)
    so it is an independent implementation of the same spec, not a rewrite of the module under test.
    """
    torch = _torch()

    eos = meta["eos_token_id"]
    ngram_size = meta["ngram_size"]
    heads_per_ngram = meta["heads_per_ngram"]
    mult = torch.tensor(meta["layer_multipliers"], dtype=torch.long)
    vocab = torch.tensor(meta["ngram_heads_vocab_sizes"], dtype=torch.long)
    offs = torch.tensor(meta["ngram_heads_offsets"], dtype=torch.long)

    input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    token_history = torch.cat([torch.as_tensor(previous_context, dtype=torch.long), input_ids], -1)

    def shift_right(token_ids, shift):
        if shift == 0:
            return token_ids
        bsz, seq_len = token_ids.shape
        positions = torch.arange(seq_len, dtype=torch.long)
        eos_positions = torch.where(token_ids == eos, positions, torch.tensor(-1))
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((bsz, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
        )
        position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(bsz, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), eos))

    shifted_tokens = [shift_right(token_history, s) for s in range(ngram_size)]
    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mixed = shifted_tokens[0] * mult[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted_tokens[position] * mult[position])
        ids = torch.remainder(mixed.unsqueeze(-1), vocab[start:end].view(1, 1, -1))
        blocks.append(ids + offs[start:end].view(1, 1, -1))
    return torch.cat(blocks, dim=-1)[:, -input_ids.shape[1] :]


@pytest.mark.parametrize("seq_len", [1, 2, 3, 17, 64])
def test_matches_the_reference_transcription(seq_len):
    torch = _torch()
    rng = np.random.default_rng(seq_len)
    eos = META["eos_token_id"]
    tokens = rng.integers(0, META["vocab_size"], size=seq_len, dtype=np.int64)
    # sprinkle EOS so the segmentation branch is exercised, not just the happy path
    if seq_len > 4:
        tokens[seq_len // 2] = eos
    prev = np.array([eos, eos], dtype=np.int64)

    want = _reference_ngram_ids(tokens[None, :], prev[None, :], META)[0].numpy()
    h, heads = _hasher(), _heads()
    got = h.row_ids(heads, np.concatenate([prev, tokens]), seq_len)
    assert np.array_equal(got, want), f"row ids differ at seq_len={seq_len}"
    assert torch.as_tensor(got).dtype == torch.int64


def test_matches_the_reference_with_a_nontrivial_previous_context():
    """Decode: the 2-token context comes from the PREVIOUS pass, not from EOS padding. This is the
    case that catches a history kept at the wrong end (or one step stale)."""
    rng = np.random.default_rng(7)
    prev = rng.integers(0, META["vocab_size"], size=2, dtype=np.int64)
    tokens = rng.integers(0, META["vocab_size"], size=5, dtype=np.int64)
    want = _reference_ngram_ids(tokens[None, :], prev[None, :], META)[0].numpy()
    got = _hasher().row_ids(_heads(), np.concatenate([prev, tokens]), 5)
    assert np.array_equal(got, want)


def test_a_wrong_multiplier_is_detected_not_used():
    with pytest.raises(ValueError, match="mismatch"):
        Qwen4ExpNGramHasher.from_checkpoint_multipliers(
            ngram_size=META["ngram_size"],
            heads_per_ngram=META["heads_per_ngram"],
            eos_token_id=META["eos_token_id"],
            checkpoint_multipliers=[1, 3, 5],
            vocab_size=META["vocab_size"],
        )


def test_checkpoint_multipliers_agree_with_the_derivation():
    h = Qwen4ExpNGramHasher.from_checkpoint_multipliers(
        ngram_size=META["ngram_size"],
        heads_per_ngram=META["heads_per_ngram"],
        eos_token_id=META["eos_token_id"],
        checkpoint_multipliers=META["layer_multipliers"],
        vocab_size=META["vocab_size"],
    )
    assert h.layer_multipliers.tolist() == META["layer_multipliers"]
    assert h.ngram_heads == META["ngram_heads"]
    assert h.context_len == META["ngram_size"] - 1


def test_an_even_multiplier_is_rejected():
    with pytest.raises(ValueError, match="ODD"):
        Qwen4ExpNGramHasher(
            ngram_size=3, heads_per_ngram=8, layer_multipliers=[2, 3, 5], eos_token_id=0
        )
