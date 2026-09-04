"""Tranche 1b — the qwen4_exp PLE block: numerics, state, staging, and graph capture.

Run in the serve image (torch does not import on this host):

    docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
      --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES=0 \
      -v <worktree>:/engine -v /home/pat/.cache/hf-ple:/ple:ro \
      -e MINISGL_PLE_FILES="$(ls /ple/model-plefp8-*.safetensors | paste -sd:)" \
      -e MINISGL_PLE_META_FILES=/ple/model-bf16-00010.safetensors \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python -m pytest /engine/tests/qwen4exp_ple_test.py \
         -q -o addopts=""'

WHAT IS BEING PINNED
--------------------
`Qwen4ExpPLE` is five pieces of arithmetic stacked on a stateful conv, and NONE of them fails
loudly. So each gets its own anchor:

  * `GroupedRMSNorm` vs a full-width RMSNorm — a full-width norm on a 4x2560 vector is a different
    function with the same shape.
  * the whole block vs a torch transcription of `modeling_qwen4_exp.py::Qwen4ExpTextPLELayer`,
    written to the reference's (B, S, C) layout so it is an independent implementation rather than a
    re-spelling of the code under test.
  * decode-step-by-step == one prefill of the same tokens. That is the state test: it catches a conv
    window kept at the wrong end, a state written before it is read, and a token history one step
    stale. It is also the only check that the dilation-9 window (not the GDN-style k-1 = 3) is
    carried.
  * a cudagraph capture + replay of the decode path, because in this repo eager-only is not done.
  * against the REAL 51.2 GB table when `MINISGL_PLE_FILES` names it: geometry, the head/width
    identity (16 x 160 == ple_embed_dim), and the measured per-step cost of the wired path.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from minisgl.distributed import set_tp_info, try_get_tp_info  # noqa: E402

# Process-global and settable once. Guarded so this file can be collected alongside the other
# module-scope qwen4exp tests (`qwen4exp_hc_parity_test.py`); an unguarded second call raises during
# COLLECTION, which aborts the whole pytest run rather than failing one module.
if try_get_tp_info() is None:
    set_tp_info(0, 1)

from minisgl.models.qwen4exp import GroupedRMSNorm, Qwen4ExpPLE  # noqa: E402
from minisgl.ple import (  # noqa: E402
    ENV_PLE_FILES,
    ENV_PLE_META_FILES,
    PLEEmbeddingSource,
    PLERuntime,
    PLEStateCache,
    Qwen4ExpNGramHasher,
)
from minisgl.ple.runtime import PLEBatch  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "qwen4exp")
META = json.load(open(os.path.join(FIXTURES, "ngram_meta.json")))

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# a small config, and a reference transcription of the upstream block
# ---------------------------------------------------------------------------


class TinyCfg:
    """Just the fields `Qwen4ExpPLE.__init__` reads. Small dims so the parity tests are exact and
    fast; the SHAPE RELATIONS (wide = hc*hidden, dilation = ngram_size, state = (k-1)*dilation) are
    the same ones the real config produces, and the real numbers are asserted separately in
    `test_real_config_geometry`."""

    def __init__(self, hidden=8, hc=4, embed=6, k=4, ngram=3, eps=1e-6):
        self.hidden_size = hidden
        self.hc_count = hc
        self.hc_hidden_size = hidden * hc
        self.ple_embed_dim = embed
        self.ple_conv_kernel_size = k
        self.ngram_size = ngram
        self.rms_norm_eps = eps


class RefRMSNorm(torch.nn.Module):
    """`Qwen4ExpTextRMSNorm`, verbatim (fp32 internals, `(1 + w)` gain, optional group_size)."""

    def __init__(self, dim, group_size=None, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.group_size = group_size

    def _norm(self, x):
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out

    def forward(self, x):
        out = self._norm(x.float())
        return (out * (1.0 + self.weight.float())).type_as(x)


class RefPLE(torch.nn.Module):
    """`Qwen4ExpTextPLELayer`, transcribed to a self-contained module with an explicit conv state.

    `conv_state` is (B, C, state_len) and is updated the way `update_conv_state` does: concatenate,
    convolve, keep the last `state_len` columns.
    """

    def __init__(self, cfg: TinyCfg):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.hc_count = cfg.hc_count
        wide = cfg.hc_hidden_size
        self.short_conv_state_len = (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size
        self.key_proj = torch.nn.Linear(cfg.ple_embed_dim, wide, bias=False)
        self.value_proj = torch.nn.Linear(cfg.ple_embed_dim, cfg.hidden_size, bias=False)
        self.norm_key = RefRMSNorm(wide, group_size=cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm_query = RefRMSNorm(wide, group_size=cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm_conv = RefRMSNorm(wide, group_size=cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.conv1d = torch.nn.Conv1d(
            wide, wide, kernel_size=cfg.ple_conv_kernel_size, groups=wide,
            dilation=cfg.ngram_size, bias=False,
        )

    def _short_conv(self, hidden_states, conv_state):
        seq_len = hidden_states.shape[1]
        h = hidden_states.transpose(1, 2)
        full = torch.cat([conv_state, h], dim=-1)
        conv_state.copy_(full[..., -self.short_conv_state_len :])
        h = F.pad(full, (self.short_conv_state_len, 0))
        h = h[..., -(self.short_conv_state_len + seq_len) :]
        return F.silu(self.conv1d(h)).transpose(1, 2)

    def forward(self, hidden_states, embeddings, conv_state):
        key_normed = self.norm_key(self.key_proj(embeddings)).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        value = self.value_proj(embeddings)
        query_normed = self.norm_query(hidden_states).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / self.hidden_size**0.5
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        return gated_value + self._short_conv(gated_value_normed, conv_state)


def _build_pair(cfg: TinyCfg, *, dtype=torch.float32, device=DEV, seed=0):
    """A `Qwen4ExpPLE` and a `RefPLE` carrying the SAME random weights."""
    torch.manual_seed(seed)
    ref = RefPLE(cfg).to(device=device, dtype=dtype)
    for p in ref.parameters():
        with torch.no_grad():
            p.copy_(torch.randn_like(p) * 0.3)
    # `Qwen4ExpPLE` declares its buffers with `torch.empty`, i.e. at the DEFAULT dtype, and
    # `BaseOP.load_state_dict` casts the incoming tensor to the declared one. Build it under the
    # dtype under test so the load is a copy, not a bf16 -> fp32 widening that would then meet bf16
    # activations at the GEMM.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        ours = Qwen4ExpPLE(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    ours.load_state_dict(
        {
            "conv1d_weight": ref.conv1d.weight.detach().clone(),
            "key_proj.weight": ref.key_proj.weight.detach().clone(),
            "value_proj.weight": ref.value_proj.weight.detach().clone(),
            "norm_key.weight": ref.norm_key.weight.detach().clone(),
            "norm_query.weight": ref.norm_query.weight.detach().clone(),
            "norm_conv.weight": ref.norm_conv.weight.detach().clone(),
            "ple_embedding.layer_multipliers": torch.tensor(
                META["layer_multipliers"], dtype=torch.int64, device=device
            ),
        }
    )
    return ours, ref


def _batch(seq_lens, slots, device=DEV, is_decode=None):
    return PLEBatch(
        embeddings=None,  # `compute` takes the embeddings explicitly; only `forward` reads this
        state_indices=torch.tensor(slots, dtype=torch.long, device=device),
        slots=list(slots),
        seq_lens=list(seq_lens),
        is_decode=all(n == 1 for n in seq_lens) if is_decode is None else is_decode,
    )


# ---------------------------------------------------------------------------
# GroupedRMSNorm
# ---------------------------------------------------------------------------


def test_grouped_norm_matches_the_reference():
    torch.manual_seed(1)
    wide, group = 32, 8
    ref = RefRMSNorm(wide, group_size=group).to(DEV)
    with torch.no_grad():
        ref.weight.copy_(torch.randn(wide, device=DEV) * 0.2)
    ours = GroupedRMSNorm(wide, group_size=group, eps=1e-6)
    ours.load_state_dict({"weight": ref.weight.detach().clone()})
    x = torch.randn(7, wide, device=DEV)
    torch.testing.assert_close(ours.forward(x), ref(x), rtol=2e-6, atol=2e-6)


def test_grouped_norm_is_not_a_full_width_norm():
    """The silent-wrong twin: same shapes, different function. If these ever agreed, the group
    reshape would have been dropped and nothing else would notice."""
    torch.manual_seed(2)
    wide, group = 32, 8
    ours = GroupedRMSNorm(wide, group_size=group, eps=1e-6)
    ours.load_state_dict({"weight": torch.zeros(wide, device=DEV)})
    full = RefRMSNorm(wide, group_size=None).to(DEV)  # weight is zeros -> gain 1
    # A vector whose groups have very different scales — the case where the two diverge most.
    x = torch.cat([torch.randn(1, group, device=DEV) * s for s in (0.01, 1.0, 10.0, 100.0)], dim=-1)
    assert not torch.allclose(ours.forward(x), full(x), rtol=1e-2, atol=1e-2)


def test_grouped_norm_rejects_an_indivisible_width():
    with pytest.raises(ValueError, match="divisible"):
        GroupedRMSNorm(30, group_size=8, eps=1e-6)


# ---------------------------------------------------------------------------
# the block vs the reference
# ---------------------------------------------------------------------------


def test_prefill_matches_the_reference():
    cfg = TinyCfg()
    ours, ref = _build_pair(cfg)
    torch.manual_seed(3)
    n = 11
    hidden = torch.randn(n, cfg.hc_hidden_size, device=DEV)
    emb = torch.randn(n, cfg.ple_embed_dim, device=DEV)

    state_ours = torch.zeros(2, cfg.hc_hidden_size, ours.conv_state_len, device=DEV)
    state_ref = torch.zeros(1, cfg.hc_hidden_size, ours.conv_state_len, device=DEV)

    b = _batch([n], [1])
    got = ours.compute(hidden, emb, b, state_ours)
    want = ref(hidden[None], emb[None], state_ref)[0]

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state_ours[1], state_ref[0], rtol=1e-5, atol=1e-5)
    # slot 0 (NULL) untouched
    assert state_ours[0].abs().max() == 0


def test_decode_matches_the_reference():
    cfg = TinyCfg()
    ours, ref = _build_pair(cfg, seed=4)
    torch.manual_seed(5)
    n_seqs = 3
    state_ours = torch.randn(n_seqs + 1, cfg.hc_hidden_size, ours.conv_state_len, device=DEV)
    state_ours[0].zero_()
    state_ref = state_ours[1:].clone()

    hidden = torch.randn(n_seqs, cfg.hc_hidden_size, device=DEV)
    emb = torch.randn(n_seqs, cfg.ple_embed_dim, device=DEV)

    b = _batch([1] * n_seqs, [1, 2, 3])
    assert b.is_decode
    got = ours.compute(hidden, emb, b, state_ours)
    want = ref(hidden[:, None], emb[:, None], state_ref)[:, 0]

    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state_ours[1:], state_ref, rtol=1e-5, atol=1e-5)


def test_varlen_prefill_matches_per_sequence_prefill():
    """Two sequences packed into one flat batch must equal two separate forwards — the conv must
    never see across a sequence boundary."""
    cfg = TinyCfg()
    ours, _ = _build_pair(cfg, seed=6)
    torch.manual_seed(7)
    lens = [5, 3]
    tot = sum(lens)
    hidden = torch.randn(tot, cfg.hc_hidden_size, device=DEV)
    emb = torch.randn(tot, cfg.ple_embed_dim, device=DEV)

    state_a = torch.zeros(3, cfg.hc_hidden_size, ours.conv_state_len, device=DEV)
    packed = ours.compute(hidden, emb, _batch(lens, [1, 2]), state_a)

    state_b = torch.zeros(3, cfg.hc_hidden_size, ours.conv_state_len, device=DEV)
    a = ours.compute(hidden[:5], emb[:5], _batch([5], [1]), state_b)
    b = ours.compute(hidden[5:], emb[5:], _batch([3], [2]), state_b)

    torch.testing.assert_close(packed, torch.cat([a, b]), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state_a, state_b, rtol=1e-5, atol=1e-5)


def test_decode_steps_equal_one_prefill():
    """THE state test. Prefilling S tokens must equal prefilling 1 then decoding S-1, token for
    token. A conv window kept at the wrong end, a state written before it is read, or a receptive
    field of 4 instead of 10 all break here and nowhere else."""
    cfg = TinyCfg()
    ours, _ = _build_pair(cfg, seed=8)
    torch.manual_seed(9)
    n = 14
    hidden = torch.randn(n, cfg.hc_hidden_size, device=DEV)
    emb = torch.randn(n, cfg.ple_embed_dim, device=DEV)

    state_p = torch.zeros(2, cfg.hc_hidden_size, ours.conv_state_len, device=DEV)
    full = ours.compute(hidden, emb, _batch([n], [1]), state_p)

    state_d = torch.zeros(2, cfg.hc_hidden_size, ours.conv_state_len, device=DEV)
    outs = [ours.compute(hidden[:1], emb[:1], _batch([1], [1], is_decode=False), state_d)]
    for t in range(1, n):
        outs.append(ours.compute(hidden[t : t + 1], emb[t : t + 1], _batch([1], [1]), state_d))
    stepwise = torch.cat(outs)

    torch.testing.assert_close(full, stepwise, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state_p, state_d, rtol=1e-5, atol=1e-5)


def test_conv_receptive_field_is_dilated_not_dense():
    """The dilation is `ngram_size`, so the state is (k-1)*ngram_size = 9, not k-1 = 3. A dense
    conv would still run, still have the right shapes, and read the wrong 4 tokens."""
    cfg = TinyCfg()
    ours, _ = _build_pair(cfg, seed=10)
    assert ours.conv_state_len == (cfg.ple_conv_kernel_size - 1) * cfg.ngram_size == 9

    # token t must depend on t-3, t-6, t-9 and NOT on t-1, t-2.
    torch.manual_seed(11)
    n = 12
    hidden = torch.randn(n, cfg.hc_hidden_size, device=DEV)
    emb = torch.zeros(n, cfg.ple_embed_dim, device=DEV)
    base_state = torch.zeros(2, cfg.hc_hidden_size, ours.conv_state_len, device=DEV)

    def run(e):
        st = base_state.clone()
        return ours.compute(hidden, e, _batch([n], [1]), st)

    ref_out = run(emb)
    for lag, expect_change in ((1, False), (2, False), (3, True), (6, True), (9, True)):
        e = emb.clone()
        e[n - 1 - lag] = 5.0
        d = (run(e)[n - 1] - ref_out[n - 1]).abs().max().item()
        assert (d > 1e-4) == expect_change, f"lag {lag}: changed={d > 1e-4}, expected {expect_change}"


def test_null_slot_padding_rows_do_not_corrupt_real_ones():
    """A captured decode graph runs at a padded batch size; the padding rows point at slot 0. Their
    output is discarded, but they must not write into a real sequence's state."""
    cfg = TinyCfg()
    ours, _ = _build_pair(cfg, seed=12)
    torch.manual_seed(13)
    state = torch.zeros(3, cfg.hc_hidden_size, ours.conv_state_len, device=DEV)
    hidden = torch.randn(3, cfg.hc_hidden_size, device=DEV)
    emb = torch.randn(3, cfg.ple_embed_dim, device=DEV)

    real = ours.compute(hidden[:1], emb[:1], _batch([1], [1]), state.clone())

    padded_state = torch.zeros_like(state)
    out = ours.compute(hidden, emb, _batch([1, 1, 1], [1, 0, 0]), padded_state)
    torch.testing.assert_close(out[:1], real, rtol=1e-5, atol=1e-5)
    assert padded_state[2].abs().max() == 0, "slot 2 was never addressed and must stay zero"


# ---------------------------------------------------------------------------
# graph capture — eager-only is not "done" in this repo
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_decode_path_is_cudagraph_capturable():
    cfg = TinyCfg()
    ours, _ = _build_pair(cfg, dtype=torch.bfloat16, seed=14)
    n_seqs = 4
    state = torch.zeros(
        n_seqs + 1, cfg.hc_hidden_size, ours.conv_state_len, dtype=torch.bfloat16, device=DEV
    )
    hidden = torch.randn(n_seqs, cfg.hc_hidden_size, dtype=torch.bfloat16, device=DEV)
    emb = torch.randn(n_seqs, cfg.ple_embed_dim, dtype=torch.bfloat16, device=DEV)
    b = _batch([1] * n_seqs, [1, 2, 3, 4])

    # eager reference, from the same starting state
    st0 = state.clone()
    eager1 = ours.compute(hidden, emb, b, state).clone()
    eager2 = ours.compute(hidden, emb, b, state).clone()

    state.copy_(st0)
    torch.cuda.synchronize()
    # warmup on a side stream, as the engine's GraphRunner does
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ours.compute(hidden, emb, b, state)
    torch.cuda.current_stream().wait_stream(s)
    state.copy_(st0)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = ours.compute(hidden, emb, b, state)
    # capture itself ran the ops once against `state`; reset and replay from the same start.
    state.copy_(st0)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(static_out, eager1, rtol=2e-2, atol=2e-2)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(static_out, eager2, rtol=2e-2, atol=2e-2)
    # the replay must have ADVANCED the persistent state, not recomputed from a stale copy
    assert not torch.equal(state, st0)


# ---------------------------------------------------------------------------
# state cache + runtime plumbing (no table needed)
# ---------------------------------------------------------------------------


def _fake_source(tmp_path, *, embed_dim, n_heads, rows_per_shard=64, row_elems=None, n_shards=4):
    """A tiny synthetic n-gram table with the real NAMES, so the loader path is exercised."""
    import struct

    from minisgl.weights.row_table import NgramHeads, ShardedRowTable, index_safetensors

    row_elems = row_elems or embed_dim // n_heads
    prefix = "model.language_model.layers.1.ple.ple_embedding"
    rng = np.random.default_rng(0)
    header, off, blobs = {}, 0, []
    for sid in range(n_shards):
        raw = rng.integers(0, 128, size=rows_per_shard * row_elems, dtype=np.uint8).tobytes()
        header[f"{prefix}.ngram_embedding.shard_{sid}.weight"] = {
            "dtype": "F8_E4M3", "shape": [rows_per_shard, row_elems],
            "data_offsets": [off, off + len(raw)],
        }
        off += len(raw)
        blobs.append(raw)
    blob = json.dumps(header).encode()
    path = os.path.join(str(tmp_path), "fake-plefp8.safetensors")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for r in blobs:
            f.write(r)

    idx = index_safetensors([path])
    locs = [idx[f"{prefix}.ngram_embedding.shard_{s}.weight"] for s in range(n_shards)]
    table = ShardedRowTable(locs, scale=0.25, shard_ids=list(range(n_shards)))
    sizes = np.array([13, 11, 7, 5, 3, 2, 17, 19][:n_heads], dtype=np.int64)
    heads = NgramHeads(
        offsets=np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64), vocab_sizes=sizes
    )
    heads.validate(table.n_rows)
    return table, heads


def test_state_cache_history_and_reset():
    st = PLEStateCache(
        num_slots=3, wide=8, state_len=9, context_len=2, eos_token_id=99,
        dtype=torch.float32, device=DEV,
    )
    assert st.history(1).tolist() == [99, 99]
    st.push_tokens(1, np.array([5]))
    assert st.history(1).tolist() == [99, 5]
    st.push_tokens(1, np.array([6, 7, 8]))
    assert st.history(1).tolist() == [7, 8]
    assert st.full_history(1, np.array([9])).tolist() == [7, 8, 9]

    st.conv_state[1].fill_(1.0)
    st.reset_slot(1)
    assert st.history(1).tolist() == [99, 99]
    assert st.conv_state[1].abs().max() == 0
    with pytest.raises(IndexError):
        st.reset_slot(0)  # the NULL slot is never handed out


def test_runtime_stages_one_h2d_and_does_not_advance_until_commit(tmp_path):
    embed_dim, n_heads = 8 * 4, 4
    table, heads = _fake_source(tmp_path, embed_dim=embed_dim, n_heads=n_heads)
    hasher = Qwen4ExpNGramHasher(
        ngram_size=3, heads_per_ngram=n_heads // 2,
        layer_multipliers=META["layer_multipliers"], eos_token_id=99,
    )
    src = PLEEmbeddingSource(
        ple_files=[], hasher=hasher, embed_dim=embed_dim, max_tokens=16, device=DEV,
        dtype=torch.float32, _table=table, _heads=heads,
    )
    state = PLEStateCache(
        num_slots=3, wide=embed_dim, state_len=9, context_len=2, eos_token_id=99,
        dtype=torch.float32, device=DEV,
    )
    rt = PLERuntime(source=src, state=state, max_seqs=4, device=DEV)

    toks = [np.array([1, 2, 3], dtype=np.int64), np.array([4, 5, 6], dtype=np.int64)]
    b = rt.prepare([1, 2], toks)
    assert b.embeddings.shape == (6, embed_dim)
    assert b.seq_lens == [3, 3] and not b.is_decode
    assert b.state_indices.tolist() == [1, 2]
    # staging must be idempotent until commit
    assert state.history(1).tolist() == [99, 99]
    again = rt.prepare([1, 2], toks)
    torch.testing.assert_close(again.embeddings, b.embeddings)
    rt.commit([1, 2], toks)
    assert state.history(1).tolist() == [2, 3]
    assert state.history(2).tolist() == [5, 6]

    # the staged values must be the dequantised table rows, in head order
    rows = src.batch_row_ids(state, [1], [np.array([7], dtype=np.int64)])
    got = src.stage_rows(rows)
    want = torch.as_tensor(table.gather(rows.reshape(-1)).reshape(1, embed_dim), device=DEV)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    src.close()


def test_staging_buffer_overflow_is_loud(tmp_path):
    embed_dim, n_heads = 8, 4
    table, heads = _fake_source(tmp_path, embed_dim=embed_dim, n_heads=n_heads, row_elems=2)
    hasher = Qwen4ExpNGramHasher(
        ngram_size=3, heads_per_ngram=n_heads // 2,
        layer_multipliers=META["layer_multipliers"], eos_token_id=99,
    )
    src = PLEEmbeddingSource(
        ple_files=[], hasher=hasher, embed_dim=embed_dim, max_tokens=4, device=DEV,
        dtype=torch.float32, _table=table, _heads=heads,
    )
    with pytest.raises(ValueError, match="staging buffer"):
        src.stage_rows(np.zeros((5, n_heads), dtype=np.int64))
    src.close()


# ---------------------------------------------------------------------------
# the real config, and the real 51.2 GB table
# ---------------------------------------------------------------------------


def test_real_config_geometry():
    """The numbers the tiny config stands in for, read from the shipping config.json."""
    import shutil
    import tempfile

    from minisgl.models.config import ModelConfig
    from minisgl.utils import cached_load_hf_config

    tmp = tempfile.mkdtemp(prefix="qwen4exp-ple-")
    shutil.copyfile(os.path.join(FIXTURES, "config.json"), os.path.join(tmp, "config.json"))
    mc = ModelConfig.from_hf(cached_load_hf_config(tmp), spec_algorithm="none")

    assert mc.ple_layer_ids == (1,)
    assert mc.hc_hidden_size == 10240
    assert mc.ple_embed_dim == 2560
    assert mc.ngram_size == 3 and mc.ple_conv_kernel_size == 4
    assert mc.heads_per_ngram == 8
    assert (mc.ngram_size - 1) * mc.heads_per_ngram == META["ngram_heads"] == 16
    assert mc.ple_embed_dim // META["ngram_heads"] == META["row_elems"] == 160
    assert mc.ngram_seed == META["seed"] == 1234
    assert mc.ngram_eos_token_id == META["eos_token_id"] == 248044

    ple = Qwen4ExpPLE(mc)
    assert ple.conv_state_len == 9
    sd = ple.state_dict()
    assert tuple(sd["conv1d_weight"].shape) == (10240, 1, 4)
    assert tuple(sd["key_proj.weight"].shape) == (10240, 2560)
    assert tuple(sd["value_proj.weight"].shape) == (2560, 2560)
    for n in ("norm_key.weight", "norm_query.weight", "norm_conv.weight"):
        assert tuple(sd[n].shape) == (10240,)
    assert sd["ple_embedding.layer_multipliers"].dtype == torch.int64


_PLE = os.environ.get(ENV_PLE_FILES, "")
_PLE_META = os.environ.get(ENV_PLE_META_FILES, "")
needs_table = pytest.mark.skipif(
    not (_PLE and _PLE_META),
    reason=f"set {ENV_PLE_FILES} (the model-plefp8-* set) and {ENV_PLE_META_FILES} (the bf16 shard)",
)


@needs_table
def test_real_table_wiring_and_cost():
    """End-to-end on the REAL table: hash -> gather -> H2D, and what a step costs.

    The numbers printed here are the measurement the tranche reports. They are wall time on this
    box's NVMe with whatever page cache the run happens to have — a cold first call is the honest
    decode figure, so it is reported separately from the warm steady state.
    """
    hasher = Qwen4ExpNGramHasher.from_checkpoint_multipliers(
        ngram_size=META["ngram_size"],
        heads_per_ngram=META["heads_per_ngram"],
        eos_token_id=META["eos_token_id"],
        checkpoint_multipliers=META["layer_multipliers"],
        vocab_size=META["vocab_size"],
    )
    src = PLEEmbeddingSource(
        ple_files=_PLE.split(":"),
        meta_files=_PLE_META.split(":"),
        hasher=hasher,
        embed_dim=META["ple_embed_dim"],
        max_tokens=4096,
        device=DEV,
        dtype=torch.bfloat16,
        workers=16,
    )
    try:
        assert src.table.row_elems == META["row_elems"]
        assert src.table.rows_per_shard == META["rows_per_shard"]
        assert src.table.n_rows == META["rows_per_shard"] * META["n_shards"]
        assert src.table.complete, "the full 128-shard set must be present"
        assert src.heads.offsets.tolist() == META["ngram_heads_offsets"]
        assert src.heads.vocab_sizes.tolist() == META["ngram_heads_vocab_sizes"]
        assert float(src.table.scale) == META["ngram_embedding_weight_scale"]

        n_slots = 16
        state = PLEStateCache(
            num_slots=n_slots, wide=10240, state_len=9, context_len=2,
            eos_token_id=META["eos_token_id"], dtype=torch.bfloat16, device=DEV,
        )
        rt = PLERuntime(source=src, state=state, max_seqs=n_slots - 1, device=DEV)
        rng = np.random.default_rng(0)

        def sample(n_seqs, n_tokens):
            toks = [
                rng.integers(0, META["vocab_size"], size=n_tokens, dtype=np.int64)
                for _ in range(n_seqs)
            ]
            return list(range(1, n_seqs + 1)), toks

        def step(n_seqs, n_tokens):
            """Time `prepare` end to end, and its three parts, so the dominant one is named rather
            than guessed. The gather is the only part that touches the NVMe."""
            slots, toks = sample(n_seqs, n_tokens)
            t0 = time.perf_counter()
            rows = src.batch_row_ids(state, slots, toks)
            t1 = time.perf_counter()
            flat = rows.reshape(-1)
            src.table.gather_into(flat, src._host_rows[: flat.size])
            t2 = time.perf_counter()
            src._dev_f32[: rows.shape[0]].copy_(src._host[: rows.shape[0]], non_blocking=True)
            src.embeddings[: rows.shape[0]].copy_(src._dev_f32[: rows.shape[0]])
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            rt.commit(slots, toks)
            return (t1 - t0, t2 - t1, t3 - t2, t3 - t0)

        # FIRST call, reported separately: it pays the mmap's first touch across ten 5.2 GB files
        # (page-table setup, not I/O the steady state repeats). Quoting it as "the" cost would be
        # dishonest in one direction; hiding it would be dishonest in the other.
        slots, toks = sample(4, 1)
        t0 = time.perf_counter()
        b = rt.prepare(slots, toks)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        first = time.perf_counter() - t0
        rt.commit(slots, toks)
        assert b.embeddings.shape == (4, 2560) and b.is_decode
        print(f"\n  PLERuntime.prepare, 4-seq decode, FIRST EVER call: {first*1e3:.1f} ms")

        # Steady state, through the real entry point (not the hand-split one used below), so the
        # reported per-step cost is the cost of the API the scheduler will actually call.
        whole = []
        for _ in range(30):
            slots, toks = sample(4, 1)
            t0 = time.perf_counter()
            rt.prepare(slots, toks)
            if DEV.type == "cuda":
                torch.cuda.synchronize()
            whole.append(time.perf_counter() - t0)
            rt.commit(slots, toks)
        print(
            f"  PLERuntime.prepare, 4-seq decode (64 rows), steady: "
            f"median {np.median(whole)*1e6:.1f} us  p90 {np.percentile(whole, 90)*1e6:.1f} us"
        )

        print("\n  breakdown (median of 30), MADV_WILLNEED prefetch path:")
        print("    shape                          hash      gather     H2D+cast     total")
        rows_seen = []
        for n_seqs, n_tokens in ((1, 1), (4, 1), (8, 1), (1, 256), (1, 2048)):
            ts = np.array([step(n_seqs, n_tokens) for _ in range(30 if n_tokens <= 1 else 5)])
            med = np.median(ts, axis=0) * 1e6
            n_rows = n_seqs * n_tokens * 16
            rows_seen.append((n_rows, med[3]))
            print(
                f"    {n_seqs:2d} seq x {n_tokens:5d} tok ({n_rows:6d} rows) "
                f"{med[0]:9.1f} {med[1]:11.1f} {med[2]:11.1f} {med[3]:10.1f} us"
            )

        # The same shapes with the row table's THREAD POOL forced on, so the crossover is measured
        # rather than inherited. `threaded_min_rows` is the row table's own tunable; restore it.
        keep = src.table.threaded_min_rows
        src.table.threaded_min_rows = 1
        try:
            print(f"\n  same shapes with the os.pread thread pool (workers={src.table.workers}):")
            for n_seqs, n_tokens in ((1, 1), (4, 1), (8, 1), (1, 256), (1, 2048)):
                ts = np.array([step(n_seqs, n_tokens) for _ in range(30 if n_tokens <= 1 else 5)])
                med = np.median(ts, axis=0) * 1e6
                print(
                    f"    {n_seqs:2d} seq x {n_tokens:5d} tok ({n_seqs*n_tokens*16:6d} rows) "
                    f"{med[0]:9.1f} {med[1]:11.1f} {med[2]:11.1f} {med[3]:10.1f} us"
                )
        finally:
            src.table.threaded_min_rows = keep

        b = rt.prepare([1], [rng.integers(0, META["vocab_size"], size=2048, dtype=np.int64)])
        assert b.embeddings.shape == (2048, 2560)

        # sanity on the VALUES: dequantised fp8 rows, so bounded and non-degenerate
        e = b.embeddings.float()
        assert torch.isfinite(e).all()
        assert e.abs().max() <= 0.09
        assert 0.004 < e.std().item() < 0.012
    finally:
        src.close()
