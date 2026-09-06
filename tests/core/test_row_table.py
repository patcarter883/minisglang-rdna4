"""Tests for the NVMe-resident n-gram row table.

Most run on a synthetic safetensors file so they need no checkpoint. The ones marked
`needs_ple_checkpoint` run against a real `model-plefp8-*.safetensors` when
`MINISGL_PLE_FILES` names one (colon-separated) — they are what proved the fp8 decode and the
shard arithmetic against the actual Qwen3.8-Flash-Next-NVFP4 weights.

Everything here targets a failure that is SILENT: a wrong fp8 decode, an off-by-one in
shard/local addressing, or a missing scale all produce plausible embeddings and surface only as
degraded output quality.
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np
import pytest

from minisgl.weights.row_table import (
    _E4M3_LUT,
    NgramHeads,
    ShardedRowTable,
    index_safetensors,
    open_qwen4exp_ngram_table,
)

PREFIX = "model.language_model.layers.1.ple.ple_embedding"


def _write_safetensors(path, tensors):
    """tensors: name -> (dtype_str, shape, raw bytes)."""
    header, off = {}, 0
    for name, (dt, shape, raw) in tensors.items():
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _n, (_d, _s, raw) in tensors.items():
            f.write(raw)


def _make_table_file(path, shard_ids, rows=8, row_elems=16, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    for sid in shard_ids:
        raw = rng.integers(0, 256, size=rows * row_elems, dtype=np.uint8).tobytes()
        data[f"{PREFIX}.ngram_embedding.shard_{sid}.weight"] = ("F8_E4M3", [rows, row_elems], raw)
    _write_safetensors(path, data)
    return data


# -- fp8 ---------------------------------------------------------------------


def test_e4m3_lut_matches_torch():
    torch = pytest.importorskip("torch")
    ref = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32).numpy()
    both_nan = np.isnan(_E4M3_LUT) & np.isnan(ref)
    assert (both_nan | (_E4M3_LUT == ref)).all()
    assert int(np.isnan(ref).sum()) == 2  # e4m3fn has exactly two NaN encodings: 0x7F / 0xFF


def test_e4m3_lut_has_no_infinities():
    """e4m3*fn* has no Inf — if this fails the exponent handling has drifted to IEEE e4m3."""
    assert not np.isinf(_E4M3_LUT).any()


# -- shard addressing --------------------------------------------------------


def test_sparse_shard_ids_address_global_rows(tmp_path):
    """The checkpoint assigns shards to files in STRING order, so one file holds shard_0, shard_1,
    shard_10, shard_100... A table built from it must still address rows by the shard's real index,
    or every lookup silently reads the wrong band."""
    p = tmp_path / "ple.safetensors"
    ids = [0, 1, 10, 100]
    _make_table_file(p, ids, rows=8, row_elems=16)
    idx = index_safetensors([str(p)])
    locs = [idx[f"{PREFIX}.ngram_embedding.shard_{i}.weight"] for i in ids]
    t = ShardedRowTable(locs, scale=1.0, shard_ids=ids)

    assert t.shard_ids == ids
    assert t.n_shards_total == 101
    assert t.n_rows == 101 * 8
    assert not t.complete

    # a row in shard 10 must read shard 10's bytes, not slot 2's
    raw_direct = np.frombuffer(
        open(p, "rb").read(), dtype=np.uint8
    )  # whole file; offsets checked below
    loc10 = idx[f"{PREFIX}.ngram_embedding.shard_10.weight"]
    expect = raw_direct[loc10.offset + 3 * 16 : loc10.offset + 4 * 16]
    got = t.gather_raw([10 * 8 + 3])[0]
    assert (got == expect).all()
    t.close()


def test_missing_shard_raises_rather_than_returning_wrong_rows(tmp_path):
    p = tmp_path / "ple.safetensors"
    ids = [0, 5]
    _make_table_file(p, ids)
    idx = index_safetensors([str(p)])
    locs = [idx[f"{PREFIX}.ngram_embedding.shard_{i}.weight"] for i in ids]
    t = ShardedRowTable(locs, shard_ids=ids)
    with pytest.raises(KeyError, match="not loaded"):
        t.gather_raw([2 * 8 + 1])  # shard 2 was never supplied
    t.close()


def test_non_uniform_shard_heights_rejected(tmp_path):
    p = tmp_path / "ple.safetensors"
    rng = np.random.default_rng(0)
    _write_safetensors(
        p,
        {
            f"{PREFIX}.ngram_embedding.shard_0.weight": (
                "F8_E4M3", [8, 16], rng.integers(0, 256, 8 * 16, dtype=np.uint8).tobytes()),
            f"{PREFIX}.ngram_embedding.shard_1.weight": (
                "F8_E4M3", [9, 16], rng.integers(0, 256, 9 * 16, dtype=np.uint8).tobytes()),
        },
    )
    idx = index_safetensors([str(p)])
    with pytest.raises(ValueError, match="height"):
        ShardedRowTable(
            [idx[f"{PREFIX}.ngram_embedding.shard_{i}.weight"] for i in (0, 1)], shard_ids=[0, 1]
        )


def test_enumeration_does_not_stop_at_the_first_gap(tmp_path):
    """Regression: probing shard_0, shard_1, shard_2, ... and stopping at the first miss found 2 of
    13 shards in the real checkpoint and silently built a table of the wrong height."""
    p = tmp_path / "ple.safetensors"
    _make_table_file(p, [0, 1, 10, 11])
    t, heads = open_qwen4exp_ngram_table([str(p)], scale_override=1.0)
    assert t.shard_ids == [0, 1, 10, 11]
    assert heads is None
    t.close()


def test_missing_scale_refuses_rather_than_defaulting_to_one(tmp_path):
    """A silent scale of 1.0 would inflate every embedding ~5000x and still 'work'."""
    p = tmp_path / "ple.safetensors"
    _make_table_file(p, [0])
    with pytest.raises(KeyError, match="weight_scale"):
        open_qwen4exp_ngram_table([str(p)])


def test_scale_is_applied(tmp_path):
    p = tmp_path / "ple.safetensors"
    _make_table_file(p, [0])
    a, _ = open_qwen4exp_ngram_table([str(p)], scale_override=1.0)
    b, _ = open_qwen4exp_ngram_table([str(p)], scale_override=0.5)
    va, vb = a.gather([0]), b.gather([0])
    assert np.allclose(np.nan_to_num(va) * 0.5, np.nan_to_num(vb))
    a.close()
    b.close()


def test_out_of_range_row_raises(tmp_path):
    p = tmp_path / "ple.safetensors"
    _make_table_file(p, [0])
    t, _ = open_qwen4exp_ngram_table([str(p)], scale_override=1.0)
    with pytest.raises(IndexError):
        t.gather([t.n_rows])
    with pytest.raises(IndexError):
        t.gather([-1])
    t.close()


# -- head addressing ---------------------------------------------------------


def _real_heads():
    vocab = np.array(
        [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
         20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171],
        dtype=np.int64,
    )
    offsets = np.concatenate([[0], np.cumsum(vocab)[:-1]])
    return NgramHeads(offsets=offsets, vocab_sizes=vocab)


def test_real_head_geometry_validates():
    h = _real_heads()
    h.validate(320_001_536)
    assert h.n_heads == 16
    assert int(h.offsets[-1] + h.vocab_sizes[-1]) == 320_001_446  # 90 rows of padding


def test_head_ids_stay_inside_their_own_band():
    """A hash collision must never cross heads — that would read another head's embeddings."""
    h = _real_heads()
    rng = np.random.default_rng(0)
    ids = h.row_ids(rng.integers(0, 2**62, size=(64, 16), dtype=np.int64))
    for k in range(16):
        lo, hi = h.offsets[k], h.offsets[k] + h.vocab_sizes[k]
        assert (ids[:, k] >= lo).all() and (ids[:, k] < hi).all()


def test_offsets_must_be_the_prefix_sum():
    bad = NgramHeads(offsets=np.array([0, 5]), vocab_sizes=np.array([4, 4]))
    with pytest.raises(ValueError, match="prefix sum"):
        bad.validate(100)


def test_heads_cannot_address_past_the_table():
    h = NgramHeads(offsets=np.array([0, 4]), vocab_sizes=np.array([4, 4]))
    with pytest.raises(ValueError, match="address"):
        h.validate(7)


# -- against the real checkpoint ---------------------------------------------

_PLE = os.environ.get("MINISGL_PLE_FILES", "")
needs_ckpt = pytest.mark.skipif(not _PLE, reason="set MINISGL_PLE_FILES to a model-plefp8-*.safetensors")


@needs_ckpt
def test_real_checkpoint_geometry_and_decode():
    files = _PLE.split(":")
    t, _ = open_qwen4exp_ngram_table(files, scale_override=0.00019931793)
    assert t.dtype == "F8_E4M3"
    assert t.row_elems == 160 and t.row_bytes == 160
    assert t.rows_per_shard == 2_500_012

    # gathered bytes must equal the offset computed by hand from the header
    sid = t.shard_ids[2]
    hdr, data_start = __import__("minisgl.weights.row_table", fromlist=["x"]).read_safetensors_header(
        t.shards[2].path
    )
    beg = hdr[f"{PREFIX}.ngram_embedding.shard_{sid}.weight"]["data_offsets"][0]
    local = 1_234_567 % t.rows_per_shard
    with open(t.shards[2].path, "rb") as f:
        f.seek(data_start + beg + local * 160)
        expect = np.frombuffer(f.read(160), dtype=np.uint8)
    assert (t.gather_raw([sid * t.rows_per_shard + local])[0] == expect).all()

    # decoded values must match the checkpoint's published distribution
    present = np.array(t.shard_ids, dtype=np.int64)
    rng = np.random.default_rng(0)
    sh = present[rng.integers(0, present.size, size=5000)]
    rows = sh * t.rows_per_shard + rng.integers(0, t.rows_per_shard, size=5000)
    samp = t.gather(rows)
    assert np.abs(samp).max() <= 0.09  # inspection: absmax 0.08935546875
    assert 0.006 < samp.std() < 0.009  # inspection: std 0.00763010
    t.close()
