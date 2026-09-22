#!/usr/bin/env python3
"""int6 group-32 n-gram rows: the decode, and the two-tensor table that feeds it.

WHY THIS TEST LOOKS LIKE THIS. An n-gram embedding has no self-evidently correct value: get the byte
order, the bit order or the +32 bias wrong and you still produce a float32 array of exactly the right
shape, full of plausible numbers, and every downstream check passes while the model's context
representation is quietly wrong. So the fixtures are built to make each of those three choices
FALSIFIABLE on its own:

  * `test_bit_order` packs codes 0,1,2,3 into one word. LSB-first and MSB-first give different
    answers, so a reversed extraction cannot pass.
  * `test_byte_order` uses a code that only appears in the high byte, so a big-endian word assembly
    moves it.
  * `test_bias` packs the stored value 32, which MUST decode to 0. Without the bias it decodes to 32
    -- and a fixture of small positive codes would not have shown it.
  * `test_group_scales_are_per_group` gives every group a distinct scale, so applying one scale to
    the whole row (or transposing the group axis) is visible.

The reference expectations are written by independent indexing, never by calling the implementation.

Run CPU-only:
    python3 tests/ple_int6_rowtable_test.py
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from minisgl.weights.row_table import (  # noqa: E402
    INT6_CODE_OFFSET,
    INT6_GROUP,
    FusedScaleRowTable,
    ShardedRowTable,
    dequant_int6_g32,
    index_safetensors,
    open_qwen4exp_ngram_table,
)

_fail = 0


def check(name, got, want, tol=0.0):
    global _fail
    got_a, want_a = np.asarray(got), np.asarray(want)
    ok = got_a.shape == want_a.shape and (
        np.array_equal(got_a, want_a) if tol == 0 else np.allclose(got_a, want_a, atol=tol)
    )
    if ok:
        print(f"  ok   {name}")
    else:
        _fail += 1
        print(f"  FAIL {name}\n       got  {got_a.ravel()[:8]} shape {got_a.shape}"
              f"\n       want {want_a.ravel()[:8]} shape {want_a.shape}")


def pack_codes(stored: np.ndarray) -> np.ndarray:
    """(n, head_dim) STORED (biased) 6-bit values -> (n, head_dim*6//8) packed bytes.

    Written from the spec, independently of the decoder: 4 codes LSB-first in a 24-bit
    little-endian word.
    """
    n, hd = stored.shape
    assert hd % 4 == 0
    out = np.zeros((n, hd // 4 * 3), dtype=np.uint8)
    for r in range(n):
        for w in range(hd // 4):
            word = 0
            for i in range(4):
                word |= int(stored[r, w * 4 + i] & 0x3F) << (6 * i)
            out[r, w * 3 + 0] = word & 0xFF
            out[r, w * 3 + 1] = (word >> 8) & 0xFF
            out[r, w * 3 + 2] = (word >> 16) & 0xFF
    return out


def test_bit_order():
    stored = np.zeros((1, 32), dtype=np.int64)
    stored[0, :4] = [0, 1, 2, 3]                      # one word, four distinct codes
    scale = np.ones((1, 1), dtype=np.float16)
    got = dequant_int6_g32(pack_codes(stored), scale)
    want = np.zeros((1, 32), dtype=np.float32)
    want[0, :4] = [0 - 32, 1 - 32, 2 - 32, 3 - 32]
    want[0, 4:] = -32.0
    check("LSB-first within the 24-bit word", got, want)
    # An MSB-first reading of the same bytes would give the reversed quartet; prove they differ.
    check("MSB-first would NOT match (fixture is discriminating)",
          np.array_equal(got[0, :4], want[0, :4][::-1]), False)


def test_byte_order():
    stored = np.zeros((1, 32), dtype=np.int64)
    stored[0, 3] = 63                                  # only the 4th code -> top of the high byte
    scale = np.ones((1, 1), dtype=np.float16)
    packed = pack_codes(stored)
    check("code 3 lives in the HIGH byte", packed[0, 2] != 0, True)
    got = dequant_int6_g32(packed, scale)
    check("little-endian word assembly", got[0, 3], np.float32(63 - 32))


def test_bias():
    stored = np.full((1, 32), INT6_CODE_OFFSET, dtype=np.int64)   # stored 32 == signed 0
    got = dequant_int6_g32(pack_codes(stored), np.ones((1, 1), dtype=np.float16))
    check("stored +32 bias decodes to zero", got, np.zeros((1, 32), dtype=np.float32))


def test_group_scales_are_per_group():
    hd, ng = 160, 160 // INT6_GROUP
    rng = np.random.default_rng(7)
    stored = rng.integers(0, 64, size=(3, hd), dtype=np.int64)
    scales = np.array([[1.0, 2.0, 4.0, 8.0, 16.0]] * 3, dtype=np.float16)
    got = dequant_int6_g32(pack_codes(stored), scales)
    want = np.empty((3, hd), dtype=np.float32)
    for r in range(3):
        for g in range(ng):
            for j in range(INT6_GROUP):                # independent indexing, not a reshape
                want[r, g * INT6_GROUP + j] = np.float32(
                    stored[r, g * INT6_GROUP + j] - INT6_CODE_OFFSET
                ) * np.float32(scales[r, g])
    check("per-group scale applied to its own 32 values", got, want)
    check("row width", got.shape, (3, hd))


def write_safetensors(path, tensors):
    """Minimal writer: {name: (dtype_str, np array)}."""
    header, blob, off = {}, bytearray(), 0
    for name, (dt, arr) in tensors.items():
        b = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    import json
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(bytes(blob))


def test_end_to_end_table():
    """Two shards x two tensors, through the real opener and the real gather path."""
    hd, ng, H = 160, 5, 4
    rng = np.random.default_rng(11)
    stored = rng.integers(0, 64, size=(2 * H, hd), dtype=np.int64)
    scales = (rng.random((2 * H, ng)).astype(np.float32) + 0.5).astype(np.float16)
    packed = pack_codes(stored)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "model-00000.safetensors")
        pre = "model.language_model.layers.1.ple.ple_embedding"
        write_safetensors(p, {
            f"{pre}.ngram_embedding.shard_0.weight_packed": ("U8", packed[:H]),
            f"{pre}.ngram_embedding.shard_0.weight_scale": ("F16", scales[:H]),
            f"{pre}.ngram_embedding.shard_1.weight_packed": ("U8", packed[H:]),
            f"{pre}.ngram_embedding.shard_1.weight_scale": ("F16", scales[H:]),
        })
        table, heads = open_qwen4exp_ngram_table([p], layer_prefix=pre)
        check("format detected from the fp16 scale dtype", table.codec, "INT6_G32")
        check("no heads metadata -> None", heads, None)
        check("row_elems", table.row_elems, hd)
        check("row_bytes = packed + scale", table.row_bytes, hd * 6 // 8 + ng * 2)
        check("n_rows spans both shards", table.n_rows, 2 * H)
        ids = np.array([0, 3, 4, 7, 5], dtype=np.int64)
        got = table.gather(ids)
        want = np.stack([
            dequant_int6_g32(packed[i : i + 1], scales[i : i + 1])[0] for i in ids
        ])
        check("gather crosses the shard boundary correctly", got, want)
        # Row 4 is local row 0 of shard 1: a row->shard division error would return row 0 instead.
        check("row 4 is NOT row 0 (shard division is exercised)",
              np.array_equal(got[2], got[0]), False)
        table.close()


def test_mxfp4_scale_is_named_not_guessed():
    """A uint8 (E8M0) scale is the MXFP4 PLE variant and must be refused BY NAME, not decoded as
    int6 -- decoding it as int6 would silently produce garbage of the right shape."""
    global _fail
    hd, H = 160, 2
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "model-00000.safetensors")
        pre = "model.language_model.layers.1.ple.ple_embedding"
        write_safetensors(p, {
            f"{pre}.ngram_embedding.shard_0.weight_packed":
                ("U8", np.zeros((H, hd // 2), dtype=np.uint8)),
            f"{pre}.ngram_embedding.shard_0.weight_scale":
                ("U8", np.zeros((H, hd // INT6_GROUP), dtype=np.uint8)),
        })
        try:
            open_qwen4exp_ngram_table([p], layer_prefix=pre)
        except NotImplementedError as e:
            check("MXFP4 scale refused by name", "MXFP4" in str(e) or "E8M0" in str(e), True)
            return
        except Exception as e:
            _fail += 1
            print(f"  FAIL MXFP4 scale raised {type(e).__name__}, wanted NotImplementedError: {e}")
            return
    _fail += 1
    print("  FAIL MXFP4 scale was accepted instead of refused")


def test_missing_scale_refused():
    global _fail
    hd, H = 160, 2
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "model-00000.safetensors")
        pre = "model.language_model.layers.1.ple.ple_embedding"
        write_safetensors(p, {
            f"{pre}.ngram_embedding.shard_0.weight_packed":
                ("U8", np.zeros((H, hd * 6 // 8), dtype=np.uint8)),
        })
        try:
            open_qwen4exp_ngram_table([p], layer_prefix=pre)
        except KeyError as e:
            check("packed without its scale refused", "weight_scale" in str(e), True)
            return
    _fail += 1
    print("  FAIL a packed shard with no scale tensor was accepted")




GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "fixtures", "qwen4exp", "ple_int6_golden.npz")


def test_golden_against_the_authors_reference():
    """REAL rows from `tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ`, with the expected output
    produced by the CHECKPOINT AUTHOR'S OWN decoder, not by ours.

    This is the check the fixtures above cannot make. They prove our decoder is self-consistent with
    the spec as we READ it; only an externally-derived expectation can catch us having read the spec
    wrong. The expectation was generated by running
    `vllm.models.qwen4_exp.common.ple.dequant_int6_fused_rows` inside `tcclaviger/vllm:dev` on these
    exact bytes (whose docstring states it is bit-identical to libr4d `ple_dequant_i6g32_f16`), and
    it is checked in so the parity survives without that image or the 108 GiB checkpoint.

    It must be BIT-exact. int6 dequant is an integer subtract and a single fp16->f32 multiply; there
    is no rounding freedom to explain a difference away, so `allclose` here would only hide a bug.
    """
    if not os.path.exists(GOLDEN):
        print("  SKIP golden fixture not present")
        return
    z = np.load(GOLDEN)
    got = dequant_int6_g32(z["packed"], np.ascontiguousarray(z["scale_bytes"]).view("<f2"))
    check("bit-exact vs the author's reference on real rows", got, z["expected"])
    # Guard against a degenerate fixture: all-zero or single-valued rows would pass any decoder.
    check("fixture is discriminating (many distinct values)",
          len(np.unique(z["expected"])) > 200, True)


test_golden_against_the_authors_reference.__module__ = __name__

for fn in (test_bit_order, test_byte_order, test_bias, test_group_scales_are_per_group,
           test_end_to_end_table, test_mxfp4_scale_is_named_not_guessed,
           test_missing_scale_refused, test_golden_against_the_authors_reference):
    print(f"\n== {fn.__name__}")
    fn()

print(f"\n{'FAILURES: %d' % _fail if _fail else 'all passed'}")
raise SystemExit(1 if _fail else 0)
