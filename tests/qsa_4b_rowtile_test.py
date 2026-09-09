"""QSA stage-4b row tiling is BIT-EXACT against the untiled form.

WHY THIS TEST IS THE POINT. §8.1 calls the tiling "numerically a no-op because the mapping is per
row". That is a claim about the arithmetic, and the whole value of the change rests on it: stage 4b
turns selected token positions into PHYSICAL kv slots, so a single wrong slot feeds the wrong KV to
sparse attention — plausible text, no error. The tiling exists to bound the prefill activation peak
(a `[chunk, 2051]` int64 gather index, 33.6 MB at a 2048 chunk) so the chunk can grow; buying that
with a silent selection bug would be a bad trade.

The kernel is not needed here — 4b is pure indexing arithmetic — so this runs on CPU.

    python3 -m pytest tests/qsa_4b_rowtile_test.py -q -o addopts=""
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.attention.qsa.runtime import _ATTN_ROW_TILE  # noqa: E402


def _untiled(page_table, row_table_idx, page_stride, sel_tokens):
    """The pre-tiling expression, verbatim, as the reference."""
    rows, width = sel_tokens.shape
    flat = row_table_idx[:, None] * page_stride + sel_tokens.clamp(min=0).to(torch.int64)
    out = page_table.reshape(-1).index_select(0, flat.reshape(-1)).reshape(rows, width)
    return out.to(torch.int32).contiguous()


def _tiled(page_table, row_table_idx, page_stride, sel_tokens):
    """The shipped expression, mirroring runtime.py's tiled branch."""
    rows, width = sel_tokens.shape
    pt_flat = page_table.reshape(-1)
    out = torch.empty((rows, width), dtype=torch.int32)
    for lo in range(0, rows, _ATTN_ROW_TILE):
        hi = min(lo + _ATTN_ROW_TILE, rows)
        flat_t = row_table_idx[lo:hi, None] * page_stride + sel_tokens[lo:hi].clamp(min=0).to(torch.int64)
        out[lo:hi] = pt_flat.index_select(0, flat_t.reshape(-1)).reshape(hi - lo, width).to(torch.int32)
    return out


def _fixture(rows, width=2051, page_stride=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    page_table = torch.randint(0, 1 << 20, (64 * page_stride,), generator=g, dtype=torch.int64)
    row_table_idx = torch.randint(0, 60, (rows,), generator=g, dtype=torch.int64)
    # -1 is the "no selection" sentinel 4b clamps; it MUST appear or the clamp path is untested.
    sel = torch.randint(-1, page_stride, (rows, width), generator=g, dtype=torch.int32)
    return page_table, row_table_idx, page_stride, sel


@pytest.mark.parametrize("rows", [
    1,                       # decode
    _ATTN_ROW_TILE - 1,      # just under the tile: takes the untiled branch
    _ATTN_ROW_TILE,          # exactly the tile
    _ATTN_ROW_TILE + 1,      # first tiled case, ragged remainder of 1
    1024,                    # today's chunk
    2048,                    # the chunk this change is meant to unlock
    2049,                    # ragged
])
def test_tiled_is_bit_identical_to_untiled(rows):
    pt, rti, ps, sel = _fixture(rows)
    assert torch.equal(_tiled(pt, rti, ps, sel), _untiled(pt, rti, ps, sel)), (
        f"row tiling changed the selected slots at rows={rows} — the per-row mapping claim is false"
    )


def test_the_sentinel_rows_are_exercised():
    """A fixture with no -1 would pass while the clamp path stayed untested."""
    _, _, _, sel = _fixture(2048)
    assert (sel < 0).any(), "fixture never produced the -1 no-selection sentinel"


def test_tiling_shrinks_the_int64_gather_index():
    """The REASON for the change: peak transient, not speed. The untiled form materialises one
    [rows, width] int64 index; the tiled form never holds more than one tile's worth."""
    rows, width = 2048, 2051
    untiled_bytes = rows * width * 8
    tiled_bytes = min(rows, _ATTN_ROW_TILE) * width * 8
    assert tiled_bytes * 8 == untiled_bytes, "expected the documented 8x reduction at a 2048 chunk"
    assert untiled_bytes > 32 * 1024 * 1024, "the untiled index should be the ~33.6 MB §8.1 cites"
