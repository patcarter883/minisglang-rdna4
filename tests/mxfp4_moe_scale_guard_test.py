"""`_check_moe_scale_pair` must tell MXFP4 E8M0 apart from an NVFP4 two-level scale.

REGRESSION THIS PINS. Both live formats are ONE BYTE wide, so dtype alone cannot separate them; the
discriminator is the zeros slot, which is the same rule the kernel's `MOE_SCALE_FMT_DISPATCH_G` uses
(`w_zeros.defined() && w_zeros.numel() > 0`). MXFP4 group scales used to be widened to fp16 at load
and returned early, so they never reached this guard. Passing the E8M0 byte through natively made
them uint8, and the guard then read EVERY MXFP4 MoE checkpoint as an e4m3 pair missing its global
and refused to boot — found 2026-09-09 on ZAYA1-8B-MXFP4.

    python3 -m pytest tests/mxfp4_moe_scale_guard_test.py -q -o addopts=""
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.quant.kernels import _check_moe_scale_pair  # noqa: E402

E, N, K, G = 4, 64, 256, 32


def _w():
    return torch.zeros((E, N, K // 8), dtype=torch.int32)


def test_mxfp4_e8m0_uint8_scales_with_no_zeros_are_accepted():
    """THE REGRESSION. A uint8 group scale and no zeros slot is MXFP4 E8M0 — the shipped MoE format
    for every mxfp4-pack-quantized checkpoint."""
    _check_moe_scale_pair(_w(), torch.zeros((E, K // G, N), dtype=torch.uint8), None, "w13")


def test_nvfp4_e4m3_without_its_global_is_still_refused():
    """The guard must not be loosened into uselessness: an e4m3 block scale with no global would
    serve the block scale alone (~1/global error per weight), fluent and finite."""
    with pytest.raises(AssertionError, match="REQUIRE the per-output-channel f32 global"):
        _check_moe_scale_pair(
            _w(), torch.zeros((E, K // 16, N), dtype=torch.float8_e4m3fn), None, "w13"
        )


def test_nvfp4_stored_as_uint8_with_a_global_still_validates_the_global():
    """uint8 + zeros present is the two-level form; the global's shape/dtype must still be checked."""
    good = torch.zeros((E, N), dtype=torch.int32)
    _check_moe_scale_pair(_w(), torch.zeros((E, K // 16, N), dtype=torch.uint8), good, "w13")
    bad = torch.zeros((E, N), dtype=torch.float32)          # not BITCAST to int32
    with pytest.raises(AssertionError):
        _check_moe_scale_pair(_w(), torch.zeros((E, K // 16, N), dtype=torch.uint8), bad, "w13")


def test_fp16_folded_scales_still_take_the_early_return():
    _check_moe_scale_pair(_w(), torch.zeros((E, K // G, N), dtype=torch.float16), None, "w13")


def test_an_unknown_scale_dtype_is_still_refused():
    with pytest.raises(AssertionError, match="scales must be"):
        _check_moe_scale_pair(_w(), torch.zeros((E, K // G, N), dtype=torch.bfloat16), None, "w13")
