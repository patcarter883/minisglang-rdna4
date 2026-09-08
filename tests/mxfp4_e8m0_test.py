"""MXFP4 native-E8M0 scale path: does it reproduce the fp16 path, and where does it beat it?

The claim being tested is narrow and checkable: passing the E8M0 byte through instead of widening it
to fp16 changes NOTHING about the dequantized weight where fp16 could represent the scale, and is
STRICTLY BETTER where it could not. Both halves matter — the first is the regression gate, the
second is the reason to do it at all.

    python3 -m pytest tests/mxfp4_e8m0_test.py -q -o addopts=""
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.quant import mxfp4  # noqa: E402

E8M0_BIAS = mxfp4.E8M0_BIAS
GROUP = mxfp4.OCP_MX_BLOCK_SIZE


def _decode_e8m0_like_the_kernel(codes: torch.Tensor) -> torch.Tensor:
    """The kernel's decode, in numpy terms: reinterpret (s << 23) as fp32.

    This is the host mirror of `E8m0GroupScale::decode` in gemv_decode.h. If the two ever disagree
    the test is worthless, so it is written as the same bit operation rather than as `2.0 ** (s-127)`
    — which is the thing being verified, not the definition.
    """
    s = codes.to(torch.int32)
    bits = (s << 23).to(torch.int32)
    out = bits.view(torch.float32).clone()
    out[s == 0] = 2.0 ** -127          # true value; a bare shift flushes it to +0
    return out


def test_shift_decode_equals_two_to_the_exponent():
    """The whole scheme rests on E8M0's bias being exactly fp32's. Check every valid code."""
    codes = torch.arange(1, 255, dtype=torch.uint8)
    got = _decode_e8m0_like_the_kernel(codes)
    want = torch.pow(torch.tensor(2.0, dtype=torch.float64), codes.to(torch.float64) - E8M0_BIAS)
    assert torch.equal(got.double(), want), "the shift decode is not 2^(s-127) somewhere in [1,254]"


def test_native_path_keeps_the_scale_byte():
    """The point of the change: 1 byte per group, not 2, and the codes untouched."""
    E, N, K = 2, 8, 128
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8)
    scale = torch.randint(100, 140, (E, N, K // GROUP), dtype=torch.uint8)
    conv = mxfp4.convert_mxfp4_moe_e8m0(packed, scale)
    assert conv["scales"].dtype == torch.uint8
    assert torch.equal(conv["scales"], scale), "E8M0 codes must pass through untouched"
    assert conv["scale_is_e8m0"] is True
    assert conv["w_zeros"] is None
    assert conv["group_size"] == GROUP
    old = mxfp4.convert_mxfp4_moe(packed, scale)
    assert old["scales"].dtype == torch.float16
    assert conv["scales"].numel() * 1 * 2 == old["scales"].numel() * 2, "scale bytes should halve"
    # the WEIGHTS are the same object either way — only the scale representation changed
    assert torch.equal(conv["w_packed"], old["w_packed"])


def test_agrees_with_fp16_where_fp16_is_exact():
    """REGRESSION GATE. Inside fp16's exponent window the two paths must produce the SAME scale, so
    switching a healthy checkpoint over cannot move a single output bit."""
    codes = torch.arange(E8M0_BIAS - 14, E8M0_BIAS + 16, dtype=torch.uint8)   # exp -14..15
    native = _decode_e8m0_like_the_kernel(codes)
    as_fp16 = mxfp4.e8m0_to_fp16_scales(codes.view(1, -1))[0].view(-1).float()
    assert torch.equal(native, as_fp16), "native and fp16 disagree INSIDE fp16's range"


def test_beats_fp16_outside_its_range():
    """THE REASON TO DO IT. Outside fp16's window the existing path saturates to inf or flushes to
    zero and the serve proceeds on a logged warning; the native path stays exact."""
    hi = torch.tensor([E8M0_BIAS + 40], dtype=torch.uint8)     # 2^40, far above fp16's 2^15
    lo = torch.tensor([E8M0_BIAS - 40], dtype=torch.uint8)     # 2^-40, far below 2^-14
    for codes, what in ((hi, "overflow"), (lo, "underflow")):
        native = _decode_e8m0_like_the_kernel(codes)
        fp16 = mxfp4.e8m0_to_fp16_scales(codes.view(1, -1))[0].view(-1).float()
        exact = torch.pow(torch.tensor(2.0, dtype=torch.float64),
                          codes.to(torch.float64) - E8M0_BIAS)
        assert torch.equal(native.double(), exact), f"native path is wrong on {what}"
        assert not torch.equal(fp16.double(), exact), (
            f"fp16 path unexpectedly survived {what} — the premise of this change is that it does not")


def test_health_report_counts_what_the_fp16_path_would_have_broken():
    """`would_saturate_fp16` is the number that says whether a given checkpoint was ever at risk."""
    scale = torch.cat([
        torch.full((30,), E8M0_BIAS, dtype=torch.uint8),        # healthy
        torch.full((2,), E8M0_BIAS + 40, dtype=torch.uint8),    # would overflow fp16
    ]).view(1, 1, 32)
    info = mxfp4.e8m0_scale_health(scale)
    assert info["would_saturate_fp16"] == 2
    assert info["e8m0_nan_groups"] == 0
    assert info["exp_max"] == 40
