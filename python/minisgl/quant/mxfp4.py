"""MXFP4 (OCP E2M1 weights + E8M0 per-32-block scale) -> W4A8-FP8-WMMA kernel-native layout.

Ported (CPU-parity-verified) from vllm-gfx1201-mxfp4/w4a8_fp8_wmma/mxfp4/convert.py.

The W4A8 fp8-WMMA kernel is a "decode a 4-bit code into fp8 e4m3, then WMMA fp8xfp8" engine. MXFP4 is
therefore NOT a new compute path — it is a different 4-bit *decode table* (the kernel's e2m1_to_e4m3
LUT, selected by `weight_is_e2m1=True`) plus a power-of-two (E8M0) group scale. This module does the
load-time format conversion so the existing kernel container ((N, K//8) packed int32 codes + (N,
K//32) fp16 group scales) carries MXFP4 data:

  * E2M1 weight nibbles are repacked into the kernel's (N, K//8) int32 container UNCHANGED — the
    *codes* are stored verbatim; the kernel's e2m1_to_e4m3 LUT (not the int4 subtract-zp path) turns
    each code into the right fp8 byte. Pure re-packing, no value remap.
  * E8M0 per-group scales (uint8, value = 2^(s-127)) are converted to the kernel's fp16 per-group
    scale array. The kernel epilogue (out_acc * wscale * a_scale) is unchanged.
  * No zero-points: MXFP4 is symmetric, so w_zeros = None (the kernel uses the implicit-symmetric
    e2m1 decode; the torch op asserts w_zeros is empty when weight_is_e2m1).

Checkpoint format (compressed-tensors "mxfp4-pack-quantized"):
  weight_packed : uint8  (N, K//2)   -- 2 E2M1 nibbles/byte, low nibble = lower K index
  weight_scale  : uint8  (N, K//32)  -- E8M0, one shared exponent per 32-element block

Output (kernel-native, matches kernels.w4a8_linear / w4a8_moe weight contract):
  w_packed : int32  (N, K//8)        -- 8 E2M1 codes/word, code j at bits [4j, 4j+3]
  scales   : fp16   (N, K//32)       -- 2^(s-127), per group
  group_size = 32 (the OCP MX block size)

Device-preserving: every op runs on the input tensor's device (post_load already places the
checkpoint tensors on-device), so the large MoE stacks never round-trip through host memory.
"""
from __future__ import annotations

import torch

# OCP E2M1 codebook, indexed by the raw 4-bit code (bit3=sign, bits2-1=exp, bit0=mantissa).
# MUST match vLLM's _FP4_E2M1_LUT and the kernel's tile_config.h::e2m1_to_e4m3 LUT.
FP4_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

OCP_MX_BLOCK_SIZE = 32  # E8M0 shares one exponent per 32-element block
E8M0_BIAS = 127


def unpack_e2m1_nibbles(weight_packed: torch.Tensor) -> torch.Tensor:
    """(N, K//2) uint8 -> (N, K) uint8 codes. Low nibble = lower (even) K index."""
    assert weight_packed.dtype == torch.uint8, weight_packed.dtype
    assert weight_packed.ndim == 2, weight_packed.shape
    n, k_half = weight_packed.shape
    codes = torch.empty((n, k_half * 2), dtype=torch.uint8, device=weight_packed.device)
    codes[:, 0::2] = weight_packed & 0x0F
    codes[:, 1::2] = (weight_packed >> 4) & 0x0F
    return codes


def pack_codes_to_int32(codes: torch.Tensor) -> torch.Tensor:
    """(N, K) uint8 codes -> (N, K//8) int32, code j at bits [4j, 4j+3] (low nibble first).

    Matches the kernel's read: ((word >> (j*4)) & 0xF) for j in 0..7.
    """
    n, k = codes.shape
    assert k % 8 == 0, f"K={k} must be a multiple of 8 for int32 packing"
    c = codes.to(torch.int64) & 0xF                      # widen so shifts don't overflow
    words = torch.zeros((n, k // 8), dtype=torch.int64, device=codes.device)
    for j in range(8):
        words |= c[:, j::8] << (4 * j)
    # Wrapping cast to int32 preserves the 32-bit pattern (incl. a set bit 31 -> negative). The
    # kernel does ((word >> 4*j) & 0xF), and the trailing &0xF recovers the right nibble regardless
    # of the word's sign / arithmetic-shift fill, so a "negative" int32 here is harmless.
    return (words & 0xFFFFFFFF).to(torch.int32)


def e8m0_to_fp16_scales(weight_scale: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """(N, K//32) uint8 E8M0 -> (N, K//32) fp16 scales = 2^(s-127).

    Returns (scales_fp16, info). E8M0 spans 2^-127..2^128, which OVERFLOWS fp16's ~2^15 range; the
    kernel stores fp16 group scales, so we surface any out-of-range exponents rather than silently
    saturating. Real trained MXFP4 block scales sit near the weight magnitude (small exponents), so
    this is informational for typical checkpoints, but a robust path may need fp32 group scales.
    """
    assert weight_scale.dtype == torch.uint8, weight_scale.dtype
    exp = weight_scale.to(torch.int32) - E8M0_BIAS        # true exponent
    # 255 is the E8M0 NaN code; flag if present.
    nan_count = int((weight_scale == 0xFF).sum())
    fp16_max_exp, fp16_min_norm_exp = 15, -14
    over = int((exp > fp16_max_exp).sum())
    under = int(((exp < fp16_min_norm_exp) & (weight_scale != 0)).sum())
    two = torch.tensor(2.0, dtype=torch.float32, device=weight_scale.device)
    scales = torch.pow(two, exp.float()).to(torch.float16)
    info = {
        "exp_min": int(exp.min()), "exp_max": int(exp.max()),
        "fp16_overflow_groups": over, "fp16_subnormal_groups": under,
        "e8m0_nan_groups": nan_count,
        "fp16_range_ok": over == 0 and nan_count == 0,
    }
    return scales, info


# ────────────────────────────────────────────────────────────────────────────────────────────────
# NATIVE E8M0 — the path that does NOT inflate the scale
# ────────────────────────────────────────────────────────────────────────────────────────────────
# `e8m0_to_fp16_scales` above widens a 1-byte block scale to 2 bytes and can saturate. The kernel
# now reads E8M0 directly (`Int4E8m0GemvLoader`, rdna4-hip-kernels 5155b68), so the conversion is
# avoidable on both counts:
#
#   BYTES   group-32 e2m1 + fp16 scale = 0.5 + 2/32 = 0.5625 B/weight
#           group-32 e2m1 + E8M0 scale = 0.5 + 1/32 = 0.53125  -> 5.6% fewer bytes read, and HALF
#           the resident scale bytes (which on an offloading serve is expert-cache slots).
#   RANGE   E8M0 spans 2^-127..2^127; fp16 saturates outside 2^-14..2^15. The fp16 path logs
#           `fp16_range_ok` and PROCEEDS, so an out-of-window checkpoint is served with saturated
#           scales and no failure. Nothing to saturate here.
#
# The E8M0 codes are passed through UNTOUCHED — this function deliberately does no arithmetic on
# them, because the decode (2^(s-127), one shift into the fp32 exponent field, E8M0's bias being
# exactly fp32's) belongs in the kernel where it is free.
def e8m0_scale_health(weight_scale: torch.Tensor) -> dict:
    """Count the codes the kernel decode treats specially. Cheap, and it runs BEFORE boot.

    The native path cannot saturate, so the only remaining special codes are the two ends of the
    domain: 0 (true value 2^-127, an fp32 subnormal the kernel spells out) and 255 (the E8M0 NaN
    code, which the kernel's shift lands on +inf). Both are counted so a checkpoint carrying them
    is a KNOWN quantity rather than a surprise in the accumulator.
    """
    assert weight_scale.dtype == torch.uint8, weight_scale.dtype
    exp = weight_scale.to(torch.int32) - E8M0_BIAS
    return {
        "exp_min": int(exp.min()), "exp_max": int(exp.max()),
        "e8m0_nan_groups": int((weight_scale == 0xFF).sum()),
        "e8m0_zero_code_groups": int((weight_scale == 0).sum()),
        # Recorded for contrast with the fp16 path: how much WOULD have been saturated.
        "would_saturate_fp16": int(((exp > 15) | ((exp < -14) & (weight_scale != 0))).sum()),
    }


def convert_mxfp4_moe_e8m0(weight_packed: torch.Tensor,
                           weight_scale: torch.Tensor) -> dict:
    """Stacked per-expert MoE conversion that KEEPS the E8M0 scale byte.

    (E, N, K//2) uint8 packed + (E, N, K//32) uint8 E8M0 ->
    {w_packed (E,N,K//8) int32, scales (E,N,K//32) uint8 E8M0, w_zeros None, group_size 32}.
    Identical to `convert_mxfp4_moe` except the scale is passed through rather than widened.
    """
    assert weight_packed.ndim == 3 and weight_scale.ndim == 3, (
        weight_packed.shape, weight_scale.shape)
    e, n, k_half = weight_packed.shape
    k = k_half * 2
    es, ns, k_groups = weight_scale.shape
    assert (es, ns) == (e, n), f"shape mismatch: weight {(e, n)} vs scale {(es, ns)}"
    assert k_groups == k // OCP_MX_BLOCK_SIZE, (
        f"scale groups {k_groups} != K//{OCP_MX_BLOCK_SIZE} = {k // OCP_MX_BLOCK_SIZE}")
    # PER-EXPERT LOOP, for the reason `convert_mxfp4_moe` documents and not for symmetry: a single
    # flattened unpack over (E*N, K) materialises the int64 nibble-widen transient for the WHOLE
    # stack, a multi-GB spike that OOMs the 16 GB card once the model is resident. The scale needs
    # no loop at all here — that is the point of this path — so only the weight is chunked.
    w_out = torch.empty((e, n, k // 8), dtype=torch.int32, device=weight_packed.device)
    for i in range(e):
        w_out[i] = pack_codes_to_int32(unpack_e2m1_nibbles(weight_packed[i]))
    return {
        "w_packed": w_out,                        # (E, N, K//8) int32, E2M1 codes
        "scales": weight_scale.contiguous(),      # (E, N, K//32) uint8 E8M0 — UNTOUCHED
        "w_zeros": None,                          # symmetric; no zero-points, no NVFP4 global
        "group_size": OCP_MX_BLOCK_SIZE,
        "scale_is_e8m0": True,                    # the dispatch flag; see the note below
        "scale_info": e8m0_scale_health(weight_scale),
        "shape": (e, n, k),
    }


def convert_mxfp4_weight(weight_packed: torch.Tensor,
                         weight_scale: torch.Tensor) -> dict:
    """Full conversion for one MXFP4 linear/expert weight matrix.

    weight_packed (N, K//2) uint8, weight_scale (N, K//32) uint8 ->
    {w_packed (N,K//8) int32, scales (N,K//32) fp16, w_zeros None, group_size 32, scale_info}.
    """
    n, k_half = weight_packed.shape
    k = k_half * 2
    ns, k_groups = weight_scale.shape
    assert ns == n, f"row mismatch: weight {n} vs scale {ns}"
    assert k_groups == k // OCP_MX_BLOCK_SIZE, (
        f"scale groups {k_groups} != K//{OCP_MX_BLOCK_SIZE} = {k // OCP_MX_BLOCK_SIZE}")

    codes = unpack_e2m1_nibbles(weight_packed)
    w_packed = pack_codes_to_int32(codes)
    scales, scale_info = e8m0_to_fp16_scales(weight_scale)
    return {
        "w_packed": w_packed,          # (N, K//8) int32, E2M1 codes (decode via e2m1_to_e4m3)
        "scales": scales,              # (N, K//32) fp16
        "w_zeros": None,               # symmetric
        "group_size": OCP_MX_BLOCK_SIZE,
        "scale_info": scale_info,
        "shape": (n, k),
    }


def convert_mxfp4_moe(weight_packed: torch.Tensor,
                      weight_scale: torch.Tensor) -> dict:
    """NOT ON THE SERVING PATH. The MoE path takes `convert_mxfp4_moe_e8m0`; this fp16-widening
    variant is kept ONLY as the parity comparand in `tests/mxfp4_e8m0_test.py`, which proves the
    native E8M0 scales agree with it bit-for-bit inside fp16's exponent window.

    It is NOT reachable from a serve and NOT selectable by any flag — deliberately, per the
    project's no-env-gating rule: if a faster path merges it is ON, and a "control leg" reproducing
    the old behaviour from the same binary is an emulated baseline, not the old code. If this ever
    needs to be measured against for real, check out the commit before the switch.

    Stacked per-expert MoE variant of convert_mxfp4_weight.

    weight_packed (E, N, K//2) uint8, weight_scale (E, N, K//32) uint8 ->
    {w_packed (E,N,K//8) int32, scales (E,N,K//32) fp16, w_zeros None, group_size 32}.
    kernels.w4a8_moe takes exactly this 3D (E,N,*) layout.

    PER-EXPERT loop (not a flatten-to-2D): the repack/scale math is per-row (E-independent), but a
    single flattened `convert_mxfp4_weight` over (E*N, K) would materialise the int64 nibble-widen
    transient for the WHOLE stack at once (a multi-GB spike that OOMs the 16 GB card when the model
    weights are already resident). Writing each expert into the pre-allocated output keeps the
    transient to one (N, K) matrix; the output is ~the same bytes as the input, so no net growth.
    """
    assert weight_packed.ndim == 3 and weight_scale.ndim == 3, (
        weight_packed.shape, weight_scale.shape)
    e, n, k_half = weight_packed.shape
    k = k_half * 2
    dev = weight_packed.device
    w_out = torch.empty((e, n, k // 8), dtype=torch.int32, device=dev)
    s_out = torch.empty((e, n, k // OCP_MX_BLOCK_SIZE), dtype=torch.float16, device=dev)
    exp_min, exp_max, over, under, nan = 127, -128, 0, 0, 0
    for i in range(e):
        conv = convert_mxfp4_weight(weight_packed[i], weight_scale[i])
        w_out[i] = conv["w_packed"]
        s_out[i] = conv["scales"]
        si = conv["scale_info"]
        exp_min = min(exp_min, si["exp_min"]); exp_max = max(exp_max, si["exp_max"])
        over += si["fp16_overflow_groups"]; under += si["fp16_subnormal_groups"]
        nan += si["e8m0_nan_groups"]
    return {
        "w_packed": w_out,
        "scales": s_out,
        "w_zeros": None,
        "group_size": OCP_MX_BLOCK_SIZE,
        "scale_info": {
            "exp_min": exp_min, "exp_max": exp_max, "fp16_overflow_groups": over,
            "fp16_subnormal_groups": under, "e8m0_nan_groups": nan,
            "fp16_range_ok": over == 0 and nan == 0,
        },
        "shape": (e, n, k),
    }


def dequant_reference(weight_packed: torch.Tensor,
                      weight_scale: torch.Tensor) -> torch.Tensor:
    """Reference dequant matching vLLM's _FP4_E2M1_LUT * 2^(s-127). (N, K) float32.

    The GOLDEN value the kernel decode path must reproduce; used by the host bit-exactness test.
    """
    codes = unpack_e2m1_nibbles(weight_packed).to(torch.int64)
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32, device=weight_packed.device)
    w = lut[codes]                                              # (N, K)
    exp = weight_scale.to(torch.int32) - E8M0_BIAS
    two = torch.tensor(2.0, device=weight_packed.device)
    scale = torch.pow(two, exp.float())                        # (N, K//32)
    scale = scale.repeat_interleave(OCP_MX_BLOCK_SIZE, dim=-1)  # (N, K)
    return w * scale
