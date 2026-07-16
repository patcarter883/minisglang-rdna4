"""NVFP4 (compressed-tensors `nvfp4-pack-quantized`) -> the MXFP4 e2m1 W4A8-WMMA path.

gfx1201 has no FP4 hardware, but the e2m1 W4A8 kernel MXFP4 rides already keeps weights 4-bit in VRAM
and decodes each E2M1 code -> fp8 e4m3 IN-REGISTER at the WMMA (fp8xfp8). NVFP4 has the IDENTICAL 4-bit
E2M1 weight codes as MXFP4; the only difference is the SCALE:

  * MXFP4 : E8M0 uint8 per-32-element block exponent (2^(s-127)).
  * NVFP4 : e4m3 fp8 per-16-element block scale  x  a per-tensor fp32 `weight_global_scale`
            (= FP8_E4M3_MAX(448) * FP4_E2M1_MAX(6) / amax, a quantization scale you DIVIDE by).

So NVFP4 is served WITHOUT any weight upconvert (4-bit stays 4-bit) by FOLDING its two-level scale into
the one per-group fp16 scale the kernel already consumes:

    fp16_group_scale[n, g] = weight_scale_e4m3[n, g].to(fp16) / weight_global_scale        (group_size 16)

This fold is exact (fp16 easily holds e4m3/global; E2M1->e4m3 decode is lossless), so NVFP4 lands at
native-NVFP4 fidelity AND native-NVFP4 memory — strictly better than upconverting to fp8 W8A8 (which
doubles VRAM and requantizes to per-channel). After the fold NVFP4 is structurally MXFP4 at group-16:
E2M1-packed weights + an fp16 per-group scale, riding the SAME generic e2m1 kernel MXFP4-at-32 uses
(only group_size==128 is compile-time specialized; every other group_size, incl. 32 today and 16 here,
is the byte-identical runtime `GSc=0` instance, with group_size derived from the scale shape).

`fold_nvfp4_scale` runs in the weight loader at the LEAF (before the GDN in_proj concat / gate-up merge
/ expert stack), which is what lets those fusions — each combining differently-scaled matrices — just
work: once the per-tensor global is folded into the per-group scale, the merges concat fp16 scales the
way they already concat MXFP4's, and no per-tensor scalar ever reaches a merge. `input_global_scale`
(the FP4 activation calibration) is dropped: the kernel quantizes activations to fp8 dynamically.
"""
from __future__ import annotations

import torch

from .mxfp4 import pack_codes_to_int32, unpack_e2m1_nibbles

NVFP4_GROUP_SIZE = 16  # NVFP4 block scale spans 16 elements (vs MXFP4's 32)


def fold_nvfp4_scale(
    weight_scale: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """(N, K//16) e4m3 block scale + per-tensor f32 global -> (N, K//16) fp16 per-group scale.

    fp16 = e4m3.to(fp16) / global. Model-agnostic and layout-agnostic (operates per element, so it
    composes with the loader's later concat/merge/stack). Run at the LEAF, before any weight fusion,
    so the per-tensor global is absorbed into the per-group scale and never has to survive a merge."""
    return (weight_scale.to(torch.float32) / global_scale.to(torch.float32).reshape(())).to(
        torch.float16
    )


def convert_nvfp4_weight(weight_packed: torch.Tensor, weight_scale: torch.Tensor) -> dict:
    """One folded NVFP4 weight matrix -> the e2m1 kernel op layout (identical to MXFP4's).

    weight_packed (N, K//2) uint8 E2M1 nibbles, weight_scale (N, K//16) fp16 (ALREADY folded by
    fold_nvfp4_scale at load) -> {w_packed (N,K//8) int32 verbatim E2M1 codes, scales (N,K//16) fp16,
    group_size 16}. The codes are the same as MXFP4's; only the group_size (16 vs 32) and the scale
    source (pre-folded fp16 vs E8M0->fp16) differ, so this reuses the MXFP4 nibble packer."""
    codes = unpack_e2m1_nibbles(weight_packed)  # (N, K) uint8
    w_packed = pack_codes_to_int32(codes)  # (N, K//8) int32
    return {
        "w_packed": w_packed,
        "scales": weight_scale.to(torch.float16).contiguous(),  # (N, K//16) fp16
        "group_size": NVFP4_GROUP_SIZE,
        "shape": (weight_packed.shape[0], weight_packed.shape[1] * 2),
    }


def convert_nvfp4_moe(weight_packed: torch.Tensor, weight_scale: torch.Tensor) -> dict:
    """Stacked per-expert variant of convert_nvfp4_weight.

    weight_packed (E, N, K//2) uint8, weight_scale (E, N, K//16) fp16 (folded) ->
    {w_packed (E,N,K//8) int32, scales (E,N,K//16) fp16, group_size 16}. Per-expert loop keeps the
    int64 nibble-widen transient to one (N, K) matrix (the OOM guard convert_mxfp4_moe uses)."""
    assert weight_packed.ndim == 3 and weight_scale.ndim == 3, (
        weight_packed.shape, weight_scale.shape
    )
    e, n, k_half = weight_packed.shape
    k = k_half * 2
    dev = weight_packed.device
    w_out = torch.empty((e, n, k // 8), dtype=torch.int32, device=dev)
    s_out = torch.empty((e, n, k // NVFP4_GROUP_SIZE), dtype=torch.float16, device=dev)
    for i in range(e):
        conv = convert_nvfp4_weight(weight_packed[i], weight_scale[i])
        w_out[i] = conv["w_packed"]
        s_out[i] = conv["scales"]
    return {"w_packed": w_out, "scales": s_out, "group_size": NVFP4_GROUP_SIZE, "shape": (e, n, k)}


def dequant_reference(
    weight_packed: torch.Tensor, weight_scale_e4m3: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Golden dequant from the RAW NVFP4 checkpoint tensors (pre-fold): E2M1_LUT[code] *
    (e4m3_block_scale / global_scale). (N, K) f32 — what the folded e2m1 kernel path must reproduce."""
    from .mxfp4 import FP4_E2M1_LUT

    codes = unpack_e2m1_nibbles(weight_packed).to(torch.int64)
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32, device=weight_packed.device)
    w = lut[codes]  # (N, K)
    bs = weight_scale_e4m3.to(torch.float32).repeat_interleave(NVFP4_GROUP_SIZE, dim=-1)  # (N, K)
    return w * bs / global_scale.to(torch.float32).reshape(())
