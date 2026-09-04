"""NVFP4 (compressed-tensors `nvfp4-pack-quantized`) -> the MXFP4 e2m1 W4A8-WMMA path.

gfx1201 has no FP4 hardware, but the e2m1 W4A8 kernel MXFP4 rides already keeps weights 4-bit in VRAM
and decodes each E2M1 code -> fp8 e4m3 IN-REGISTER at the WMMA (fp8xfp8). NVFP4 has the IDENTICAL 4-bit
E2M1 weight codes as MXFP4; the only difference is the SCALE:

  * MXFP4 : E8M0 uint8 per-32-element block exponent (2^(s-127)).
  * NVFP4 : e4m3 fp8 per-16-element block scale  x  a per-tensor fp32 global scale. TWO PRODUCERS
            SPELL THAT GLOBAL DIFFERENTLY AND RECIPROCALLY: compressed-tensors
            `weight_global_scale` = 448*6/amax (DIVIDE by it), modelopt `weight_scale_2` =
            amax/(448*6) (MULTIPLY by it). See NVFP4_GLOBAL_SCALE_IS_RECIPROCAL.

So NVFP4 is served WITHOUT any weight upconvert (4-bit stays 4-bit) by FOLDING its two-level scale into
the one per-group fp16 scale the kernel already consumes:

    fp16_group_scale[n, g] = weight_scale_e4m3[n, g].to(fp16) / weight_global_scale        (group_size 16)
    fp16_group_scale[n, g] = weight_scale_e4m3[n, g].to(fp16) * weight_scale_2             (modelopt)

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

# THE TWO PRODUCERS OF NVFP4 SHIP RECIPROCAL PER-TENSOR GLOBALS, AND THE NAME IS THE ONLY TELL.
#
#   compressed-tensors  `.weight_global_scale` = 448*6/amax   -> a LARGE number you DIVIDE by
#   modelopt            `.weight_scale_2`      = amax/(448*6) -> a SMALL number you MULTIPLY by
#
# Measured on RadixArk/Qwen3.8-Flash-Next-NVFP4 (modelopt 0.46.0), layer 0 routed experts:
# weight_scale_2 = 2.078e-4, e4m3 block scales 7..256. Multiplying gives |w| mean 0.0105 / max 0.18,
# which matches the same layer's UNQUANTIZED bf16 shared expert (|w| mean 0.0071) — the anchor that
# settles the direction by measurement. Dividing gives |w| mean 242,377, and folded scales up to
# 4.2e6 that OVERFLOW the fp16 the kernel consumes -> inf -> NaN logits from the first MoE block.
#
# `global_field` is therefore REQUIRED, and it is the literal checkpoint suffix the caller read the
# global out of — not a bool and not a producer name. A new NVFP4 loader cannot call this without
# stating which spelling it found, and a spelling that is not listed here raises instead of guessing.
NVFP4_GLOBAL_SCALE_IS_RECIPROCAL = {
    "weight_global_scale": True,  # compressed-tensors: divide
    "weight_scale_2": False,  # modelopt: multiply
}


def fold_nvfp4_scale(
    weight_scale: torch.Tensor, global_scale: torch.Tensor, *, global_field: str
) -> torch.Tensor:
    """(N, K//16) e4m3 block scale + per-tensor f32 global -> (N, K//16) fp16 per-group scale.

    Model-agnostic and layout-agnostic (operates per element, so it composes with the loader's later
    concat/merge/stack). Run at the LEAF, before any weight fusion, so the per-tensor global is
    absorbed into the per-group scale and never has to survive a merge.

    `global_field` names the CHECKPOINT tensor suffix `global_scale` came from and selects the
    convention — see `NVFP4_GLOBAL_SCALE_IS_RECIPROCAL`.
    """
    try:
        reciprocal = NVFP4_GLOBAL_SCALE_IS_RECIPROCAL[global_field]
    except KeyError:
        raise ValueError(
            f"unknown NVFP4 global-scale field {global_field!r}. The per-tensor global is a DIVISOR "
            f"in compressed-tensors ('weight_global_scale') and a MULTIPLIER in modelopt "
            f"('weight_scale_2'); the two are reciprocals and picking wrong is a ~7-orders-of-"
            f"magnitude error that shows up as NaN, not as a load failure. Add the new spelling to "
            f"NVFP4_GLOBAL_SCALE_IS_RECIPROCAL with its measured direction."
        ) from None
    ws = weight_scale.to(torch.float32)
    g = global_scale.to(torch.float32).reshape(())
    folded = (ws / g if reciprocal else ws * g).to(torch.float16)
    # The fp16 store is the kernel's scale dtype, and it is exactly where a wrong convention stops
    # being an arithmetic error and becomes an inf. Checking here makes that class of mistake a loud
    # load-time failure instead of NaN logits 48 layers later, which is the failure mode this repo
    # has been burned by before.
    if not bool(folded.isfinite().all()):
        raise ValueError(
            f"NVFP4 scale fold overflowed fp16 for global_field={global_field!r}: "
            f"{int((~folded.isfinite()).sum())}/{folded.numel()} folded scales are inf/nan "
            f"(e4m3 block scale max {float(ws.max()):.4g}, global {float(g):.6e}). The kernel's "
            f"per-group scale is fp16 (max 65504), so this is a real dequant error and not a "
            f"precision nit — most likely the global-scale convention is inverted for this producer."
        )
    return folded


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
    weight_packed: torch.Tensor,
    weight_scale_e4m3: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    global_field: str = "weight_global_scale",
) -> torch.Tensor:
    """Golden dequant from the RAW NVFP4 checkpoint tensors (pre-fold): E2M1_LUT[code] *
    fold(e4m3_block_scale, global). (N, K) f32 — what the folded e2m1 kernel path must reproduce.

    Takes the same `global_field` convention selector as `fold_nvfp4_scale` and reads the SAME
    `NVFP4_GLOBAL_SCALE_IS_RECIPROCAL` table, so the golden and the served path cannot disagree about
    the direction — but it stays in fp32 rather than calling the fold, because the fold's fp16 store
    is precisely the kernel-side rounding this reference exists to measure. It keeps a default
    (unlike the fold) because it is a diagnostic and every current caller is compressed-tensors."""
    from .mxfp4 import FP4_E2M1_LUT

    if global_field not in NVFP4_GLOBAL_SCALE_IS_RECIPROCAL:
        raise ValueError(f"unknown NVFP4 global-scale field {global_field!r}")
    codes = unpack_e2m1_nibbles(weight_packed).to(torch.int64)
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32, device=weight_packed.device)
    w = lut[codes]  # (N, K)
    bs = weight_scale_e4m3.to(torch.float32).repeat_interleave(NVFP4_GROUP_SIZE, dim=-1)  # (N, K)
    g = global_scale.to(torch.float32).reshape(())
    return w * bs / g if NVFP4_GLOBAL_SCALE_IS_RECIPROCAL[global_field] else w * bs * g


# --- ENCODER: bf16/fp32 -> NVFP4 -----------------------------------------------------------------
# The decoders above consume a checkpoint someone else quantized. This is the other direction, and it
# exists because the Muse-Glimmer DFlash drafter ships bf16 ONLY: there is no NVFP4 build of it, and
# on 2x16 GB the drafter has to be 4-bit to fit beside the target at all (bf16 reserves 5.5 GiB, fp8
# 3.08 GiB, and neither leaves a KV pool).
#
# Doing this by RTN needs no calibration data, and that is not a shortcut — it is the recipe the
# shipping Muse-Glimmer checkpoint itself declares for weights: `observer: memoryless_minmax`,
# `dynamic: false`, i.e. scales derived from each tensor's own amax. (Its `input_activations` block
# asks for FP4 activations, which this engine does not do and cannot: gfx1201 has no FP4 math, so
# NVFP4 here means W4A8 — 4-bit weights decoded to fp8 e4m3 in-register at the WMMA.)
#
# For a DRAFT model the error budget is unusually forgiving: the target verifies every drafted token,
# so a worse drafter costs ACCEPTANCE RATE, never correctness.

# Magnitudes of the OCP E2M1 codebook, i.e. FP4_E2M1_LUT[0:8]. Midpoints between consecutive entries
# are the round-to-nearest bucket edges.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_EDGES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0


def quantize_nvfp4_rtn(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, K) float weight -> (w_packed (N, K//8) int32 E2M1 codes, scales (N, K//16) fp16).

    Returns the ALREADY-FOLDED per-group scale the e2m1 kernel consumes — the same thing
    `fold_nvfp4_scale` produces for a real NVFP4 checkpoint — so the output drops straight into the
    existing W4A8 path with no per-tensor scalar left over.

    Two-level scale, matching compressed-tensors:
        global      = FP8_E4M3_MAX * FP4_E2M1_MAX / amax(|w|)        (per tensor, a DIVISOR)
        scale[n,g]  = e4m3( amax(|w[n,g]|) / FP4_E2M1_MAX * global ) (per 16-element group)
        w ~= E2M1_LUT[code] * scale / global
    The e4m3 ROUND-TRIP is applied here rather than kept in fp32, because that rounding is part of
    the format — leaving it out would produce scales the real decode path could not reproduce.
    """
    assert w.ndim == 2, w.shape
    n, k = w.shape
    assert k % NVFP4_GROUP_SIZE == 0, f"K={k} must be a multiple of {NVFP4_GROUP_SIZE}"
    wf = w.detach().to(torch.float32)
    amax = wf.abs().amax().clamp_min(1e-12)
    global_scale = (FP8_E4M3_MAX * FP4_E2M1_MAX) / amax

    g = wf.reshape(n, k // NVFP4_GROUP_SIZE, NVFP4_GROUP_SIZE)
    group_amax = g.abs().amax(dim=-1)  # (N, K//16)
    s = (group_amax / FP4_E2M1_MAX) * global_scale
    # Round the scale THROUGH e4m3, then read it back: this is the value a real checkpoint stores.
    s_e4m3 = s.clamp(min=1e-12, max=FP8_E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    eff = (s_e4m3 / global_scale).clamp_min(1e-12)  # (N, K//16) effective per-group scale

    q = g / eff.unsqueeze(-1)
    mag = q.abs().clamp(max=FP4_E2M1_MAX)
    edges = torch.tensor(_E2M1_EDGES, dtype=torch.float32, device=wf.device)
    idx = torch.bucketize(mag, edges)  # 0..7, round-to-nearest over the codebook magnitudes
    codes = (idx | (q < 0).to(torch.int64) * 8).to(torch.uint8).reshape(n, k)
    return pack_codes_to_int32(codes), eff.to(torch.float16).contiguous()


def dequantize_nvfp4_folded(w_packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of `quantize_nvfp4_rtn`, for tests: (N,K//8) int32 + (N,K//16) fp16 -> (N,K) float32."""
    from .mxfp4 import FP4_E2M1_LUT

    n, kw = w_packed.shape
    k = kw * 8
    words = w_packed.to(torch.int64)
    codes = torch.stack([(words >> (4 * j)) & 0xF for j in range(8)], dim=-1).reshape(n, k)
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32, device=w_packed.device)
    return lut[codes] * scales.to(torch.float32).repeat_interleave(NVFP4_GROUP_SIZE, dim=-1)
