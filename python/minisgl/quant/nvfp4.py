"""NVFP4 (compressed-tensors `nvfp4-pack-quantized`) -> the MXFP4 e2m1 W4A8-WMMA path.

gfx1201 has no FP4 hardware, but the e2m1 W4A8 kernel MXFP4 rides already keeps weights 4-bit in VRAM
and decodes each E2M1 code -> fp8 e4m3 IN-REGISTER at the WMMA (fp8xfp8). NVFP4 has the IDENTICAL 4-bit
E2M1 weight codes as MXFP4; the only difference is the SCALE:

  * MXFP4 : E8M0 uint8 per-32-element block exponent (2^(s-127)).
  * NVFP4 : e4m3 fp8 per-16-element block scale  x  a per-tensor fp32 global scale. TWO PRODUCERS
            SPELL THAT GLOBAL DIFFERENTLY AND RECIPROCALLY: compressed-tensors
            `weight_global_scale` = 448*6/amax (DIVIDE by it), modelopt `weight_scale_2` =
            amax/(448*6) (MULTIPLY by it). See NVFP4_GLOBAL_SCALE_IS_RECIPROCAL.

TWO WAYS TO SERVE THAT SCALE, AND THEY ARE NOT EQUIVALENT.

**(1) SPLIT — `split_nvfp4_scale`, the default for MoE experts.** Keep both levels: the e4m3 block
scale passes through BYTE-VERBATIM (1 B per 16 weights) and the per-tensor global is normalised to a
MULTIPLIER and broadcast to a per-OUTPUT-CHANNEL f32 vector. The kernel consumes them as two levels
(`w4a8_tile::E4m3GroupScaleGlobal` in `rdna4-hip-kernels/fp8_wmma/fp8_wmma_rocm/tile_config.h`), with
the (E, N) f32 global riding the existing `w_zeros` pointer slot — NVFP4 is symmetric, so that slot is
null and there is no op-schema change (precedent: `NlInt8Loader` threads an NL codebook the same
way). This is the ACCURATE arm and it is also the SMALL one: 1 byte of block scale instead of 2.

**(2) FOLD — `fold_nvfp4_scale`, the legacy arm, still the DENSE-linear path.** Collapse both levels
into one fp16 per-group scale:

    fp16_group_scale[n, g] = weight_scale_e4m3[n, g].to(fp16) / weight_global_scale        (group_size 16)
    fp16_group_scale[n, g] = weight_scale_e4m3[n, g].to(fp16) * weight_scale_2             (modelopt)

**THIS FOLD IS LOSSY. THE DOCSTRING HERE USED TO CLAIM IT WAS EXACT; IT IS NOT.** fp16 carries an
11-bit significand, and `e4m3_block * global` is a 4-bit significand times an arbitrary f32 one — the
product does not land on an fp16 grid point, so every single group scale is rounded. MEASURED against
an fp64 golden dequant on real `Qwen3.8-Flash-Next-NVFP4` layer-0 expert tensors: **rel-err max
4.37e-04 / mean 1.4-2.4e-04 on EVERY weight**, and **1.2e-03..3.2e-03 through a 4-row GEMV**. The
split arm is exact to f32 on the same tensors. Independently corroborated by the CPU MoE port, which
measured the checkpoint-native e4m3 layout at 1.99e-07 vs the fp16 fold's 3.58e-04 (~1800x) on a
working kernel. So the split is an ACCURACY FIX that also saves bytes, not a memory optimisation that
costs accuracy — and the old claim of exactness is the reason nobody looked.

Either way NVFP4 is served WITHOUT any weight upconvert (4-bit stays 4-bit) and is structurally MXFP4
at group-16: E2M1-packed weights + a per-group scale, riding the SAME generic e2m1 kernel MXFP4-at-32
uses (only group_size==128 is compile-time specialized; every other group_size, incl. 32 today and 16
here, is the byte-identical runtime `GSc=0` instance, with group_size derived from the scale shape).

WHY THE GLOBAL IS A PER-OUTPUT-CHANNEL VECTOR AND NOT A SCALAR. Both entry points run in the weight
loader at the LEAF — before the GDN in_proj concat, the gate/up merge and the per-expert stack —
because those fusions each combine DIFFERENTLY-SCALED matrices and a surviving per-tensor scalar could
not describe the result. The fold discharged that by absorbing the scalar. The split discharges it by
shape instead: after the gate|up merge the global is constant on each contiguous OUTPUT-CHANNEL range,
which is exactly what an N-vector expresses, so `torch.cat(dim=0)` and the expert stack carry it with
no special case. (Measured across 1536 experts of layers 0/23/47: `gate_proj` and `up_proj` share the
same `weight_scale_2` in 1536/1536 cases, and `down_proj` has 132-281 distinct per-expert values. On
THIS checkpoint the vector is therefore constant per expert per container — but that is a PRODUCER
property, not a format guarantee, so the contract stays the N-vector.)

`input_global_scale` (the FP4 activation calibration) is dropped on both arms: the kernel quantizes
activations to fp8 dynamically.
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


_GLOBAL_FIELD_HELP = (
    "The per-tensor global is a DIVISOR in compressed-tensors ('weight_global_scale') and a "
    "MULTIPLIER in modelopt ('weight_scale_2'); the two are reciprocals and picking wrong is a "
    "~7-orders-of-magnitude error that shows up as NaN, not as a load failure. Add the new spelling "
    "to NVFP4_GLOBAL_SCALE_IS_RECIPROCAL with its measured direction."
)


def _global_is_reciprocal(global_field: str) -> bool:
    try:
        return NVFP4_GLOBAL_SCALE_IS_RECIPROCAL[global_field]
    except KeyError:
        raise ValueError(
            f"unknown NVFP4 global-scale field {global_field!r}. {_GLOBAL_FIELD_HELP}"
        ) from None


def nvfp4_global_multiplier(global_scale: torch.Tensor, *, global_field: str) -> torch.Tensor:
    """The per-tensor global as a DIRECTION-NORMALISED f32 MULTIPLIER, 0-dim.

    THE KERNEL ONLY EVER MULTIPLIES. `E4m3GroupScaleGlobal::epi` does `acc * global[nc]` and has no
    divide, deliberately — a policy that could divide would have to carry the convention into the
    kernel, where it is unobservable. So direction normalisation is a host-side loader job and this
    is the ONE place it happens: compressed-tensors' divisor is reciprocated here, modelopt's
    multiplier passes through, and everything downstream (loader, merge, stack, container, kernel)
    handles exactly one convention.

    THE DIRECTION IS ASSERTED, NOT ASSUMED. Using the documented (divisor) sign on this checkpoint's
    `weight_scale_2` yields |W| ~ 4.2e6 against a true rms of 0.0135 — finite, non-NaN in f32, and
    therefore capable of loading "successfully" and serving garbage. `NVFP4_GLOBAL_SCALE_IS_RECIPROCAL`
    is the only tell, and the caller must name the checkpoint field it read.
    """
    reciprocal = _global_is_reciprocal(global_field)
    g = global_scale.to(torch.float32).reshape(())
    if not bool(torch.isfinite(g)) or float(g) <= 0.0:
        raise ValueError(
            f"NVFP4 global scale for field {global_field!r} is {float(g)!r}; it must be finite and "
            f"strictly positive (it is amax-derived on both producers). {_GLOBAL_FIELD_HELP}"
        )
    mul = (1.0 / g) if reciprocal else g
    if not bool(torch.isfinite(mul)) or float(mul) <= 0.0:
        raise ValueError(
            f"NVFP4 global scale for field {global_field!r} normalised to a non-finite/non-positive "
            f"multiplier {float(mul)!r} from raw {float(g)!r} (reciprocal={reciprocal})."
        )
    return mul


def split_nvfp4_scale(
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    global_field: str,
    out_features: "int | None" = None,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """(N, K//16) e4m3 block scale + per-tensor global -> (e4m3 block scale VERBATIM, (N,) f32 global).

    THE ACCURATE ARM, and the default for MoE experts. Nothing is computed on the block scale: the
    checkpoint's e4m3 bytes are the kernel's `E4m3GroupScaleGlobal::ScaleT` bytes, so this is a dtype
    assertion and a passthrough, and there is no rounding step to measure. The whole two-level product
    is reconstructed in the kernel in f32.

    The global comes back as a per-OUTPUT-CHANNEL f32 VECTOR of length N, not a scalar, so that the
    loader's gate|up merge (`torch.cat(dim=0)`) and its per-expert stack carry it with NO special
    case — see the module docstring. `out_features` overrides the length when the caller knows N
    independently (e.g. it already sharded the block scale on the K axis, which leaves N unchanged);
    by default N is `weight_scale.shape[0]`.

    Returns `(block_scale, global_vec)`:
        block_scale (N, K//16) float8_e4m3fn — the checkpoint tensor, untouched.
        global_vec  (N,)       float32       — a MULTIPLIER, direction already normalised.
    """
    if weight_scale.ndim != 2:
        raise ValueError(
            f"NVFP4 leaf block scale must be (N, K//{NVFP4_GROUP_SIZE}); got {tuple(weight_scale.shape)}"
        )
    mul = nvfp4_global_multiplier(global_scale, global_field=global_field)
    # BYTE-VERBATIM. A checkpoint that ships the block scale as raw uint8 bytes (some exporters do)
    # is bitcast, never converted: `.to(float8_e4m3fn)` on a uint8 tensor would VALUE-convert (byte
    # 126 -> 126.0 -> inf/448), which is the one mistake here that produces plausible-looking finite
    # scales for small bytes and only explodes for large ones.
    if weight_scale.dtype == torch.uint8:
        block = weight_scale.view(torch.float8_e4m3fn)
    elif weight_scale.dtype == torch.float8_e4m3fn:
        block = weight_scale
    else:
        raise ValueError(
            f"NVFP4 block scale must be float8_e4m3fn (or its raw uint8 bytes); got "
            f"{weight_scale.dtype}. A fp16 tensor here means the caller already FOLDED, and folding "
            f"then splitting would bake the fold's 4.37e-04 error into the 'exact' path."
        )
    n = int(weight_scale.shape[0]) if out_features is None else int(out_features)
    global_vec = torch.full((n,), float(mul), dtype=torch.float32, device=weight_scale.device)
    return block.contiguous(), global_vec


def fold_nvfp4_scale(
    weight_scale: torch.Tensor, global_scale: torch.Tensor, *, global_field: str
) -> torch.Tensor:
    """(N, K//16) e4m3 block scale + per-tensor f32 global -> (N, K//16) fp16 per-group scale.

    LEGACY, LOSSY, AND STILL LIVE FOR DENSE LINEARS. See the module docstring: the fp16 collapse
    carries a MEASURED 4.37e-04 max / 1.4-2.4e-04 mean relative error on every weight of a real
    checkpoint (1.2e-03..3.2e-03 through a 4-row GEMV), where `split_nvfp4_scale` is exact. This
    function used to be documented as exact; it is not, and that claim is why the split was never
    written. Prefer `split_nvfp4_scale` for anything whose kernel can consume two levels.

    It remains the path for NVFP4 DENSE linears (`quant.method.NvFp4LinearMethod` ->
    `w4a8_fp8_wmma_kernel.hip` / `gemm_tiled.h`), which have not been templated on the WScale policy
    yet — the dense core still hardcodes a `const __half*` scale pointer, so handing it e4m3 bytes
    would reinterpret them as halves and return plausible garbage. Converting the dense core is the
    named follow-up, not a silent omission.

    Model-agnostic and layout-agnostic (operates per element, so it composes with the loader's later
    concat/merge/stack). Run at the LEAF, before any weight fusion, so the per-tensor global is
    absorbed into the per-group scale and never has to survive a merge.

    `global_field` names the CHECKPOINT tensor suffix `global_scale` came from and selects the
    convention — see `NVFP4_GLOBAL_SCALE_IS_RECIPROCAL`.
    """
    reciprocal = _global_is_reciprocal(global_field)
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


# The two repo-native leaf names an NVFP4 scale pair can land on. `weight_scale` is shared by both
# arms (its DTYPE says which — see `_scale_op_dtype`); `weight_global` exists only on the split arm.
NVFP4_BLOCK_SCALE_LEAF = "weight_scale"
NVFP4_GLOBAL_LEAF = "weight_global"


def nvfp4_leaf_splits(base: str) -> bool:
    """Does this module's NVFP4 scale stay TWO-LEVEL (split) or get folded to one fp16 scale?

    ROUTED MoE EXPERTS SPLIT; EVERYTHING ELSE FOLDS. This is not a preference — it is a statement
    about which KERNELS have been templated on the WScale policy. The grouped-MoE cores
    (`moe_kernel.hip`, `moe_gemm_tiled.h`, `gemv_decode.h`) take `WLoad` as a template parameter and
    have an `E4m3GroupScaleGlobal` instantiation; the DENSE cores (`w4a8_fp8_wmma_kernel.hip`,
    `gemm_tiled.h`) still hardcode `const __half* w_scales`, so handing them e4m3 bytes reads them as
    halves and returns finite, plausible, wrong numbers. Splitting for a module whose kernel cannot
    consume the split is therefore silent corruption, and this predicate is the fence.

    Keyed on `.experts.` in the CHECKPOINT base, which is the same substring
    `models/weight.py::_EXPERT_PATTERN` uses to decide a tensor is per-expert and stackable — so the
    "gets split" set and the "gets stacked into an (E, ...) MoE container" set are the same set by
    construction, rather than by two rules that could drift.

    When the dense cores are templated too, this becomes `return True` and `fold_nvfp4_scale` goes
    away — that is the whole of the follow-up.
    """
    return ".experts." in base


def nvfp4_leaf_scales(
    base: str,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    global_field: str,
    device: "torch.device | None" = None,
) -> "list[tuple[str, torch.Tensor]]":
    """One NVFP4 scale/global checkpoint pair -> the repo-native LEAF tensors, `[(name, tensor), ...]`.

    THE ONE TEXT every weight loader calls, so the fold/split decision, the direction normalisation
    and the leaf naming exist once. Returns TWO entries for a routed MoE expert (block scale + the
    per-output-channel global) and ONE for everything else (the folded fp16 scale) — see
    `nvfp4_leaf_splits`.

    Callers feed each returned `(name, tensor)` through their EXISTING per-leaf path (remap -> TP
    shard -> GDN concat -> gate|up merge -> expert stack). That is deliberate and is the reason the
    global is an N-vector: it needs no special case anywhere downstream.

    `device` is used ONLY by the fold arm, which is fp32 arithmetic on an e4m3 input and torch has no
    CPU float8 math. The split arm is a dtype passthrough plus a `full()`, so it stays on whatever
    device the checkpoint read landed on — which lets the loader keep doing its TP slice host-side.
    """
    if nvfp4_leaf_splits(base):
        block, global_vec = split_nvfp4_scale(
            weight_scale, global_scale, global_field=global_field
        )
        return [
            (f"{base}.{NVFP4_BLOCK_SCALE_LEAF}", block),
            (f"{base}.{NVFP4_GLOBAL_LEAF}", global_vec),
        ]
    ws = weight_scale if device is None else weight_scale.to(device)
    gs = global_scale if device is None else global_scale.to(device)
    return [
        (f"{base}.{NVFP4_BLOCK_SCALE_LEAF}", fold_nvfp4_scale(ws, gs, global_field=global_field))
    ]


def _scale_op_dtype(weight_scale: torch.Tensor) -> torch.dtype:
    """The op-layout dtype for a leaf scale, and the whole of the fold-vs-split dispatch.

    SELECTION IS BY DTYPE, END TO END. The kernel binding gates on exactly this
    (`torch_binding.cpp`: kHalf -> `Fp16GroupScale`, kFloat8_e4m3fn/kByte -> `E4m3GroupScaleGlobal`),
    so the container, this converter and the kernel all read the same one fact off the same tensor and
    cannot disagree. There is no flag, no env knob and no scheme string in this path — a mismatch
    would be silent (e4m3 bytes read as halves are finite and plausible), so it must not be
    expressible.
    """
    if weight_scale.dtype == torch.float16:
        return torch.float16  # legacy fold (dense linears)
    if weight_scale.dtype in (torch.float8_e4m3fn, torch.uint8):
        return torch.float8_e4m3fn  # NVFP4-native block scale; global travels separately
    raise ValueError(
        f"NVFP4 scale must be fp16 (folded) or float8_e4m3fn/uint8 (native block scale); got "
        f"{weight_scale.dtype}"
    )


def convert_nvfp4_weight(weight_packed: torch.Tensor, weight_scale: torch.Tensor) -> dict:
    """One NVFP4 weight matrix -> the e2m1 kernel op layout (identical to MXFP4's).

    weight_packed (N, K//2) uint8 E2M1 nibbles -> w_packed (N, K//8) int32 verbatim E2M1 codes. The
    codes are the same as MXFP4's, so this reuses the MXFP4 nibble packer; only the group_size (16 vs
    32) differs.

    `weight_scale` passes through in ITS OWN dtype, which is the fold/split selector (see
    `_scale_op_dtype`):
        (N, K//16) fp16              -> the legacy folded single-level scale (dense linears);
        (N, K//16) float8_e4m3fn/u8  -> the NVFP4-native block scale, BYTE-VERBATIM, whose companion
                                        per-output-channel f32 global travels as its own tensor.
    The e4m3 arm does no arithmetic on the scale at all — that is the point of it."""
    codes = unpack_e2m1_nibbles(weight_packed)  # (N, K) uint8
    w_packed = pack_codes_to_int32(codes)  # (N, K//8) int32
    dt = _scale_op_dtype(weight_scale)
    scales = (
        weight_scale.view(dt) if weight_scale.dtype == torch.uint8 else weight_scale.to(dt)
    )
    return {
        "w_packed": w_packed,
        "scales": scales.contiguous(),  # (N, K//16) fp16 | float8_e4m3fn
        "group_size": NVFP4_GROUP_SIZE,
        "shape": (weight_packed.shape[0], weight_packed.shape[1] * 2),
    }


def convert_nvfp4_moe(weight_packed: torch.Tensor, weight_scale: torch.Tensor) -> dict:
    """Stacked per-expert variant of convert_nvfp4_weight.

    weight_packed (E, N, K//2) uint8, weight_scale (E, N, K//16) fp16 (folded) OR float8_e4m3fn (the
    native block scale) -> {w_packed (E,N,K//8) int32, scales (E,N,K//16) in the SAME scale dtype,
    group_size 16}. Per-expert loop keeps the int64 nibble-widen transient to one (N, K) matrix (the
    OOM guard convert_mxfp4_moe uses)."""
    assert weight_packed.ndim == 3 and weight_scale.ndim == 3, (
        weight_packed.shape, weight_scale.shape
    )
    e, n, k_half = weight_packed.shape
    k = k_half * 2
    dev = weight_packed.device
    w_out = torch.empty((e, n, k // 8), dtype=torch.int32, device=dev)
    s_out = torch.empty(
        (e, n, k // NVFP4_GROUP_SIZE), dtype=_scale_op_dtype(weight_scale), device=dev
    )
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


def dequantize_nvfp4_split(
    w_packed: torch.Tensor, scales: torch.Tensor, global_vec: torch.Tensor
) -> torch.Tensor:
    """HOST MIRROR OF `E4m3GroupScaleGlobal`, for tests: op-layout codes + e4m3 block scale + the
    per-output-channel f32 global -> (N, K) float32.

    Same ARITHMETIC ORDER as the kernel, deliberately: the group fold `acc * decode(block)` runs
    first and the per-channel `epi(global)` multiplies the result, because the kernel applies
    `wscale_epi` in its EPILOGUE, once per output channel, not per group. Doing it in the other order
    here would make this reference agree with the kernel only up to f32 rounding and would hide
    exactly the class of ordering mistake a reference exists to catch.

        w_packed   (N, K//8) int32              — verbatim E2M1 codes
        scales     (N, K//16) float8_e4m3fn/u8  — the block scale, byte-verbatim
        global_vec (N,) float32                 — direction-normalised MULTIPLIER
    """
    from .mxfp4 import FP4_E2M1_LUT

    n, kw = w_packed.shape
    k = kw * 8
    words = w_packed.to(torch.int64)
    codes = torch.stack([(words >> (4 * j)) & 0xF for j in range(8)], dim=-1).reshape(n, k)
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32, device=w_packed.device)
    bs = scales
    if bs.dtype == torch.uint8:
        bs = bs.view(torch.float8_e4m3fn)
    per_group = lut[codes] * bs.to(torch.float32).repeat_interleave(NVFP4_GROUP_SIZE, dim=-1)
    return per_group * global_vec.to(torch.float32).reshape(n, 1)
