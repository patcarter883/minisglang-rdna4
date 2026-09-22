from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Protocol, Tuple, runtime_checkable

import torch
import torch.nn.functional as F

from . import kernels
from .config import QuantConfig

if TYPE_CHECKING:
    from minisgl.layers.base import BaseOP

# Fused dense gate_up + silu_and_mul for a MERGED gate_up projection at decode: one kernel writes
# silu(gate)*up, dropping the separate silu launch + the [.., 2*inter] HBM round-trip. TWO arms, one
# per activation format -- `mmq_fp8_gemm_silu` (W4A8) and `mmq_regdirect_w4a16_gemv_silu` (W4A16) --
# each bit-exact to its OWN unfused pair. They are not interchangeable: fusing a W4A16 model through
# the W4A8 kernel would quantize activations at decode only, splitting precision by M inside one
# request. Otherwise apply_swiglu returns None and Linear.forward_swiglu falls back unfused.
# NO REVERT KNOB. This was gated on MINISGL_DENSE_FUSED_SILU, which was wrong twice over: a
# merged fast path that has proved itself is simply ON -- the worktree is the isolation, not a
# runtime flag -- and the knob was not reachable anyway, because a compose serve forwards only
# the MINISGL_* variables docker-compose.yml enumerates by hand and this was never one of them.
# Setting it to 0 changed nothing, silently, which invalidated an A/B of this very path twice in
# one session: both legs engaged the fused arm, and only the hip-engage ledger showed it.


def _fused_swiglu_ok(x: torch.Tensor, w_packed: torch.Tensor, group_size: int) -> bool:
    """Shape gate for the fused dense gemm+silu decode kernels. BOTH arms share it: the W4A8 and
    W4A16 fused GEMVs are the same `gemv_decode_core` sweep under different loaders, so they have
    the same shape contract.

    K % 32 and group % 16 are the KERNEL's constraints, taken from its own TORCH_CHECKs. This used
    to say `K % 512` and `group_size % 32`, which were stale: the kernel relaxed both (512 was an
    inheritance from a retired LDS K-tiling; group 32 predates the per-16-K-half scale fold) and the
    engine-side gate was not updated with it. A gate stricter than the kernel is not conservative
    here -- it silently sends shapes the kernel computes correctly back to the unfused path, and for
    the plain GEMV the same staleness cost Gemma4's 2816-wide projections (2816 % 512 == 256) a
    fall-through to the WMMA prefill body at M=1. Group 16 is NVFP4's native grouping.
    """
    N = w_packed.shape[0]
    if not (x.shape[0] <= 16 and x.shape[1] % 32 == 0
            and group_size % 16 == 0 and group_size <= 128 and N % 2 == 0):
        return False
    # NO SCALE-FORMAT RESTRICTION. Both fused arms take all three WSP policies -- fp16 group scale,
    # MXFP4's E8M0 byte, and NVFP4's e4m3 block + f32 global.
    #
    # This briefly gated the W4A8 arm to fp16 only, because an MXFP4/NVFP4 layer reaching it died
    # with `TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be fp16")` and took the
    # serve down at the first decode step. That check, and a neighbouring `group_size % 32`, were
    # STALE GUARDS IN THE BINDING: `launch_mmq_fp8_gemm_silu_gfx1201` has run
    # W4A8_DENSE_SCALE_FMT_DISPATCH (Fp16GroupScale / E8m0GroupScale / E4m3GroupScaleGlobal) and
    # required only `group_size % 16` since the WSP templating landed. Restricting the ENGINE would
    # have made this permanent -- a gate stricter than the kernel, which is the exact defect being
    # fixed everywhere else in this series. Both binding guards now match their launcher, verified
    # bit-exact for AWQ/MXFP4/NVFP4 in fp8_wmma/tests/test_w4a16_gemv_silu.py.
    return True


# Packed WORDS (elements of the container dtype, whatever its width) drawn for the sign-convention
# decision. 65536 int32 words = 524,288 nibbles, which resolves a real mode beyond any doubt; a
# narrower container yields proportionally fewer nibbles from the same word count, which the
# `min_margin` floor then judges on its own terms rather than by a fixed count.
_CT_SIGN_SAMPLE_WORDS = 1 << 16

# ...and WHERE those words are drawn from, which is exactly as load-bearing as how many. The sample
# is BLOCKS, not a strided lattice, because a lattice gets the "where" wrong twice on the shipped
# shapes and both failures are silent:
#
#   * TRUNCATION. `raw[::n // 65536][:65536]` keeps only the first 65536 strided rows, so for
#     `65536 < n < 131072` the sample is a PREFIX covering as little as 50% of the tensor — the very
#     prefix bias this detector was rewritten to remove. A TP=2-sharded dense CT linear
#     (N=768, K=1024 -> n=98304 int32 words) covers 66.7% and decides from its first two thirds.
#   * ROW ALIASING, which is worse, and which the shipped MoE shapes hit exactly. The stride comes
#     out an exact multiple of the packed row length: w13 at E=128 / N=1536 / K=4096 gives
#     n=100,663,296 and stride=1536 = 3 x (K/8), so `i % (K/8) == 0` for EVERY sampled index; w2
#     (n=50,331,648, stride=768, K/8=96) is the same. "Sampled with a fixed stride across all E
#     experts" is then true and useless — the sample sees exactly ONE of the 512 (resp. 96) packed
#     input groups, the same 8 input channels of the model, and the whole checkpoint's decode
#     convention is decided from them. Real checkpoints have outlier input channels that quantize to
#     the int4 rail (`tests/gemma4_parity_test.py` measures that rail fraction), and a rail-saturated
#     column has its mode at neither 8 nor 0: at best `c8 ~ c0` refuses a healthy checkpoint, at
#     worst it picks the wrong convention and XORs every nibble of every expert into plausible
#     garbage.
#
# So: `_CT_SIGN_SAMPLE_BLOCKS` contiguous runs, starts spread evenly across `[0, n - block]` and
# INCLUDING the tail, so the sampled index range is exactly `[0, n)`. Each run covers 128 consecutive
# packed columns and the runs cover every expert and every output row, so neither axis can be aliased
# away by an unlucky stride. Integer arithmetic, no RNG, one `index_select`.
_CT_SIGN_SAMPLE_BLOCKS = 512
# Minimum |counts[8] - counts[0]| / sampled nibbles for the histogram to be a DECISION rather than a
# coin flip. A genuine symmetric int4 stack puts ~12% of its nibbles on the mode and a small fraction
# of that on the opposite value, so real checkpoints clear this by orders of magnitude; 0.1% fires
# only on a near-perfect tie, which is exactly the case where guessing XOR-corrupts every weight.
_CT_SIGN_MIN_MARGIN = 1e-3


class CtSignConvention(NamedTuple):
    """The packed-nibble sign convention of ONE compressed-tensors stack, decided once.

    Threaded rather than re-derived, because the derivation is a SAMPLE: two calls that look at
    different parts of the same stack can disagree, and disagreeing means one contiguous block of
    experts gets XORed and the rest does not — plausible text, no crash, and nothing downstream can
    detect it. A chunked per-expert-range repack (weight offload Stage B, plan §5.2) MUST carry this
    object into every chunk instead of calling the detector again."""

    uint4b8: bool
    margin: float
    sampled_words: int
    # Spacing between the starts of consecutive sampled BLOCKS, in packed words (1 when the whole
    # tensor was sampled). NOT an element-by-element stride: see `_CT_SIGN_SAMPLE_BLOCKS` for why a
    # lattice sample is unsafe on the shipped shapes.
    stride: int
    # How many contiguous runs the sample was drawn from. 1 == "the whole tensor", which is the
    # only case in which `stride` is meaningful as a step of one.
    blocks: int = 1


def _ct_sign_sample(raw: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """`(sampled_rows, blocks, block_start_stride)` for the `(n, elem)` byte view `raw`.

    Draws at most `_CT_SIGN_SAMPLE_WORDS` packed words spread over ALL of `raw` as
    `_CT_SIGN_SAMPLE_BLOCKS` contiguous runs — never a prefix, never a single-column lattice. See
    `_CT_SIGN_SAMPLE_BLOCKS` for the two shipped shapes that forced this.

    Blocks cannot overlap or duplicate: the start step is `floor((n - block) / (blocks - 1))`, and
    `n > _CT_SIGN_SAMPLE_WORDS = blocks * block` makes that at least `block`. The last start is
    exactly `n - block`, so the sampled index range is `[0, n)` with no truncation.
    """
    n = int(raw.shape[0])
    if n <= _CT_SIGN_SAMPLE_WORDS:
        return raw, 1, 1
    blocks = _CT_SIGN_SAMPLE_BLOCKS
    block = max(1, _CT_SIGN_SAMPLE_WORDS // blocks)
    span = n - block  # last legal start; > 0 because n > _CT_SIGN_SAMPLE_WORDS >= block
    starts = torch.arange(blocks, dtype=torch.int64, device=raw.device) * span // (blocks - 1)
    within = torch.arange(block, dtype=torch.int64, device=raw.device)
    idx = (starts[:, None] + within[None, :]).reshape(-1)
    return raw.index_select(0, idx), blocks, span // (blocks - 1)


def ct_packed_sign_convention(
    packed: torch.Tensor,
    *,
    name: str = "weight_packed",
    min_margin: float = _CT_SIGN_MIN_MARGIN,
) -> CtSignConvention:
    """Decide a compressed-tensors int4 checkpoint's packed sign convention from the nibble
    distribution: uint4b8 (q+8, mode at 8 for symmetric weights) -> pass through; two's-complement
    (mode at 0) -> XOR 0x88.

    Samples the WHOLE stack, not the leading 65536 words. The prefix sample read only the first
    ~0.5% of a 512-expert stack, so the decision was made from expert 0 and applied to all 512 — and,
    worse, it made the answer depend on WHICH SLICE the caller happened to hold, which is what breaks
    the moment a chunked loader repacks expert ranges separately.

    "The whole stack" means both axes. The sample is contiguous BLOCKS spread across `[0, n)`, not a
    strided lattice: on every shipped MoE shape the lattice stride came out an exact multiple of the
    packed row length, so a sample that spanned all 128 experts still only ever read packed column 0
    — one input group out of 512. See `_CT_SIGN_SAMPLE_BLOCKS`.

    DETERMINISM IS NOT CROSS-RANK AGREEMENT, and reading it as such is the trap this paragraph
    replaces. There is no RNG here, so the same tensor always yields the same answer — but under TP
    the ranks do not hold the same tensor. Plain TP gives rank r a `w13` of shape (E, 2*I/tp, H);
    EP-over-TP gives it experts [r*E/ep, (r+1)*E/ep). Different bytes, sampled independently,
    decided independently. On a homogeneously-packed checkpoint (every real one) both ranks land on
    the same answer because the convention is a property of the PRODUCER and not of a slice; on a
    mixed-packing stack they can diverge, and then one rank's half of a TP-split GEMM is dequantized
    in the uint4b8 domain and the other's in two's-complement — plausible text, no error anywhere.

    What would close it: compare the returned `CtSignConvention` across the CPU group at post_load,
    or decide once per checkpoint and thread the object (which is also what Stage B's chunked repack
    needs). Neither exists yet, and `tests/core/test_ct_sign_convention.py::TestCrossRankHazard`
    pins the hazard so the "every rank agrees" claim cannot quietly come back.

    Raises on an ambiguous histogram instead of coin-flipping: the old `counts[8] >= counts[0]`
    silently resolved a tie toward pass-through, and a wrong resolution XORs every nibble.
    """
    # BYTE-WISE, NOT int32-WISE, and that is a correctness property rather than a refactor. The
    # previous form hardcoded a 32-bit word: it masked `& 0xFFFFFFFF` and unpacked `range(8)`
    # nibbles per element. Hand it a stack whose int4 codes are packed into uint8 — which is exactly
    # how `_GroupedMxFp4Experts` and `_GroupedNvFp4Experts` already ship 4-bit
    # weights in this file's own sibling module, and how any future int4 loader policy on the shared
    # kernel core would arrive — and nibble positions 2..7 of every element read as 0. `counts[0]`
    # is then inflated by 6/8 of the sample, the histogram decides "two's-complement" with a huge
    # (and entirely fake) margin, and `apply_ct_sign` XORs a stack that needed no XOR: every weight
    # off by 8 quanta, right shapes, no error. Deriving the element width from the tensor makes the
    # decision a property of the BYTES, which is what it always claimed to be, and the repo's
    # "dtype-agnostic, template the core" rule spells out as the general form.
    #
    # Sampling stays WORD-ALIGNED (a whole packed element at a time, every byte lane of it), so a
    # layout in which one lane is systematically different cannot be missed by a byte stride that
    # happens to be a multiple of the element size.
    if packed.numel() == 0:
        raise ValueError(f"{name}: empty packed tensor, no sign convention to decide")
    elem = packed.element_size()
    raw = packed.contiguous().view(torch.uint8).reshape(-1, elem)
    n = int(raw.shape[0])
    sample, blocks, stride = _ct_sign_sample(raw)
    sample = sample.to(torch.int16)
    nib = torch.cat([sample & 0xF, (sample >> 4) & 0xF]).reshape(-1).to(torch.int64)
    counts = torch.bincount(nib, minlength=16)
    c8, c0 = int(counts[8]), int(counts[0])
    total = int(nib.numel())
    margin = abs(c8 - c0) / total if total else 0.0
    if margin < min_margin:
        raise ValueError(
            f"{name}: cannot decide the compressed-tensors int4 packing. Sampled "
            f"{int(sample.shape[0])} of {n} packed {elem * 8}-bit words ({blocks} contiguous "
            f"block(s), block-start stride {stride}, spanning the whole tensor); "
            f"nibble counts at 8 and 0 are {c8} and {c0}, a "
            f"margin of {margin:.2e} against a {min_margin:.0e} floor. uint4b8 and two's-complement "
            f"are indistinguishable here and guessing XORs every nibble of the stack, which produces "
            f"plausible text rather than an error. Inspect the checkpoint's quantization_config."
        )
    return CtSignConvention(
        uint4b8=c8 > c0,
        margin=margin,
        sampled_words=int(sample.shape[0]),
        stride=stride,
        blocks=blocks,
    )


def apply_ct_sign(t: torch.Tensor, conv: CtSignConvention) -> torch.Tensor:
    """Put a packed int4 tensor into the op's uint4b8 domain under an already-decided convention.

    The ONE place the transform lives, so a weight and its zero-point can never end up in different
    domains: they share a quantizer, so `scale*(W_u - Z_u) == scale*(q - zp)` holds only if both got
    the same treatment. Callers pass the SAME `conv` for both.

    DTYPE-PRESERVING, not int32-pinned. The flip is `^ 0x88` over the raw BYTES — every packed
    format's nibble pair, whatever container dtype it arrived in — so the result is restored to
    `t.dtype` rather than reinterpreted as int32. Hardcoding int32 was not merely inelegant: given a
    uint8-packed stack (how `_GroupedMxFp4Experts` / `_GroupedNvFp4Experts`
    already ship 4-bit weights, and how any new int4 loader policy on the shared kernel core would
    arrive) `.view(torch.int32)` silently returns a tensor of one quarter the last dimension, in the
    wrong dtype, whenever that dimension happens to divide by 4 — and raises a shape error, blamed
    on the checkpoint, when it does not. `t.dtype` is the identity transform for the int32 case that
    exists today, so this is byte-identical on every shipped path."""
    if conv.uint4b8:
        return t
    return (t.contiguous().view(torch.uint8) ^ 0x88).view(t.dtype).contiguous()


def _ct_packed_is_uint4b8(packed: torch.Tensor) -> bool:
    """Boolean form of `ct_packed_sign_convention`, for call sites that need nothing else."""
    return ct_packed_sign_convention(packed).uint4b8


class CtSignRankDivergence(RuntimeError):
    """Two TP ranks put the SAME logical weights in different int4 sign domains."""


def collect_ct_sign_decisions(model) -> Dict[str, bool]:
    """`{op path: conv.uint4b8}` for every container that made a packed-int4 sign decision.

    Uses `weights.moe_interpose._iter_ops`, which is the repo's ONE op-tree walk and produces the
    same dotted paths `BaseOP.state_dict` emits. That matters twice over: `BaseOP` is not an
    `nn.Module`, so `named_modules()` does not exist on most of this tree; and the paths are the
    strings the two ranks compare, so they have to be a pure function of the module tree rather than
    of a construction order that an MTP head or a differing shard count could renumber.

    The VALUE is the boolean and nothing else. `margin`, `sampled_words`, `stride` and `blocks` are
    properties of the SHARD each rank happened to sample and legitimately differ between ranks —
    comparing them would fail every healthy TP=2 boot on a perfectly uniform checkpoint. `uint4b8`
    is the only field that has to agree, because it is the only one that changes bytes.
    """
    # local: quant <- weights <- layers would be an import cycle at module load.
    from minisgl.weights.moe_interpose import _iter_ops

    out: Dict[str, bool] = {}
    for path, mod in _iter_ops(model, "", set()):
        conv = getattr(mod, "_ct_sign", None)
        if conv is not None and hasattr(conv, "uint4b8"):
            out[path] = bool(conv.uint4b8)
    return out


def verify_ct_sign_across_ranks(model, group, tp_size: int, tp_rank: int) -> Dict[str, bool]:
    """CLOSE the cross-rank hazard `ct_packed_sign_convention` documents. Call at post_load.

    THE HAZARD, restated because the failure has no symptom. The sign convention is decided from a
    SAMPLE of the packed nibbles, and under TP no two ranks hold the same bytes: plain TP gives rank
    r a `w13` of shape (E, 2*I/tp, H), EP-over-TP gives it experts [r*E/ep, (r+1)*E/ep). Each rank
    runs `post_load` on its own container and decides independently. On a homogeneously packed
    checkpoint — every real one — both land on the same answer, because the convention is a property
    of the PRODUCER rather than of a slice. On a mixed-packing stack they diverge, and then one
    rank's half of a TP-split GEMM is dequantized in the uint4b8 domain and the other's in
    two's-complement. The shapes are right, no kernel errors, and the model produces fluent text
    that is wrong. Nothing downstream can see it.

    `ct_packed_sign_convention` itself REFUSES a tie (it raises below `_CT_SIGN_MIN_MARGIN`) — but
    that refusal is evaluated per shard, and a mixed stack is only a tie when you can see all of it.
    Neither rank ever does, so the refusal cannot fire. A collective is the only place the whole
    stack is observable, which is why this lives here and not inside the detector.

    Cheap enough to be unconditional: one `all_gather_object` of a dict of bools at boot, on the
    gloo CPU group the engine already builds for its control messages. `tp_size == 1` returns
    immediately — there is nothing to disagree with — and so does any model with no CT containers
    (the NVFP4 / MXFP4 MoE paths declare no `_ct_sign` at all, so qwen4_exp's NVFP4 body walks
    straight through this and only its AWQ sibling is actually gated).

    Raises rather than warns. A warning here is a serve that answers questions wrongly for as long
    as it is up.
    """
    local = collect_ct_sign_decisions(model)
    if tp_size <= 1 or group is None:
        return local
    import torch.distributed as dist

    gathered: List[Dict[str, bool] | None] = [None] * tp_size
    dist.all_gather_object(gathered, local, group=group)
    ref = gathered[0] or {}
    bad_keys: List[str] = []
    missing: List[str] = []
    for d in gathered:
        d = d or {}
        # A path present on one rank and absent on another is ALSO a divergence: it means one rank
        # built a CT container where its peer built something else, so the two are not running the
        # same model. Reported separately because the fix differs (a build/config skew, not a
        # mixed-packing checkpoint).
        if set(d) != set(ref):
            missing.extend(sorted(set(d) ^ set(ref)))
        for k in sorted(set(d) & set(ref)):
            if d[k] != ref[k] and k not in bad_keys:
                bad_keys.append(k)
    if bad_keys or missing:
        detail = "\n".join(
            f"    {k}: " + ", ".join(f"rank{r}={(g or {}).get(k)}" for r, g in enumerate(gathered))
            for k in (bad_keys + sorted(set(missing)))[:20]
        )
        raise CtSignRankDivergence(
            f"compressed-tensors int4 SIGN CONVENTION DIVERGED ACROSS TP RANKS "
            f"(tp_size={tp_size}, this rank={tp_rank}). Each rank samples only its own shard, so a "
            f"mixed-packing stack is decided independently and one rank dequantizes in the uint4b8 "
            f"domain while the other uses two's-complement. Every weight in the disagreeing stack "
            f"is then off by 8 quanta on one rank: right shapes, no kernel error, fluent and wrong "
            f"text.\n"
            f"  {len(bad_keys)} container(s) disagreed"
            + (f", {len(set(missing))} present on some ranks only" if missing else "")
            + f":\n{detail}\n"
            f"  This is a CHECKPOINT property, not a runtime one — inspect its quantization_config "
            f"and repack it with a single convention, or serve it at TP=1, where one rank sees the "
            f"whole stack and the detector's own tie-refusal can fire."
        )
    return local


@runtime_checkable
class LinearMethod(Protocol):
    """How a parallel-linear layer allocates its weights and computes its matmul.
    The layer owns sharding + collectives; the method owns weight layout + the GEMM."""

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        """Declare weight tensors on `layer` as plain (non-underscore) attributes so
        BaseOP's __dict__ introspection serializes/loads them. Shapes/dtypes MUST match
        the checkpoint exactly (BaseOP.load_state_dict asserts both)."""
        ...

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor: ...


class UnquantizedLinearMethod:
    """bf16/f16 dense linear. Routes through the engine's M-invariant WMMA GEMM (layers/minv.py) so a
    chunked / prefix-cached / spec-verify forward matches a fresh one bit-for-bit; falls back to
    F.linear for dtypes/shapes/contexts the kernel doesn't cover (fp32, IN%16!=0, cudagraph capture).
    This is the single chokepoint every unquantized Linear in every model flows through."""

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        layer.weight = torch.empty(out_features, in_features)

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        from minisgl.layers.minv import minv_linear

        return minv_linear(x, layer.weight, bias)


class W4A8LinearMethod:
    """int4-weight / fp8-activation WMMA via the swappable kernel provider.

    Phase 2 scaffold. The op consumes weights in its native layout
    (w_packed (N, K/8) i32, scales (N, K/32) f16, zeros (N/8, K/32) i32 | None).
    Checkpoints arrive in AWQ (g128, AutoGPTQ bit order, K-major) or compressed-tensors
    (g32) layout, so weights are declared in CHECKPOINT layout (to load) then converted
    to op layout once after load.

    TODO Phase 2c: implement create_weights (exact AWQ/CT buffer shapes) +
    process_weights_after_load (port the conversion from
    vllm-gfx1201/w4a8_fp8_wmma/{vllm_adapter.py:_awq_to_op_layout, moe_experts.py}).
    apply() below is the final call shape and is ready once op-layout buffers exist.
    """

    def __init__(self, quant: QuantConfig) -> None:
        self.quant = quant

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        # Declare buffers in CHECKPOINT layout so BaseOP load matches. (N=out, K=in are the
        # LOCAL/per-TP sizes; TP-quant sharding is a follow-up.)
        pf = 32 // self.quant.bits
        g = self.quant.group_size
        N, K = out_features, in_features
        if self.quant.is_gptq:
            # GPTQ "gemm" layout: qweight int32 packed along INPUT (K//pf, N), per-group
            # scales (K//g, N), and qzeros (K//g, N//pf) ALWAYS present (even symmetric — the
            # constant zero is stored explicitly; MoeWNA16 likewise loads then folds it).
            assert K % pf == 0 and K % g == 0 and N % pf == 0, (
                f"GPTQ needs K%{pf}==0,K%{g}==0,N%{pf}==0; got N={N},K={K}"
            )
            layer.qweight = torch.empty((K // pf, N), dtype=torch.int32)
            layer.scales = torch.empty((K // g, N), dtype=torch.float16)
            layer.qzeros = torch.empty((K // g, N // pf), dtype=torch.int32)
            if self.quant.desc_act:
                layer.g_idx = torch.empty((K,), dtype=torch.int32)
            return
        if self.quant.is_compressed_tensors:
            # compressed-tensors W4A16 DENSE linear: weight_packed (N, K//pf) int32 (8 SIGNED int4 per
            # int32, natural K order) + weight_scale (N, K//g). Already the op's natural nibble order,
            # so post_load is a whole-tensor fixup (XOR 0x88 + constant zero-point 8) — the same
            # conversion _GroupedCompressedTensorsExperts uses, minus the E dim. `weight_shape` in the
            # checkpoint is ignored by the loader.
            assert K % pf == 0 and K % g == 0 and N % pf == 0, (
                f"CT dense needs K%{pf}==0,K%{g}==0,N%{pf}==0; got N={N},K={K}"
            )
            layer.weight_packed = torch.empty((N, K // pf), dtype=torch.int32)
            # pack-quantized scales ship fp16 (some heads ship bf16); the loader normalizes both to
            # fp16 (engine._cast) so this one declared dtype matches every CT checkpoint.
            layer.weight_scale = torch.empty((N, K // g), dtype=torch.float16)
            if not self.quant.sym:
                # ASYMMETRIC: per-group weight_zero_point, int4-packed 8-per-int32 along the OUTPUT
                # dim (shape [N//pf, G]) — already the op's zeros layout. Loaded + used in process().
                layer.weight_zero_point = torch.empty((N // pf, K // g), dtype=torch.int32)
            return
        # AWQ "gemm" layout: qweight (K, N//pf) i32, scales (K//group, N) f16,
        # qzeros (K//group, N//pf) i32 (asymmetric only).
        assert N % pf == 0 and K % g == 0, f"W4A8 needs N%{pf}==0,K%{g}==0; got N={N},K={K}"
        layer.qweight = torch.empty((K, N // pf), dtype=torch.int32)
        layer.scales = torch.empty((K // g, N), dtype=torch.float16)
        if not self.quant.sym:
            layer.qzeros = torch.empty((K // g, N // pf), dtype=torch.int32)

    def process_weights_after_load(self, layer: "BaseOP") -> None:
        if self.quant.is_compressed_tensors:
            # CT DENSE -> op layout (constant zero-point 8; zeros_op all 0x88; scales as-is).
            # The op wants weights as uint4b8 (nibble = q + 8). compressed-tensors "pack-quantized"
            # ships int4 in one of TWO packings, per producer, that we must distinguish per checkpoint:
            #   * two's-complement signed int4 (nibble = q & 0xF): convert to uint4b8 by flipping each
            #     nibble's top bit — XOR 0x88 per byte — since (q&0xF)^8 == q+8 for q in [-8,7].
            #   * already-offset uint4b8 (nibble = q + 8; AWQ-style zero_point=8, e.g. cyankiwi's
            #     Qwen3.5/3.6 "AWQ-*-INT4" dense checkpoints): pass through UNCHANGED — an XOR here
            #     would scramble it (it re-flips the top bit) and produce garbage.
            # Symmetric weights make the two trivially separable by the packed nibble distribution:
            # uint4b8 is a bell curve with its mode at 8 (q=0); two's-complement's mode is at 0.
            pf = 32 // self.quant.bits
            N, Kp = layer.weight_packed.shape  # type: ignore[attr-defined]
            G = layer.weight_scale.shape[-1]  # type: ignore[attr-defined]
            wp = layer.weight_packed.contiguous()  # type: ignore[attr-defined]
            # ONE decision for this stack, kept on the layer. `_ct_sign` is what a chunked repack
            # (weight offload Stage B) must reuse; re-deriving it per chunk XOR-corrupts whichever
            # chunks disagree, with no crash and plausible output.
            conv = ct_packed_sign_convention(wp, name=f"{type(layer).__name__}.weight_packed")
            layer._ct_sign = conv  # type: ignore[attr-defined]
            layer._w_packed_op = apply_ct_sign(wp, conv)
            # GROUP-MAJOR: the op indexes scales `[g*N + n]` / zeros `[g*(N/8) + n/8]`, so N is the
            # contiguous axis and a fragment's 16 lanes coalesce into one request.
            layer._scales_op = layer.weight_scale.to(torch.float16).transpose(0, 1).contiguous()  # type: ignore[attr-defined]
            zp = getattr(layer, "weight_zero_point", None)
            if zp is None:
                # SYMMETRIC: constant zero-point 8 (uint4b8), zeros_op all 0x88.
                zeros = torch.empty((G, N // pf), dtype=torch.int32)
                zeros.view(torch.uint8).fill_(0x88)
                layer._zeros_op = zeros.to(wp.device)
            else:
                # ASYMMETRIC: real per-group zero_point, already int4-packed [N//pf, G] along N (the
                # op's zeros layout). It shares the weight's sign convention (same quantizer), so apply
                # the SAME uint4b8-vs-two's-complement transform: W_u and Z_u then live in one unsigned
                # domain and the op computes scale*(W_u - Z_u) = scale*(q - zp), exact.
                zp = apply_ct_sign(zp.contiguous(), conv)
                # The 4-bit packing runs along N *within* each int32, so transposing (N//pf, G) is safe.
                layer._zeros_op = zp.transpose(0, 1).contiguous()  # (N//pf, G) -> (G, N//pf)
                del layer.weight_zero_point
            del layer.weight_packed, layer.weight_scale
            return
        qz = getattr(layer, "qzeros", None)
        if self.quant.is_gptq:
            # GPTQ -> op layout (2M-2). desc_act is asserted off (g_idx identity) by the converter.
            assert not self.quant.desc_act, "GPTQ desc_act (act-order) not supported"
            w_packed, scales_op, zeros_op = kernels.gptq_to_op_layout(
                layer.qweight, layer.scales, qz, bits=self.quant.bits  # type: ignore[attr-defined]
            )
        else:
            w_packed, scales_op, zeros_op = kernels.awq_to_op_layout(
                layer.qweight, layer.scales, qz, bits=self.quant.bits  # type: ignore[attr-defined]
            )
        # op-layout buffers are derived (underscore -> not re-serialized); free the loaded ones.
        layer._w_packed_op = w_packed
        layer._scales_op = scales_op
        layer._zeros_op = zeros_op
        if kernels.MOE_W4A16 != "0":
            # W4A16 (unquantized-act) dense path. It reads the SAME `_w_packed_op` the W4A8 arm
            # reads — no repack, no second layout. This used to build a register-direct `w_rep` and
            # DELETE the op-layout, which is why W4A16 cost a full extra copy of every weight and
            # still had no unquantized arm below the WMMA band.
            layer._w4a16 = True
            layer._n_out = w_packed.shape[0]
        del layer.qweight, layer.scales
        if qz is not None:
            del layer.qzeros

    # This method's GEMM takes the producer's (x_fp8, act_scales) pair; `_LinearTPImpl.forward`
    # reads this flag rather than probing the signature, so a method that cannot use the pair needs
    # no keyword argument and no branch.
    #
    # Under W4A16 it must be FALSE. The A16 arm consumes bf16/fp16 activations directly and never
    # reads the pair, so advertising it makes the PRODUCER quantize every token for a consumer that
    # discards the result — exactly the cost this path exists to remove. `apply` merely ignored the
    # pair, which is not the same thing: the work had already been done upstream. Both sibling
    # methods (MxFp4 / NvFp4) already declare False with this reasoning; this was the sibling the
    # fix missed. MOE_W4A16 is a process-wide knob read at import, so a property is exact.
    @property
    def supports_producer_actquant(self) -> bool:  # type: ignore[override]
        return kernels.MOE_W4A16 == "0"

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None,
        *, x_fp8: torch.Tensor | None = None, act_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "_w4a16", False):
            out = kernels.w4a16_linear(
                x,
                layer._w_packed_op,  # type: ignore[attr-defined]
                layer._scales_op,  # type: ignore[attr-defined]
                layer._zeros_op,  # type: ignore[attr-defined]
                self.quant.group_size,
                layer._n_out,  # type: ignore[attr-defined]
            )
        else:
            out = kernels.w4a8_linear(
                x,
                layer._w_packed_op,  # type: ignore[attr-defined]
                layer._scales_op,  # type: ignore[attr-defined]
                layer._zeros_op,  # type: ignore[attr-defined]
                self.quant.group_size,
                x_fp8=x_fp8,
                act_scales=act_scales,
            )
        out = out.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out

    def apply_swiglu(self, layer: "BaseOP", x: torch.Tensor) -> torch.Tensor | None:
        """FUSED gate_up + silu_and_mul at decode (this linear is a merged gate_up). Returns None ->
        caller falls back to the unfused silu_and_mul(apply(...)) when the fused kernel doesn't apply:
        the W4A16 path, prefill (M>16), or an unsupported shape. Bit-exact when it fires."""
        w = layer._w_packed_op  # type: ignore[attr-defined]
        if not _fused_swiglu_ok(x, w, self.quant.group_size):
            return None
        # Pick the fused arm that MATCHES this layer's activation format. Firing the W4A8 one under
        # W4A16 would quantize activations at decode only, while prefill stayed unquantized — an
        # activation-format split by M inside a single request. Both arms read the same
        # `_w_packed_op` and both are bit-exact to their own unfused pair.
        if getattr(layer, "_w4a16", False):
            return kernels.w4a16_linear_silu(
                x, w, layer._scales_op, layer._zeros_op, self.quant.group_size  # type: ignore[attr-defined]
            ).to(x.dtype)
        return kernels.w4a8_linear_silu(
            x, w, layer._scales_op, layer._zeros_op, self.quant.group_size  # type: ignore[attr-defined]
        ).to(x.dtype)


class MxFp4LinearMethod:
    """MXFP4 (OCP E2M1 weights + E8M0 per-32-block scale) dense linear, served through the SAME
    W4A8 fp8-WMMA kernel as int4 with `weight_is_e2m1=True` (per-token fp8 activations). The
    checkpoint (compressed-tensors `mxfp4-pack-quantized`) ships weights ALREADY in a compact
    packed form — weight_packed uint8 (N, K//2) 2 E2M1 nibbles/byte + weight_scale uint8 (N, K//32)
    E8M0 group exponent — so `process_weights_after_load` runs the MXFP4 converter (nibbles ->
    (N,K//8) int32 codes verbatim; E8M0 kept as the checkpoint's own byte, G14) and drops the checkpoint copies.
    Symmetric (no zero-points). Config-selected purely from `quant.weight_is_e2m1`."""

    def __init__(self, quant: QuantConfig) -> None:
        self.quant = quant

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        N, K = out_features, in_features
        g = self.quant.group_size  # 32 (OCP MX block)
        assert K % 2 == 0 and K % g == 0 and N % 8 == 0, (
            f"MXFP4 needs K%2==0,K%{g}==0,N%8==0; got N={N},K={K}"
        )
        # CHECKPOINT layout (uint8), so BaseOP load matches. E8M0 scale is an integer exponent, NOT
        # a float — declared uint8 so the engine's _cast leaves it untouched (see engine._cast).
        layer.weight_packed = torch.empty((N, K // 2), dtype=torch.uint8)
        layer.weight_scale = torch.empty((N, K // g), dtype=torch.uint8)

    def process_weights_after_load(self, layer: "BaseOP") -> None:
        from . import mxfp4

        conv = mxfp4.convert_mxfp4_weight(layer.weight_packed, layer.weight_scale)  # type: ignore[attr-defined]
        info = conv["scale_info"]
        # The fp16-overflow warning is GONE because the overflow is gone: the kernel now takes the
        # checkpoint's E8M0 byte directly (FORMAT_MATRIX.md G14), and E8M0's bias is 127 — exactly
        # fp32's — so the device decode is one shift into the exponent field, not a convert. What
        # remains worth reporting is the e8m0 NaN code (255), which is a property of the CHECKPOINT
        # and poisons its group whatever the kernel does.
        if info["e8m0_nan_groups"]:
            from minisgl.utils import init_logger

            init_logger("mxfp4").info_rank0(
                f"[mxfp4] {info['e8m0_nan_groups']} e8m0-NaN group scale(s) on "
                f"{getattr(layer, 'prefix', '<linear>')} (exp {info['exp_min']}..{info['exp_max']});"
                f" those groups decode to +inf."
            )
        layer._w_packed_op = conv["w_packed"]  # (N, K//8) int32
        # GROUP-MAJOR: the op indexes scales `[g*N + n]` so N is the contiguous axis (coalesced read).
        layer._scales_op = conv["scales"].transpose(0, 1).contiguous()  # (K//32, N) uint8 E8M0
        if kernels.MOE_W4A16 != "0":
            # W4A16: activations stay bf16/fp16 and are NEVER quantized. The E2M1 codes and the E8M0
            # scale tensor go to the kernel UNCHANGED — its `bool E2M1` template flag picks the
            # codebook decode and its WSP policy reads uint8-without-zeros as E8M0. So this is a
            # load policy on the existing core, not a second kernel (KERNEL_CORE_POLICY) — and it
            # reads the op-layout `w_packed` at every M, so there is nothing to repack.
            layer._w4a16 = True
            layer._n_out = conv["w_packed"].shape[0]
        del layer.weight_packed, layer.weight_scale

    # The W4A16 arm consumes unquantized activations, so it must not advertise the producer's
    # (x_fp8, act_scales) pair — accepting it would make the producer quantize per token for nothing,
    # which is exactly the cost this path exists to remove.
    supports_producer_actquant = False

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        if getattr(layer, "_w4a16", False):
            out = kernels.w4a16_linear(
                x,
                layer._w_packed_op,  # type: ignore[attr-defined]
                layer._scales_op,  # type: ignore[attr-defined]
                None,  # symmetric — E8M0 block scales, no zero-points and no global
                self.quant.group_size,
                layer._n_out,  # type: ignore[attr-defined]
                weight_is_e2m1=True,
            )
            if bias is not None:
                out = out + bias
            return out
        out = kernels.w4a8_linear(
            x,
            layer._w_packed_op,  # type: ignore[attr-defined]
            layer._scales_op,  # type: ignore[attr-defined]
            None,  # symmetric — no zero-points
            self.quant.group_size,
            weight_is_e2m1=True,
        )
        out = out.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out

    def apply_swiglu(self, layer: "BaseOP", x: torch.Tensor) -> torch.Tensor | None:
        """FUSED gate_up + silu (MXFP4 / E2M1, symmetric) at decode; None -> caller falls back."""
        w = getattr(layer, "_w_packed_op", None)
        if w is None or not _fused_swiglu_ok(x, w, self.quant.group_size):
            return None
        # See W4A8LinearMethod.apply_swiglu: the fused arm must match the layer's activation format.
        if getattr(layer, "_w4a16", False):
            return kernels.w4a16_linear_silu(
                x, w, layer._scales_op, None, self.quant.group_size, weight_is_e2m1=True  # type: ignore[attr-defined]
            ).to(x.dtype)
        return kernels.w4a8_linear_silu(
            x, w, layer._scales_op, None, self.quant.group_size, weight_is_e2m1=True  # type: ignore[attr-defined]
        ).to(x.dtype)


def create_linear_method(
    quant: QuantConfig | None, *, quantized: bool = True
) -> LinearMethod:
    """Pick the method for a linear layer. `quantized=False` (e.g. lm_head) always
    stays unquantized even when the model is quantized."""
    if quant is None or not quantized:
        return UnquantizedLinearMethod()
    if quant.is_nvfp4:
        # gfx1201 has no FP4 hardware -> upconvert NVFP4 to the fp8 W8A8 path at load (see NvFp4LinearMethod).
        return NvFp4LinearMethod(quant)
    # MXFP4 (compressed-tensors float-quantized 4-bit, OCP E2M1) -> the W4A8 kernel with the e2m1
    # decode. Config-selected from the DECLARED scheme (no model-name branch); disjoint from the int4
    # W4A8 path below (weight_type=="int") and the fp8 W8A8 path (bits==8).
    if quant.weight_is_e2m1:
        return MxFp4LinearMethod(quant)
    # fp8 W8A8 (compressed-tensors float-quantized 8-bit) dense linear — served through the SAME
    # W8A8 WMMA core as the fp8 MoE experts (single-expert grouped GEMM). ZAYA's dense/attn linears
    # stay in the quant `ignore` list (-> unquantized), so this only fires for a checkpoint that
    # actually declares fp8-W8A8 dense linears (e.g. RedHatAI *-FP8-dynamic).
    if quant.is_fp8_block:
        # BEFORE is_fp8_w8a8: a blockwise checkpoint satisfies both, and the per-channel method would
        # declare a (N,1) scale where the file ships (N/128, K/128).
        return Fp8BlockDequantLinearMethod(quant)
    if quant.is_fp8_w8a8:
        return Fp8W8A8LinearMethod(quant)
    return W4A8LinearMethod(quant)


class NvFp4LinearMethod:
    """NVFP4 (compressed-tensors 'nvfp4-pack-quantized') dense linear, served through the SAME e2m1
    W4A8 kernel as MXFP4 — weights stay 4-bit; the E2M1 codes decode to fp8 e4m3 in-register at the
    WMMA (no VRAM upconvert). NVFP4 differs from MXFP4 only in the scale, which the WEIGHT LOADER folds
    to one fp16 per-group scale at the leaf (`nvfp4.fold_nvfp4_scale`: e4m3 block / per-tensor global),
    dropping the global tensors. So by the time this method loads, the checkpoint is MXFP4-shaped:
    weight_packed uint8 (N,K//2) 2 E2M1 nibbles/byte + weight_scale fp16 (N,K//16). `process_weights_
    after_load` packs the nibbles to (N,K//8) int32 codes (verbatim) and passes the fp16 scale through;
    `apply` calls the e2m1 kernel at group_size 16. Symmetric (no zero-points). From quant.is_nvfp4.

    NATIVE TWO-LEVEL SINCE 2026-09-12 (FORMAT_MATRIX.md G14). It was on the fp16 fold until then,
    and the reason was a kernel fact rather than an oversight: the MoE cores were templated on a
    WScale policy and carried an `E4m3GroupScaleGlobal` instantiation while the DENSE cores
    (`w4a8_fp8_wmma_kernel.hip`, `gemm_tiled.h`) hardcoded `const __half* w_scales`, so handing
    them e4m3 bytes would have read them as halves and returned finite, plausible, wrong numbers.
    The dense cores now carry the same policy on every arm, so the fold — a measured 4.37e-04 max
    relative error, in MORE bytes than the native form — is gone from the load path."""

    def __init__(self, quant: QuantConfig) -> None:
        self.quant = quant

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        # CHECKPOINT-NATIVE TWO-LEVEL layout, the same pair `_GroupedNVFP4Experts` declares: a
        # 1-byte e4m3 block scale per (channel, group) PLUS a per-output-channel f32 global. This
        # used to be a single fp16 `weight_scale` because the dense cores had no WScale policy;
        # they do now (FORMAT_MATRIX.md G14), so `nvfp4_leaf_splits` is True for every module and
        # the loader delivers both leaves here exactly as it does for routed experts.
        N, K = out_features, in_features
        g = self.quant.group_size  # 16 (NVFP4 block)
        assert K % 2 == 0 and K % g == 0 and N % 8 == 0, (
            f"NVFP4 needs K%2==0,K%{g}==0,N%8==0; got N={N},K={K}"
        )
        layer.weight_packed = torch.empty((N, K // 2), dtype=torch.uint8)
        # float8_e4m3fn, NOT uint8 — the SAME dtype `_GroupedNVFP4Experts` declares (layers/moe.py:384)
        # and the same one `nvfp4.split_nvfp4_scale` emits (it bitcasts a uint8 checkpoint tensor to
        # e4m3 so every consumer sees one encoding). Both are 1 byte and byte-verbatim, so this is an
        # ENCODING declaration, not a precision one — and `layers/base.py::_coerce_dtype` hard-fails a
        # quantized-dtype mismatch rather than casting, precisely so a disagreement here is loud.
        # Declaring uint8 shipped briefly in 84b1ede and failed at load with
        #   "weight dtype mismatch ... model torch.uint8 vs checkpoint torch.float8_e4m3fn".
        layer.weight_scale = torch.empty((N, K // g), dtype=torch.float8_e4m3fn)  # e4m3 block, byte-verbatim
        layer.weight_global = torch.empty((N,), dtype=torch.float32)       # per-OUTPUT-CHANNEL

    def process_weights_after_load(self, layer: "BaseOP") -> None:
        from . import nvfp4

        conv = nvfp4.convert_nvfp4_weight(layer.weight_packed, layer.weight_scale)  # type: ignore[attr-defined]
        layer._w_packed_op = conv["w_packed"]  # (N, K//8) int32 E2M1 codes
        # GROUP-MAJOR: the op indexes scales `[g*N + n]` so N is the contiguous axis (coalesced read).
        layer._scales_op = conv["scales"].transpose(0, 1).contiguous()  # (K//16, N) e4m3 bytes
        # (N,) f32 -> the SAME BYTES viewed as int32, because the op's global slot is `w_zeros`,
        # typed `const int*`, and the kernel does `reinterpret_cast<const float*>` on the other
        # side. `.view()` is a zero-copy bitcast; `.to(torch.int32)` would VALUE-convert
        # (2.078e-04 -> 0) and give an all-zero layer. The two spellings are one character apart —
        # this is the same guard `_GroupedNVFP4Experts.post_load` carries, for the same reason.
        layer._global_op = layer.weight_global.contiguous().view(torch.int32)  # type: ignore[attr-defined]
        if kernels.MOE_W4A16 != "0":
            # W4A16: unquantized activations. Same E2M1 codes, same two-level scale (e4m3 block +
            # the f32 global in the zeros slot) -- the register-direct kernel's WSP policy reads
            # exactly that combination as E4m3GroupScaleGlobal, so nothing is normalised or folded.
            #
            # NVFP4's group_size is 16. That used to exclude it from the WIDE register-direct twin
            # (which needs `wide` to divide group_size/16) and strand it on the narrow arm; the
            # tiled A16 core carries a RUNTIME group size, so g=16 is served like any other.
            layer._w4a16 = True
            layer._n_out = conv["w_packed"].shape[0]
        del layer.weight_packed, layer.weight_scale, layer.weight_global

    # See MxFp4LinearMethod: the W4A16 arm must not advertise the producer act-quant pair.
    supports_producer_actquant = False

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        if getattr(layer, "_w4a16", False):
            out = kernels.w4a16_linear(
                x,
                layer._w_packed_op,  # type: ignore[attr-defined]
                layer._scales_op,  # type: ignore[attr-defined]
                layer._global_op,  # type: ignore[attr-defined]  f32 global rides the zeros slot
                self.quant.group_size,
                layer._n_out,  # type: ignore[attr-defined]
                weight_is_e2m1=True,
            )
            if bias is not None:
                out = out + bias
            return out
        out = kernels.w4a8_linear(
            x,
            layer._w_packed_op,  # type: ignore[attr-defined]
            layer._scales_op,  # type: ignore[attr-defined]
            layer._global_op,  # type: ignore[attr-defined]  the f32 global rides the zeros slot
            self.quant.group_size,
            weight_is_e2m1=True,
        )
        out = out.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out

    def apply_swiglu(self, layer: "BaseOP", x: torch.Tensor) -> torch.Tensor | None:
        """FUSED gate_up + silu (NVFP4 / E2M1, two-level scale) at decode; None -> caller falls back.

        THIS METHOD HAD NO apply_swiglu AT ALL, so an NVFP4 model never reached a fused swiglu on
        any path — it was the sibling both earlier fusions skipped. Two things had to be true before
        it could: the A16 GEMV had to fold per 16-K half (NVFP4's native group is 16, and the arm
        refused group % 32 outright), and `_fused_swiglu_ok` had to stop carrying the stale % 32.
        Both landed, so the gap is now just a missing method.

        The f32 per-output-channel global rides the `w_zeros` slot, as everywhere else in this class
        — the kernel's `wscale_epi` reads it per half, which is required here and not merely nice:
        gate and up are independently quantised and have DIFFERENT globals.
        """
        w = getattr(layer, "_w_packed_op", None)
        if w is None or not _fused_swiglu_ok(x, w, self.quant.group_size):
            return None
        if getattr(layer, "_w4a16", False):
            return kernels.w4a16_linear_silu(
                x, w, layer._scales_op, layer._global_op,  # type: ignore[attr-defined]
                self.quant.group_size, weight_is_e2m1=True,
            ).to(x.dtype)
        return kernels.w4a8_linear_silu(
            x, w, layer._scales_op, layer._global_op,  # type: ignore[attr-defined]
            self.quant.group_size, weight_is_e2m1=True,
        ).to(x.dtype)


class Fp8BlockDequantLinearMethod:
    """DeepSeek-style BLOCKWISE fp8 (e4m3 weights + a 2-D scale per `block_structure` tile),
    dequantized to bf16 ONCE at load and then served as an ordinary unquantized linear.

    WHY DEQUANTIZE RATHER THAN SERVE IT QUANTIZED. A blockwise scale varies along K as well as N, so
    unlike the per-output-channel `Fp8W8A8LinearMethod` it cannot fold into the GEMM epilogue, and no
    kernel in this engine consumes a 2-D weight scale. The options were a new blockwise fp8 kernel or
    an exact one-time dequantization; the second is chosen because it is EXACT (an fp8 value times its
    block scale, widened — no second approximation), needs no kernel, and lands the module on
    `minv_linear`, the same M-invariant chokepoint every other unquantized Linear uses, so a
    chunked / prefix-cached / spec-verify forward still matches a fresh one bit-for-bit.

    THE MEMORY IS AFFORDABLE HERE, which is the only reason this is a reasonable trade. It applies to
    Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ's attention and GDN projections, and the NVFP4 sibling of that
    same checkpoint SPARES those modules entirely — i.e. serves them bf16 already, at the same size
    this produces. The bulk of the model (the routed experts) stays 4-bit either way. Do NOT reach for
    this method on a checkpoint whose blockwise-fp8 modules are the bulk: there it would double the
    weight budget and a real blockwise kernel is the answer.

    The declared per-group-128 dynamic fp8 ACTIVATION scheme is deliberately dropped with the weight
    quantization: once the weight is bf16 there is nothing to pair a quantized activation with, and
    quantizing x for a bf16 GEMM would be a pure loss. Same reasoning the W4A16 arms use.
    """

    def __init__(self, quant: QuantConfig) -> None:
        self.quant = quant

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        # CHECKPOINT layout so the BaseOP loader lands tensors directly. N, K are the LOCAL (per-TP)
        # sizes, and the scale is declared in the SAME sharded space so the generic sharder splits it
        # with the weight: block 128 divides every shipped dimension of this checkpoint (2560/128=20,
        # 10240/128=80, 12288/128=96, 6144/128=48) and it must keep dividing after the TP split, so
        # that is asserted rather than assumed — a non-dividing shard would silently misalign every
        # scale tile against the rows it scales.
        bn, bk = (int(self.quant.block_structure[0]), int(self.quant.block_structure[-1]))
        N, K = out_features, in_features
        if N % bn or K % bk:
            raise ValueError(
                f"blockwise fp8: local shape ({N}, {K}) is not divisible by block {(bn, bk)}. The "
                f"scale tiles would misalign against the rows they scale. This is usually a TP split "
                f"that cut a block in half."
            )
        layer.weight = torch.empty((N, K), dtype=torch.float8_e4m3fn)
        # `weight_scale_inv`, not `weight_scale`: the DeepSeek blockwise name, which is what the
        # checkpoint ships (BF16 [N/128, K/128]).
        layer.weight_scale_inv = torch.empty((N // bn, K // bk), dtype=torch.bfloat16)

    def process_weights_after_load(self, layer: "BaseOP") -> None:
        bn, bk = (int(self.quant.block_structure[0]), int(self.quant.block_structure[-1]))
        w = layer.weight
        N, K = w.shape
        # Expand the tile scale to full (N, K) by repeating each tile value over its block, then
        # multiply in f32 and narrow once. `repeat_interleave` on both axes rather than a broadcast
        # reshape: the reshape spelling only works when N/bn and K/bk tile exactly in that memory
        # order, and getting it wrong transposes the scale field silently.
        sc = layer.weight_scale_inv.to(torch.float32)
        sc = sc.repeat_interleave(bn, dim=0).repeat_interleave(bk, dim=1)[:N, :K]
        layer.weight = (w.to(torch.float32) * sc).to(torch.bfloat16).contiguous()
        del layer.weight_scale_inv

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        from minisgl.layers.minv import minv_linear

        return minv_linear(x, layer.weight, bias)


class Fp8W8A8LinearMethod:
    """Dense fp8 W8A8 linear: per-output-channel fp8 (e4m3) weights + dynamic per-token fp8
    activations (the RedHatAI *-FP8-dynamic scheme: weights `strategy:channel`, activations
    `dynamic:token`). Served through kernels.w8a8_dense_linear — the same validated W8A8 WMMA core as
    the fp8 MoE experts, reused as a single-expert grouped GEMM (no dedicated dense kernel needed).
    Config-selected purely from `quant.is_fp8_w8a8` (float-quantized, 8-bit)."""

    def __init__(self, quant: QuantConfig) -> None:
        self.quant = quant

    def create_weights(self, layer: "BaseOP", out_features: int, in_features: int) -> None:
        # CHECKPOINT layout so the BaseOP loader lands tensors directly: fp8 e4m3 weight (N,K) +
        # per-output-channel scale (N,1). compressed-tensors fp8 ships the scale fp16 (some heads
        # bf16); declare fp16 like the CT W4A8 path — the engine._cast normalizes both to fp16 —
        # then process_weights_after_load promotes it to the f32 the kernel ABI wants. (N=out, K=in
        # are the LOCAL/per-TP sizes.)
        N, K = out_features, in_features
        layer.weight = torch.empty((N, K), dtype=torch.float8_e4m3fn)
        layer.weight_scale = torch.empty((N, 1), dtype=torch.float16)

    def process_weights_after_load(self, layer: "BaseOP") -> None:
        # op layout == natural row-major e4m3 (the GEMM indexes [n*K+k]); the kernel takes the e4m3
        # bytes as uint8 (zero-copy bitcast preserving the bit pattern) + a flat (N,) f32 channel
        # scale — the same op layout _GroupedFP8Experts.post_load builds, minus the E dim.
        layer._w_op = layer.weight.contiguous().view(torch.uint8)  # type: ignore[attr-defined]
        layer._scales_op = layer.weight_scale.squeeze(-1).contiguous().float()  # (N,1)->(N,)
        del layer.weight, layer.weight_scale

    def apply(
        self, layer: "BaseOP", x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        out = kernels.w8a8_dense_linear(
            x, layer._w_op, layer._scales_op  # type: ignore[attr-defined]
        )
        if bias is not None:
            out = out + bias
        return out
