from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import (
    DistributedCommunicator,
    get_dp_info,
    get_ep_rank,
    get_ep_size,
    get_tp_info,
    is_ep_enabled,
    is_ep_over_tp,
)
from minisgl.quant import kernels
from minisgl.utils import div_even
from minisgl.weights.granule import (
    ExpertContainer,
    GranuleSpec,
    assert_decode_policy_agrees,
    assert_granule_pair_consistent,
    decode_policy,
    spec_for_container,
)
from minisgl.weights.granule import offload_refusal as granule_offload_refusal

from .base import BaseOP

if TYPE_CHECKING:
    from minisgl.quant.config import QuantConfig


# The MoE all-reduce's comms/compute overlap used to live here as MINISGL_MOE_ASYNC_AR, a Qwen3.5-MoE-
# only side-stream chunking of this one collective. It is now the model-agnostic primitive in
# layers/tp_overlap.py (MINISGL_TP_OVERLAP / MINISGL_TP_AR_CHUNKS), which Qwen3.5-MoE and Gemma4 both
# call. Nothing MoE-specific was lost: the trick was never about experts, only about having independent
# compute to hide a collective behind.


class _GroupedGPTQExperts(ExpertContainer, BaseOP):
    """Per-expert grouped GPTQ buffers for one of the two MoE GEMMs (w13 or w2).

    Declared in CHECKPOINT layout, STACKED over the E experts so the streaming loader's
    merge->stack path lands tensors here directly (keys `<...>.{qweight,scales,qzeros}`):
        qweight (E, K//pf, N) i32   scales (E, K//g, N) f16   qzeros (E, K//g, N//pf) i32
    where N=out, K=in are per-expert. `post_load` converts each expert to the op's native
    grouped layout (the same gptq_to_op_layout used for dense linears) and stacks:
        _w_op (E, N, K//pf) i32   _scales_op (E, N, K//g) f16   _zeros_op (E, N//pf, K//g) i32
    which is exactly what `kernels.w4a8_moe` consumes for w13 / w2."""

    def __init__(self, num_experts: int, out_features: int, in_features: int, quant: "QuantConfig"):
        pf = 32 // quant.bits
        g = quant.group_size
        N, K = out_features, in_features
        assert K % pf == 0 and K % g == 0 and N % pf == 0, (
            f"grouped GPTQ needs K%{pf}==0,K%{g}==0,N%{pf}==0; got N={N},K={K}"
        )
        self.qweight = torch.empty((num_experts, K // pf, N), dtype=torch.int32)
        self.scales = torch.empty((num_experts, K // g, N), dtype=torch.float16)
        self.qzeros = torch.empty((num_experts, K // g, N // pf), dtype=torch.int32)
        self._quant = quant
        # E recorded, never inferred — see weights/granule.py::ExpertContainer.
        self._num_experts = num_experts

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedGPTQExperts holds weights; call kernels.w4a8_moe instead")

    def post_load(self) -> None:
        assert not self._quant.desc_act, "GPTQ desc_act (act-order) not supported"
        E = self.qweight.shape[0]
        w_op, s_op, z_op = [], [], []
        for e in range(E):
            w, s, z = kernels.gptq_to_op_layout(
                self.qweight[e], self.scales[e], self.qzeros[e], bits=self._quant.bits
            )
            w_op.append(w)
            s_op.append(s)
            z_op.append(z)
        self._w_op = torch.stack(w_op, dim=0)
        self._scales_op = torch.stack(s_op, dim=0)
        self._zeros_op = torch.stack(z_op, dim=0)
        if kernels.MOE_W4A16 != "0":
            # W4A16 (fp16-act) path: repack int4 op-layout -> register-direct w_rep_wide and DROP the
            # fp8 op-layout (frees the memory; scales/zeros are shared). See kernels.w4a16_moe.
            import fp8_wmma

            N, K8 = self._w_op.shape[1], self._w_op.shape[2]
            wide = kernels._w4a16_wide(self._quant.group_size)
            w_rep = fp8_wmma.repack_int4_to_w_rep_moe(self._w_op, N, K8 * 8)
            self._w_rep = fp8_wmma.repack_w_rep_wide_moe(w_rep, wide)
            del self._w_op
        del self.qweight, self.scales, self.qzeros


class _GroupedAWQExperts(ExpertContainer, BaseOP):
    """AWQ-gemm experts for one MoE GEMM (w13 or w2), STACKED over E (checkpoint layout).

    AWQ packs int4 along the OUTPUT N with the GEMM interleave: qweight (E, K, N//pf) int32,
    per-group scales (E, K//g, N), qzeros (E, K//g, N//pf) (AWQ is asymmetric -> zeros ALWAYS
    present). `post_load` runs the proven `awq_to_op_layout` per expert (undo interleave,
    transpose, repack along K) and stacks to the op's grouped layout
    (_w_op (E, N, K//pf), _scales_op (E, N, K//g), _zeros_op (E, N//pf, K//g)). N=out, K=in."""

    def __init__(self, num_experts: int, out_features: int, in_features: int, quant: "QuantConfig"):
        pf = 32 // quant.bits
        g = quant.group_size
        N, K = out_features, in_features
        assert N % pf == 0 and K % g == 0, f"AWQ experts need N%{pf}==0,K%{g}==0; got N={N},K={K}"
        self.qweight = torch.empty((num_experts, K, N // pf), dtype=torch.int32)
        self.scales = torch.empty((num_experts, K // g, N), dtype=torch.float16)
        self.qzeros = torch.empty((num_experts, K // g, N // pf), dtype=torch.int32)
        self._quant = quant
        # E recorded, never inferred — see weights/granule.py::ExpertContainer.
        self._num_experts = num_experts

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedAWQExperts holds weights; call kernels.w4a8_moe instead")

    def post_load(self) -> None:
        E = self.qweight.shape[0]
        w_op, s_op, z_op = [], [], []
        for e in range(E):
            w, s, z = kernels.awq_to_op_layout(
                self.qweight[e], self.scales[e], self.qzeros[e], bits=self._quant.bits
            )
            w_op.append(w)
            s_op.append(s)
            z_op.append(z)
        self._w_op = torch.stack(w_op, dim=0)
        self._scales_op = torch.stack(s_op, dim=0)
        self._zeros_op = torch.stack(z_op, dim=0)
        if kernels.MOE_W4A16 != "0":
            # W4A16 (fp16-act) path: repack int4 op-layout -> register-direct w_rep_wide and DROP the
            # fp8 op-layout (frees the memory; scales/zeros are shared). See kernels.w4a16_moe.
            import fp8_wmma

            N, K8 = self._w_op.shape[1], self._w_op.shape[2]
            wide = kernels._w4a16_wide(self._quant.group_size)
            w_rep = fp8_wmma.repack_int4_to_w_rep_moe(self._w_op, N, K8 * 8)
            self._w_rep = fp8_wmma.repack_w_rep_wide_moe(w_rep, wide)
            del self._w_op
        del self.qweight, self.scales, self.qzeros


class _GroupedRXFExperts(ExpertContainer, BaseOP):
    """RXF W4(NL)-A8 experts for one MoE GEMM (w13 or w2), STACKED over E.

    RXF ships weights op-layout already (no AWQ/GPTQ unpack-transpose-repack): weight_packed
    (E, N, K/2) uint8 NL indices, weight_scale (E, N, K/32) fp16 per-group scale, group=32,
    symmetric NL codebook (no zero-points). N=out, K=in per expert. `post_load` transposes the SCALE
    to group-major (E, K/32, N) — the op indexes it `[g*N + n]`, so N must be the contiguous axis for
    the 16-lane fragment read to coalesce. Consumed by kernels.rxf_moe."""

    def __init__(self, num_experts: int, out_features: int, in_features: int, quant: "QuantConfig"):
        N, K = out_features, in_features
        span = quant.rotation_span
        assert K % 32 == 0 and K % span == 0 and K % 2 == 0, (
            f"grouped RXF needs K%32==0,K%span({span})==0,K%2==0; got N={N},K={K}"
        )
        self.weight_packed = torch.empty((num_experts, N, K // 2), dtype=torch.uint8)
        self.weight_scale = torch.empty((num_experts, N, K // 32), dtype=torch.float16)
        self._quant = quant
        # E recorded, never inferred — see weights/granule.py::ExpertContainer.
        self._num_experts = num_experts

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedRXFExperts holds weights; call kernels.rxf_moe instead")

    def post_load(self) -> None:
        # GROUP-MAJOR scale, for BOTH the LDS and register-direct paths (they share the same kernel
        # scale contract, so this must NOT sit behind the RXF_REGDIRECT gate — that would feed the
        # LDS path a channel-major tensor and silently compute wrong numbers).
        self.weight_scale = self.weight_scale.transpose(1, 2).contiguous()  # (E, N, K/32) -> (E, K/32, N)
        # Register-direct b128: pre-permute the NL codes into WMMA-B lane order (repack_rxf_w_rep_moe)
        # for kernels.rxf_moe_regdirect (~2x the LDS-staged rxf_moe at decode). Drop weight_packed
        # (the LDS path's buffer) — never both. Off -> keep the as-is buffers for the LDS rxf_moe.
        if not kernels.RXF_REGDIRECT:
            return
        import fp8_wmma  # rxf folded into fp8_wmma

        E, N, Kp = self.weight_packed.shape
        ktiles = (Kp * 2) // 16
        wide = 4 if ktiles % 4 == 0 else 2  # b128 when K%64==0 (ZAYA), else b64
        self._w_rep = fp8_wmma.rxf_repack_w_rep_moe(self.weight_packed.contiguous(), wide)
        self._wide = wide
        del self.weight_packed


class _GroupedCompressedTensorsExperts(ExpertContainer, BaseOP):
    """compressed-tensors int4 *weight-only* (W4A16) experts for one MoE GEMM (w13 or w2), STACKED
    over E. The checkpoint ships (N=out, K=in per expert):
        weight_packed (E, N, K//pf) int32 — 8 int4 per int32, packed along INPUT K in natural order
            (K-index k -> column k//pf, nibble k%pf).
        weight_scale  (E, N, K//g)  fp16 or bf16 — per-(output-row, input-group) scale, group g=32.
            Checkpoints ship EITHER (35B: bf16; Agents-A1: fp16); engine._cast normalizes both to
            fp16, which is also what the op consumes, so the buffer is declared fp16.
        weight_zero_point (E, N//pf, K//g) int32 — ASYMMETRIC checkpoints only (config_groups
            `symmetric: false`, e.g. `cyankiwi/Agents-A1-AWQ-INT4`); int4-packed 8-per-int32 along
            the OUTPUT N, i.e. already the op's zeros layout. Symmetric checkpoints omit it.
    This is structurally the op's grouped `_w_op (E,N,K//pf)` / `_scales_op (E,N,K//g)` layout ALREADY
    (same natural nibble order as gptq_to_op_layout's output), so `post_load` is a cheap whole-tensor
    fixup rather than a per-expert unpack/transpose. It mirrors the DENSE compressed-tensors linear
    (`W4A8LinearMethod.process_weights_after_load`) exactly, one E dimension up:
      * The op wants nibbles as uint4b8 (nibble = q + 8). "pack-quantized" ships int4 in one of TWO
        packings per producer, so the convention is DETECTED from the nibble distribution rather than
        assumed: two's-complement (mode at 0) is converted by flipping each nibble's top bit — XOR
        0x8 per nibble == XOR 0x88 per byte via a uint8 view — since `(q & 0xF) ^ 8 == q + 8` for q in
        [-8,7]; already-offset uint4b8 (mode at 8) passes through UNCHANGED, because an XOR there
        re-flips the top bit and yields garbage weights.
      * zeros_op is the real per-group `weight_zero_point` when the checkpoint is asymmetric (it
        shares the weight's sign convention — same quantizer — so it gets the SAME transform, putting
        W_u and Z_u in one unsigned domain where the op's `scale*(W_u - Z_u) == scale*(q - zp)` is
        exact), else the constant 8 (every nibble 8 -> every int32 0x88888888), shape (E,N//pf,K//g).
    Then `kernels.w4a8_moe` consumes `_w_op/_scales_op/_zeros_op` exactly as for GPTQ/AWQ (the
    activations are quantized to int8 by that kernel — same W4A16-weights-through-W4A8-kernel path the
    AWQ experts already use)."""

    # This is the one shipped format whose post_load makes a DECODE decision that no tensor records:
    # `ct_packed_sign_convention` reads the nibble histogram and either XORs the whole stack to
    # uint4b8 or leaves it. w13 and w2 come from one checkpoint and one quantizer, so they must
    # resolve identically; a disagreement decodes one GEMM `q+8` and the other two's-complement, with
    # right shapes and plausible text. Declaring the DECISION path (not `_ct_sign` itself, whose
    # margin/sample counts differ per stack) puts it in `GranuleSpec.policy`, in `fingerprint()` and
    # under `assert_granule_pair_consistent`, which `MoELayer.granule_specs` runs over the pair.
    _granule_policy = ("_ct_sign.uint4b8",)

    def __init__(self, num_experts: int, out_features: int, in_features: int, quant: "QuantConfig"):
        pf = 32 // quant.bits  # 8
        g = quant.group_size  # 32
        N, K = out_features, in_features
        assert K % pf == 0 and K % g == 0 and N % pf == 0, (
            f"grouped compressed-tensors needs K%{pf}==0,K%{g}==0,N%{pf}==0; got N={N},K={K}"
        )
        self.weight_packed = torch.empty((num_experts, N, K // pf), dtype=torch.int32)
        # fp16, matching the DENSE compressed-tensors linear (quant/method.py create_weights) and
        # engine._cast, which normalizes every non-fp32 `.weight_scale` to fp16. Declaring bf16 here
        # (as this did) both refused every CT-MoE checkpoint at the load-time dtype check AND, once
        # that check coerces, would round an fp16 scale through bf16 and lose 2 mantissa bits before
        # post_load casts it straight back to fp16.
        self.weight_scale = torch.empty((num_experts, N, K // g), dtype=torch.float16)
        if not quant.sym:
            # ASYMMETRIC: real per-group zero-points, already the op's packed [N//pf, G] layout.
            self.weight_zero_point = torch.empty((num_experts, N // pf, K // g), dtype=torch.int32)
        self._quant = quant
        # E recorded, never inferred — see weights/granule.py::ExpertContainer.
        self._num_experts = num_experts

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedCompressedTensorsExperts holds weights; call kernels.w4a8_moe")

    def post_load(self) -> None:
        from minisgl.quant import kernels
        from minisgl.quant.method import apply_ct_sign, ct_packed_sign_convention

        pf = 32 // self._quant.bits
        E, N, Kp = self.weight_packed.shape
        G = self.weight_scale.shape[-1]
        wp = self.weight_packed.contiguous()
        # Detected, not assumed — see the class docstring. XOR only for two's-complement packing.
        # ONE decision for the WHOLE (E, N, K/8) stack, sampled with a fixed stride across all E
        # experts and kept on the container. `_ct_sign` is what a chunked per-expert-range repack
        # (weight offload Stage B, plan §5.2) must reuse: re-deriving it per chunk lets two chunks
        # disagree, which XORs one contiguous block of experts and not the rest — plausible text, no
        # crash, and undetectable downstream.
        conv = ct_packed_sign_convention(wp, name="_GroupedCompressedTensorsExperts.weight_packed")
        self._ct_sign = conv
        self._w_op = apply_ct_sign(wp, conv)
        # GROUP-MAJOR scales/zeros: the op indexes `[g*N + n]`, so N must be the CONTIGUOUS axis (a
        # fragment's 16 lanes differ only in n, and then coalesce into one request). The
        # compressed-tensors checkpoint ships channel-major, so this path transposes; AWQ/GPTQ already
        # ship group-major and no longer transpose at all (quant/kernels.py awq_to_op_layout).
        self._scales_op = self.weight_scale.to(torch.float16).transpose(1, 2).contiguous()  # (E, G, N)
        zp = getattr(self, "weight_zero_point", None)
        if zp is None:
            # SYMMETRIC: zero-point == 8 for every (output, group); every packed nibble 8 -> 0x88888888.
            zeros = torch.empty((E, G, N // pf), dtype=torch.int32)
            zeros.view(torch.uint8).fill_(0x88)
            self._zeros_op = zeros.to(wp.device)
            # DECLARE the residency exemption HERE, in the branch that made the buffer constant.
            # Every expert row is the same 0x88 fill, so it need not travel per granule (~3% of w13).
            # It has to be a DECLARATION rather than something the granule walker detects from the
            # bytes, because each TP rank walks its OWN shard (w13 is column-split, w2 row-split) and
            # a content-derived exemption is a decision two ranks can make differently with no
            # collective to catch it — they would then hold different granule_bytes, different
            # fingerprints and different placement (`weights/placement.py:24`,
            # `weights/host_capacity.py:178`). `quant.sym` is config, so this branch is taken on
            # every rank or on none. `derive_granule_spec` re-verifies it bitwise, so if a future
            # checkpoint or repack makes these rows differ the boot fails loudly instead of
            # dequantizing every expert against expert 0's zeros.
            self._residency_shared = ("_zeros_op",)
        else:
            # ASYMMETRIC: same sign convention as the weight, so the same transform. The 4-bit packing
            # runs along N *within* each int32, so transposing the (N//pf, G) axes leaves it intact.
            zp = apply_ct_sign(zp.contiguous(), conv)
            self._zeros_op = zp.transpose(1, 2).contiguous()  # (E, N//pf, G) -> (E, G, N//pf)
            del self.weight_zero_point
        del self.weight_packed, self.weight_scale
        if kernels.MOE_W4A16 != "0":
            # W4A16 (fp16-act) path: repack int4 op-layout -> register-direct w_rep_wide and DROP the
            # fp8 op-layout (same as _GroupedAWQExperts). g=32 -> wide 2 (b64, kernel 13fba94).
            import fp8_wmma

            N2, K8 = self._w_op.shape[1], self._w_op.shape[2]
            wide = kernels._w4a16_wide(self._quant.group_size)
            w_rep = fp8_wmma.repack_int4_to_w_rep_moe(self._w_op, N2, K8 * 8)
            self._w_rep = fp8_wmma.repack_w_rep_wide_moe(w_rep, wide)
            del self._w_op


class _GroupedMxFp4Experts(ExpertContainer, BaseOP):
    """MXFP4 (OCP E2M1 weights + E8M0 per-32-block scale) experts for one MoE GEMM (w13 or w2),
    STACKED over E. The compressed-tensors `mxfp4-pack-quantized` checkpoint ships (per expert,
    merged gate|up into w13 / down into w2 by the loader):
        weight_packed (E, N, K//2) uint8 — 2 E2M1 nibbles/byte, low nibble = lower K index.
        weight_scale  (E, N, K//32) uint8 — E8M0, one shared exponent per 32-element block.
    (N=out, K=in per expert.) `post_load` runs the MXFP4 converter (nibbles -> (E,N,K//8) int32 codes
    verbatim; E8M0 -> fp16 group scale) so `kernels.w4a8_moe(..., weight_is_e2m1=True)` consumes
    `_w_op/_scales_op` exactly as the int4 experts do. Symmetric -> no zero-points."""

    def __init__(self, num_experts: int, out_features: int, in_features: int, quant: "QuantConfig"):
        g = quant.group_size  # 32
        N, K = out_features, in_features
        assert K % 2 == 0 and K % g == 0 and N % 8 == 0, (
            f"grouped MXFP4 needs K%2==0,K%{g}==0,N%8==0; got N={N},K={K}"
        )
        # CHECKPOINT layout (uint8); E8M0 scale is an integer exponent, NOT a float, so the engine's
        # _cast leaves both uint8 buffers untouched.
        self.weight_packed = torch.empty((num_experts, N, K // 2), dtype=torch.uint8)
        self.weight_scale = torch.empty((num_experts, N, K // g), dtype=torch.uint8)
        self._quant = quant
        # E recorded, never inferred — see weights/granule.py::ExpertContainer.
        self._num_experts = num_experts

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedMxFp4Experts holds weights; call kernels.w4a8_moe instead")

    def post_load(self) -> None:
        from minisgl.quant import kernels

        N = self.weight_packed.shape[1]
        K = self.weight_packed.shape[2] * 2
        if kernels.MOE_MXFP4_REGDIRECT:
            # Register-direct b128: E2M1 nibbles -> WMMA-B lane-order w_rep_wide + E8M0->fp16 group
            # scales, consumed by kernels.w4a16_moe(weight_is_e2m1=True). fp16 acts DIRECT (no
            # act-quant) — faster AND higher quality than the LDS w4a8_moe e2m1 path. group=32 -> wide
            # 2 (b64). Symmetric MXFP4 -> no zero-points.
            import fp8_wmma

            wide = kernels._w4a16_wide(self._quant.group_size)  # g=32 -> 2
            self._w_rep, _scales_rd = fp8_wmma.mxfp4_to_w_rep_moe(
                self.weight_packed, self.weight_scale, N, K, wide
            )
            # mxfp4_to_w_rep_moe passes the E8M0->fp16 scale through in CHECKPOINT (E, N, K//32) order;
            # the op wants it GROUP-MAJOR so its `[g*N + n]` read coalesces.
            self._scales_rd = _scales_rd.transpose(1, 2).contiguous()  # (E, K//32, N)
            del self.weight_packed, self.weight_scale
            return

        from minisgl.quant import mxfp4

        # NATIVE E8M0. The op reads the checkpoint's 1-byte block scale directly
        # (`w4a8_tile::E8m0GroupScale` / `FmtMxfp4E8m0`, rdna4-hip-kernels ebcb0d3) instead of the
        # fp16 widening this used to do, which was worse on two axes:
        #   BYTES  0.5625 -> 0.53125 B/weight (5.6% fewer bytes read on a bandwidth-bound decode)
        #          and HALF the resident scale bytes.
        #   EXACT  E8M0 spans 2^-127..2^127 and fp16 saturates outside 2^-14..2^15, so the old path
        #          could serve SATURATED scales on an out-of-window checkpoint — and the branch
        #          below only LOGGED it and carried on. There is nothing to saturate now.
        # `MINISGL_MXFP4_FP16_SCALES=1` restores the old widening; it exists so a numerics report can
        # be bisected onto the previous path, not as a tuning knob.
        import os as _os

        if (_os.environ.get("MINISGL_MXFP4_FP16_SCALES") or "").strip() == "1":
            conv = mxfp4.convert_mxfp4_moe(self.weight_packed, self.weight_scale)
            info = conv["scale_info"]
            if not info["fp16_range_ok"]:
                from minisgl.utils import init_logger

                init_logger("mxfp4").info_rank0(
                    f"[mxfp4-moe] E8M0 group scales exceed the fp16 store "
                    f"(exp {info['exp_min']}..{info['exp_max']}, {info['fp16_overflow_groups']} overflow "
                    f"/ {info['e8m0_nan_groups']} e8m0-NaN groups); an fp32 group-scale path may be needed."
                )
        else:
            conv = mxfp4.convert_mxfp4_moe_e8m0(self.weight_packed, self.weight_scale)
            info = conv["scale_info"]
            if info["would_saturate_fp16"] or info["e8m0_nan_groups"]:
                from minisgl.utils import init_logger

                init_logger("mxfp4").info_rank0(
                    f"[mxfp4-moe] native E8M0 scales: exp {info['exp_min']}..{info['exp_max']}, "
                    f"{info['would_saturate_fp16']} group(s) the OLD fp16 store would have saturated, "
                    f"{info['e8m0_nan_groups']} e8m0-NaN group(s)."
                )
        self._w_op = conv["w_packed"]  # (E, N, K//8) int32
        # GROUP-MAJOR for the op's coalesced `[g*N + n]` scale read (see _GroupedAWQExperts).
        # GROUP-MAJOR for the op's coalesced `[g*N + n]` read. dtype now follows the path:
        # uint8 E8M0 by default, fp16 under the bisect knob.
        self._scales_op = conv["scales"].transpose(1, 2).contiguous()  # (E, K//32, N)
        del self.weight_packed, self.weight_scale


class _GroupedNvFp4Experts(ExpertContainer, BaseOP):
    """NVFP4 (compressed-tensors 'nvfp4-pack-quantized') experts for one MoE GEMM (w13 or w2), STACKED
    over E. NVFP4 has the IDENTICAL 4-bit E2M1 weight codes as MXFP4; only the scale differs, and this
    container keeps that scale in the CHECKPOINT'S OWN TWO LEVELS rather than folding it:

        weight_packed (E, N, K//2)  uint8          — 2 E2M1 nibbles/byte.
        weight_scale  (E, N, K//16) float8_e4m3fn  — the per-16-element block scale, BYTE-VERBATIM.
        weight_global (E, N)        float32        — the per-OUTPUT-CHANNEL global MULTIPLIER.

    WHY TWO TENSORS AND NOT ONE FOLDED fp16 SCALE (which is what this was until 2026-09-05). The fold
    is LOSSY — a measured 4.37e-04 max / 1.4-2.4e-04 mean relative error on every weight of a real
    checkpoint, where the two-level form is exact — and it is BIGGER, because an fp16 group scale is
    2 bytes per 16 weights where an e4m3 block scale is 1. See `quant/nvfp4.py`'s module docstring;
    the old claim that the fold was exact is the reason this sat unfixed.

    The global is a per-output-channel VECTOR, not a scalar, because the loader's gate|up merge and
    per-expert stack combine differently-scaled matrices: after the merge the global is constant on
    each contiguous output-channel RANGE, which an (N,) vector expresses and a scalar cannot. That
    shape is what lets `torch.cat(dim=0)` and `_ExpertStacker` carry it with no special case.

    `post_load` packs the nibbles to (E,N,K//8) int32 codes, transposes the block scale group-major,
    and bitcasts the global to int32 so it can ride the kernel's `w_zeros` POINTER SLOT — NVFP4 is
    symmetric, so that slot is otherwise null and there is no op-schema change (precedent:
    `_GroupedRXFExperts` threads the RXF NL codebook the same way). `kernels.w4a8_moe(...,
    weight_is_e2m1=True)` at group_size 16 then consumes `_w_op/_scales_op/_global_op`; the kernel
    picks `w4a8_tile::E4m3GroupScaleGlobal` vs `Fp16GroupScale` off the SCALES DTYPE, so there is no
    flag to get out of sync. Symmetric — there are no zero-points to displace."""

    def __init__(self, num_experts: int, out_features: int, in_features: int, quant: "QuantConfig"):
        g = quant.group_size  # 16
        N, K = out_features, in_features
        assert K % 2 == 0 and K % g == 0 and N % 8 == 0, (
            f"grouped NVFP4 needs K%2==0,K%{g}==0,N%8==0; got N={N},K={K}"
        )
        self.weight_packed = torch.empty((num_experts, N, K // 2), dtype=torch.uint8)
        # float8_e4m3fn, not uint8: this is the checkpoint tensor's own dtype, so the loader's
        # BaseOP dtype assertion passes on a VERBATIM tensor and no cast is possible. A uint8
        # declaration would accept a value-converted tensor (byte 126 -> 126.0) just as happily.
        self.weight_scale = torch.empty((num_experts, N, K // g), dtype=torch.float8_e4m3fn)
        self.weight_global = torch.empty((num_experts, N), dtype=torch.float32)
        self._quant = quant
        # E recorded, never inferred — see weights/granule.py::ExpertContainer.
        self._num_experts = num_experts

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedNvFp4Experts holds weights; call kernels.w4a8_moe instead")

    def post_load(self) -> None:
        from minisgl.quant import nvfp4

        conv = nvfp4.convert_nvfp4_moe(self.weight_packed, self.weight_scale)
        self._w_op = conv["w_packed"]  # (E, N, K//8) int32
        # GROUP-MAJOR for the op's coalesced `[g*N + n]` scale read (see _GroupedAWQExperts).
        self._scales_op = conv["scales"].transpose(1, 2).contiguous()  # (E, K//16, N) e4m3
        # (E, N) f32 -> the SAME BYTES viewed as int32, because the op's global slot is `w_zeros`,
        # typed `const int*`. `Tensor.view(dtype)` is a zero-copy bitcast at equal itemsize, and the
        # kernel does `reinterpret_cast<const float*>` on the other side — so the f32 values survive
        # exactly. A `.to(torch.int32)` here would VALUE-convert (2.078e-04 -> 0) and produce a model
        # whose every expert output is zero; the two spellings are one character apart, so this
        # comment is the guard.
        self._global_op = self.weight_global.contiguous().view(torch.int32)  # (E, N)
        del self.weight_packed, self.weight_scale, self.weight_global


class _GroupedFP8Experts(ExpertContainer, BaseOP):
    """Weight-only fp8 (F8_E4M3) experts for one MoE GEMM (w13 or w2), STACKED over E.

    ZAYA's experts are compressed-tensors *float-quant*: each expert weight is F8_E4M3 with a
    per-output-channel (dim-0) F32 `weight_scale`. Storing the raw fp8 (~8 GB total) instead of
    dequantizing to bf16 (~16 GB) is what lets the 8B fit a single 16 GB gfx1201 card. At compute the
    fp8 weights feed the native W8A8 grouped-MoE kernel DIRECTLY (no dequant): `post_load` bitcasts
    them to the kernel's uint8 op layout (`_w_op`/`_scales_op`) and drops the checkpoint copies.
    `dequant()` (the full fp8->bf16 stack) is the gated A/B reference ONLY (`MINISGL_ZAYA_OLDMOE=1`) —
    it re-materializes the WHOLE (E,N,K) bf16 stack per GEMM per forward, so it is decidedly NOT the
    hot path. Buffers are declared in CHECKPOINT dtype/shape so the BaseOP loader's dtype assertion
    passes and the streaming stack path lands tensors here directly:
        weight (E, N, K) f8_e4m3   weight_scale (E, N, 1) f32     (N=out, K=in per expert)."""

    def __init__(self, num_experts: int, out_features: int, in_features: int):
        N, K = out_features, in_features
        self.weight = torch.empty((num_experts, N, K), dtype=torch.float8_e4m3fn)
        self.weight_scale = torch.empty((num_experts, N, 1), dtype=torch.float32)
        # E recorded, never inferred — see weights/granule.py::ExpertContainer.
        self._num_experts = num_experts

    def forward(self, *args, **kwargs):  # pragma: no cover - storage container, never called
        raise RuntimeError("_GroupedFP8Experts holds weights; dequant per-expert at compute")

    # Which whole-stack A/B knob was LIVE when `post_load` built these buffers, or None. RECORDED at
    # the point of decision, never re-read: `offload_refusal` runs at placement time, arbitrarily
    # later than both `_FP8MoEMethod.__init__` (which snapshots MINISGL_ZAYA_OLDMOE at construction)
    # and `post_load` (which snapshots both to decide whether to repack). Three reads of a mutable
    # process global at three times can disagree, and one direction of disagreement is silent: a
    # container BUILT under the knob but ASKED after it was cleared answers "offloadable", becomes
    # host-resident, and then streams the whole (E,N,K) stack over PCIe every step — which presents
    # as "the offload mechanism does not work", not as a stale env read.
    _whole_stack_knob: "str | None" = None

    def offload_refusal(self) -> "str | None":
        """Refuse host residency under the two whole-stack A/B toggles.

        `MINISGL_ZAYA_OLDMOE=1` routes the forward through `dequant()`, which materializes the ENTIRE
        (E, N, K) bf16 stack every GEMM every forward, and `MINISGL_ZAYA_W8A16=1` reads `_w_op` for
        every expert rather than the routed ones. Either one pulls the whole stack across PCIe per
        step regardless of routing, so an offloaded serve would be catastrophically slow for a reason
        that has nothing to do with the mechanism — and somebody would reasonably conclude the
        mechanism does not work. Refuse at boot with the knob named."""
        knob = self._whole_stack_knob
        if knob is None:
            return None
        return (
            f"{knob}=1 was live when these buffers were built, so the forward reads EVERY expert's "
            f"weights (whole-stack dequant / full _w_op scan) and host residency would stream the "
            f"entire stack per step. Unset {knob} and rebuild the model to offload these experts."
        )

    def dequant(self, dtype: torch.dtype) -> torch.Tensor:
        """A/B-reference ONLY (`MINISGL_ZAYA_OLDMOE=1`): dequantize ALL experts to `dtype` -> (E,N,K).
        This materializes the full bf16 stack (every expert, both GEMMs) per forward — the transient
        the fp8 storage scheme exists to avoid. The default path never calls this (native W8A8 kernel
        consumes `_w_op`/`_scales_op` directly).

        VECTORIZED (2026-07-04): one whole-tensor dequant instead of a Python per-expert loop+stack.
        The old `torch.stack([... for e in range(E)])` issued ~3E tiny kernels PER dequant × 2 GEMMs ×
        40 layers = the ~9,300-launch/step op-flood that made the fused OLDMOE step 284ms (vs 55ms
        native fp8); this collapses it to 3 ops. `weight_scale` (E,N,1) broadcasts over `weight` (E,N,K)
        exactly as the per-expert `[e]` slices did — bit-identical result."""
        return (self.weight.float() * self.weight_scale).to(dtype)

    def post_load(self) -> None:
        """Build the native W8A8 kernel's op-layout buffers and drop the checkpoint copies.

        The op layout IS the natural (E, N, K) f8_e4m3 row-major checkpoint layout (the GEMM reads
        `w_fp8 + e*N*K` and indexes `[n*K + k]`), so `_w_op` is just a contiguous view; the kernel
        wants the per-output-channel scale as a flat (E, N) f32. Underscore-prefixed so the BaseOP
        state walk skips them. The kernel binding takes the e4m3 bytes as a uint8 tensor (it copies
        them straight to LDS for the fp8 WMMA intrinsic), so reinterpret the f8_e4m3 storage as uint8
        — a zero-copy bitcast that preserves the exact e4m3 bit pattern."""
        self._w_op = self.weight.contiguous().view(torch.uint8)
        self._scales_op = self.weight_scale.squeeze(-1).contiguous().float()  # (E, N, 1) -> (E, N)
        # Register-direct b128: pre-permute the per-expert fp8 weights into WMMA-B lane order (the fp8
        # twin of the W4A16 _w_rep path) and DROP the plain op-layout copy — same ~8 GB footprint, just
        # reordered, so we must never hold BOTH or the 8B OOMs the 16 GB card. Consumed by
        # kernels.w8a8_moe_regdirect. Skipped when an env opt-out (W8A16/OLDMOE) still needs plain
        # _w_op/weight. (Per-container repack transient is one GEMM's ~128 MB, not the whole model.)
        _oldmoe = os.environ.get("MINISGL_ZAYA_OLDMOE", "0") == "1"
        _w8a16 = os.environ.get("MINISGL_ZAYA_W8A16", "0") == "1"
        # Snapshot the whole-stack knob HERE, where it is actually read, so `offload_refusal` reports
        # what this container WAS BUILT AS rather than what the environment happens to say later.
        self._whole_stack_knob = (
            "MINISGL_ZAYA_OLDMOE" if _oldmoe else ("MINISGL_ZAYA_W8A16" if _w8a16 else None)
        )
        if kernels.MOE_W8A8_REGDIRECT and not _oldmoe and not _w8a16:
            import fp8_wmma

            E, N, K = self._w_op.shape
            w_rep = fp8_wmma.repack_fp8_to_w_rep_moe(self._w_op, N, K)
            self._w_rep = fp8_wmma.repack_w_rep_wide_moe(w_rep, 2)  # b128
            del self._w_op
        # A/B toggle: MINISGL_ZAYA_OLDMOE=1 keeps the checkpoint fp8 weights for the legacy
        # dequant->Triton path (forward branch). Default drops them (native W8A8 kernel only).
        if not _oldmoe:
            del self.weight, self.weight_scale


# =====================================================================================
# Config-driven MoE-expert quant selector (mirrors quant.method.create_linear_method for the
# dense linears and the GDN in_proj dispatch). ALL three module families now pick their scheme +
# kernel from the SAME declared QuantConfig the same way — no scattered per-scheme `if`s in the
# layer, no env var that substitutes a different scheme than the checkpoint declares, no model-name
# branches. A `MoEQuantMethod` owns one scheme family: which per-expert weight CONTAINER to
# allocate (__init__) and which grouped kernel to run (forward, both the plain TP path and the
# per-rank EP shard). `create_moe_quant_method` maps the config to the subclass.
# =====================================================================================
class MoEQuantMethod:
    """How a MoE expert GEMM pair (w13 gate|up, w2 down) allocates its weights and runs its matmul.
    The MoELayer owns routing, EP dispatch/combine and the TP all-reduce; the method owns the
    per-expert weight layout + the grouped GEMM."""

    supports_ep: bool = False  # can this scheme run the EP all_gather/mask/all_reduce shard path?
    needs_precomputed_route: bool = False  # True -> forward MUST be handed topk_weights/topk_ids
    # Can GEMM1 consume the PRODUCER's (x_fp8, act_scales) pair from the feeding RMSNorm?
    # This is opt-IN and checked at the call site rather than letting every scheme silently accept
    # and drop the pair: a dropped pair reads as "producer fusion is free" while the pre-kernel
    # quietly still ran, which is precisely the shape of A/B lie this repo has killed five of.
    # False here means the sparse block does not even build the pair, so nothing is wasted either.
    supports_producer_actquant: bool = False

    def create_experts(self, num_experts: int, out_features: int, in_features: int):
        """Allocate the per-expert weight container for ONE GEMM (STACKED over `num_experts` on
        dim 0; caller passes the LOCAL count under EP). Returns a BaseOP container (quantized) or a
        plain stacked bf16/fp16 tensor (unquantized)."""
        raise NotImplementedError

    def apply(
        self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
        top_k: int, renormalize: bool, activation: str, apply_router_weight_on_input: bool,
        x_fp8=None, act_scales=None,
    ) -> "torch.Tensor":
        """Plain (non-EP) forward over the full replicated expert stack.

        `x_fp8`/`act_scales`: the PRODUCER-quantized activation pair from the RMSNorm feeding this
        sparse block (tail_hip `rms_norm_quant`). A scheme that cannot use it must IGNORE it, never
        assert on it — but a scheme that can use it must actually pass it down, because a silently
        dropped pair reads as "producer fusion is free" while the pre-kernel quietly still ran.
        """
        raise NotImplementedError

    def ep_local(
        self, w13, w2, g_hidden, local_weights, local_ids, *, top_k: int, renormalize: bool,
        activation: str = "silu",
    ) -> "torch.Tensor":
        """EP per-rank shard kernel: run THIS rank's local expert stack (w13/w2 already the
        [E_local,...] shard) over the all_gather'd tokens with local-remapped ids/weights.

        `activation` mirrors `apply`'s and MUST be honoured: this path used to take no activation at
        all, so an EP serve of a gelu model produced silu experts — same weights, same shapes, no
        error, just wrong numbers on one deployment topology only. Every override either implements
        the activation or rejects it; none may ignore it."""
        raise NotImplementedError(f"{type(self).__name__} does not support expert parallelism")


# Gated activations the `kernels.w4a8_moe` family serves. "gelu" is HF `gelu_pytorch_tanh` (Gemma4's
# routed experts), which that kernel applies on its UNFUSED gemm1 path. Kept as one shared gate so a
# scheme cannot drift into accepting an activation its kernel does not actually implement — the
# failure mode being silent (right shapes, wrong numbers), not a crash.
_W4A8_ACTIVATIONS = ("silu", "gelu")


def _check_activation(scheme: str, activation: str, allowed=_W4A8_ACTIVATIONS) -> None:
    if activation not in allowed:
        raise NotImplementedError(
            f"MoE {scheme} path supports activation {'|'.join(allowed)}; got {activation!r}"
        )


class _UnquantizedMoEMethod(MoEQuantMethod):
    """bf16/fp16 stacked experts — the fused moe_backend (or the precomputed-route stacked kernel)."""

    def create_experts(self, num_experts, out_features, in_features):
        return torch.empty(num_experts, out_features, in_features)

    def apply(self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
              top_k, renormalize, activation, apply_router_weight_on_input,
              x_fp8=None, act_scales=None):
        if topk_ids is not None:
            # Route computed in the model (Zaya top-1 + MOD, GLM noaux_tc). The moe_backend fuses
            # softmax+topk internally so it can't take a precomputed route — call the stacked kernel.
            from minisgl.moe.fused import fused_experts_impl

            return fused_experts_impl(
                hidden_states, w13, w2, topk_weights, topk_ids,
                activation=activation, apply_router_weight_on_input=apply_router_weight_on_input,
            )
        ctx = get_global_ctx()
        return ctx.moe_backend.forward(
            hidden_states=hidden_states, w1=w13, w2=w2, gating_output=router_logits,
            topk=top_k, renormalize=renormalize, activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


class _W4A8MoEMethod(MoEQuantMethod):
    """int4-weight grouped experts through the shared `kernels.w4a8_moe` (int4 weight x per-token fp8
    act). Covers GPTQ (K-major qweight), AWQ-gemm (N-major, interleaved, asymmetric) and
    compressed-tensors int4 (W4A16 weights served through the same W4A8 kernel) — they differ only in
    CHECKPOINT layout (the container's post_load converts each to the op's grouped triple)."""

    supports_ep = True
    supports_producer_actquant = True

    def __init__(self, quant: "QuantConfig"):
        self._quant = quant
        if quant.is_gptq:
            self._cls = _GroupedGPTQExperts
        elif quant.is_awq:
            self._cls = _GroupedAWQExperts
        elif quant.is_compressed_tensors:
            self._cls = _GroupedCompressedTensorsExperts
        else:
            raise AssertionError(f"W4A8 MoE unsupported quant method: {quant.method}")

    def create_experts(self, num_experts, out_features, in_features):
        return self._cls(num_experts, out_features, in_features, self._quant)

    def apply(self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
              top_k, renormalize, activation, apply_router_weight_on_input,
              x_fp8=None, act_scales=None):
        _check_activation("W4A8", activation)
        assert not apply_router_weight_on_input, "MoE W4A8 path has no router-weight-on-input"
        if getattr(w13, "_w_rep", None) is not None:
            self._reject_w4a16_activation(activation)
            # NOTE the producer pair is deliberately NOT forwarded here. `kernels.w4a16_moe` is a
            # FP16-ACTIVATION path -- it never quantizes activations at all -- so there is no
            # act-quant dispatch to delete and nothing the e4m3 rows could be handed to. Dropping
            # the pair is correct, and it is visible: the engage ledger prints `w4a16_moe` with no
            # `+prequant`, so an A/B cannot mistake this branch for a fused one.
            # W4A16 (fp16-act) path: int4 weights repacked to register-direct _w_rep in post_load;
            # scales/zeros stay op-layout (w4a16_moe consumes _scales_op/_zeros_op directly, see its
            # (E,2*inter,K//g) / (E,(2*inter)//8,K//g) signature). AWQ/GPTQ int4 is asymmetric -> pass
            # zeros; weight_is_e2m1=False (true int4, not MXFP4). Same route fallback as w4a8_moe.
            inter = w13._scales_op.shape[2] // 2  # (E, G, 2*inter) -> group-major, N is last
            return kernels.w4a16_moe(
                hidden_states, w13._w_rep, w13._scales_op, w13._zeros_op,
                w2._w_rep, w2._scales_op, w2._zeros_op,
                hidden_states.shape[1], inter, self._quant.group_size,
                topk_weights=topk_weights, topk_ids=topk_ids,
                router_logits=router_logits, top_k=top_k, renormalize=renormalize,
                weight_is_e2m1=False,
            )
        return kernels.w4a8_moe(
            hidden_states, w13._w_op, w13._scales_op, w13._zeros_op,
            w2._w_op, w2._scales_op, w2._zeros_op,
            router_logits, top_k, renormalize, topk_weights=topk_weights, topk_ids=topk_ids,
            activation=activation, x_fp8=x_fp8, act_scales=act_scales,
        )

    @staticmethod
    def _reject_w4a16_activation(activation: str) -> None:
        # `kernels.w4a16_moe` (the MINISGL_MOE_W4A16=1 register-direct path, selected in post_load by
        # building _w_rep) still hard-codes silu on its tail. Reaching it with gelu would compute silu
        # and return a perfectly well-formed wrong answer, and the env that selects it is a PERF knob
        # nobody associates with numerics — so refuse loudly instead. Fix is the same one-liner as
        # w4a8_moe's (thread the activation to its tail) plus a gelu tail kernel.
        if activation != "silu":
            raise NotImplementedError(
                f"MoE W4A16 (MINISGL_MOE_W4A16=1) is silu-only; got activation {activation!r}. "
                "Unset MINISGL_MOE_W4A16 to serve this model through the W4A8 path."
            )

    def ep_local(self, w13, w2, g_hidden, local_weights, local_ids, *, top_k, renormalize,
                 activation="silu"):
        _check_activation("W4A8", activation)
        if getattr(w13, "_w_rep", None) is not None:
            self._reject_w4a16_activation(activation)
            inter = w13._scales_op.shape[2] // 2  # (E, G, 2*inter) -> group-major, N is last
            return kernels.w4a16_moe(
                g_hidden, w13._w_rep, w13._scales_op, w13._zeros_op,
                w2._w_rep, w2._scales_op, w2._zeros_op,
                g_hidden.shape[1], inter, self._quant.group_size,
                topk_weights=local_weights, topk_ids=local_ids,
                top_k=top_k, renormalize=renormalize, weight_is_e2m1=False,
            )
        return kernels.w4a8_moe(
            g_hidden, w13._w_op, w13._scales_op, w13._zeros_op,
            w2._w_op, w2._scales_op, w2._zeros_op,
            None, top_k, renormalize, topk_weights=local_weights, topk_ids=local_ids,
            activation=activation,
        )


class _MxFp4MoEMethod(MoEQuantMethod):
    """MXFP4 (OCP E2M1) grouped experts through the shared `kernels.w4a8_moe` with
    `weight_is_e2m1=True` — the same W4A8 fp8-WMMA kernel the int4 experts use, only a different
    4-bit decode table + E8M0->fp16 group scale (done at load in `_GroupedMxFp4Experts.post_load`).
    Symmetric, so no zero-points (None). EP-capable exactly like the int4 W4A8 path (E on dim 0 of
    every expert buffer; a shard is a pure dim-0 slice)."""

    supports_ep = True
    supports_producer_actquant = True

    def __init__(self, quant: "QuantConfig"):
        self._quant = quant

    def create_experts(self, num_experts, out_features, in_features):
        return _GroupedMxFp4Experts(num_experts, out_features, in_features, self._quant)

    def apply(self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
              top_k, renormalize, activation, apply_router_weight_on_input,
              x_fp8=None, act_scales=None):
        _check_activation("MXFP4", activation)
        assert not apply_router_weight_on_input, "MoE MXFP4 path has no router-weight-on-input"
        if getattr(w13, "_w_rep", None) is not None:
            _W4A8MoEMethod._reject_w4a16_activation(activation)
            # Register-direct b128, fp16 acts direct (see _GroupedMxFp4Experts.post_load). Qwen3.5-MoE
            # hands raw router_logits (no model-side route), so forward them + top_k/renormalize and let
            # w4a16_moe fuse softmax+topk (topk_ids stays None); a noaux_tc precomputed route passes through.
            inter = w13._scales_rd.shape[2] // 2  # (E, G, 2*inter) group-major -> inter
            return kernels.w4a16_moe(
                hidden_states, w13._w_rep, w13._scales_rd, None, w2._w_rep, w2._scales_rd, None,
                hidden_states.shape[1], inter, self._quant.group_size,
                topk_weights=topk_weights, topk_ids=topk_ids,
                router_logits=router_logits, top_k=top_k, renormalize=renormalize,
                weight_is_e2m1=True,
            )
        return kernels.w4a8_moe(
            hidden_states, w13._w_op, w13._scales_op, None,
            w2._w_op, w2._scales_op, None,
            router_logits, top_k, renormalize, topk_weights=topk_weights, topk_ids=topk_ids,
            weight_is_e2m1=True, activation=activation, x_fp8=x_fp8, act_scales=act_scales,
        )

    def ep_local(self, w13, w2, g_hidden, local_weights, local_ids, *, top_k, renormalize,
                 activation="silu"):
        _check_activation("MXFP4", activation)
        if getattr(w13, "_w_rep", None) is not None:
            _W4A8MoEMethod._reject_w4a16_activation(activation)
            inter = w13._scales_rd.shape[2] // 2  # (E, G, 2*inter) group-major -> inter
            return kernels.w4a16_moe(
                g_hidden, w13._w_rep, w13._scales_rd, None, w2._w_rep, w2._scales_rd, None,
                g_hidden.shape[1], inter, self._quant.group_size,
                topk_weights=local_weights, topk_ids=local_ids, weight_is_e2m1=True,
            )
        return kernels.w4a8_moe(
            g_hidden, w13._w_op, w13._scales_op, None,
            w2._w_op, w2._scales_op, None,
            None, top_k, renormalize, topk_weights=local_weights, topk_ids=local_ids,
            weight_is_e2m1=True, activation=activation,
        )


class _NvFp4MoEMethod(MoEQuantMethod):
    """NVFP4 (compressed-tensors 'nvfp4-pack-quantized') grouped experts through `kernels.w4a8_moe`
    with `weight_is_e2m1=True` at group_size 16 — the SAME e2m1 kernel MXFP4 uses (weights stay
    4-bit), but consuming the checkpoint's TWO-LEVEL scale directly: an e4m3 block scale plus the
    per-output-channel f32 global, which rides the `w_zeros` pointer slot. See
    `_GroupedNvFp4Experts`. Always the LDS path (the register-direct b128 wide-load requires
    group_size%32, which NVFP4's 16 is not — see `kernels._w4a16_wide`), so no `_w_rep`. Qwen3.5-MoE
    hands raw router_logits, which w4a8_moe routes internally (topk_ids None). EP-capable like the
    other e2m1 experts (E on dim 0 of every buffer, including the global)."""

    supports_producer_actquant = True

    supports_ep = True

    def __init__(self, quant: "QuantConfig"):
        self._quant = quant

    def create_experts(self, num_experts, out_features, in_features):
        return _GroupedNvFp4Experts(num_experts, out_features, in_features, self._quant)

    def apply(self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
              top_k, renormalize, activation, apply_router_weight_on_input,
              x_fp8=None, act_scales=None):
        _check_activation("NVFP4", activation)
        assert not apply_router_weight_on_input, "MoE NVFP4 path has no router-weight-on-input"
        # `_global_op` occupies the ZEROS argument. That is not a hack in the shape of one: the
        # kernel's WScale policy owns the slot (`E4m3GroupScaleGlobal::wz_base/epi`) and its
        # `uses_zeros=false` is constexpr, so the zero-point read is dead-code-eliminated under this
        # policy and cannot decode the global's f32 bits as packed nibbles.
        return kernels.w4a8_moe(
            hidden_states, w13._w_op, w13._scales_op, w13._global_op,
            w2._w_op, w2._scales_op, w2._global_op,
            router_logits, top_k, renormalize, topk_weights=topk_weights, topk_ids=topk_ids,
            weight_is_e2m1=True, activation=activation, x_fp8=x_fp8, act_scales=act_scales,
        )

    def ep_local(self, w13, w2, g_hidden, local_weights, local_ids, *, top_k, renormalize,
                 activation="silu"):
        _check_activation("NVFP4", activation)
        return kernels.w4a8_moe(
            g_hidden, w13._w_op, w13._scales_op, w13._global_op,
            w2._w_op, w2._scales_op, w2._global_op,
            None, top_k, renormalize, topk_weights=local_weights, topk_ids=local_ids,
            weight_is_e2m1=True, activation=activation,
        )


class _RXFMoEMethod(MoEQuantMethod):
    """RXF W4(NL)-A8 grouped experts (`kernels.rxf_moe`). No EP path (RXF has no precomputed-topk
    shard route, which EP requires) — stays replicated."""

    supports_ep = False

    def __init__(self, quant: "QuantConfig"):
        self._quant = quant

    def create_experts(self, num_experts, out_features, in_features):
        return _GroupedRXFExperts(num_experts, out_features, in_features, self._quant)

    def apply(self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
              top_k, renormalize, activation, apply_router_weight_on_input,
              x_fp8=None, act_scales=None):
        # Still silu-ONLY, deliberately: both `kernels.rxf_moe` and `rxf_moe_regdirect` hard-code
        # silu_and_mul on their tail, and no RXF checkpoint we serve declares a gelu. Rejecting is the
        # point — the alternative is a gelu model quietly getting silu experts. Adding gelu here is
        # the same one-line policy threading w4a8_moe just got, on those two functions.
        _check_activation("RXF", activation, allowed=("silu",))
        assert not apply_router_weight_on_input, "MoE RXF path has no router-weight-on-input"
        if getattr(w13, "_w_rep", None) is not None:
            # Register-direct b128 (LDS-bypass) — the default RXF path (~2x rxf_moe at decode,
            # bit-exact). post_load built _w_rep/_wide and dropped weight_packed.
            return kernels.rxf_moe_regdirect(
                hidden_states, w13._w_rep, w13.weight_scale, w2._w_rep, w2.weight_scale,
                top_k, topk_weights=topk_weights, topk_ids=topk_ids,
                span=self._quant.rotation_span, wide=w13._wide,
            )
        return kernels.rxf_moe(
            hidden_states, w13.weight_packed, w13.weight_scale, w2.weight_packed, w2.weight_scale,
            router_logits, top_k, renormalize, topk_weights=topk_weights, topk_ids=topk_ids,
            span=self._quant.rotation_span,
        )


class _FP8MoEMethod(MoEQuantMethod):
    """fp8 W8A8 experts (ZAYA): F8_E4M3 weights + a per-output-channel f32 scale, fed to the native
    `kernels.w8a8_moe` fp8-WMMA kernel with per-token fp8 activations — the CHECKPOINT-DECLARED
    scheme, and the DEFAULT. Two opt-outs, both env-gated PERF toggles (never a scheme substitution
    picked silently): MINISGL_ZAYA_W8A16=1 -> fp8 weights dequantized in-register to bf16 acts
    (W8A16, quality/latency trade); MINISGL_ZAYA_OLDMOE=1 -> legacy fp8->bf16 dequant->fused Triton
    A/B reference. W8A16 used to be the default (an env silently swapping the declared act scheme) —
    that was a bug; it is now an explicit opt-in."""

    supports_ep = True
    needs_precomputed_route = True  # ZAYA top-1 + MOD route is computed model-side

    def __init__(self):
        self._oldmoe = os.environ.get("MINISGL_ZAYA_OLDMOE", "0") == "1"
        # W8A16 is now OPT-IN (=1); default is the checkpoint-declared native W8A8 fp8-act kernel.
        self._w8a16_fn = None
        if not self._oldmoe and os.environ.get("MINISGL_ZAYA_W8A16", "0") == "1":
            try:  # fail-safe: an env without the built extension falls back to native W8A8
                from fp8_wmma import fused_moe_w8a16

                self._w8a16_fn = fused_moe_w8a16
            except ImportError:
                self._w8a16_fn = None

    def create_experts(self, num_experts, out_features, in_features):
        return _GroupedFP8Experts(num_experts, out_features, in_features)

    def apply(self, w13, w2, hidden_states, *, router_logits, topk_weights, topk_ids,
              top_k, renormalize, activation, apply_router_weight_on_input,
              x_fp8=None, act_scales=None):
        assert topk_ids is not None, "fp8 experts use the precomputed-route path (ZAYA top-1 + MOD)"
        if self._w8a16_fn is not None:
            # W8A16 opt-in: dequant the fp8 weight tile to bf16 IN-REGISTER (no full-stack
            # materialize), routed experts only; bf16 acts. Uses the always-present op-layout buffers.
            # The kernel requires bf16 activations, so guard the cast (no-op when the model is already
            # bf16 — the norm for W8A16 checkpoints) instead of hardcoding an unconditional conversion.
            acts = hidden_states if hidden_states.dtype == torch.bfloat16 else hidden_states.to(torch.bfloat16)
            return self._w8a16_fn(
                acts,
                w13._w_op, w13._scales_op, w2._w_op, w2._scales_op,
                topk_weights, topk_ids.to(torch.int32),
            )
        if self._oldmoe:
            # A/B reference: legacy fp8->bf16-dequant->Triton path (weights kept in post_load).
            from minisgl.moe.fused import fused_experts_impl

            return fused_experts_impl(
                hidden_states, w13.dequant(hidden_states.dtype), w2.dequant(hidden_states.dtype),
                topk_weights, topk_ids,
                activation=activation, apply_router_weight_on_input=apply_router_weight_on_input,
            )
        if getattr(w13, "_w_rep", None) is not None:
            # Register-direct b128 (LDS-bypass) — the default native W8A8 path (~2x the LDS-staged
            # w8a8_moe at decode, bit-exact). post_load built _w_rep and dropped _w_op.
            return kernels.w8a8_moe_regdirect(
                hidden_states, w13._w_rep, w13._scales_op, w2._w_rep, w2._scales_op,
                top_k, topk_weights=topk_weights, topk_ids=topk_ids,
            )
        return kernels.w8a8_moe(
            hidden_states, w13._w_op, w13._scales_op, w2._w_op, w2._scales_op,
            None, top_k, renormalize, topk_weights=topk_weights, topk_ids=topk_ids,
        )

    def ep_local(self, w13, w2, g_hidden, local_weights, local_ids, *, top_k, renormalize,
                 activation="silu"):
        # Every kernel below (w8a16 / w8a8_moe / regdirect) has a silu-only tail.
        _check_activation("fp8 W8A8", activation, allowed=("silu",))
        if self._w8a16_fn is not None:
            # Kernel requires bf16 acts; guard the cast (no-op when already bf16) — see forward().
            acts = g_hidden if g_hidden.dtype == torch.bfloat16 else g_hidden.to(torch.bfloat16)
            return self._w8a16_fn(
                acts,
                w13._w_op, w13._scales_op, w2._w_op, w2._scales_op, local_weights, local_ids,
            )
        if getattr(w13, "_w_rep", None) is not None:
            return kernels.w8a8_moe_regdirect(
                g_hidden, w13._w_rep, w13._scales_op, w2._w_rep, w2._scales_op,
                top_k, topk_weights=local_weights, topk_ids=local_ids,
            )
        return kernels.w8a8_moe(
            g_hidden, w13._w_op, w13._scales_op, w2._w_op, w2._scales_op,
            None, top_k, renormalize, topk_weights=local_weights, topk_ids=local_ids,
        )


def create_moe_quant_method(
    quant: "QuantConfig | None", *, fp8_experts: bool = False
) -> MoEQuantMethod:
    """Pick the MoE-expert quant method from the checkpoint's DECLARED scheme (the analogue of
    quant.method.create_linear_method for dense linears). Route:
      * fp8 W8A8 (compressed-tensors float-quantized 8-bit, or the explicit `fp8_experts` signal) ->
        native w8a8_moe (per-token fp8 acts; W8A16 is an env opt-in, never the default);
      * RXF -> rxf_moe;
      * MXFP4 (compressed-tensors float-quantized 4-bit, OCP E2M1) -> the shared w4a8_moe kernel with
        weight_is_e2m1=True (same kernel, e2m1 decode + E8M0->fp16 group scale);
      * int4 AWQ / GPTQ / compressed-tensors int4 -> the shared w4a8_moe kernel;
      * no quant -> the unquantized fused backend.
    Selection is purely config-driven: no model-name branch, and no env that substitutes a different
    scheme than the checkpoint declares."""
    if fp8_experts or (quant is not None and quant.is_fp8_w8a8):
        return _FP8MoEMethod()
    if quant is None:
        return _UnquantizedMoEMethod()
    if quant.is_nvfp4:
        # NVFP4 -> the MXFP4 e2m1 kernel at group-16 (weights stay 4-bit). The loader keeps the
        # checkpoint's TWO scale levels — an e4m3 block scale + a per-OUTPUT-CHANNEL f32 global — and
        # that shape is what lets the gate/up merge and the expert stack carry the global with no
        # special case. See _GroupedNvFp4Experts / nvfp4.nvfp4_leaf_scales.
        return _NvFp4MoEMethod(quant)
    if quant.is_rxf:
        return _RXFMoEMethod(quant)
    if quant.weight_is_e2m1:
        return _MxFp4MoEMethod(quant)
    if quant.is_int4:
        return _W4A8MoEMethod(quant)
    raise AssertionError(
        f"MoE: unsupported declared quant scheme (method={quant.method}, bits={quant.bits}, "
        f"weight_type={quant.weight_type})"
    )


class MoELayer(BaseOP):
    # ── WEIGHT-OFFLOAD INTERPOSITION POINT ──────────────────────────────────────────────────────
    # A `weights/moe_interpose.MoEWeightSeam`, or None. Declared as a CLASS attribute so that a
    # serve with no offload pays exactly one `is not None` test per MoE forward and stores nothing
    # per layer — and, more importantly, so it stays out of `vars(self)`, which is what
    # `BaseOP.state_dict` / `load_state_dict` / `post_load` and the granule walk all iterate.
    # `moe_interpose.attach_seams` sets it on the INSTANCE (never on the class) after `post_load()`,
    # and freezes it before the first graph capture.
    _weight_offload = None

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        quant: "QuantConfig | None" = None,
        fp8_experts: bool = False,
        force_no_ep: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        # Expert parallelism: each DP replica OWNS only experts [dp_rank*E/dp : (dp_rank+1)*E/dp], so
        # the per-expert weight buffers are sized to the LOCAL count (E/dp); the dispatch/combine
        # collective in forward() reconstructs the full result. enable_ep is the process-global toggle
        # (set by the Engine before model build); off => full replicated count. Every expert buffer
        # (fp8 AND quantized: _w_op/_scales_op/_zeros_op / weight_packed) puts E on dim 0, and every
        # grouped kernel reads E = w13.shape[0], so a shard is a pure dim-0 slice — EP is quant-agnostic
        # for the W4A8 (GPTQ/AWQ) and W4A16 (compressed-tensors) op layouts (shared w4a8_moe kernel) and
        # the fp8 W8A8/W8A16 layouts. RXF is excluded (no precomputed-topk path, which EP requires).
        # `force_no_ep` keeps a specific layer replicated even under EP — used for the tiny MTP draft
        # head, whose EP-sharding would make spec-decode propose issue data-dependent collectives.
        # Config-driven expert quant method (mirrors create_linear_method for the dense linears): it
        # owns the container class + the grouped kernel, and declares whether the scheme can run EP.
        self._moe_method = create_moe_quant_method(quant, fp8_experts=fp8_experts)
        self.enable_ep = (
            is_ep_enabled() and self._moe_method.supports_ep and not force_no_ep
        )
        # Expert-sharding group: DP replicas (DP+EP) OR the TP ranks (EP-over-TP, vllm-style TP+EP).
        # get_ep_size/get_ep_rank abstract the two; ep_rank indexes this rank's expert shard (used for
        # the _ep_dispatch output slice). ep_dp_rank kept as an alias for the existing dispatch code.
        self.ep_size = get_ep_size()
        self.ep_rank = get_ep_rank()
        self.ep_dp_rank = self.ep_rank
        self.ep_dp_size = self.ep_size
        if self.enable_ep:
            assert num_experts % self.ep_size == 0, (
                f"EP needs num_experts ({num_experts}) divisible by ep_size ({self.ep_size})"
            )
            self.local_num_experts = num_experts // self.ep_size
            self.local_expert_offset = self.ep_rank * self.local_num_experts
        else:
            self.local_num_experts = num_experts
            self.local_expert_offset = 0
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.quant = quant
        self.fp8_experts = fp8_experts
        # Under EP each rank holds FULL experts (its expert subset); only plain TP tensor-splits each
        # expert's FFN across the TP ranks. EP-over-TP therefore keeps the full intermediate size.
        intermediate_size_per_partition = (
            intermediate_size if self.enable_ep else div_even(intermediate_size, tp_size))
        # The method allocates the per-expert container for each GEMM (config-driven: fp8 F8_E4M3,
        # int4 W4A8/W4A16 grouped, RXF NL, or a plain stacked bf16/fp16 tensor). EP: size to the LOCAL
        # expert shard (E/dp) so this replica loads + runs only its experts; the streaming loader skips
        # non-local ids (weight.py mirror). enable_ep off (unquantized, RXF, force_no_ep) =>
        # local_num_experts == num_experts (full replicated). Both GEMMs, w13 = gate|up (2*inter), w2 =
        # down (hidden), share the method and the silu_and_mul convention.
        self.gate_up_proj = self._moe_method.create_experts(
            self.local_num_experts, 2 * intermediate_size_per_partition, hidden_size
        )
        self.down_proj = self._moe_method.create_experts(
            self.local_num_experts, hidden_size, intermediate_size_per_partition
        )

    # ── granule descriptor ──────────────────────────────────────────────────────────────────────
    # The MoE seam for weight residency. `forward` reads `w13, w2 = self.gate_up_proj, self.down_proj`
    # and hands them to `method.apply`/`method.ep_local` AS ARGUMENTS — no format reads weights off
    # `self` — so a residency layer that knows what one expert IS needs nothing format-specific here.
    # Seven model families (qwen2_moe, qwen3_5_moe, gemma4, glm4_moe_lite, models/utils, laguna, zaya)
    # share this one MoELayer, so what lands here is general by construction.

    def expert_containers(self) -> "dict[str, object]":
        """The two per-expert weight containers of this layer, keyed by attribute name."""
        return {"gate_up_proj": self.gate_up_proj, "down_proj": self.down_proj}

    def post_load(self) -> None:
        """Finalize both containers, then prove they made the SAME decode decisions.

        UNCONDITIONAL, and that is the whole point of overriding here. The w13/w2 decode-policy
        comparison already existed — inside `granule_specs()` via `assert_granule_pair_consistent` —
        but `granule_specs()` is called by `weights/moe_interpose.attach_seams` and by nothing else,
        so it ran only on a serve that offloads weights. On every other serve the two containers of a
        layer could resolve the compressed-tensors sign convention in opposite directions and nothing
        looked: `_GroupedCompressedTensorsExperts.post_load` samples w13's nibble histogram and w2's
        independently, and the two stacks are differently shaped (w13 is 2*inter x hidden, w2 is
        hidden x inter, and under TP they are split on different axes), so they are genuinely two
        samples of two different tensors. A disagreement decodes one GEMM as `q + 8` and the other as
        two's-complement — every weight of that GEMM off by 8 quanta, right shapes, no kernel fault,
        plausible text.

        Cheap: it reads the declared `_granule_policy` paths off each container (two `getattr`
        chains), with no tensor walk, no `torch.equal` and no synchronize — see
        `granule.decode_policy` for why it deliberately does not go through spec derivation.
        """
        super().post_load()
        assert_decode_policy_agrees(
            {name: decode_policy(c) for name, c in self.expert_containers().items()},
            where=f"{type(self).__name__} w13/w2",
        )

    def granule_specs(self, **kw) -> "dict[str, GranuleSpec]":
        """Derive both GEMMs' granule descriptors. Call AFTER `post_load()` — before it, the
        quantized containers still hold checkpoint buffers that `post_load` deletes.

        `local_num_experts` is passed explicitly rather than inferred: under EP each rank holds a
        SHARD, so the container's dim 0 is E/ep_size and an inferred count would be right by accident
        on one topology and silently wrong on another.

        BOOT-TIME ONLY, and never from `forward`: derivation walks attributes, launches the
        declaration check's `torch.equal` and synchronizes, so it is illegal under graph capture and
        a per-step stall in eager decode. `derive_granule_spec` refuses under capture; this is the
        note for the eager case.
        """
        specs = {
            name: spec_for_container(c, self.local_num_experts, **kw)
            for name, c in self.expert_containers().items()
        }
        assert_granule_pair_consistent(
            specs["gate_up_proj"], specs["down_proj"], where=type(self).__name__
        )
        return specs

    def co_demanded_granule_bytes(self, **kw) -> int:
        """Bytes moved by routing ONE expert in this layer: its slice of w13 AND of w2.

        This is the unit the route oracle counts and the placement plan prices — the two GEMMs are
        always demanded together, so they are one granule for accounting even though each component
        still gets its own contiguous slab (see `plan_component_major`).
        """
        return sum(s.granule_bytes for s in self.granule_specs(**kw).values())

    def offload_refusal(self) -> "str | None":
        """Why this layer's experts cannot be host-resident, or None. First refusal wins.

        Surfaced at the LAYER because that is the object a placement pass iterates; the refusal
        itself lives on the container (`_GroupedFP8Experts` under `MINISGL_ZAYA_OLDMOE=1` /
        `MINISGL_ZAYA_W8A16=1`, whose forwards read EVERY expert). `derive_granule_spec` only
        consults it when asked (`for_offload=True`), and `granule_specs` deliberately does not ask —
        a spec is also derived for pure sizing, which must work for a layer that will never move. So
        a planner has to call this before it places a layer on host, or it will happily place one
        whose forward streams the whole stack per step.

        Rank-uniform: the refusal is a function of WHAT `post_load` BUILT (snapshotted into
        `_GroupedFP8Experts._whole_stack_knob` at the point the knob is actually read), and every
        rank runs the same `post_load` under the same launch environment. Deliberately NOT a live
        `os.environ` read here: the environment can move between build and placement, and the silent
        direction of that disagreement is a layer built under the knob being reported offloadable,
        then streaming the whole stack per step. A per-rank knob would place a layer on host on one
        rank and on device on the other — different budgets, different placement, no collective to
        catch it — so if that ever becomes settable per rank it must be agreed, not read.
        """
        for name, c in self.expert_containers().items():
            why = granule_offload_refusal(c)
            if why is not None:
                return f"{name}: {why}"
        return None

    def _ep_route(
        self,
        router_logits: "torch.Tensor | None",
        topk_weights: "torch.Tensor | None",
        topk_ids: "torch.Tensor | None",
    ):
        """Return (topk_weights f32, topk_ids i32) for the EP dispatch. EP must all_gather a route, so
        it can't defer to the kernel's fused softmax+topk — precompute it here. A model that already
        provides a route (GLM/DeepSeek noaux_tc) passes it through unchanged; renormalize must happen
        over ALL top_k here, BEFORE the per-rank local-expert masking in _ep_dispatch.

        NOT bit-identical to the non-EP route, and it must NOT be used to PREDICT which experts the
        kernel will read. This used to claim it matched "the kernel's own torch route (kernels.py
        w4a8_moe._route)". That function no longer exists — the served route is
        `moe_hip.moe_route_align`, which does softmax + top-k INSIDE the kernel — and the two break
        EXACT TIES at the k-th boundary opposite ways. Ties are common, not exotic: the gate logits are
        bf16 (8 mantissa bits) spread over hundreds of experts. MEASURED on Qwen3.8-Flash-Next
        (E=512, top_k=10), layer 0 of an 8-token prefill: experts 324 and 366 both scored exactly
        -5.09375 at ranks 9 and 10, straddling the cut; the kernel kept the LOWER index (324) and
        `torch.topk` kept the higher (366). Two of the eight MoE calls in that one forward diverged by
        exactly one expert this way.

        This is self-consistent for EP — `_ep_dispatch` hands these ids to the kernel, so an EP serve
        uses this route end to end — but it does mean an EP and a non-EP serve of the same model can
        select a different expert on a tie row. Anything that needs the set of experts the GEMM will
        actually dereference (a routed weight gather, an offload prefetch) must call
        `quant.kernels._route_align` and read its `topk_ids`/`expert_ids`, not re-derive it here."""
        if topk_ids is not None:
            assert topk_weights is not None, "topk_weights required when topk_ids is given"
            return topk_weights.to(torch.float32), topk_ids.to(torch.int32)
        assert router_logits is not None, "EP route needs router_logits or a precomputed topk"
        probs = torch.softmax(router_logits.float(), dim=-1)
        tw, ti = torch.topk(probs, self.top_k, dim=-1)
        if self.renormalize:
            tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
        return tw.contiguous(), ti.to(torch.int32).contiguous()

    def _ep_dispatch(self, hidden_states, topk_weights, topk_ids, local_kernel):
        """Expert-parallel dispatch/combine, quant-agnostic. all_gather every replica's token rows +
        route so each rank sees ALL tokens -> remap global expert id to local (mask non-local by zeroing
        the route weight) -> run the format-specific grouped kernel over THIS rank's expert shard ->
        all_reduce(SUM) (each token's k experts each live on exactly one rank, so the sum reconstructs
        the full top-k result) -> slice our rows. `local_kernel(g_hidden, local_weights, local_ids_i32)
        -> (dp*N, H)` is the only per-format part (fp8 W8A16/W8A8, W4A8/W4A16 w4a8_moe). Replaces the TP
        all_reduce epilogue. See MoELayer.forward for the closures; N self-coordination is Part A."""
        ep = get_global_ctx().ep
        assert ep is not None, "EP enabled but ctx.ep group not built"
        real_n = hidden_states.shape[0]
        ep_w = topk_weights.contiguous()
        ep_i = topk_ids.to(torch.int32).contiguous()
        hs = hidden_states
        if is_ep_over_tp():
            # EP-over-TP: the TP ranks share IDENTICAL tokens (replicated hidden after the attention
            # all-reduce) and an identical route (replicated gate), so there is NO all_gather — each rank
            # already sees every token. Run its expert shard (non-local masked to weight 0) and all_reduce
            # the partials (each token's k experts each live on exactly one rank → the sum is exact). No
            # pad/self-coordination (all ranks have the same real_n) and no output slice (all rows are ours).
            lo, hi = self.local_expert_offset, self.local_expert_offset + self.local_num_experts
            is_local = (ep_i >= lo) & (ep_i < hi)
            local_ids = torch.where(is_local, ep_i - lo, torch.zeros_like(ep_i))
            local_weights = torch.where(is_local, ep_w, torch.zeros_like(ep_w))
            partial = local_kernel(hs, local_weights, local_ids.to(torch.int32))
            return ep.all_reduce(partial)
        # Common token count N every replica pads to before the all_gather (RCCL needs equal shapes):
        #  1. graph decode: pre-padded to a captured bs -> already equal, no host sync.
        #  2. eager with a scheduler pre-agreement (ep_loop prefill): ep.pad_tokens.
        #  3. eager, no pre-agreement (spec-verify / prefix-seed): SELF-COORDINATE via one tiny
        #     all_gather of real_n. This is what lets EP survive spec-decode (Part A).
        if torch.cuda.is_current_stream_capturing():
            common_n = real_n
        elif ep.pad_tokens is not None:
            common_n = max(ep.pad_tokens, real_n)
        else:
            counts = ep.all_gather(
                torch.tensor([real_n], device=hs.device, dtype=torch.int64)
            )
            # Inherent host sync: `common_n` sizes the `pad` for the torch.cat below, so the padded
            # tensor's shape must be known host-side — it cannot be removed without a device sync.
            common_n = int(counts.max().item())
        if common_n > real_n:
            pad = common_n - real_n
            hs = torch.cat([hs, hs.new_zeros(pad, hs.shape[1])], dim=0)
            ep_w = torch.cat([ep_w, ep_w.new_zeros(pad, ep_w.shape[1])], dim=0)
            ep_i = torch.cat([ep_i, ep_i.new_zeros(pad, ep_i.shape[1])], dim=0)
        N = hs.shape[0]  # common token count, identical on every rank
        # FUSED EP all-gather: hs(bf16)/weights(f32)/ids(i32) are TINY latency-bound decode collectives
        # (~27us each, tensor-size-independent). Byte-pack all three into ONE all_gather, then split —
        # halves the per-MoE-layer collective count (3 gathers -> 1). Bit-identical (pure reinterpret +
        # a device-local unpack copy). Shapes are static under graph capture, so it captures cleanly.
        H = hs.shape[1]
        hs_b = hs.contiguous().view(torch.uint8)        # (N, H*2)
        w_b = ep_w.contiguous().view(torch.uint8)       # (N, top_k*4)
        i_b = ep_i.contiguous().view(torch.uint8)        # (N, top_k*4)
        g = ep.all_gather(torch.cat([hs_b, w_b, i_b], dim=1))  # (dp*N, H*2 + top_k*8)
        o1, o2 = H * 2, H * 2 + ep_w.shape[1] * 4
        g_hidden = g[:, :o1].contiguous().view(hs.dtype)       # (dp*N, H)
        g_weights = g[:, o1:o2].contiguous().view(ep_w.dtype)  # (dp*N, top_k)
        g_ids = g[:, o2:].contiguous().view(ep_i.dtype)        # (dp*N, top_k)
        lo, hi = self.local_expert_offset, self.local_expert_offset + self.local_num_experts
        is_local = (g_ids >= lo) & (g_ids < hi)
        local_ids = torch.where(is_local, g_ids - lo, torch.zeros_like(g_ids))
        local_weights = torch.where(is_local, g_weights, torch.zeros_like(g_weights))
        partial = local_kernel(g_hidden, local_weights, local_ids.to(torch.int32))  # (dp*N, H)
        partial = ep.all_reduce(partial)  # SUM across ranks -> full result for every token
        return partial[self.ep_dp_rank * N : self.ep_dp_rank * N + real_n]  # drop padding

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        *,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        reduce: bool = True,
        x_fp8: torch.Tensor | None = None,
        act_scales: torch.Tensor | None = None,
    ):
        # Either pass raw `router_logits` (fused softmax+topk inside the kernel) OR a precomputed
        # `topk_weights`/`topk_ids` route (GLM/DeepSeek noaux_tc, ZAYA top-1 computed in the model).
        # The expert scheme + kernel are owned by the config-driven `self._moe_method`; MoELayer only
        # owns routing + the EP dispatch/combine + the TP all-reduce.
        method = self._moe_method
        w13, w2 = self.gate_up_proj, self.down_proj
        # WEIGHT-OFFLOAD SEAM. Every scheme's `apply`/`ep_local` takes the pair AS ARGUMENTS and no
        # format reads its weights off `self`, so substituting the containers HERE is format-agnostic
        # by construction and covers all seven model families that share this MoELayer.
        #
        # Under the layer-granular plan the substitution already happened at bind time (a layer is
        # entirely device- or entirely host-resident, decided once at boot and frozen), so this is an
        # identity plus two `is` checks — no allocation, no launch, no host sync, nothing that could
        # differ between an eager warmup and a captured replay. The checks are load-bearing: the MTP
        # draft head builds its own MoELayer, so a seam bound to the wrong layer would read
        # identically-shaped containers and produce plausible logits with no crash.
        if self._weight_offload is not None:
            # THIRD TIER: CPU-COMPUTE. The experts of this layer are executed by AVX-512 cores on
            # the host, where the weights already are, instead of being streamed to the GPU over
            # PCIe. Only the activation crosses the bus (~5 KB down, ~10 KB up, against 30.7 MB of
            # weights), which is the whole thesis — see weights/cpu_tier.py.
            #
            # Tested BEFORE resolve(), not after: a CPU-tier seam's containers are PAGEABLE host
            # memory with no device mapping, so `resolve()` deliberately refuses on one rather than
            # handing the grouped kernel a pointer the device cannot dereference.
            #
            # The route has to be materialised here — the non-EP path normally defers softmax+topk
            # into the kernel, and there is no kernel on this path. This is BLOCK mode: submit and
            # join in one call, GPU idle for the duration. It is still 3.6-4.3x cheaper per layer
            # than streaming the same layer (0.583 ms at the measured 6-thread knee vs 2.12-2.49 ms
            # over card 1's Gen4 x8 link). The concurrent composition — submit, run GPU work, join
            # — is `seam.cpu_submit`/`cpu_join` and needs SPLIT mode to have any GPU work to hide
            # behind, because the residual stream is sequential.
            if self._weight_offload.computes_on_cpu:
                assert not self.enable_ep, (
                    "the CPU-compute tier is not wired through the EP all_gather: _ep_dispatch "
                    "re-orders rows across ranks, so a CPU partial computed from pre-gather rows "
                    "would be added to the wrong tokens. Plan CPU layers only on an EP-free rank, "
                    "or pack the CPU partial into the gather first."
                )
                tw, ti = self._ep_route(router_logits, topk_weights, topk_ids)
                out = self._weight_offload.cpu_forward(hidden_states, tw, ti)
                if self.tp_size > 1 and reduce:
                    out = self._comm.all_reduce(out)
                return out
            w13, w2 = self._weight_offload.resolve(w13, w2)
        # PRODUCER-SIDE act quant: the pair describes `hidden_states` ROW FOR ROW. Under EP the rows
        # are all_gather'd and re-ordered before the local kernel sees them (see _ep_dispatch), so a
        # pair that was not gathered alongside them would be silently mismatched -- every token would
        # be scaled by another token's amax. That is a wrong-numbers bug with no crash, so refuse it
        # here rather than dropping it quietly. Packing the pair INTO the byte-packed gather is the
        # real fix and is tractable (it is +K bytes and +4 bytes per row on an existing cat), but it
        # is unmeasured, so the call sites simply do not build a pair when EP is on.
        if x_fp8 is not None and self.enable_ep:
            raise NotImplementedError(
                "MoE producer-side act-quant is not wired through the EP all_gather: the gather "
                "re-orders rows, so (x_fp8, act_scales) must be packed into it or not supplied. "
                "Pass x_fp8=None when enable_ep is set."
            )
        if x_fp8 is not None and not method.supports_producer_actquant:
            raise NotImplementedError(
                f"{type(method).__name__} cannot consume a producer-quantized activation pair; "
                "passing one would silently drop it and make an A/B read as if fusion applied."
            )
        if self.enable_ep:
            # Expert-parallel dispatch/combine (only schemes with method.supports_ep reach here). EP
            # can't defer routing to the kernel (it must all_gather a route), so precompute topk here —
            # renormalized over ALL top_k BEFORE the per-rank local-expert masking, then passed
            # precomputed. fp8 already REQUIRES the model-side route; W4A8/W4A16 derive it via _ep_route.
            # The all_gather/mask/all_reduce scaffold + N self-coordination live in _ep_dispatch; only
            # the local-shard kernel (method.ep_local) is scheme-specific.
            if method.needs_precomputed_route:
                assert topk_ids is not None, "EP fp8 experts need the model-side precomputed route"
                ep_w, ep_i = topk_weights, topk_ids
            else:
                ep_w, ep_i = self._ep_route(router_logits, topk_weights, topk_ids)
            final_hidden_states = self._ep_dispatch(
                hidden_states, ep_w, ep_i,
                lambda gh, lw, li: method.ep_local(
                    w13, w2, gh, lw, li, top_k=self.top_k, renormalize=self.renormalize,
                    activation=self.activation,
                ),
            )
        else:
            final_hidden_states = method.apply(
                w13, w2, hidden_states,
                router_logits=router_logits, topk_weights=topk_weights, topk_ids=topk_ids,
                top_k=self.top_k, renormalize=self.renormalize,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                x_fp8=x_fp8, act_scales=act_scales,
            )
        # EP already all_reduce'd over the dp/EP group (which subsumes any per-replica TP reduce —
        # ZAYA is tp_size=1 anyway), so skip the TP epilogue when the EP path ran. reduce=False also
        # skips it so the caller can fuse this partial with another row-parallel partial (shared expert)
        # and all_reduce once — the row-parallel down-proj output is a per-rank partial either way.
        if self.tp_size > 1 and not self.enable_ep and reduce:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states


# ── [ADAPTIVE-K] per-token expert-count gating for spec-verify-sized batches ─────────────────────
# Verify cost is superlinear in M because the UNION of top-k experts over the M rows grows fast
# (measured `distinct(qlen)` = 6.94x at qlen 16), and the MoE block is 19% of our verify time. This
# keeps each row's leading experts until their cumulative combine weight reaches tau and drops the
# rest, shrinking that union. Ported from lucebox's [TAG_MMID_ADAPTIVE_K].
#
# NOT LOSSLESS. It changes the TARGET's own output, so accepted tokens differ from true greedy
# target output — unlike the adaptive verify WIDTH, which only truncates drafts. It is therefore a
# quality/throughput trade and defaults OFF (tau=0), exactly as lucebox ship it. Gate any rollout on
# output quality, not just tok/s.
#
# No kernel change: dropped slots get weight 0 AND are pointed at a slot the row already keeps, so
# they contribute exactly zero and add no expert to the union. A -1 sentinel would be cleaner (one
# fewer padded row per drop) but moe_align does `atomicAdd(&cnt[topk_ids[t]], 1)` with no bounds
# check, so a negative id is an out-of-bounds shared-memory write — that needs a 4-line guard in the
# canonical kernel first.
_ADAPTIVE_K_TAU = float(os.environ.get("MINISGL_MOE_ADAPTIVE_K_TAU") or "0")


def adaptive_k_gate(
    topk_weights: torch.Tensor,  # (M, top_k) f32, already normalized
    topk_ids: torch.Tensor,      # (M, top_k) int
    tau: float = _ADAPTIVE_K_TAU,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop each row's trailing experts once cumulative combine weight reaches `tau`; renormalize.

    Verify-sized batches only (2..16 rows): at M=1 there is no union to shrink, and past ~16 the
    expert set is already saturated so gating buys nothing and only costs accuracy.
    """
    if tau <= 0.0 or tau >= 1.0:
        return topk_weights, topk_ids
    M = topk_weights.shape[0]
    if M < 2 or M > 16:
        return topk_weights, topk_ids
    order = topk_weights.argsort(dim=-1, descending=True)
    w = topk_weights.gather(-1, order)
    ids = topk_ids.gather(-1, order)
    # Keep slot i iff the mass BEFORE it has not yet reached tau — so the slot that crosses tau is
    # kept and slot 0 is always kept (cum-before = 0), i.e. a row can never end up with no expert.
    keep = (w.cumsum(-1) - w) < tau
    w = w * keep
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
    ids = torch.where(keep, ids, ids[:, :1].expand_as(ids))
    return w.contiguous(), ids.to(torch.int32).contiguous()
