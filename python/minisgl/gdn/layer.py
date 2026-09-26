"""Phase 3b-2: a clean minisgl `QwenGatedDeltaNet` linear-attention layer.

Reimplements vLLM's `QwenGatedDeltaNetAttention` (Qwen3.5 / Qwen3-Next GDN) forward
COMPUTE, stripped of all vLLM coupling (CustomOp / forward_context / distributed /
MergedColumnParallelLinear / the torch.ops dispatch). The compute now runs entirely on
native HIP kernels (`torch.ops.gdn_hip.*`) — conv, gated-delta-rule prefill/decode, and
the gated RMSNorm — with no Triton dependency.

Scope (3b): NUMERICS of one layer, in isolation.
  * TP=1, unquantized bf16 projections (the 35B keeps GDN projections in bf16; only
    the routed MoE experts are W4A8).
  * Non-interleaved qkvz/ba layout (`gqa_interleaved_layout=False`, Qwen3.5).
  * No speculative decode / MTP.
  * State + indices are passed EXPLICITLY (not pulled from a ForwardContext), so the
    layer is testable standalone (3b-3) and wired to `GDNStateCache` later in 3c/3d.

Faithful to the reference's `_forward_core` (prefill = causal_conv1d_fn ->
fused_post_conv_prep -> chunk_gated_delta_rule; decode = causal_conv1d_update ->
rearrange -> fused_sigmoid_gating_delta_rule_update) and `_output_projection`
(gated RMSNorm(core, z) -> out_proj).

conv_state convention: this layer expects the dim-first / "DS" layout
`(num_slots, conv_dim, conv_kernel-1)` — exactly what `GDNStateCache` allocates. On a
backend where `is_conv_state_dim_first()` is False the caller must transpose; that is
re-verified against the live RDNA4 path in 3b-3.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch import nn

from minisgl._hip_engage import engaged

# Fuse the gated RMSNorm into gdn_decode's epilogue (gdn_hip.gdn_decode_gated) — one kernel instead of
# gdn_decode + rmsnorm_gated (−1 launch + core HBM round-trip). Bit-exact vs the two-kernel path. Default
# on; falls back automatically if the loaded gdn_hip .so predates the op. Set MINISGL_GDN_FUSED_NORM=0 to
# force the unfused path.
_GDN_FUSED_NORM = os.environ.get("MINISGL_GDN_FUSED_NORM", "1") == "1"
# Fuse the whole DECODE chain — causal_conv1d_update + gdn_decode + gated RMSNorm — into ONE kernel
# (gdn_hip.gdn_decode_conv_gated): reads mixed_qkv directly (no python q/k/v split, no conv_out HBM
# round-trip), 2 launches -> 1. Bit-exact vs the two-kernel path; falls back automatically if the loaded
# gdn_hip .so predates the op. Set MINISGL_GDN_FUSED_CONV=0 to force the separate path.
_GDN_FUSED_CONV = os.environ.get("MINISGL_GDN_FUSED_CONV", "1") == "1"

# Route the UNQUANTIZED GDN projections (in_proj_qkvz / in_proj_ba / out_proj) through the shared
# decode GEMV at small M instead of nn.Linear -> F.linear -> rocBLAS.
#
# WHY: these are the largest kernel group in bs=1 decode. Profiled on the 35B AWQ (TP=2, 30 GDN
# layers): rocBLAS MT128x128x32 3.185 ms/step (60 calls, 53.1 us) + MT16x16x32 0.422 ms/step
# (30 calls) = 3.61 ms/step = 24% of the whole 14.75 ms kernel step. Measured per-shape at M=1:
#     in_proj_qkvz 2048x6144   F.linear 70.6us/356 GB/s  ->  gemv 44.8us/561 GB/s (83% of peak)
#     out_proj     2048x2048   F.linear 36.0us/233       ->  gemv 17.3us/484
#     x30 layers               F.linear 3.542 ms/step    ->  gemv 2.013 ms/step   (1.76x)
#
# THRESHOLD, and why it is small: the GEMV is M=1-shaped. At M=8 it LOSES to rocBLAS (4.42 vs
# 3.23 ms/step) and it cannot run at all above the core's MMAX cap of 16. Cutting its VALU 4x via
# packed dot2 did not move M=8, so M>=4 is not VALU-bound there — above the crossover the work
# belongs on WMMA, which rocBLAS already does well (403 GB/s at M=8).
#
# CORRECTNESS: the GEMV is M-invariant (row 0 at M=1 vs M=2/8/16 is bit-identical, max|d|=0.0), but
# it is NOT bit-identical to F.linear (rel 1.7e-5..8.2e-4 on these shapes). F.linear itself measured
# M-invariant here, so a threshold INTRODUCES a crossing that does not exist today: a token computed
# at M<=MAXM and the same token computed at M>MAXM would differ. That is the hazard layers/minv.py
# exists to prevent (ZAYA GSM8K 45->25).
#
# UNCONDITIONAL, MAXM=16 — no env gate. The hazard above is a THRESHOLD hazard, not a kernel
# hazard: it appears only where two different kernels serve the same shape at different M. MAXM=16
# is the M ceiling this GEMV accepts (torch_binding checks M<=16), so putting the threshold at the
# ceiling lands BOTH ordinary decode (M=1..max_running) and spec-decode VERIFY (M=K+1) on the SAME
# kernel — which is what makes verify match sequential decode. At the old MAXM=2 they sat on
# opposite sides and spec losslessness was silently broken. Above 16 the fallback is F.linear,
# which is not M-invariant at all, so a HIGHER MAXM strictly widens the M-invariant region.
_GDN_PROJ_GEMV_MAXM = 16

if TYPE_CHECKING:
    from minisgl.quant.method import LinearMethod

# The GDN compute kernels (conv, gated-delta-rule prefill/decode, the gated RMSNorm) are now native
# HIP (torch.ops.gdn_hip.*), AOT-compiled, no Triton JIT. Importing this layer no longer drags in the
# vendored Triton tree at all.


class _MethodLinear(nn.Module):
    """A quantized linear that lives INSIDE the GDN nn.Module (so the existing nn.Module state
    bridge — GDNLinearAttn delegating to the wrapped module with assign=True — loads it), but
    delegates weight layout + the GEMM to the GENERIC minisgl `LinearMethod` (the same
    `create_linear_method` the dense/MoE linears use: W4A8 for AWQ/GPTQ/compressed-tensors int4,
    any future scheme for free). It registers the method's CHECKPOINT buffers
    (weight_packed/scale[/zero_point]) as nn buffers so load reaches them; `process_quant()`
    converts them to op layout (post-load); forward runs the method's quantized matmul.

    No dequant-to-bf16 at load: the int4 packs + scales load as-is and the kernel dequants
    in-register during the GEMM (int4 weight x fp8 activation, exactly like the dense linears)."""

    def __init__(self, in_features: int, out_features: int, method: "LinearMethod", *, device) -> None:
        super().__init__()
        self._method = method

        class _Holder:  # create_weights sets plain attrs; we lift them to nn buffers
            pass

        holder = _Holder()
        with torch.device(device):  # build on meta -> no real allocation until load(assign=True)
            method.create_weights(holder, out_features, in_features)
        for name, t in vars(holder).items():
            self.register_buffer(name, t)

    def process_quant(self) -> None:
        proc = getattr(self._method, "process_weights_after_load", None)
        if proc is not None:
            proc(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._method.apply(self, x, None)


def _make_proj(
    in_features: int, out_features: int, method: "LinearMethod | None", dtype, device
) -> nn.Module:
    """Build one GDN projection. Unquantized (or no method) -> a plain bf16/fp16 `nn.Linear` (the
    35B path, and the CAM-training path which needs a differentiable `F.linear` on `.weight`).
    A quantized method -> `_MethodLinear` (int4 packs + the shared quant kernel)."""
    from minisgl.quant.method import UnquantizedLinearMethod

    if method is None or isinstance(method, UnquantizedLinearMethod):
        return _GemvLinear(in_features, out_features, bias=False, dtype=dtype, device=device)
    return _MethodLinear(in_features, out_features, method, device=device)


class _GemvLinear(nn.Linear):
    """nn.Linear that routes SMALL-M forwards through the shared decode GEMV
    (fp8_wmma.dense_bf16_gemv = gemv_decode_core<Bf16GemvLoader>) and everything else through the
    normal nn.Linear path — see the note on _GDN_PROJ_GEMV_MAXM for the measured crossover and why
    the threshold sits at the GEMV's M ceiling rather than lower.

    SUBCLASSES nn.Linear rather than wrapping one: a wrapper renames the parameter to
    `<proj>.lin.weight` and the checkpoint loader (models/qwen3_5.py:248 pops by the module's own
    state_dict names) then dies with KeyError on `...in_proj_qkvz.lin.weight`. Subclassing keeps
    `weight` a direct Parameter, so the state_dict key, the TP sharding, and the differentiable
    F.linear the CAM-training path needs are all unchanged.
    """

    _gemv = None
    _probed = False

    @classmethod
    def _fn(cls):
        if not cls._probed:
            cls._probed = True
            try:
                from fp8_wmma import dense_bf16_gemv

                cls._gemv = dense_bf16_gemv
            except Exception:
                cls._gemv = None
        return cls._gemv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if (x.dim() == 2 and x.shape[0] <= _GDN_PROJ_GEMV_MAXM and self.bias is None
                and w.dtype in (torch.bfloat16, torch.float16) and x.dtype == w.dtype
                and w.shape[-1] % 8 == 0):
            fn = self._fn()
            if fn is not None:
                engaged("fp8_wmma.dense_bf16_gemv[gdn_proj]")
                return fn(x.contiguous(), w)
        return super().forward(x)   # F.linear -> rocBLAS (also the >MAXM / prefill path)


class GatedRMSNormWeight(nn.Module):
    """Pure parameter holder for the gated-RMSNorm weight + eps. The gated norm itself runs through
    torch.ops.gdn_hip.rmsnorm_gated (norm-before-gate + SiLU, both hardcoded in the HIP kernel), so
    this module's forward is never called — it exists only to own `.weight` (state_dict key
    `…linear_attn.norm.weight`) and `.eps`."""

    def __init__(self, hidden_size: int, eps: float, *, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))


class QwenGatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        *,
        tp_size: int = 1,
        eps: float = 1e-6,
        activation: str = "silu",
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
        qkvz_method: "LinearMethod | None" = None,
        ba_method: "LinearMethod | None" = None,
        out_proj_method: "LinearMethod | None" = None,
    ) -> None:
        super().__init__()
        # Tensor parallel: GDN is purely HEAD-parallel — each rank owns num_*_heads/tp_size key &
        # value heads, and every downstream dim (key_dim, value_dim, conv_dim, the qkvz/ba/conv
        # projections, the per-v-head A_log/dt_bias, the ssm/conv state) follows the head split. The
        # forward below uses these LOCAL counts unchanged, so it computes the rank's shard; the
        # bridge (GDNLinearAttn) all-reduces the row-parallel out_proj. head_*_dim and the gated
        # `norm` (over head_v_dim) are per-head and stay replicated. tp_size=1 -> unchanged (the
        # standalone Phase-3b numerics tests build with the default).
        assert num_k_heads % tp_size == 0 and num_v_heads % tp_size == 0, (
            f"GDN heads must divide tp_size={tp_size}: k={num_k_heads}, v={num_v_heads}"
        )
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads // tp_size
        self.num_v_heads = num_v_heads // tp_size
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation
        # OUTPUT-GATE activation id for the gated-RMSNorm epilogues (kernel `gate_act` policy:
        # 0=silu, 1=sigmoid). Distinct from the CONV activation, which stays SiLU. Was hardcoded
        # SiLU in every kernel while this parameter sat dead — the exact silent-quality-poison
        # shape: a sigmoid checkpoint would have produced grammatical-but-degenerate output through
        # all 48 GDN layers with no error anywhere. Unknown spellings raise HERE, at build time.
        _GATE_IDS = {"silu": 0, "swish": 0, "sigmoid": 1}
        if activation not in _GATE_IDS:
            raise ValueError(
                f"unsupported GDN output-gate activation {activation!r} "
                f"(kernels implement {sorted(_GATE_IDS)})"
            )
        self._gate_act = _GATE_IDS[activation]
        self._proj_dtype = dtype  # output dtype of the projections (== model dtype)

        self.key_dim = head_k_dim * self.num_k_heads
        self.value_dim = head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Projections (bias-free, like the reference). in_proj_qkvz packs q,k,v,z; in_proj_ba packs
        # b,a (per-v-head scalars). Each is either bf16 `nn.Linear` (35B / CAM) or a quantized
        # `_MethodLinear` — config-driven, decided by the caller via create_linear_method. in_proj_ba
        # (the tiny per-head beta/decay scalars) is kept full precision in every shipped GDN
        # checkpoint, so its method is normally unquantized; qkvz/out_proj follow the config.
        self.in_proj_qkvz = _make_proj(
            hidden_size, self.key_dim * 2 + self.value_dim * 2, qkvz_method, dtype, device
        )
        self.in_proj_ba = _make_proj(
            hidden_size, 2 * self.num_v_heads, ba_method, dtype, device
        )
        # conv1d weight mirrors the checkpoint shape (conv_dim, 1, kernel); the kernels
        # take a (conv_dim, kernel) view. Depthwise causal short-conv, bias-free here.
        self.conv1d_weight = nn.Parameter(
            torch.empty(self.conv_dim, 1, conv_kernel_size, dtype=dtype, device=device)
        )
        self.conv1d_bias: torch.Tensor | None = None

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, dtype=torch.float32, device=device))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32, device=device))

        self.norm = GatedRMSNormWeight(head_v_dim, eps=eps, device=device, dtype=dtype)
        self.out_proj = _make_proj(self.value_dim, hidden_size, out_proj_method, dtype, device)

        # fp32 caches of the two FROZEN weights the gdn_hip kernels consume at fp32: the depthwise
        # conv weight and the gated-RMSNorm weight. Both are model parameters — constant after the
        # state-dict load — so the previous per-call `.float()` was recomputing a constant, i.e. one
        # extra elementwise launch per layer per token (felt as launch overhead in the eager,
        # cudagraph-disabled decode loop). Built once lazily on first forward (which runs post-load,
        # before any graph capture) and reused. Assumes weights are frozen + not device-moved after
        # warmup, which is the inference contract here.
        self._conv_w_fp32: torch.Tensor | None = None
        self._norm_w_fp32: torch.Tensor | None = None

    # ---- input projections: qkvz and ba read the SAME hidden row ----
    def fuse_input_projections(self) -> None:
        """After load: run in_proj_qkvz + in_proj_ba as ONE decode GEMV when both are unquantised
        (layers/same_input_gemv.py). A checkpoint that quantises one and not the other keeps two."""
        from minisgl.layers.same_input_gemv import fuse_same_input

        self._qkvz_ba = fuse_same_input("gdn.in_proj_qkvz+ba", (self.in_proj_qkvz, self.in_proj_ba))

    def _in_proj(self, x: torch.Tensor):
        fused = getattr(self, "_qkvz_ba", None)
        if fused is not None:
            qkvz, ba = fused.forward(x, lambda h: [self.in_proj_qkvz(h), self.in_proj_ba(h)])
            return qkvz, ba
        return self.in_proj_qkvz(x), self.in_proj_ba(x)

    # ---- input split (non-interleaved Qwen3.5 layout) ----
    def _split_qkvz_ba(self, qkvz: torch.Tensor, ba: torch.Tensor, n: int):
        """qkvz -> (mixed_qkv, z); ba -> (b, a). Mirrors
        prepare_gdn_attention_core_inputs for gqa_interleaved_layout=False."""
        qkv_size = self.key_dim * 2 + self.value_dim
        z_size = self.value_dim
        mixed_qkv, z_flat = qkvz.split([qkv_size, z_size], dim=-1)
        z = z_flat.reshape(n, -1, self.head_v_dim)  # (n, num_v_heads, head_v_dim)
        b, a = ba.chunk(2, dim=-1)  # each (n, num_v_heads)
        return mixed_qkv, z, b, a

    def _conv_weights(self) -> torch.Tensor:
        return self.conv1d_weight.view(self.conv_dim, self.conv_kernel_size)

    def _conv_weights_fp32(self) -> torch.Tensor:
        """Cached fp32 (conv_dim, kernel) conv weight — built once (lazily, post weight-load), not
        re-cast per step. Replaces the per-call `self._conv_weights().float()` constant-recompute."""
        if self._conv_w_fp32 is None:
            self._conv_w_fp32 = self._conv_weights().float().contiguous()
        return self._conv_w_fp32

    def _norm_weight_fp32(self) -> torch.Tensor:
        """Cached fp32 gated-RMSNorm weight — same constant-recompute fix as the conv weight."""
        if self._norm_w_fp32 is None:
            self._norm_w_fp32 = self.norm.weight.float().contiguous()
        return self._norm_w_fp32

    def _split_conv_qkv(self, conv_out: torch.Tensor, n: int):
        """Split the conv output [n, conv_dim] = [q|k|v] into q,k [n, num_k_heads, head_k_dim] and
        v [n, num_v_heads, head_v_dim] — the layout gdn_hip's recurrent kernels consume."""
        q, k, v = conv_out.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(n, self.num_k_heads, self.head_k_dim).contiguous()
        k = k.reshape(n, self.num_k_heads, self.head_k_dim).contiguous()
        v = v.reshape(n, self.num_v_heads, self.head_v_dim).contiguous()
        return q, k, v

    def _gate_args(self) -> tuple:
        """Trailing `gate_activation` arg for the gated-norm ops, as a splat.

        Empty for SiLU so the call is BYTE-IDENTICAL to the pre-policy form — which is also what
        keeps an OLD gdn_hip .so (schema without the arg, e.g. the serve image's baked build)
        working for every silu/swish checkpoint. A sigmoid checkpoint on an old .so fails the op's
        schema match with an argument-count error at first forward: loud, and the fix is a kernel
        rebuild, not a silent SiLU."""
        return (self._gate_act,) if self._gate_act else ()

    # ---- output projection: rmsnorm_gated(core, z) -> flatten -> out_proj ----
    def _output_projection(self, core_attn_out: torch.Tensor, z: torch.Tensor, n: int) -> torch.Tensor:
        import gdn_hip as gdn  # lazy: only the engine forward needs the HIP .so (canonical callables)

        out_dtype = self._proj_dtype
        # bf16-native rmsnorm_gated: reads x/z at the input dtype, up-casts to fp32 for the norm, writes
        # back at the input dtype. .contiguous() (was implicit in the old .float() copy) is required:
        # core is a reshape of the gdn output, and z is a strided slice of the qkvz projection.
        core = core_attn_out.reshape(-1, core_attn_out.shape[-1]).contiguous()  # [n*HV, head_v_dim]
        z_flat = z.reshape(-1, z.shape[-1]).contiguous()
        engaged("gdn_hip.rmsnorm_gated")
        normed = gdn.rmsnorm_gated(core, z_flat, self._norm_weight_fp32(), self.norm.eps,
                                   *self._gate_args())
        normed = normed.reshape(n, self.value_dim)  # (n, num_v_heads, head_v_dim) -> (n, value_dim)
        return self.out_proj(normed.to(out_dtype))

    # ---- differentiable training prefill: native fwd + recompute-backward via the *_train wrappers ----
    def _prefill_train_one_seq(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """One sequence, zero initial state, DIFFERENTIABLE. Mirrors forward_prefill's compute but uses
        the gdn_hip *_train autograd wrappers (recompute-backward) for conv + gated-delta prefill, so
        grad flows back through the native path to hidden_states. No conv_state/ssm_state carry —
        training binds a fresh sequence from zero state."""
        from gdn_hip import autograd as gdn_bwd  # lazy: only the training path needs the wrappers

        n = hidden_states.shape[0]
        qkvz, ba = self._in_proj(hidden_states)
        mixed_qkv, z, b, a = self._split_qkvz_ba(qkvz, ba, n)
        conv_out = gdn_bwd.causal_conv1d_fwd_train(
            mixed_qkv.contiguous(), self._conv_weights_fp32(), None, 1)  # SiLU
        q, k, v = self._split_conv_qkv(conv_out, n)
        train_op = gdn_bwd.gdn_prefill_train if os.environ.get("GDN_HIP_WMMA_PREFILL") == "0" \
            else gdn_bwd.gdn_prefill_wmma_train
        core = train_op(q, k, v, a.contiguous(), b.contiguous(), self.A_log, self.dt_bias,
                        self.head_k_dim ** -0.5, 1)  # [T, num_v_heads, head_v_dim]
        # differentiable output projection: the raw gdn.rmsnorm_gated (used by the serve-path
        # _output_projection) is only differentiable if gdn_hip.autograd.enable() has been called to
        # register its formula. The training path must NOT depend on that process-wide global, so use
        # the self-contained rmsnorm_gated_train wrapper here (this was the 24-layer backward-cos drop).
        out_dtype = self._proj_dtype
        core = core.reshape(-1, core.shape[-1]).contiguous()          # [T*num_v_heads, head_v_dim]
        z_flat = z.reshape(-1, z.shape[-1]).contiguous()
        normed = gdn_bwd.rmsnorm_gated_train(core, z_flat, self._norm_weight_fp32(), self.norm.eps,
                                             *self._gate_args())
        return self.out_proj(normed.reshape(n, self.value_dim).to(out_dtype))

    def _prefill_train_batch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """BATCHED [B,T,H] differentiable prefill (all sequences length T). Kills the per-sequence Python
        loop: position-wise ops (in_proj, split, rmsnorm) run flattened over B*T; the conv and gated-delta
        recurrence run as ONE varlen op call over all B sequences (batch folded into heads in the backward
        reference). Algebraically identical to stacking _prefill_train_one_seq over the batch."""
        from gdn_hip import autograd as gdn_bwd

        B, T, _ = hidden_states.shape
        n = B * T
        qkvz, ba = self._in_proj(hidden_states.reshape(n, -1))
        mixed_qkv, z, b, a = self._split_qkvz_ba(qkvz, ba, n)
        conv_out = gdn_bwd.causal_conv1d_batch_train(
            mixed_qkv.reshape(B, T, -1).contiguous(), self._conv_weights_fp32(), None, 1)  # [B,T,conv_dim]
        q, k, v = self._split_conv_qkv(conv_out.reshape(n, -1), n)
        train_batch = gdn_bwd.gdn_prefill_batch_train if os.environ.get("GDN_HIP_WMMA_PREFILL") == "0" \
            else gdn_bwd.gdn_prefill_wmma_batch_train
        core = train_batch(
            q.reshape(B, T, self.num_k_heads, self.head_k_dim),
            k.reshape(B, T, self.num_k_heads, self.head_k_dim),
            v.reshape(B, T, self.num_v_heads, self.head_v_dim),
            a.reshape(B, T, self.num_v_heads).contiguous(), b.reshape(B, T, self.num_v_heads).contiguous(),
            self.A_log, self.dt_bias, self.head_k_dim ** -0.5, 1)  # [B,T,num_v_heads,head_v_dim]
        out_dtype = self._proj_dtype
        core = core.reshape(n * self.num_v_heads, self.head_v_dim).contiguous()
        z_flat = z.reshape(n * self.num_v_heads, self.head_v_dim).contiguous()
        normed = gdn_bwd.rmsnorm_gated_train(core, z_flat, self._norm_weight_fp32(), self.norm.eps,
                                             *self._gate_args())
        return self.out_proj(normed.reshape(B, T, self.value_dim).to(out_dtype))

    def _forward_prefill_train(self, hidden_states: torch.Tensor,
                               query_start_loc: torch.Tensor) -> torch.Tensor:
        """Differentiable prefill over a (possibly multi-sequence) varlen batch: run each sequence
        independently through the single-seq differentiable path and concatenate. Per-sequence keeps
        each doc's causal conv + zero-init recurrence isolated (no cross-sequence leakage)."""
        cu = query_start_loc.tolist()
        if len(cu) == 2:  # single sequence — the common training/eval case
            return self._prefill_train_one_seq(hidden_states)
        return torch.cat(
            [self._prefill_train_one_seq(hidden_states[cu[i]:cu[i + 1]]) for i in range(len(cu) - 1)],
            dim=0)

    # ---- ReplaySSM ring: the two rules any non-decode use of ssm_state must obey ----------------
    # Under ReplaySSM the true state is fold(ssm_state, ring), so every OTHER kernel that touches
    # ssm_state has to bracket itself: materialise before reading, invalidate after writing. Both
    # prefill entry points below do exactly that; the cross-layer sites (snapshot / radix clone /
    # verify-state install / fresh-slot reset) do it through GDNStateCache.
    @staticmethod
    def _replay_flush(gdn, ring, ssm_state, state_idx) -> None:
        """Fold the ring into ssm_state for these slots, so the prefill kernel reads the TRUE initial
        state. A no-op dispatch when the ring is empty (the kernel returns on its own cursor), which
        is the common case — a slot only has entries if plain decode ran on it since the last write."""
        if ring is not None:
            gdn.gdn_replay_flush(ssm_state, state_idx, ring["k"], ring["vr"], ring["g"], ring["len"],
                                 ring["s0n"])

    @staticmethod
    def _replay_invalidate(ring, state_idx) -> None:
        """The prefill just overwrote ssm_state: drop the buffered entries and mark ||S0||_F unknown.
        -1 makes the next decode step re-establish the checkpoint norm itself."""
        if ring is not None:
            ring["len"].index_fill_(0, state_idx, 0)
            ring["s0n"].index_fill_(0, state_idx, -1.0)

    # ---- prefill: chunk-scan over the full sequence, writes final SSM state ----
    def forward_prefill(
        self,
        hidden_states: torch.Tensor,  # (T, hidden)
        conv_state: torch.Tensor,  # (num_slots, conv_dim, kernel-1), updated in place
        ssm_state: torch.Tensor,  # (num_slots, num_v_heads, head_v_dim, head_k_dim)
        query_start_loc: torch.Tensor,  # cu_seqlens, int32 (num_seqs+1,)
        state_indices: torch.Tensor,  # slot id per sequence, int32
        has_initial_state: torch.Tensor,  # bool per sequence
        conv_metadata=None,  # GDN conv metadata (nums_dict/batch_ptr/token_chunk_offset_ptr)
        ring=None,  # ReplaySSM ring for this GDN layer (GDNStateCache.ring(lid)), or None
    ) -> torch.Tensor:
        # Differentiable native path: when autograd is tracking the input (tap / LM-loss training), the
        # in-place conv/prefill ops below cannot carry a backward (Tensor(a!) state; torch rejects a raw
        # autograd formula on a non-functional op), so route conv+prefill through the *_train recompute
        # wrappers. Serving runs under no_grad / inference_mode with frozen weights, so requires_grad is
        # False and this never fires on the hot path (nor during graph capture).
        if torch.is_grad_enabled() and hidden_states.requires_grad:
            return self._forward_prefill_train(hidden_states, query_start_loc)

        import gdn_hip as gdn  # lazy: only the engine forward needs the HIP .so (canonical callables)

        n = hidden_states.shape[0]
        qkvz, ba = self._in_proj(hidden_states)
        mixed_qkv, z, b, a = self._split_qkvz_ba(qkvz, ba, n)
        state_idx = state_indices.long()  # int32->int64 once, reused by conv + prefill kernels
        has_init = has_initial_state.to(torch.uint8)  # bool->uint8 once, reused likewise
        self._replay_flush(gdn, ring, ssm_state, state_idx)  # READ of ssm_state -> materialise first

        # Depthwise causal conv (varlen) + SiLU; conv_state (fp32) updated in place per slot. The HIP
        # kernel takes token-major [T, conv_dim] contiguous (vs the Triton path's transposed view).
        # bf16-native: the gdn_hip kernels are templated on the I/O dtype and up-cast to fp32
        # in-register, so mixed_qkv/a/b/core/z flow through at the model dtype (no .float() HBM
        # round-trip). .contiguous() is still required — the conv kernel reads token-major contiguous,
        # and it also replaces the contiguity the old .float() copy used to provide for the views below.
        engaged("gdn_hip.causal_conv1d_fwd")
        conv_out = gdn.causal_conv1d_fwd(
            mixed_qkv.contiguous(),
            self._conv_weights_fp32(),
            None,  # bias-free
            query_start_loc,
            state_idx,
            has_init,
            conv_state,
            1,  # SiLU
        )
        # Gated-delta-rule prefill: l2norm(q,k) + g/beta from (a,b,A_log,dt_bias) folded INTO the
        # kernel (replacing fused_post_conv_prep + chunk_gated_delta_rule). State written in place.
        # Two validated kernels (tools/gdn_hip_parity.py, both vs the recurrent oracle):
        #   - gdn_prefill_wmma (DEFAULT): matrix-core chunked, 5-6.7x FASTER than recurrent at
        #     T=256..16384 (tools/gdn_hip_bench.py); fp16 matmul operands -> max|Δ|~1e-3 vs recurrent.
        #   - gdn_prefill (recurrent): the per-token fp32 reference; exact but slow. Fallback via
        #     GDN_HIP_WMMA_PREFILL=0 (e.g. if a real-decay regime stresses the fp16 absorption).
        # (gdn_prefill_chunked, the scalar chunked op, is kept only as a parity oracle — it was ~4x
        # SLOWER than recurrent, which is why the WMMA reformulation exists.)
        q, k, v = self._split_conv_qkv(conv_out, n)
        prefill_op = gdn.gdn_prefill if os.environ.get("GDN_HIP_WMMA_PREFILL") == "0" \
            else gdn.gdn_prefill_wmma
        engaged("gdn_hip.gdn_prefill" if os.environ.get("GDN_HIP_WMMA_PREFILL") == "0"
                else "gdn_hip.gdn_prefill_wmma")
        core = prefill_op(
            q, k, v, a.contiguous(), b.contiguous(), self.A_log, self.dt_bias,
            query_start_loc, state_idx, has_init,
            ssm_state, self.head_k_dim ** -0.5, 1,
        )  # [T, num_v_heads, head_v_dim] at the input (model) dtype
        self._replay_invalidate(ring, state_idx)  # WRITE of ssm_state -> the ring is now stale
        return self._output_projection(core, z, n)

    # ---- verify: varlen recurrent prefill that ALSO captures the per-token recurrent state ----
    # WHY THERE IS NO WMMA VERIFY, and what it costs. Normal prefill dispatches gdn_prefill_wmma
    # (matrix-core chunked, 5-6.7x faster than recurrent, ZERO scratch). Verify cannot: it needs the
    # SSM state after EVERY token so the scheduler can install the state at the accepted prefix, and
    # a chunked matrix-core formulation only yields the end-of-chunk state. So the spec path runs the
    # recurrent kernel, which measures 528 B/lane of scratch across all 72 instantiations
    # (-Rpass-analysis=kernel-resource-usage, identical on clang 22 and 23) where prefill pays none.
    #
    # HOW BIG IS THE VERIFY WINDOW? This decides whether a WMMA verify could ever pay, and the
    # answer depends entirely on the drafter -- an earlier version of this note got it wrong by
    # quoting the N-GRAM default (spec_num_draft=4, so ~5 tokens, where matrix cores have nothing to
    # work with). DFlash is completely different:
    #     GDN_WC (gdn_kernels.hip)                        = 16   <- WMMA chunk, one 16x16 tile dim
    #     z-lab Qwen3.6-35B-A3B-DFlash  block_size        = 16
    #     z-lab Qwen3.6-27B-DFlash / Qwen3.5-4B-DFlash    = 16
    #     poolside Laguna-XS-2.1, Muse-Glimmer-30B        = 16
    #     ZAYA CCA ns15                                   = 16 (num_draft+1)
    #     RadixArk Qwen3.8-27B-DSpark                     =  7
    # A DFlash verify window is num_draft+1 = 16 tokens = EXACTLY ONE GDN_WC CHUNK. So for every
    # block-16 drafter the "not enough volume for matrix cores" objection does not apply at all --
    # the window is precisely one unit of the fast kernel's work. The only real blocker is the
    # per-token state capture above.
    #
    # Still DO NOT conclude that this is why spec is slow. The recorded root cause of the 1.57x LOSS
    # on the offload arm is different and larger -- cost scaling with QUERY TOKENS through MoE expert
    # streaming -- and that was measured while this asymmetry was also present.
    #
    # WHAT IS ACTUALLY UNKNOWN: nobody has attributed spec's cost between GDN-verify and MoE
    # streaming. If spec decode is revisited, measure that split FIRST -- it decides whether a
    # per-token-capturing WMMA verify would be worth writing at all.
    def forward_prefill_verify(
        self,
        hidden_states: torch.Tensor,  # (T, hidden)
        conv_state: torch.Tensor,  # (num_slots, conv_dim, kernel-1), updated in place
        ssm_state: torch.Tensor,  # (num_slots, num_v_heads, head_v_dim, head_k_dim)
        query_start_loc: torch.Tensor,  # cu_seqlens, int32 (num_seqs+1,)
        state_indices: torch.Tensor,  # slot id per sequence, int32
        has_initial_state: torch.Tensor,  # bool per sequence
        max_qlen: int,  # = max extend_len (= max K+1) across the batch's verify windows
        ring=None,  # ReplaySSM ring for this GDN layer (GDNStateCache.ring(lid)), or None
    ):
        """Spec-decode GDN verify forward. Identical recurrence to ``forward_prefill`` (same bit-stable
        recurrent kernels), but captures the conv + ssm state AFTER EACH of the per-seq verify tokens
        into scratch buffers. The scheduler then installs the state after the accepted prefix
        (index = accepted_count-1) directly into the slot — no snapshot, no 2x re-advance, BIT-EXACT.

        Returns ``(out, conv_scratch, ssm_scratch)``:
          conv_scratch: [max_qlen, num_seqs, conv_dim, kernel-1] (fp32)
          ssm_scratch:  [max_qlen, num_seqs, num_v_heads, head_v_dim, head_k_dim] (ssm dtype)
        """
        import gdn_hip as gdn  # lazy: only the engine forward needs the HIP .so (canonical callables)

        n = hidden_states.shape[0]
        qkvz, ba = self._in_proj(hidden_states)
        mixed_qkv, z, b, a = self._split_qkvz_ba(qkvz, ba, n)
        state_idx = state_indices.long()
        has_init = has_initial_state.to(torch.uint8)
        # REPLAY VERIFY (the ring path). The draft window is APPENDED to the ring instead of
        # materialised into ssm_state, so there is no flush before (the checkpoint + ring already ARE
        # the state) and no invalidate after (the ring is not stale — it now holds this window). The
        # accept step rewinds rejected drafts with a cursor decrement (GDNStateCache.rollback_ring)
        # instead of scattering a per-token state snapshot back into the slot, so this path also
        # returns ssm_scratch=None and never allocates it.
        # Gated on the window fitting the ring: a mid-window flush would destroy the checkpoint the
        # rollback rewinds to, so the kernel refuses q_len > L and we fall back to the materialising
        # verify below.
        use_replay = (
            ring is not None
            and hasattr(gdn, "gdn_verify_replay")
            and int(max_qlen) <= int(ring["k"].shape[2])
        )
        if not use_replay:
            self._replay_flush(gdn, ring, ssm_state, state_idx)  # READ of ssm_state -> materialise

        # Conv: bit-identical to forward_prefill's causal_conv1d_fwd, plus per-token window capture.
        engaged("gdn_hip.causal_conv1d_fwd_verify")
        conv_out, conv_scratch = gdn.causal_conv1d_fwd_verify(
            mixed_qkv.contiguous(),
            self._conv_weights_fp32(),
            None,
            query_start_loc,
            state_idx,
            has_init,
            conv_state,
            int(max_qlen),
            1,  # SiLU
        )
        # Gated-delta-rule verify: the RECURRENT oracle. Bit-stable, and it stays the default.
        #
        # A CHUNKED/WMMA verify EXISTS and is NOT used here: gdn_hip.gdn_prefill_verify_wmma. It is
        # correct (rdna4-hip-kernels tests/test_verify_wmma.py) and 3.07x faster than this kernel at
        # the 16-token window — 7.73 -> 2.52 ms per forward across the 30 linear_attention layers.
        # It was wired here, built, and A/B'd against old code back-to-back on 2026-09-16, and moved
        # serve throughput by NOTHING: bs=1 108.5 -> 108.4, bs=2 131.6 -> 130.9, bs=6 202.1 -> 202.2,
        # TPOT 9.19 -> 9.16 ms.
        #
        # WHY IT CHANGED NOTHING, which is the part worth keeping. `k_dflash=15` is a CEILING that
        # sizes a width LADDER (spec/width.py verify_width_ladder -> [3, 7, 15]); an adaptive
        # controller picks a rung per step from measured acceptance. At the served acceptance
        # (accept-len ~2.1) it picks rung 7 in 98% of steps and rung 15 in none:
        #     verify-width[0:17(1%) 3:11(1%) 7:1172(98%) 15:0(0%)]
        # and the boot log says which kernel each rung commits through:
        #     [gdn] spec-verify commit per width: qlen=4->ring-rollback, qlen=8->ring-rollback,
        #                                         qlen=16->materialised
        # Rung 7 is an 8-token window, which FITS the ReplaySSM ring, so `use_replay` above is true
        # and gdn_verify_replay carries it. Only rung 15 (qlen 16) reaches THIS path, and the
        # controller picks it in 0% of steps -- so a 3x kernel here is 3x of nothing.
        #
        # NOT a capture problem, which is the first thing to suspect and is worth ruling out in
        # writing: all three widths ARE captured ("Capturing spec-verify CUDA graphs (widths=[3, 7,
        # 15] -> qlens=[4, 8, 16])"), the width-15 graph included. It is captured and never replayed.
        # Do NOT read the `verify-graph replay=N eager=M` counter as evidence about this dispatch --
        # it counts CUDA-graph replay vs eager execution (scheduler.py `_m_vgraph`), and its
        # "replay" has nothing to do with gdn_verify_replay. Misreading exactly that pair is how the
        # first write-up of this reached the right conclusion from the wrong evidence.
        #
        # So the win is NOT in this kernel. At the width that actually runs, gdn_verify_replay is
        # already the faster kernel at bs=1 (0.073 ms vs 0.123 ms for the WMMA verify). It loses to
        # WMMA only at N=4 (0.113 vs 0.056 ms), which `use_replay` never lets it compare — that gate
        # asks "does the window FIT the ring", not "which kernel is cheaper". Measuring whether
        # routing wide-batch width-7 windows away from replay pays is the open follow-up; it is a
        # scheduling question, not a kernel one.
        #
        # Read verify-width[...] out of the serve log BEFORE optimising anything on a verify path.
        q, k, v = self._split_conv_qkv(conv_out, n)
        if use_replay:
            engaged("gdn_hip.gdn_verify_replay")
            core = gdn.gdn_verify_replay(
                q, k, v, a.contiguous(), b.contiguous(), self.A_log, self.dt_bias,
                query_start_loc, state_idx, has_init, ssm_state, *ring.args(),
                self.head_k_dim ** -0.5, 1, int(max_qlen),
            )
            # ssm_scratch is None BY CONTRACT here: the per-token states the materialising path hands
            # back exist only so the caller can pick the accepted one, and the ring makes that a
            # cursor rewind. Verified bit-exact against the served decode trajectory
            # (rdna4-hip-kernels tests/test_verify_replay.py).
            return self._output_projection(core, z, n), conv_scratch, None
        # hasattr, not an env knob: an older kernel package simply has no WMMA verify to call, and
        # the recurrent kernel stays the correct answer there.
        engaged("gdn_hip.gdn_prefill_verify")
        core, ssm_scratch = gdn.gdn_prefill_verify(
            q, k, v, a.contiguous(), b.contiguous(), self.A_log, self.dt_bias,
            query_start_loc, state_idx, has_init,
            ssm_state, int(max_qlen), self.head_k_dim ** -0.5, 1,
        )
        self._replay_invalidate(ring, state_idx)  # WRITE of ssm_state -> the ring is now stale
        return self._output_projection(core, z, n), conv_scratch, ssm_scratch

    # ---- decode: single-step recurrent update per sequence, advances state in place ----
    def forward_decode(
        self,
        hidden_states: torch.Tensor,  # (B, hidden), one token per sequence
        conv_state: torch.Tensor,  # (num_slots, conv_dim, kernel-1), updated in place
        ssm_state: torch.Tensor,  # (num_slots, num_v_heads, head_v_dim, head_k_dim)
        query_start_loc: torch.Tensor,  # int32 (num_decodes+1,)
        state_indices: torch.Tensor,  # slot id per sequence, int32
        ring=None,  # ReplaySSM ring for this GDN layer (GDNStateCache.ring(lid)), or None
    ) -> torch.Tensor:
        import gdn_hip as gdn  # lazy: only the engine forward needs the HIP .so (canonical callables)

        n = hidden_states.shape[0]
        qkvz, ba = self._in_proj(hidden_states)
        mixed_qkv, z, b, a = self._split_qkvz_ba(qkvz, ba, n)
        # SSM-NORM PROBE (MINISGL_SSM_NORM_LOG=N, debug): every Nth decode call on layer 0, log the
        # max per-head ||S||_F across active slots. The gdn_hip decode kernels Frobenius-clamp the
        # state at SSM_STATE_MAX_NORM (1000, compile-time) — a nonlinear intervention llama.cpp does
        # not perform — so if a model's real rumination-era states brush the cap, minisgl distorts
        # them where llama.cpp lets them ride. This logger answers whether the clamp ENGAGES in vivo.
        import os as _os
        _nl = int(_os.environ.get("MINISGL_SSM_NORM_LOG", "0") or 0)
        if _nl:
            self._norm_calls = getattr(self, "_norm_calls", 0) + 1
            if self._norm_calls % _nl == 0 and not torch.cuda.is_current_stream_capturing():
                try:
                    idx = state_indices.long()
                    st = ssm_state[idx].float()          # [B, HV, V, K]
                    nrm = st.reshape(st.shape[0], st.shape[1], -1).norm(dim=-1)
                    from minisgl.utils import init_logger
                    init_logger(__name__).info_rank0(
                        f"[ssm-norm] call={self._norm_calls} max={nrm.max().item():.1f} "
                        f"p50={nrm.median().item():.1f} (cap 1000)")
                except Exception as e:
                    from minisgl.utils import init_logger
                    init_logger(__name__).warning_rank0(f"[ssm-norm] probe failed: {e!r}")

        state_idx = state_indices.long()  # int32->int64 once, reused by both kernels below

        if ring is not None and hasattr(gdn, "gdn_decode_conv_gated_replay"):
            # FUSED + REPLAY: the same conv_update + gated-RMSNorm fusion as the rung below, with the
            # ReplaySSM state step in place of the materialised recurrence. The per-step ssm_state
            # read-modify-WRITE disappears: the step reads the checkpoint, probes it, and appends
            # (k, vr, g) to the ring; ssm_state is written only when the ring fills (or its Frobenius
            # bound trips), on a device-side branch, so this is capture-safe and needs no host
            # round-trip. Everything that reads ssm_state elsewhere flushes first — see
            # GDNStateCache's ring lifecycle. Measured 1.13x/2.60x/2.90x/1.50x on the kernel at
            # B=1/2/4/8 (bf16 state, served head dims).
            z_flat = z.reshape(-1, z.shape[-1]).contiguous()
            engaged("gdn_hip.gdn_decode_conv_gated_replay")
            normed = gdn.gdn_decode_conv_gated_replay(
                mixed_qkv.contiguous(), self._conv_weights_fp32(), None, conv_state,
                a.contiguous(), b.contiguous(), self.A_log, self.dt_bias,
                ssm_state, state_idx, ring["k"], ring["vr"], ring["g"], ring["vn"], ring["len"],
                ring["s0n"], z_flat, self._norm_weight_fp32(), self.norm.eps,
                1, self.head_k_dim ** -0.5, 1, *self._gate_args(),
            )  # [B, num_v_heads, head_v_dim], conv+replay-recurrence+gated-RMS-norm in one
            return self.out_proj(normed.reshape(n, self.value_dim).to(self._proj_dtype))

        if _GDN_FUSED_CONV and hasattr(gdn, "gdn_decode_conv_gated"):
            # FUSED: conv_update + gdn_decode + gated-RMSNorm in ONE kernel (was 2 launches:
            # causal_conv1d_update + gdn_decode_gated). Reads mixed_qkv directly — no python q/k/v split,
            # no conv_out HBM round-trip. Bit-exact vs the two-kernel path (max|Δ|=0); ~1.3x per-token.
            z_flat = z.reshape(-1, z.shape[-1]).contiguous()
            engaged("gdn_hip.gdn_decode_conv_gated")
            normed = gdn.gdn_decode_conv_gated(
                mixed_qkv.contiguous(), self._conv_weights_fp32(), None, conv_state,
                a.contiguous(), b.contiguous(), self.A_log, self.dt_bias,
                ssm_state, state_idx, z_flat, self._norm_weight_fp32(), self.norm.eps,
                1, self.head_k_dim ** -0.5, 1, *self._gate_args(),
            )  # [B, num_v_heads, head_v_dim], conv+recurrence+gated-RMS-norm in one
            return self.out_proj(normed.reshape(n, self.value_dim).to(self._proj_dtype))

        # One-step depthwise causal conv update (state roll) + SiLU; conv_state (fp32) in place.
        engaged("gdn_hip.causal_conv1d_update")
        conv_out = gdn.causal_conv1d_update(
            mixed_qkv.contiguous(),  # bf16-native; .contiguous() supplies the token-major layout
            self._conv_weights_fp32(),
            None,  # bias-free
            conv_state,
            state_idx,
            1,  # SiLU
        )
        # One-step gated-delta-rule (l2norm + g/beta folded in); ssm_state updated in place per slot.
        q, k, v = self._split_conv_qkv(conv_out, n)
        if _GDN_FUSED_NORM and hasattr(gdn, "gdn_decode_gated"):
            # FUSED: gdn_decode + the gated-RMSNorm output projection in ONE kernel (skips the separate
            # rmsnorm_gated launch + the core HBM round-trip). Bit-exact vs the two-kernel path;
            # z_flat/norm_weight/eps are exactly what _output_projection feeds the standalone rmsnorm_gated.
            z_flat = z.reshape(-1, z.shape[-1]).contiguous()
            engaged("gdn_hip.gdn_decode_gated")
            normed = gdn.gdn_decode_gated(
                q, k, v, a.contiguous(), b.contiguous(), self.A_log, self.dt_bias,
                ssm_state, state_idx, z_flat, self._norm_weight_fp32(), self.norm.eps,
                self.head_k_dim ** -0.5, 1, *self._gate_args(),
            )  # [B, num_v_heads, head_v_dim], already gated-RMS-normed
            return self.out_proj(normed.reshape(n, self.value_dim).to(self._proj_dtype))
        engaged("gdn_hip.gdn_decode")
        core = gdn.gdn_decode(
            q, k, v, a.contiguous(), b.contiguous(), self.A_log, self.dt_bias,
            ssm_state, state_idx, self.head_k_dim ** -0.5, 1,
        )  # [B, num_v_heads, head_v_dim] at the input (model) dtype
        return self._output_projection(core, z, n)

    # ---- post-load: convert any quantized projection to its op layout (int4 packs -> kernel buffers) ----
    def process_quant(self) -> None:
        """Finalize the quantized projections after the checkpoint load (called by the
        GDNLinearAttn bridge's post_load). A no-op for bf16 (`nn.Linear`) projections."""
        for proj in (self.in_proj_qkvz, self.in_proj_ba, self.out_proj):
            if isinstance(proj, _MethodLinear):
                proj.process_quant()

    # ---- warmup hook — no-op now that the conv is AOT HIP (no Triton autotune to settle) ----
    @torch.no_grad()
    def warmup_conv(self, num_tokens: int, *, iters: int = 2) -> None:
        """The Triton causal_conv1d_fn autotuned in place on its first call (NaN/0/OOM risk), so it
        had to be warmed per process. gdn_hip's conv is AOT-compiled HIP — no JIT, no autotune —
        so there is nothing to warm. Kept as a no-op for engine API compatibility."""
        return
