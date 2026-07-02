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

import torch
from torch import nn

# The GDN compute kernels (conv, gated-delta-rule prefill/decode, the gated RMSNorm) are now native
# HIP (torch.ops.gdn_hip.*), AOT-compiled, no Triton JIT. Importing this layer no longer drags in the
# vendored Triton tree at all.


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

        self.key_dim = head_k_dim * self.num_k_heads
        self.value_dim = head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Projections (bias-free, like the reference). in_proj_qkvz packs q,k,v,z;
        # in_proj_ba packs b,a (per-v-head scalars).
        self.in_proj_qkvz = nn.Linear(
            hidden_size, self.key_dim * 2 + self.value_dim * 2, bias=False, dtype=dtype, device=device
        )
        self.in_proj_ba = nn.Linear(
            hidden_size, 2 * self.num_v_heads, bias=False, dtype=dtype, device=device
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
        self.out_proj = nn.Linear(
            self.value_dim, hidden_size, bias=False, dtype=dtype, device=device
        )

        # fp32 caches of the two FROZEN weights the gdn_hip kernels consume at fp32: the depthwise
        # conv weight and the gated-RMSNorm weight. Both are model parameters — constant after the
        # state-dict load — so the previous per-call `.float()` was recomputing a constant, i.e. one
        # extra elementwise launch per layer per token (felt as launch overhead in the eager,
        # cudagraph-disabled decode loop). Built once lazily on first forward (which runs post-load,
        # before any graph capture) and reused. Assumes weights are frozen + not device-moved after
        # warmup, which is the inference contract here.
        self._conv_w_fp32: torch.Tensor | None = None
        self._norm_w_fp32: torch.Tensor | None = None

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

    # ---- output projection: rmsnorm_gated(core, z) -> flatten -> out_proj ----
    def _output_projection(self, core_attn_out: torch.Tensor, z: torch.Tensor, n: int) -> torch.Tensor:
        from gdn_hip import op as gdn  # lazy: only the engine forward needs the HIP .so

        out_dtype = self.out_proj.weight.dtype
        # bf16-native rmsnorm_gated: reads x/z at the input dtype, up-casts to fp32 for the norm, writes
        # back at the input dtype. .contiguous() (was implicit in the old .float() copy) is required:
        # core is a reshape of the gdn output, and z is a strided slice of the qkvz projection.
        core = core_attn_out.reshape(-1, core_attn_out.shape[-1]).contiguous()  # [n*HV, head_v_dim]
        z_flat = z.reshape(-1, z.shape[-1]).contiguous()
        normed = gdn.rmsnorm_gated(core, z_flat, self._norm_weight_fp32(), self.norm.eps)
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
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
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
        out_dtype = self.out_proj.weight.dtype
        core = core.reshape(-1, core.shape[-1]).contiguous()          # [T*num_v_heads, head_v_dim]
        z_flat = z.reshape(-1, z.shape[-1]).contiguous()
        normed = gdn_bwd.rmsnorm_gated_train(core, z_flat, self._norm_weight_fp32(), self.norm.eps)
        return self.out_proj(normed.reshape(n, self.value_dim).to(out_dtype))

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
    ) -> torch.Tensor:
        # Differentiable native path: when autograd is tracking the input (tap / LM-loss training), the
        # in-place conv/prefill ops below cannot carry a backward (Tensor(a!) state; torch rejects a raw
        # autograd formula on a non-functional op), so route conv+prefill through the *_train recompute
        # wrappers. Serving runs under no_grad / inference_mode with frozen weights, so requires_grad is
        # False and this never fires on the hot path (nor during graph capture).
        if torch.is_grad_enabled() and hidden_states.requires_grad:
            return self._forward_prefill_train(hidden_states, query_start_loc)

        from gdn_hip import op as gdn  # lazy: only the engine forward needs the HIP .so

        n = hidden_states.shape[0]
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        mixed_qkv, z, b, a = self._split_qkvz_ba(qkvz, ba, n)
        state_idx = state_indices.long()  # int32->int64 once, reused by conv + prefill kernels
        has_init = has_initial_state.to(torch.uint8)  # bool->uint8 once, reused likewise

        # Depthwise causal conv (varlen) + SiLU; conv_state (fp32) updated in place per slot. The HIP
        # kernel takes token-major [T, conv_dim] contiguous (vs the Triton path's transposed view).
        # bf16-native: the gdn_hip kernels are templated on the I/O dtype and up-cast to fp32
        # in-register, so mixed_qkv/a/b/core/z flow through at the model dtype (no .float() HBM
        # round-trip). .contiguous() is still required — the conv kernel reads token-major contiguous,
        # and it also replaces the contiguity the old .float() copy used to provide for the views below.
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
        core = prefill_op(
            q, k, v, a.contiguous(), b.contiguous(), self.A_log, self.dt_bias,
            query_start_loc, state_idx, has_init,
            ssm_state, self.head_k_dim ** -0.5, 1,
        )  # [T, num_v_heads, head_v_dim] at the input (model) dtype
        return self._output_projection(core, z, n)

    # ---- verify: varlen recurrent prefill that ALSO captures the per-token recurrent state ----
    def forward_prefill_verify(
        self,
        hidden_states: torch.Tensor,  # (T, hidden)
        conv_state: torch.Tensor,  # (num_slots, conv_dim, kernel-1), updated in place
        ssm_state: torch.Tensor,  # (num_slots, num_v_heads, head_v_dim, head_k_dim)
        query_start_loc: torch.Tensor,  # cu_seqlens, int32 (num_seqs+1,)
        state_indices: torch.Tensor,  # slot id per sequence, int32
        has_initial_state: torch.Tensor,  # bool per sequence
        max_qlen: int,  # = max extend_len (= max K+1) across the batch's verify windows
    ):
        """Spec-decode GDN verify forward. Identical recurrence to ``forward_prefill`` (same bit-stable
        recurrent kernels), but captures the conv + ssm state AFTER EACH of the per-seq verify tokens
        into scratch buffers. The scheduler then installs the state after the accepted prefix
        (index = accepted_count-1) directly into the slot — no snapshot, no 2x re-advance, BIT-EXACT.

        Returns ``(out, conv_scratch, ssm_scratch)``:
          conv_scratch: [max_qlen, num_seqs, conv_dim, kernel-1] (fp32)
          ssm_scratch:  [max_qlen, num_seqs, num_v_heads, head_v_dim, head_k_dim] (ssm dtype)
        """
        from gdn_hip import op as gdn  # lazy: only the engine forward needs the HIP .so

        n = hidden_states.shape[0]
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        mixed_qkv, z, b, a = self._split_qkvz_ba(qkvz, ba, n)
        state_idx = state_indices.long()
        has_init = has_initial_state.to(torch.uint8)

        # Conv: bit-identical to forward_prefill's causal_conv1d_fwd, plus per-token window capture.
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
        # Gated-delta-rule verify: the RECURRENT (non-WMMA) oracle — bit-stable, the whole point of
        # verify. Captures the ssm state after each token. (No WMMA path: the chunk-size dependence is
        # exactly the non-bit-exactness this kernel removes.)
        q, k, v = self._split_conv_qkv(conv_out, n)
        core, ssm_scratch = gdn.gdn_prefill_verify(
            q, k, v, a.contiguous(), b.contiguous(), self.A_log, self.dt_bias,
            query_start_loc, state_idx, has_init,
            ssm_state, int(max_qlen), self.head_k_dim ** -0.5, 1,
        )
        return self._output_projection(core, z, n), conv_scratch, ssm_scratch

    # ---- decode: single-step recurrent update per sequence, advances state in place ----
    def forward_decode(
        self,
        hidden_states: torch.Tensor,  # (B, hidden), one token per sequence
        conv_state: torch.Tensor,  # (num_slots, conv_dim, kernel-1), updated in place
        ssm_state: torch.Tensor,  # (num_slots, num_v_heads, head_v_dim, head_k_dim)
        query_start_loc: torch.Tensor,  # int32 (num_decodes+1,)
        state_indices: torch.Tensor,  # slot id per sequence, int32
    ) -> torch.Tensor:
        from gdn_hip import op as gdn  # lazy: only the engine forward needs the HIP .so

        n = hidden_states.shape[0]
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        mixed_qkv, z, b, a = self._split_qkvz_ba(qkvz, ba, n)
        state_idx = state_indices.long()  # int32->int64 once, reused by both kernels below

        # One-step depthwise causal conv update (state roll) + SiLU; conv_state (fp32) in place.
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
        core = gdn.gdn_decode(
            q, k, v, a.contiguous(), b.contiguous(), self.A_log, self.dt_bias,
            ssm_state, state_idx, self.head_k_dim ** -0.5, 1,
        )  # [B, num_v_heads, head_v_dim] at the input (model) dtype
        return self._output_projection(core, z, n)

    # ---- warmup hook — no-op now that the conv is AOT HIP (no Triton autotune to settle) ----
    @torch.no_grad()
    def warmup_conv(self, num_tokens: int, *, iters: int = 2) -> None:
        """The Triton causal_conv1d_fn autotuned in place on its first call (NaN/0/OOM risk), so it
        had to be warmed per process. gdn_hip's conv is AOT-compiled HIP — no JIT, no autotune —
        so there is nothing to warm. Kept as a no-op for engine API compatibility."""
        return
