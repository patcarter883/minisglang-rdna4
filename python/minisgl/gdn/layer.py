"""Phase 3b-2: a clean minisgl `QwenGatedDeltaNet` linear-attention layer.

Reimplements vLLM's `QwenGatedDeltaNetAttention` (Qwen3.5 / Qwen3-Next GDN) forward
COMPUTE, stripped of all vLLM coupling (CustomOp / forward_context / distributed /
MergedColumnParallelLinear / the torch.ops dispatch). It consumes the vendored FLA
kernels (`minisgl.gdn.fla.ops` + `minisgl.gdn.mamba.ops`) directly.

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
(RMSNormGated(core, z) -> out_proj).

conv_state convention: this layer expects the dim-first / "DS" layout
`(num_slots, conv_dim, conv_kernel-1)` — exactly what `GDNStateCache` allocates. On a
backend where `is_conv_state_dim_first()` is False the caller must transpose; that is
re-verified against the live RDNA4 path in 3b-3.
"""

from __future__ import annotations

import torch
from torch import nn

from minisgl.gdn.fla.ops import (
    RMSNormGated,
    chunk_gated_delta_rule,
    fused_post_conv_prep,
    fused_sigmoid_gating_delta_rule_update,
)
from minisgl.gdn.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update


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
        eps: float = 1e-6,
        activation: str = "silu",
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation

        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Projections (bias-free, like the reference). in_proj_qkvz packs q,k,v,z;
        # in_proj_ba packs b,a (per-v-head scalars).
        self.in_proj_qkvz = nn.Linear(
            hidden_size, self.key_dim * 2 + self.value_dim * 2, bias=False, dtype=dtype, device=device
        )
        self.in_proj_ba = nn.Linear(
            hidden_size, 2 * num_v_heads, bias=False, dtype=dtype, device=device
        )
        # conv1d weight mirrors the checkpoint shape (conv_dim, 1, kernel); the kernels
        # take a (conv_dim, kernel) view. Depthwise causal short-conv, bias-free here.
        self.conv1d_weight = nn.Parameter(
            torch.empty(self.conv_dim, 1, conv_kernel_size, dtype=dtype, device=device)
        )
        self.conv1d_bias: torch.Tensor | None = None

        self.dt_bias = nn.Parameter(torch.ones(num_v_heads, dtype=torch.float32, device=device))
        self.A_log = nn.Parameter(torch.empty(num_v_heads, dtype=torch.float32, device=device))

        self.norm = RMSNormGated(
            head_v_dim,
            eps=eps,
            group_size=None,
            norm_before_gate=True,
            activation=("silu" if activation == "swish" else activation),
            device=device,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            self.value_dim, hidden_size, bias=False, dtype=dtype, device=device
        )

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

    # ---- output projection: RMSNormGated(core, z) -> flatten -> out_proj ----
    def _output_projection(self, core_attn_out: torch.Tensor, z: torch.Tensor, n: int) -> torch.Tensor:
        z_shape_og = z.shape
        core = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core = self.norm(core, z)
        core = core.reshape(z_shape_og)
        core = core.flatten(-2)  # (n, num_v_heads, head_v_dim) -> (n, value_dim)
        return self.out_proj(core)

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
        n = hidden_states.shape[0]
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        mixed_qkv, z, b, a = self._split_qkvz_ba(qkvz, ba, n)

        # `mixed_qkv` is a `.split()` VIEW into the wider qkvz (token-stride = qkvz_dim).
        # causal_conv1d_fn handles that non-unit token-stride correctly (verified vs a CPU
        # reference conv at slot>=1: gapped split-view == contiguous, rel ~7e-3 bf16), so no
        # explicit .contiguous() is needed here — the reference passes the analogous view too.
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            self._conv_weights(),
            self.conv1d_bias,
            activation=self.activation,
            conv_states=conv_state,
            has_initial_state=has_initial_state,
            cache_indices=state_indices,
            query_start_loc=query_start_loc,
            metadata=conv_metadata,  # precomputed nums_dict/batch_ptr/token_chunk_offset_ptr
        ).transpose(0, 1)

        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=mixed_qkv,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=self.num_k_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        q, k, v, g, beta = (t.unsqueeze(0) for t in (q, k, v, g, beta))

        initial_state = ssm_state[state_indices].contiguous()
        initial_state[~has_initial_state, ...] = 0
        core_attn_out, last_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=False,  # l2norm already applied in fused_post_conv_prep
        )
        ssm_state[state_indices] = last_state.to(ssm_state.dtype)
        return self._output_projection(core_attn_out.squeeze(0), z, n)

    # ---- decode: single-step recurrent update per sequence, advances state in place ----
    def forward_decode(
        self,
        hidden_states: torch.Tensor,  # (B, hidden), one token per sequence
        conv_state: torch.Tensor,  # (num_slots, conv_dim, kernel-1), updated in place
        ssm_state: torch.Tensor,  # (num_slots, num_v_heads, head_v_dim, head_k_dim)
        query_start_loc: torch.Tensor,  # int32 (num_decodes+1,)
        state_indices: torch.Tensor,  # slot id per sequence, int32
    ) -> torch.Tensor:
        n = hidden_states.shape[0]
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        mixed_qkv, z, b, a = self._split_qkvz_ba(qkvz, ba, n)

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self._conv_weights(),
            self.conv1d_bias,
            self.activation,
            conv_state_indices=state_indices,
            validate_data=True,
        )
        q, k, v = self._rearrange_mixed_qkv(mixed_qkv, n)

        core_attn_out, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log,
            a=a,
            b=b,
            dt_bias=self.dt_bias,
            q=q,
            k=k,
            v=v,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=query_start_loc,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
        return self._output_projection(core_attn_out.squeeze(0), z, n)

    def _rearrange_mixed_qkv(self, mixed_qkv: torch.Tensor, seq_len: int):
        """Split packed [.., 2*key_dim + value_dim] into (1, seq, heads, dim) q/k/v."""
        q_dim = k_dim = self.key_dim
        v_dim = self.value_dim
        query, key, value = mixed_qkv.split([q_dim, k_dim, v_dim], dim=-1)
        query = query.reshape(1, seq_len, -1, self.head_k_dim).contiguous()
        key = key.reshape(1, seq_len, -1, self.head_k_dim).contiguous()
        value = value.reshape(1, seq_len, -1, self.head_v_dim).contiguous()
        return query, key, value
