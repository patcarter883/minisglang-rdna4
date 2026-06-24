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

# RMSNormGated is kept ONLY as the norm-weight container (its .weight / .eps); its Triton forward is
# never called — the gated norm runs through torch.ops.gdn_hip.rmsnorm_gated. The GDN compute kernels
# (conv, gated-delta-rule prefill/decode) are now native HIP (gdn_hip), AOT-compiled, no Triton JIT.
from minisgl.gdn.fla.ops import RMSNormGated


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
        core = core_attn_out.reshape(-1, core_attn_out.shape[-1]).float()  # [n*HV, head_v_dim]
        z_flat = z.reshape(-1, z.shape[-1]).float()
        normed = gdn.rmsnorm_gated(core, z_flat, self.norm.weight.float(), self.norm.eps)
        normed = normed.reshape(n, self.value_dim)  # (n, num_v_heads, head_v_dim) -> (n, value_dim)
        return self.out_proj(normed.to(out_dtype))

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
        from gdn_hip import op as gdn  # lazy: only the engine forward needs the HIP .so

        n = hidden_states.shape[0]
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        mixed_qkv, z, b, a = self._split_qkvz_ba(qkvz, ba, n)

        # Depthwise causal conv (varlen) + SiLU; conv_state (fp32) updated in place per slot. The HIP
        # kernel takes token-major [T, conv_dim] contiguous (vs the Triton path's transposed view).
        conv_out = gdn.causal_conv1d_fwd(
            mixed_qkv.float().contiguous(),
            self._conv_weights().float(),
            None,  # bias-free
            query_start_loc,
            state_indices.long(),
            has_initial_state.to(torch.uint8),
            conv_state,
            1,  # SiLU
        )
        # Chunked gated-delta-rule: l2norm(q,k) + g/beta from (a,b,A_log,dt_bias) are folded INTO
        # the kernel (replacing fused_post_conv_prep + chunk_gated_delta_rule). State written in
        # place. The chunked form is the throughput path (intra-chunk parallel); it is numerically
        # equal to the recurrent gdn_prefill (validated max|Δ|~2e-7, the oracle), which remains the
        # fallback/reference op.
        q, k, v = self._split_conv_qkv(conv_out, n)
        core = gdn.gdn_prefill_chunked(
            q, k, v, a.float(), b.float(), self.A_log, self.dt_bias,
            query_start_loc, state_indices.long(), has_initial_state.to(torch.uint8),
            ssm_state, self.head_k_dim ** -0.5, 1,
        )  # [T, num_v_heads, head_v_dim] fp32
        return self._output_projection(core, z, n)

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

        # One-step depthwise causal conv update (state roll) + SiLU; conv_state (fp32) in place.
        conv_out = gdn.causal_conv1d_update(
            mixed_qkv.float().contiguous(),
            self._conv_weights().float(),
            None,  # bias-free
            conv_state,
            state_indices.long(),
            1,  # SiLU
        )
        # One-step gated-delta-rule (l2norm + g/beta folded in); ssm_state updated in place per slot.
        q, k, v = self._split_conv_qkv(conv_out, n)
        core = gdn.gdn_decode(
            q, k, v, a.float(), b.float(), self.A_log, self.dt_bias,
            ssm_state, state_indices.long(), self.head_k_dim ** -0.5, 1,
        )  # [B, num_v_heads, head_v_dim] fp32
        return self._output_projection(core, z, n)

    # ---- warmup hook — no-op now that the conv is AOT HIP (no Triton autotune to settle) ----
    @torch.no_grad()
    def warmup_conv(self, num_tokens: int, *, iters: int = 2) -> None:
        """The Triton causal_conv1d_fn autotuned in place on its first call (NaN/0/OOM risk), so it
        had to be warmed per process. gdn_hip's conv is AOT-compiled HIP — no JIT, no autotune —
        so there is nothing to warm. Kept as a no-op for engine API compatibility."""
        return
