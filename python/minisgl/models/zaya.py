"""ZAYA1-8B — CCA-hybrid MoE decoder (Step 1: scaffold + registration + model shell).

The 80-layer tower interleaves two mixer families by global layer index:
  * EVEN layer (lid % 2 == 0) -> CCA cross-channel attention: a depthwise+grouped causal-conv
    front-end (the vendored `torch.ops.zaya_cca` kernels) feeding partial-rotary paged GQA. Conv
    recurrent state lives in `ctx.cca_state`, indexed by the layer's CCA-attn id (cca_layer_id),
    which is ALSO the AttentionLayer layer_id (paged KV pool slice). 40 CCA layers.
  * ODD layer (lid % 2 == 1) -> EDA/MOD MoE: a `ZayaRouter` (down_proj -> EDA add -> RMSNorm ->
    GELU MLP -> top-1 over 16 experts + a "skip" MOD expert) gating `MoELayer` experts. The router
    hidden state threads forward to the NEXT MoE layer's EDA. 40 MoE layers.

ZAYA's residual scheme is NON-STANDARD (residual_in_fp32 + scale_residual_merge): the residual is
the full fp32 stream, each layer merges (optionally affine-scaled) residual + hidden in fp32 then
RMSNorms the MERGED stream as the mixer input, and RETURNS (mixer_output, residual) — the add back
happens at the TOP of the next layer, not in-layer. So this file does NOT reuse `RMSNormFused`'s
pre-norm pattern; it carries an explicit fp32 accumulator and a plain `RMSNorm` per layer.

Step 1 lands the shell: config plumbing, the class hierarchy, the residual + EDA forward loop, the
final fp32 merge + final_norm + tied lm_head, and the engine seams (`iter_cca_layers`). The CCA and
MoE compute bodies are stubbed (raise NotImplementedError) — Steps 2-3 fill them in (see the module
docstring's "remaining stubs" and PORT_PLAN.md §6-§7).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn as nn
from minisgl.core import get_global_ctx
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearOProj,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
)
from minisgl.layers.norm import _rms_norm
from minisgl.utils import init_logger, nvtx_annotate

from .base import BaseLLMModel

logger = init_logger(__name__)

if TYPE_CHECKING:
    from .config import ModelConfig


def _concat(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


# =====================================================================================
# Residual scaling (scale_residual_merge) — affine on the fp32 residual stream.
# =====================================================================================
class ResidualScaling(BaseOP):
    """Per-layer affine applied to the fp32 residual stream BEFORE the merge+norm (PORT_PLAN §3.1).

        hidden_states = (hidden_states.float() + hs_bias) * hs_scale
        if layer_n != 0 and residual is not None:
            residual  = (residual.float()  + res_bias) * res_scale

    Layer 0 owns ONLY the hidden_states affine (no residual params — `not_first_layer=False`). The
    FINAL top-level merge (layer_n == num_hidden_layers) is a full one (has residual params).
    Checkpoint keys: `layers.{i}.res_scale.*` and top-level `model.res_scale.*`."""

    def __init__(self, hidden_size: int, layer_n: int):
        self._layer_n = layer_n
        self._has_residual = layer_n != 0
        self.hidden_states_scale = torch.empty(hidden_size, dtype=torch.float32)
        self.hidden_states_bias = torch.empty(hidden_size, dtype=torch.float32)
        if self._has_residual:
            self.residual_scale = torch.empty(hidden_size, dtype=torch.float32)
            self.residual_bias = torch.empty(hidden_size, dtype=torch.float32)

    def forward(
        self, residual: torch.Tensor | None, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor | None, torch.Tensor]:
        hidden_states = (hidden_states.float() + self.hidden_states_bias) * self.hidden_states_scale
        if self._has_residual and residual is not None:
            residual = (residual.float() + self.residual_bias) * self.residual_scale
        return residual, hidden_states


# =====================================================================================
# CCA conv front-end bridge (nn.Module so its fp32-preserving params load via assign=True).
# =====================================================================================
class CCAConv(nn.Module):
    """The CCA `qkv` module: input projections, the two causal convs, and the per-head temperature.

    Held as plain `nn.Parameter`s (loaded with assign=True, the GDN bridge pattern) so dtype is
    preserved — the conv weights / `temp` are consumed fp32 by the kernel and must NOT be bf16-cast.
    `post_load` pre-computes the cached fp32 kernel weights (w0, b0, w1 transposed, b1, temp_eff)
    ONCE so the hot path never re-derives them (PORT_PLAN §1.1).

    Checkpoint prefix: `self_attn.qkv.*` (the module is literally named `qkv` in the checkpoint)."""

    def __init__(self, config: ModelConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        hidden = config.hidden_size
        nq, nk, hd = config.cca_num_q_heads, config.cca_num_k_heads, config.cca_head_dim
        latent_q = nq * hd  # 1024
        latent_k = nk * hd  # 256
        conv_dim = config.cca_conv_dim  # C = 1280
        k0, k1 = config.cca_time0, config.cca_time1  # 2, 2
        ng = nq + nk  # grouped-conv groups H = 10

        f = dict(dtype=dtype, device=device)
        # Input projections (bf16 dense). 2048 -> {q:1024, k:256, v1:128, v2:128}.
        self.linear_q = nn.Parameter(torch.empty(latent_q, hidden, **f))
        self.linear_k = nn.Parameter(torch.empty(latent_k, hidden, **f))
        self.val_proj1 = nn.Parameter(torch.empty(hd, hidden, **f))
        self.val_proj2 = nn.Parameter(torch.empty(hd, hidden, **f))
        # Causal convs over the q|k concat (depthwise conv_qk.0, grouped conv_qk.1).
        self.conv_qk_0_weight = nn.Parameter(torch.empty(conv_dim, 1, k0, **f))
        self.conv_qk_0_bias = nn.Parameter(torch.empty(conv_dim, **f))
        self.conv_qk_1_weight = nn.Parameter(torch.empty(conv_dim, hd, k1, **f))
        self.conv_qk_1_bias = nn.Parameter(torch.empty(conv_dim, **f))
        # Per-head temperature (fp32-preserving; kernel exponentiates if clamp_temp).
        self.temp = nn.Parameter(torch.empty(nk, **f))  # [2]

        self._nq, self._nk, self._hd, self._ng = nq, nk, hd, ng
        self._latent_q, self._latent_k = latent_q, latent_k
        self._conv_dim = conv_dim
        self._clamp_temp = bool(getattr(config, "cca_clamp_temp", False))
        # fp32 kernel-weight cache, filled by post_load (constants — derived once, reused every step).
        self._w0: torch.Tensor | None = None  # [C, K0]
        self._b0: torch.Tensor | None = None  # [C]
        self._w1: torch.Tensor | None = None  # [H, d_out, d_in, K1] (pre-transposed for coalescing)
        self._b1: torch.Tensor | None = None  # [C]
        self._temp_eff: torch.Tensor | None = None  # [num_k_heads] fp32

    @torch.no_grad()
    def post_load(self) -> None:
        # Cache the fp32 kernel weights ONCE (they are frozen after the state-dict load). Mirrors the
        # reference `CCA._conv_weights_fp32` / `_temp_eff` (cca.py:1036-1074):
        #   w0 = conv_qk.0.weight.squeeze(1)               -> [C, K0]
        #   w1 = conv_qk.1.weight [C=H*d, d_in, K1] viewed [H, d_out, d_in, K1] (dim 1 comes from
        #        splitting C and IS the output channel) then permuted [H, d_in, d_out, K1] (swaps dims
        #        1 & 2) for the coalesced layout the built cca_kernel.hip expects. d_out == d_in ==
        #        head_dim, so the two 128-dims look alike but the permute is load-bearing.
        #   temp_eff = exp(clamp(temp,1e-7,2.0)) if clamp_temp else temp.
        self._w0 = self.conv_qk_0_weight.squeeze(1).float().contiguous()  # [C, K0]
        self._b0 = self.conv_qk_0_bias.float().contiguous()  # [C]
        self._b1 = self.conv_qk_1_bias.float().contiguous()  # [C]
        w1 = self.conv_qk_1_weight.float()  # [C = H*d, d_in, K1]
        c_out, d_in, k1 = w1.shape
        num_heads = c_out // d_in  # groups == heads, d_out == d_in == head_dim
        self._w1 = (
            w1.view(num_heads, d_in, d_in, k1).permute(0, 2, 1, 3).contiguous()
        )  # [H, d_out, d_in, K1]
        t = self.temp.float()
        if self._clamp_temp:
            t = torch.exp(torch.clamp(t, 1e-7, 2.0))
        self._temp_eff = t.contiguous()  # [num_k_heads]

    def conv_weights_fp32(self):
        """(w0, b0, w1, b1) fp32 kernel weights — built by post_load."""
        assert self._w0 is not None, "CCAConv.post_load() must run before forward"
        return self._w0, self._b0, self._w1, self._b1

    def temp_eff(self) -> torch.Tensor:
        assert self._temp_eff is not None, "CCAConv.post_load() must run before forward"
        return self._temp_eff


# =====================================================================================
# Router bridge (nn.Module — never quantized; bf16/fp32 params via assign=True load).
# =====================================================================================
class ZayaRouter(nn.Module):
    """The EDA/MOD router: down_proj -> (EDA add) -> RMSNorm(eda) -> GELU MLP -> top-1 over
    num_experts + a MOD "skip" expert (PORT_PLAN §7.1). Threads `router_states` forward to the next
    MoE layer's EDA. EDA is OFF on the FIRST MoE layer (layer_number == 1).

    Checkpoint prefix: `zaya_block.router.*`."""

    def __init__(self, config: ModelConfig, *, use_eda: bool, dtype: torch.dtype,
                 device: torch.device):
        super().__init__()
        hidden = config.hidden_size
        r = config.zaya_mlp_expansion  # 256
        ne = config.num_experts  # 16
        self._use_eda = use_eda
        self._num_experts = ne
        self._eps = config.rms_norm_eps

        f = dict(dtype=dtype, device=device)
        self.down_proj_weight = nn.Parameter(torch.empty(r, hidden, **f))
        self.down_proj_bias = nn.Parameter(torch.empty(r, **f))
        self.rmsnorm_eda_weight = nn.Parameter(torch.empty(r, **f))
        if use_eda:
            self.router_states_scale = nn.Parameter(torch.empty(r, **f))
        # router_mlp: Linear(r->r)+b, GELU, Linear(r->r)+b, GELU, Linear(r->ne+1) NO bias.
        self.router_mlp_0_weight = nn.Parameter(torch.empty(r, r, **f))
        self.router_mlp_0_bias = nn.Parameter(torch.empty(r, **f))
        self.router_mlp_2_weight = nn.Parameter(torch.empty(r, r, **f))
        self.router_mlp_2_bias = nn.Parameter(torch.empty(r, **f))
        self.router_mlp_4_weight = nn.Parameter(torch.empty(ne + 1, r, **f))  # +1 = MOD skip
        # balancing_biases [ne+1] fp32 (the skip/MOD slot's bias is baked into the checkpoint value);
        # affects CHOICE only. fp32 to match the engine's fp32 upcast of `.balancing_biases`. zeros
        # (not empty) so an unloaded buffer is inert rather than NaN; load via assign=True overwrites.
        self.register_buffer("balancing_biases", torch.zeros(ne + 1, dtype=torch.float32))

    def forward(
        self, hidden_states: torch.Tensor, prev_router_states: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-1 EDA/MOD route (mirror reference ZayaRouter.forward, zaya.py:384-447).

        Returns (route_prob[N,1] fp32, expert_idx[N,1] int64, router_states_next[N,r]).
        `expert_idx` may equal `num_experts` (the MOD skip slot); the caller clamps it for the
        expert kernel and masks the skip rows. `balancing_biases` steer the CHOICE only — the
        returned probability is the UN-biased softmax prob at the chosen expert. route_prob is fp32
        (the fused-MoE topk-weight convention; the kernel casts to compute dtype internally)."""
        import torch.nn.functional as F

        hs = F.linear(hidden_states, self.down_proj_weight, self.down_proj_bias)  # [N, r]
        if self._use_eda and prev_router_states is not None:
            hs = hs + prev_router_states * self.router_states_scale
        # Stash the PRE-norm router state — this is what threads to the next MoE layer's EDA.
        router_states_next = hs.clone()

        hs_norm = _rms_norm(hs, self.rmsnorm_eda_weight, self._eps)  # RMSNorm(r), no residual
        x = F.linear(hs_norm, self.router_mlp_0_weight, self.router_mlp_0_bias)
        x = F.gelu(x)
        x = F.linear(x, self.router_mlp_2_weight, self.router_mlp_2_bias)
        x = F.gelu(x)
        logits = F.linear(x, self.router_mlp_4_weight)  # [N, ne+1] (no bias)

        # zaya_high_prec -> fp32 softmax (selection-stable); biases affect CHOICE only.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)  # [N, ne+1]
        # biases steer the CHOICE only; detach so they never feed gradients (reference zaya.py:427:
        # `expert_prob.detach().to(torch.float32) + self.balancing_biases`). probs is already fp32.
        biased = probs.detach() + self.balancing_biases  # [N, ne+1] fp32
        expert_idx = torch.topk(biased, 1, dim=-1).indices  # [N, 1] int64 (may select skip == ne)
        route_prob = torch.gather(probs, 1, expert_idx).contiguous()  # [N, 1] fp32 un-biased prob
        return route_prob, expert_idx, router_states_next


# =====================================================================================
# CCA attention mixer (conv front-end + partial-rotary paged GQA).
# =====================================================================================
class ZayaCCAAttn(BaseOP):
    """EVEN-layer mixer. Holds the `CCAConv` bridge (checkpoint `self_attn.qkv.*`), the paged
    `AttentionLayer` (partial-rotary GQA; q/k already RMS-normed by the conv kernel), and `o_proj`
    (checkpoint `self_attn.o_proj.*`, 1024 -> 2048)."""

    def __init__(self, config: ModelConfig, cca_layer_id: int):
        self._cca = CCAConv(config, dtype=torch.get_default_dtype(), device=torch.device("meta"))
        self._cca_layer_id = cca_layer_id
        nqo, nkv, hd = config.cca_num_q_heads, config.cca_num_k_heads, config.cca_head_dim
        # q/k are pre-normed by the CCA kernel -> q_norm/k_norm = None. The CCA-attn id is BOTH the
        # conv-state index and the paged-KV layer_id (contiguous over attention-bearing layers).
        self.attn = AttentionLayer(
            layer_id=cca_layer_id,
            head_dim=hd,
            num_qo_heads=nqo,
            num_kv_heads=nkv,
            rotary_config=config.rotary_config,  # partial rotary (rotary_dim = head_dim * 0.5)
            q_norm=None,
            k_norm=None,
        )
        self.o_proj = LinearOProj(nqo * hd, config.hidden_size, has_bias=False)
        self._sqrt_head_dim = float(hd) ** 0.5  # kernel sqrt_d arg (= sqrt(head_dim), NOT inverse)

    @nvtx_annotate("CCA")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """CCA mixer forward (PORT_PLAN §6). minisgl is token-major ([N, hidden], no batch dim) and
        HOMOGENEOUS (prefill XOR decode), so this dispatches purely on ``batch.phase`` — no mixed
        batch, no decode/prefill split. The vendored kernel bakes grouped-means + per-head RMS-norm +
        key-temperature into its normalized q|k output (``qk_out`` columns [0:latent_q]=q,
        [latent_q:C]=k); v is built model-side from val_proj1(current hs) ‖ val_proj2(previous hs)."""
        import torch.nn.functional as F

        ctx = get_global_ctx()
        state = ctx.cca_state
        md = ctx.batch.cca_metadata
        assert state is not None and md is not None, "CCA layer needs ctx.cca_state + batch.cca_metadata"
        conv = state.conv(self._cca_layer_id)  # [num_slots, C, conv_width] fp32 (mutated in place)
        prev = state.prev(self._cca_layer_id)  # [num_slots, hidden] fp32

        cca = self._cca
        nq, gqa = cca._nq, cca._nq // cca._nk
        latent_q, latent_k = cca._latent_q, cca._latent_k
        sqrt_d = float(self._sqrt_head_dim)
        w0, b0, w1, b1 = cca.conv_weights_fp32()
        temp_eff = cca.temp_eff()

        hs = hidden_states  # [N, hidden]
        model_dtype = hs.dtype
        N = hs.shape[0]

        # Packed q|k for the conv kernel (fp32). v is computed AFTER the kernel (it needs prev_hs).
        q = F.linear(hs, cca.linear_q)  # [N, latent_q]
        k = F.linear(hs, cca.linear_k)  # [N, latent_k]
        qk_new = torch.cat([q, k], dim=-1).float().contiguous()  # [N, C] fp32

        if md.is_prefill:
            qsl = md.query_start_loc  # int32 [num_seqs+1]
            device = hs.device
            num_seqs = md.num_seqs
            # The per-token tensors below are sized by N; they only line up with the cu_seqlens if the
            # batch is exactly token-major (no padding rows — eager v0). Catch a padded/malformed
            # batch here rather than as a cryptic kernel IndexError downstream.
            assert int(qsl[-1].item()) == N, f"prefill cu_seqlens end {int(qsl[-1])} != num_tokens {N}"
            assert md.state_indices.shape[0] == num_seqs, (
                f"state_indices {tuple(md.state_indices.shape)} must have num_seqs={num_seqs} entries"
            )
            seq_lens = (qsl[1:] - qsl[:-1]).to(torch.int64)  # [num_seqs]
            req_id = torch.repeat_interleave(
                torch.arange(num_seqs, device=device, dtype=torch.int32), seq_lens, output_size=N
            )  # [N] int32
            rid = req_id.long()  # gather index, reused below
            tok = torch.arange(N, device=device, dtype=torch.int32)  # [N] local token index
            seg_pos = (tok - qsl[:-1][rid]).to(torch.int32)  # position within each sequence
            slot = md.state_indices.to(torch.int64)[rid]  # [N] conv slot per token
            is_last = tok == (qsl[1:] - 1)[rid]  # [N] bool — last token of each seq
            # init_states: gather cached conv per seq, zeroed where no initial state (fresh request).
            has_init = md.has_initial_state  # bool [num_seqs]
            init_states = conv[md.state_indices.to(torch.int64)].float()  # [num_seqs, C, W]
            init_states = torch.where(
                has_init.view(-1, 1, 1), init_states, init_states.new_zeros(())
            ).contiguous()
            qk_out = torch.ops.zaya_cca.cca_prefill_qk(
                qk_new, conv, init_states, seg_pos, req_id, slot, is_last,
                w0, b0, w1, b1, temp_eff, nq, gqa, latent_q, sqrt_d,
            )  # [N, C] normalized q|k; conv updated in place (only each seq's last token)
            # hs2 = previous-token hidden, per-seq shift; first token seeded by prev[slot] (zero fresh).
            hs2 = torch.empty_like(hs)
            hs2[1:] = hs[:-1]
            init_hs = torch.where(
                has_init.view(-1, 1), prev[md.state_indices.to(torch.int64)].to(model_dtype),
                hs.new_zeros(()),
            )
            seg_start = qsl[:-1].to(torch.int64)
            hs2[seg_start] = init_hs
            # store each seq's LAST hidden for the next chunk/step.
            prev[md.state_indices.to(torch.int64)] = hs[(qsl[1:] - 1).to(torch.int64)].float()
        else:
            slot = md.state_indices.to(torch.int64)  # [N]; real seqs >= 1, graph-pad rows == 0
            # Slot 0 is the reserved NULL block, used for graph-capture padding rows; the kernel skips
            # them via is_pad. Compute is_pad BEFORE the (host-syncing) sanity checks so the all-pad
            # dummy capture batch validates vacuously, and skip those checks entirely under cudagraph
            # capture/replay (a .all()/[mask] host sync is illegal inside a captured stream).
            is_pad = slot == 0  # graph-capture padding rows (none in eager)
            if not torch.cuda.is_current_stream_capturing():
                real = slot[~is_pad]  # real (non-pad) sequences only
                assert bool((real >= 1).all()), "decode: non-pad state_indices must be >= 1 (slot 0 is NULL)"
                assert bool((real < state.num_slots).all()), (
                    f"decode: state_indices out of range [1, {state.num_slots})"
                )
            qk_out = torch.ops.zaya_cca.cca_decode_qk(
                qk_new, conv, slot, is_pad,
                w0, b0, w1, b1, temp_eff, nq, gqa, latent_q, sqrt_d,
            )  # [N, C] normalized q|k; conv rolled+appended in place
            # previous-hidden for val_proj2 = the cached prev_hs of each slot, THEN store current.
            hs2 = prev[slot].to(model_dtype)  # [N, hidden] (OLD prev_hs)
            prev[slot] = hs.float()

        # Values from the two time streams: val_proj1 on current hs, val_proj2 on previous hs.
        v1 = F.linear(hs, cca.val_proj1)  # [N, head_dim] = latent_k/2
        v2 = F.linear(hs2, cca.val_proj2)  # [N, head_dim]
        v = torch.cat([v1, v2], dim=-1)  # [N, latent_k]

        # Assemble qkv for the paged GQA attention (q|k pre-normed by the kernel -> q_norm/k_norm None).
        qf = qk_out[:, :latent_q].to(model_dtype)
        kf = qk_out[:, latent_q : latent_q + latent_k].to(model_dtype)
        qkv = torch.cat([qf, kf, v.to(model_dtype)], dim=-1)  # [N, latent_q + 2*latent_k]
        o = self.attn.forward(qkv)  # partial RoPE(0.5) + store K/V to paged pool + GQA attention
        return self.o_proj.forward(o)  # [N, hidden]

    def warmup_conv(self, num_tokens: int) -> None:
        # CCA kernel has no autotune; warmup is a no-op (kept for engine-loop symmetry with GDN).
        pass

    # ---- BaseOP <-> nn.Module state bridge (the CCAConv params live in nn._parameters) ----
    def state_dict(self, *, prefix: str = "", result=None):
        result = result if result is not None else {}
        # o_proj is a BaseOP -> default walk; the CCAConv params go under the `qkv.*` checkpoint name.
        self.o_proj.state_dict(prefix=_concat(prefix, "o_proj"), result=result)
        self.attn.state_dict(prefix=_concat(prefix, "attn"), result=result)
        for name, tensor in self._cca.state_dict().items():
            result[_concat(prefix, _concat("qkv", name))] = tensor
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        self.o_proj.load_state_dict(state_dict, prefix=_concat(prefix, "o_proj"), _internal=True)
        self.attn.load_state_dict(state_dict, prefix=_concat(prefix, "attn"), _internal=True)
        qkv_prefix = _concat(prefix, "qkv")
        sub = {name: state_dict.pop(_concat(qkv_prefix, name)) for name in self._cca.state_dict()}
        missing, unexpected = self._cca.load_state_dict(sub, strict=True, assign=True)
        assert not missing and not unexpected, (missing, unexpected)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def post_load(self) -> None:
        # Importing cca_hip registers torch.ops.zaya_cca.{cca_decode_qk,cca_prefill_qk}. Done at
        # load time (not lazily in forward) so a missing/unbuilt .so fails during model load with a
        # clear error, not cryptically on the first decode step. Kept out of module import so
        # `import minisgl.models.zaya` stays GPU-free (the .so only loads when a model is built).
        import cca_hip.cca_op  # noqa: F401

        self.o_proj.post_load()
        self._cca.post_load()


# =====================================================================================
# MoE mixer (EDA/MOD router + grouped experts).
# =====================================================================================
class ZayaMoEBlock(BaseOP):
    """ODD-layer mixer (checkpoint `zaya_block.*`). The `ZayaRouter` computes the top-1 route in the
    model; `MoELayer` runs the experts via the precomputed-route path. MOD: a "skip" expert (index
    num_experts) whose output is `input * route_prob` (PORT_PLAN §7.2)."""

    def __init__(self, config: ModelConfig, *, use_eda: bool):
        self._use_eda = use_eda
        self._use_mod = config.zaya_use_mod
        self._num_experts = config.num_experts  # skip slot (MOD) == this index
        self.router = ZayaRouter(
            config, use_eda=use_eda, dtype=torch.get_default_dtype(),
            device=torch.device("meta"),
        )
        # Experts stay fp8 (F8_E4M3 + per-channel F32 scale, ~8 GB): dequant-to-bf16 at load is
        # ~16 GB and OOMs the 16 GB card. Weights are dequantized per-expert at compute and run on
        # the unquantized fused Triton path (PORT_PLAN §7.3; native W8A8-fp8 kernel is Step 3 v1).
        # renormalize=False, silu.
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,  # 1
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,  # ffn_hidden_size // 2 = 2048
            renormalize=False,
            activation="silu",
            fp8_experts=True,
        )

    @nvtx_annotate("ZayaMoE")
    def forward(
        self, hidden_states: torch.Tensor, prev_router_states: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-1 expert dispatch + MOD skip (mirror reference ZayaBlock.forward, zaya.py:514-539).

        The route is computed model-side; experts run via the precomputed-topk path. MOD: the
        "skip" expert (index num_experts) outputs `input * route_prob` instead of an expert MLP.
        Returns (mixer_output[N,H], router_states_next[N,r] threaded to the next MoE layer)."""
        route_prob, expert_idx, router_states_next = self.router.forward(
            hidden_states, prev_router_states
        )  # [N,1] fp32, [N,1] int64, [N,r]

        ne = self._num_experts
        if self._use_mod:
            # Skip slot (== ne) has no real expert weights -> clamp to a valid id for the kernel,
            # then mask its rows back out and replace with the scaled-input MOD output.
            clamped_idx = torch.clamp(expert_idx, 0, ne - 1).to(torch.int32)
            experts_out = self.experts.forward(
                hidden_states, topk_weights=route_prob, topk_ids=clamped_idx
            )  # [N,H] model-dtype
            prob = route_prob.to(hidden_states.dtype)  # [N,1] gate, in compute dtype
            mod_out = hidden_states * prob  # [N,H] skip-expert output (gated residual)
            mask = (expert_idx != ne).to(hidden_states.dtype)  # [N,1] 1.0 where a real expert ran
            out = mask * experts_out + (1.0 - mask) * mod_out
        else:
            # The router always emits ne+1 logits (the MOD skip slot at index ne exists even when
            # MOD is disabled), so top-1 can still land on `ne`. Without MOD there is no skip output
            # to substitute, but the id MUST be clamped to a real expert before the kernel gathers
            # weights — an unclamped `ne` indexes past the last expert (OOB gather / device assert).
            clamped_idx = torch.clamp(expert_idx, 0, ne - 1).to(torch.int32)
            out = self.experts.forward(
                hidden_states, topk_weights=route_prob, topk_ids=clamped_idx
            )
        return out, router_states_next

    # ---- state bridge: router params live in nn._parameters; experts is a BaseOP ----
    def state_dict(self, *, prefix: str = "", result=None):
        result = result if result is not None else {}
        self.experts.state_dict(prefix=_concat(prefix, "experts"), result=result)
        for name, tensor in self.router.state_dict().items():
            result[_concat(prefix, _concat("router", name))] = tensor
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        self.experts.load_state_dict(state_dict, prefix=_concat(prefix, "experts"), _internal=True)
        router_prefix = _concat(prefix, "router")
        sub = {name: state_dict.pop(_concat(router_prefix, name)) for name in self.router.state_dict()}
        missing, unexpected = self.router.load_state_dict(sub, strict=True, assign=True)
        assert not missing and not unexpected, (missing, unexpected)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def post_load(self) -> None:
        self.experts.post_load()


# =====================================================================================
# Decoder layer — one class, branches on is_cca. res_scale is LAYER-level.
# =====================================================================================
class ZayaDecoderLayer(BaseOP):
    """One tower layer. The fp32 merge-and-norm (PORT_PLAN §3) runs at the TOP of every layer:

        residual, hidden = res_scale(residual, hidden)        # affine on the fp32 stream
        residual = (residual or 0) + hidden                   # fp32 merge
        hidden   = input_norm(residual).to(dtype)             # RMSNorm over the MERGED stream
        hidden   = mixer(hidden)                              # CCA or MoE
        return hidden, residual[, router_states_next]         # mixer output -> next layer's input

    The add-back happens at the NEXT layer's top, not in-layer."""

    def __init__(self, config: ModelConfig, layer_id: int, *, is_cca: bool,
                 cca_layer_id: int | None, use_eda: bool):
        self._layer_id = layer_id
        self._is_cca = is_cca
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # scale_residual_merge: the per-layer affine on the fp32 residual stream is OPTIONAL — it
        # exists only when the config flags it (reference ZayaDecoder*Layer.__init__, zaya.py:256/577).
        # When off, the merge is a plain fp32 add (no affine). Default True (ZAYA1-8B ships it on).
        self._scale_residual_merge = bool(getattr(config, "scale_residual_merge", True))
        if self._scale_residual_merge:
            self.res_scale = ResidualScaling(config.hidden_size, layer_n=layer_id)
        if is_cca:
            assert cca_layer_id is not None
            self.self_attn = ZayaCCAAttn(config, cca_layer_id)
            self._mixer: BaseOP = self.self_attn
        else:
            self.zaya_block = ZayaMoEBlock(config, use_eda=use_eda)
            self._mixer = self.zaya_block

    def _merge_and_norm(
        self, residual: torch.Tensor | None, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._scale_residual_merge:
            # ResidualScaling already upcasts both streams to fp32 (its affine runs in fp32).
            residual, hidden_states = self.res_scale.forward(residual, hidden_states)
            residual = hidden_states if residual is None else residual + hidden_states
        else:
            # No affine: plain fp32 merge (reference zaya.py:274-277). Force fp32 so the residual
            # stream stays fp32 even on layer 0 (residual is None -> seed = hidden.float()).
            residual = (
                hidden_states.float()
                if residual is None
                else residual.float() + hidden_states.float()
            )
        # RMSNorm runs over the fp32 merged stream; the mixer (bf16 projections) needs the normed
        # input in the model compute dtype (reference: input_norm(residual).to(layer_dtype)).
        normed = self.input_norm.forward(residual.float()).to(self.input_norm.weight.dtype)
        return normed, residual

    @nvtx_annotate("ZayaLayer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        prev_router_states: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden_states, residual = self._merge_and_norm(residual, hidden_states)
        if self._is_cca:
            hidden_states = self.self_attn.forward(hidden_states)
            return hidden_states, residual, prev_router_states  # CCA passes router state through
        hidden_states, router_states_next = self.zaya_block.forward(
            hidden_states, prev_router_states
        )
        return hidden_states, residual, router_states_next


# =====================================================================================
# Model tower.
# =====================================================================================
class ZayaModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        cca_pos = {lid: pos for pos, lid in enumerate(config.cca_layer_ids)}
        # EDA is OFF on the FIRST MoE layer (global layer_id == 1); on for every later MoE layer.
        first_moe_lid = next((lid for lid in range(config.num_layers) if lid % 2 == 1), None)
        self.layers = OPList(
            [
                ZayaDecoderLayer(
                    config, lid,
                    is_cca=(lid % 2 == 0),
                    cca_layer_id=cca_pos.get(lid),
                    use_eda=(config.zaya_use_eda and lid % 2 == 1 and lid != first_moe_lid),
                )
                for lid in range(config.num_layers)
            ]
        )
        # The final top-level res_scale merge (layer_n == num_hidden_layers) is a FULL one — but only
        # when scale_residual_merge is on (reference ZayaModel.__init__, zaya.py:674-675).
        self._scale_residual_merge = bool(getattr(config, "scale_residual_merge", True))
        if self._scale_residual_merge:
            self.res_scale_final = ResidualScaling(config.hidden_size, layer_n=config.num_layers)
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Aux-hidden capture seam (spec-decode); OFF by default (zero cost).
        self._capture_layer_ids: List[int] | None = None

    def set_capture_layers(self, ids: List[int] | None) -> None:
        self._capture_layer_ids = list(ids) if ids else None

    def forward(
        self, input_ids: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        prev_router_states: torch.Tensor | None = None  # EDA: threaded across MoE layers only
        cap = self._capture_layer_ids if return_hidden else None
        cap_set = set(cap) if cap else None
        grabbed: dict[int, torch.Tensor] = {}
        for lid, layer in enumerate(self.layers.op_list):
            hidden_states, residual, prev_router_states = layer.forward(
                hidden_states, residual, prev_router_states
            )
            if cap_set is not None and lid in cap_set:
                grabbed[lid] = residual.clone()
        # Final fp32 merge + final_norm (PORT_PLAN §3 final block; reference zaya.py:723-736). The
        # merged sum is normed; whether it lands in `residual` or `hidden_states` is irrelevant
        # (addition is commutative and the norm sees the same fp32 value either way).
        if self._scale_residual_merge:
            residual, hidden_states = self.res_scale_final.forward(residual, hidden_states)
            merged = hidden_states if residual is None else residual + hidden_states
        else:
            merged = (
                hidden_states.float()
                if residual is None
                else hidden_states.float() + residual.float()
            )
        # final_norm over the fp32 merged stream; lm_head (bf16, tied to embed) needs compute dtype.
        final = self.final_norm.forward(merged.float()).to(self.final_norm.weight.dtype)
        if return_hidden:
            aux_stack = torch.stack([grabbed[i] for i in cap], dim=0) if cap else None
            return final, aux_stack
        return final


# =====================================================================================
# Top-level *ForCausalLM.
# =====================================================================================
class ZayaForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = ZayaModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    # no_grad mirrors the GDN layer idiom (gdn/layer.py): the CCA/router nn.Parameters default to
    # requires_grad=True, so without this the cudagraph-capture forward (which, unlike the scheduler's
    # @inference_mode forward, has no grad guard) tracks grad and in-place KV/state writes raise
    # "a leaf Variable that requires grad is being used in an in-place operation".
    @torch.no_grad()
    def forward(self, return_hidden: bool = False):
        input_ids = get_global_ctx().batch.input_ids
        if return_hidden:
            last_hidden, aux_hidden = self.model.forward(input_ids, return_hidden=True)
            return self.lm_head.forward(last_hidden), last_hidden, aux_hidden
        return self.lm_head.forward(self.model.forward(input_ids))

    def set_capture_layers(self, ids: List[int] | None) -> None:
        self.model.set_capture_layers(ids)

    def iter_cca_layers(self) -> List[ZayaCCAAttn]:
        """CCA mixers in cca_layer_id order (engine uses this to size cca_state + warmup)."""
        return [
            layer.self_attn
            for layer in self.model.layers.op_list
            if isinstance(getattr(layer, "self_attn", None), ZayaCCAAttn)
        ]


__all__ = ["ZayaForCausalLM"]
