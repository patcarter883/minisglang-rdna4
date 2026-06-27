"""Phase 3d-1 — the Qwen3.5 GDN-hybrid model (text path of Qwen3_5ForConditionalGeneration).

A pre-norm decoder of `num_layers` blocks where `config.layer_types[i]` selects the mixer:
  * "linear_attention" (the 3-in-4 majority) -> the parity-validated `QwenGatedDeltaNet`
    (gdn/layer.py), wrapped in a BaseOP bridge so its nn.Module params live in the minisgl
    state-dict/meta-load system. Per-sequence recurrent state (conv + ssm) is read from
    `ctx.gdn_state` by the layer's own GDN-slot index (gdn_layer_id), with the per-batch
    `ctx.batch.gdn_metadata` built by the scheduler.
  * "full_attention" (the 1-in-4) -> `Qwen3_5Attn`: gated, partial-rotary GQA — NOT the
    Phase-1 RopeAttn. q_proj emits 2*num_qo_heads*head_dim (q + a per-head sigmoid gate
    applied to the attention output); partial rotary (rotary_dim<head_dim) is handled by
    RotaryEmbedding; per-head q_norm/k_norm precede rotary.

MLP is the dense SwiGLU (`GatedMLP`); no MoE in the 4B (MoE variants reuse the Phase-2-MoE
path later). TP=1, GDN/attention projections unquantized bf16. Weight-name remap (HF
`model.language_model.*`, qkv/z + b/a concat, conv1d) is Phase 3d-3; this file fixes the
minisgl-native key layout the loader targets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.gdn.layer import QwenGatedDeltaNet
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    OPList,
    ParallelLMHead,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.quant import create_linear_method
from minisgl.utils import div_even, init_logger, nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP

logger = init_logger(__name__)

# --- env-gated per-layer GPU-time attribution (diagnostics only) -------------------------------
# MINISGL_LAYER_PROF=<N> times the mixer (GDN/attention) and the FFN (MoE) per layer with CUDA
# events, bucketed, and logs the per-step split every N forward steps. Splits the decode wall time
# across gdn / attn / ffn so we know which kernel family owns it. Inert (zero overhead) when unset.
import os as _os
from collections import defaultdict as _dd

_LP_EVERY = int(_os.environ["MINISGL_LAYER_PROF"]) if _os.environ.get("MINISGL_LAYER_PROF", "").isdigit() else 0
_lp_buckets: dict = _dd(float)
_lp_step = 0


def _lp_timed(bucket: str, fn, arg):
    if not _LP_EVERY:
        return fn(arg)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    r = fn(arg)
    e.record()
    e.synchronize()
    _lp_buckets[bucket] += s.elapsed_time(e)
    return r


def _lp_tick(layer_id: int) -> None:
    global _lp_step
    if not _LP_EVERY or layer_id != 0:
        return
    _lp_step += 1
    if _lp_step % _LP_EVERY == 0 and _lp_buckets:
        tot = sum(_lp_buckets.values()) or 1.0
        parts = "  ".join(
            f"{k}={v / _LP_EVERY:.2f}ms({100 * v / tot:.0f}%)"
            for k, v in sorted(_lp_buckets.items(), key=lambda x: -x[1])
        )
        logger.info_rank0(f"[layer-prof] per-step GPU over {_LP_EVERY} steps: {parts}")
        _lp_buckets.clear()

if TYPE_CHECKING:
    from .config import ModelConfig


def _concat(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class Qwen3_5Attn(BaseOP):
    """Gated, partial-rotary GQA. q_proj carries a per-head sigmoid output gate."""

    def __init__(self, config: ModelConfig, layer_id: int):
        head_dim = config.head_dim
        nqo, nkv = config.num_qo_heads, config.num_kv_heads
        qm = create_linear_method(config.quant)
        # q_proj emits q + gate (2x), interleaved per head: [head_h q | head_h gate].
        self.q_proj = LinearColParallelMerged(
            config.hidden_size, [2 * nqo * head_dim], has_bias=False, quant_method=qm
        )
        self.k_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False, quant_method=qm
        )
        self.v_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False, quant_method=qm
        )
        # Qwen3.5 RMSNorm uses the (1 + weight) gain convention (weight init 0), UNLIKE the dense
        # Qwen3 plain-weight norm. Applies to all Qwen3_5RMSNorm sites (q/k norm, input/post/final
        # decoder norms); the GDN's gated norm weight keeps the plain-weight convention (init 1).
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, plus_one=True)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, plus_one=True)
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=nqo,
            num_kv_heads=nkv,
            rotary_config=config.rotary_config,  # rotary_dim=64 -> partial rotary
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * nqo, config.hidden_size, has_bias=False, quant_method=qm
        )
        self._head_dim = head_dim
        # LOCAL qo-head count: q_proj is column-parallel, so under TP each rank emits nqo/tp heads
        # (q+gate). The forward reshape MUST use the local count, not the full nqo. (This TP path
        # was masked until the GDN compile stall was removed — Phase 4 / gdn_hip.)
        self._num_qo_heads = div_even(nqo, get_tp_info().size)

    @nvtx_annotate("MHA_gated")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        hd = self._head_dim
        qg = self.q_proj.forward(x).view(n, self._num_qo_heads, 2 * hd)
        q = qg[..., :hd].reshape(n, self._num_qo_heads * hd)
        gate = qg[..., hd:].reshape(n, self._num_qo_heads * hd)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        # AttentionLayer applies q_norm/k_norm (over head_dim) then partial rotary, then attn.
        o = self.attn.forward(torch.cat([q, k, v], dim=-1))
        o = o * torch.sigmoid(gate)
        return self.o_proj.forward(o)


class GDNLinearAttn(BaseOP):
    """BaseOP bridge around the nn.Module `QwenGatedDeltaNet`.

    The wrapped module's params are NOT discoverable by BaseOP's __dict__ walk (they live in
    nn.Module's _parameters), so state_dict / load_state_dict are delegated to the module.
    load uses assign=True so meta params are replaced by the real-device checkpoint tensors
    AND their dtype is preserved (A_log / dt_bias must stay fp32 — the engine's bf16 cast must
    skip them; enforced in Phase 3d-3's weight map)."""

    def __init__(self, gdn: QwenGatedDeltaNet, gdn_layer_id: int):
        self._gdn = gdn  # leading "_" -> hidden from BaseOP's default state-dict walk
        self._gdn_layer_id = gdn_layer_id
        # out_proj is row-parallel (each rank contracts its value_dim/tp shard): the GDN module
        # returns a PARTIAL hidden-size output per rank, so the bridge all-reduces it (a no-op at
        # tp_size=1). The collective lives here, not in the standalone GDN compute module.
        self._comm = DistributedCommunicator()
        self._tp_size = get_tp_info().size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        state = ctx.gdn_state
        assert state is not None, "GDN layer ran without ctx.gdn_state (engine wiring missing)"
        md = ctx.batch.gdn_metadata
        conv = state.conv(self._gdn_layer_id)
        ssm = state.ssm(self._gdn_layer_id)
        # A spec-decode VERIFY batch (phase "decode", extend_len = K+1 per seq) uses the varlen
        # recurrent path, like a prefill. The GDN state it leaves is "after the last verify token";
        # the scheduler snapshots + re-advances it to the accepted position (see SPEC_DECODE.md).
        if ctx.batch.spec_verify and getattr(md, "capture_verify_state", False):
            # Spec verify with per-token-state capture: bit-stable recurrent verify kernels that ALSO
            # emit the conv/ssm state after each token, so the scheduler can install the accepted-prefix
            # state directly (no snapshot, no 2x re-advance, bit-exact vs 1-token decode). The scratch
            # is stashed per gdn_layer_id on the metadata for the scheduler to read post-forward.
            out, conv_scr, ssm_scr = self._gdn.forward_prefill_verify(
                x, conv, ssm, md.query_start_loc, md.state_indices, md.has_initial_state,
                md.verify_max_qlen,
            )
            md.conv_scratch[self._gdn_layer_id] = conv_scr
            md.ssm_scratch[self._gdn_layer_id] = ssm_scr
        elif ctx.batch.is_prefill or ctx.batch.spec_verify:
            out = self._gdn.forward_prefill(
                x, conv, ssm, md.query_start_loc, md.state_indices, md.has_initial_state
            )
        else:
            out = self._gdn.forward_decode(x, conv, ssm, md.query_start_loc, md.state_indices)
        if self._tp_size > 1:
            out = self._comm.all_reduce(out)
        return out

    def warmup_conv(self, num_tokens: int) -> None:
        self._gdn.warmup_conv(num_tokens)

    # ---- BaseOP <-> nn.Module state bridge ----
    def state_dict(self, *, prefix: str = "", result=None):
        result = result if result is not None else {}
        for name, tensor in self._gdn.state_dict().items():
            result[_concat(prefix, name)] = tensor
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        sub = {name: state_dict.pop(_concat(prefix, name)) for name in self._gdn.state_dict()}
        missing, unexpected = self._gdn.load_state_dict(sub, strict=True, assign=True)
        assert not missing and not unexpected, (missing, unexpected)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def post_load(self) -> None:  # GDN keeps bf16/fp32 weights as-loaded; nothing to finalize
        pass


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        is_gdn: bool,
        gdn_layer_id: int | None,
        mlp_factory: Callable[[ModelConfig], BaseOP] = Qwen3MLP,
    ):
        if is_gdn:
            assert gdn_layer_id is not None
            gdn = QwenGatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                tp_size=get_tp_info().size,  # head-parallel: local heads + out_proj all-reduce
                eps=config.rms_norm_eps,
                dtype=torch.get_default_dtype(),  # bf16 under the engine's build context
                device=torch.device("meta"),  # built on meta; real tensors via load(assign=True)
            )
            self.linear_attn = GDNLinearAttn(gdn, gdn_layer_id)
            self._attn_op: BaseOP = self.linear_attn
        else:
            self.self_attn = Qwen3_5Attn(config, layer_id)
            self._attn_op = self.self_attn
        # Dense SwiGLU for the 4B; the MoE variants pass a sparse-block factory (the MLP is the
        # ONLY structural difference between qwen3_5 and qwen3_5_moe decoder layers).
        self.mlp = mlp_factory(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps, plus_one=True
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size, eps=config.rms_norm_eps, plus_one=True
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _lp_tick(self._layer_id)
        x, residual = self.input_layernorm.forward(x, residual)
        mixer = "gdn" if type(self._attn_op).__name__ == "GDNLinearAttn" else "attn"
        x = _lp_timed(mixer, self._attn_op.forward, x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = _lp_timed("ffn", self.mlp.forward, x)
        return x, residual


class Qwen3_5Model(BaseOP):
    def __init__(
        self, config: ModelConfig, *, mlp_factory: Callable[[ModelConfig], BaseOP] = Qwen3MLP
    ):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        gdn_pos = {gid: pos for pos, gid in enumerate(config.gdn_layer_ids)}
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(
                    config, lid, is_gdn=lid in gdn_pos, gdn_layer_id=gdn_pos.get(lid),
                    mlp_factory=mlp_factory,
                )
                for lid in range(config.num_layers)
            ]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps, plus_one=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3_5ForConditionalGeneration(BaseLLMModel):
    def __init__(
        self, config: ModelConfig, *, mlp_factory: Callable[[ModelConfig], BaseOP] = Qwen3MLP
    ):
        self.model = Qwen3_5Model(config, mlp_factory=mlp_factory)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)

    def iter_gdn_layers(self) -> List[GDNLinearAttn]:
        """GDN bridge ops in gdn_layer_id order (engine uses this to size gdn_state + warmup)."""
        out: List[GDNLinearAttn] = [
            layer.linear_attn
            for layer in self.model.layers.op_list
            if isinstance(getattr(layer, "linear_attn", None), GDNLinearAttn)
        ]
        return out


__all__ = ["Qwen3_5ForConditionalGeneration"]
