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
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.gdn.layer import QwenGatedDeltaNet
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearReplicated,
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

    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        name_prefix: str | None = None,
        attn_kv_id: int | None = None,
    ):
        head_dim = config.head_dim
        nqo, nkv = config.num_qo_heads, config.num_kv_heads
        # Per-projection quant is CONFIG-DRIVEN (like the GDN in_proj below): a self_attn projection
        # is quantized iff the checkpoint's quant config declares its module quantized (NOT in the
        # `ignore` list). The AWQ 35B keeps q/k/v/o bf16 (whole backbone in modules_to_not_convert);
        # the MXFP4 checkpoint ALSO keeps self_attn bf16 (its q/k/v/o sit in the ignore list) while
        # quantizing the GDN in_proj — the precision falls out of the config, no model-name branch.
        # `name_prefix` is the checkpoint namespace for is_module_quantized: the backbone decoder uses
        # `model.layers.<id>`, but the MTP head's self_attn lives under `mtp.layers.0.*` in the
        # checkpoint (its ignore entries are `mtp.layers.0.self_attn.*`), so Qwen3_5MTPAttn overrides
        # it — else the MTP layer id (== num_layers) matches no ignore entry and mis-quantizes a bf16
        # MTP head (both AWQ and MXFP4 ship the whole MTP head unquantized).
        q = config.quant
        prefix = name_prefix if name_prefix is not None else f"model.layers.{layer_id}"

        def _attn_method(module: str) -> "object":
            name = f"{prefix}.self_attn.{module}"
            quantized = q is not None and q.is_module_quantized(name)
            return create_linear_method(q, quantized=quantized)

        # q_proj emits q + gate (2x), interleaved per head: [head_h q | head_h gate].
        self.q_proj = LinearColParallelMerged(
            config.hidden_size, [2 * nqo * head_dim], has_bias=False, quant_method=_attn_method("q_proj")
        )
        self.k_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False, quant_method=_attn_method("k_proj")
        )
        self.v_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False, quant_method=_attn_method("v_proj")
        )
        # Qwen3.5 RMSNorm uses the (1 + weight) gain convention (weight init 0), UNLIKE the dense
        # Qwen3 plain-weight norm. Applies to all Qwen3_5RMSNorm sites (q/k norm, input/post/final
        # decoder norms); the GDN's gated norm weight keeps the plain-weight convention (init 1).
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, plus_one=True)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps, plus_one=True)
        self.attn = AttentionLayer(
            # Index the paged KV pool by the COMPACT full-attn position (GDN hybrid: 0..9 over the
            # 10 full-attn layers), not the global layer_id (3,7,..,39) — the pool has one slot per
            # full-attn layer. Non-hybrid: attn_kv_id is None -> identity layer_id.
            layer_id=attn_kv_id if attn_kv_id is not None else layer_id,
            head_dim=head_dim,
            num_qo_heads=nqo,
            num_kv_heads=nkv,
            rotary_config=config.rotary_config,  # rotary_dim=64 -> partial rotary
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * nqo, config.hidden_size, has_bias=False, quant_method=_attn_method("o_proj")
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
            # cudagraph capture: if the verify-graph capturer pre-bound a PERSISTENT scratch buffer for
            # this layer (GDNVerifyGraphCapture), COPY the fresh kernel output into it IN PLACE so the
            # captured graph's pointer stays valid across replays and the scheduler (which reads the
            # static replay-time metadata) sees the fresh state. Eager path: no pre-bound buffer, so
            # just stash the fresh kernel tensors as before.
            pre_conv = md.conv_scratch.get(self._gdn_layer_id)
            if pre_conv is not None:
                pre_conv.copy_(conv_scr)
                md.ssm_scratch[self._gdn_layer_id].copy_(ssm_scr)
            else:
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

    def post_load(self) -> None:  # convert any quantized GDN projection to op layout (bf16: no-op)
        self._gdn.process_quant()


class Qwen3_5DecoderLayer(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        is_gdn: bool,
        gdn_layer_id: int | None,
        attn_kv_id: int | None = None,
        mlp_factory: Callable[[ModelConfig], BaseOP] = Qwen3MLP,
    ):
        if is_gdn:
            assert gdn_layer_id is not None
            # Per-projection quant is CONFIG-DRIVEN, via the same generic dispatcher the dense
            # linears use: a projection is quantized iff the checkpoint's quant config declares its
            # module quantized (i.e. it is NOT in the `ignore` list). The 35B (qwen3_5_moe) nulls
            # config.quant for the backbone, so create_linear_method returns UnquantizedLinearMethod
            # and the GDN stays bf16; the dense 27B (compressed-tensors) quantizes in_proj_qkv/z +
            # out_proj (int4 packs served through the W4A8 kernel) and keeps in_proj_a/b bf16 (they
            # sit in the ignore list). No model-name branch — the precision falls out of the config.
            q = config.quant

            def _gdn_method(module: str) -> "object":
                name = f"model.layers.{layer_id}.linear_attn.{module}"
                quantized = q is not None and q.is_module_quantized(name)
                return create_linear_method(q, quantized=quantized)

            gdn = QwenGatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                tp_size=get_tp_info().size,  # head-parallel: local heads + out_proj all-reduce
                eps=config.rms_norm_eps,
                dtype=torch.get_default_dtype(),  # bf16/fp16 under the engine's build context
                device=torch.device("meta"),  # built on meta; real tensors via load(assign=True)
                qkvz_method=_gdn_method("in_proj_qkv"),
                ba_method=_gdn_method("in_proj_b"),
                out_proj_method=_gdn_method("out_proj"),
            )
            self.linear_attn = GDNLinearAttn(gdn, gdn_layer_id)
            self._attn_op: BaseOP = self.linear_attn
        else:
            assert attn_kv_id is not None
            self.self_attn = Qwen3_5Attn(config, layer_id, attn_kv_id=attn_kv_id)
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
        # Compact KV index per FULL-attention layer (mirrors gdn_pos): the paged KV pool has one
        # slot per full-attn layer, so a full-attn layer at global id `lid` stores/reads at
        # attn_pos[lid]. For a non-hybrid model this is the identity (every layer is full-attn).
        attn_pos = {aid: pos for pos, aid in enumerate(config.full_attn_layer_ids)}
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(
                    config, lid, is_gdn=lid in gdn_pos, gdn_layer_id=gdn_pos.get(lid),
                    attn_kv_id=attn_pos.get(lid), mlp_factory=mlp_factory,
                )
                for lid in range(config.num_layers)
            ]
        )
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps, plus_one=True)
        # Spec-decode aux capture: decoder-layer ids whose output hidden is stashed (None = off).
        self._capture_layer_ids: List[int] | None = None
        self._aux_hidden: List[torch.Tensor] | None = None
        # Capture the FULL post-layer residual stream (x + residual = z-lab's hidden_states[lid+1]) —
        # this fused layer returns residual = stream+attn with the MLP output still in x, so `residual`
        # alone is missing the captured layer's MLP add and is OOD for a drafter trained on the full
        # stream. MINISGL_AUX_POSTMLP=0 restores the legacy residual-only capture (for A/B).
        self._aux_postmlp = _os.environ.get("MINISGL_AUX_POSTMLP", "1") not in ("0", "false", "no")

    def set_capture_layers(self, ids: List[int] | None) -> None:
        self._capture_layer_ids = list(ids) if ids else None

    def forward(
        self, input_ids: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        # Aux capture is OFF unless return_hidden AND layers are programmed: zero cost otherwise.
        cap = self._capture_layer_ids if return_hidden else None
        cap_set = set(cap) if cap else None
        grabbed: dict[int, torch.Tensor] = {}
        for lid, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(x, residual)
            if cap_set is not None and lid in cap_set:
                # Full post-layer stream (x + residual) == z-lab hidden_states[lid+1] (attn AND mlp
                # folded in); `residual` alone omits this layer's MLP. Legacy path = residual only.
                grabbed[lid] = ((x + residual) if self._aux_postmlp else residual).clone()
        # MTP seed = the PRE-final-norm residual stream (x + residual), the Qwen3.5 MTP
        # `previous_hidden_states` input (the MTP's own pre_fc_norm_hidden re-normalizes it).
        pre_norm = (x + residual).clone() if return_hidden else None
        final = self.norm.forward(x, residual)[0]
        if return_hidden:
            # stack in the programmed id order so a consumer can index aux by position; None if empty.
            aux_stack = torch.stack([grabbed[i] for i in cap], dim=0) if cap else None
            return final, pre_norm, aux_stack
        return final


class Qwen3_5MTPAttn(Qwen3_5Attn):
    """Gated partial-rotary GQA for the self-contained MTP draft chain. Reuses the q/k/v/o
    projections + q_norm/k_norm of Qwen3_5Attn but runs a MATERIALIZED causal attention over the
    SHORT per-request draft chain (no paged KV / attn backend)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        # The MTP head's self_attn lives under `mtp.layers.0.*` in the checkpoint — gate its quant on
        # THAT namespace (not `model.layers.<num_layers>`, which matches no ignore entry) so a bf16
        # MTP head stays bf16 under a quantized backbone (AWQ + MXFP4 both ship it unquantized).
        super().__init__(config, layer_id, name_prefix="mtp.layers.0")
        self._num_kv_heads = div_even(config.num_kv_heads, get_tp_info().size)
        self._scale = float(self._head_dim) ** -0.5

    def forward_draft(
        self, x: torch.Tensor, positions: torch.Tensor, cache: "list", step: int
    ) -> torch.Tensor:
        T = x.shape[0]
        hd, nq, nkv = self._head_dim, self._num_qo_heads, self._num_kv_heads
        qg = self.q_proj.forward(x).view(T, nq, 2 * hd)
        q = qg[..., :hd].reshape(T, nq * hd)
        gate = qg[..., hd:].reshape(T, nq * hd)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x).view(T, nkv, hd)
        # q_norm/k_norm over head_dim, then partial rotary (same as AttentionLayer).
        self.q_norm.forward_inplace(q.view(T, nq, hd))
        self.k_norm.forward_inplace(k.view(T, nkv, hd))
        q, k = self.attn.rotary.forward(positions, q, k)
        q = q.view(T, nq, hd)
        k = k.view(T, nkv, hd)
        cache.append((k, v))
        Ks = torch.stack([c[0] for c in cache], dim=0)  # [S,T,nkv,hd]
        Vs = torch.stack([c[1] for c in cache], dim=0)  # [S,T,nkv,hd]
        # GQA: each q-head maps to kv-head (h // (nq//nkv)). Expand kv heads to q heads.
        rep = nq // nkv
        Ks = Ks.repeat_interleave(rep, dim=2)  # [S,T,nq,hd]
        Vs = Vs.repeat_interleave(rep, dim=2)
        scores = torch.einsum("thd,sthd->ths", q, Ks) * self._scale  # [T,nq,S]
        probs = scores.softmax(dim=-1).to(Vs.dtype)
        o = torch.einsum("ths,sthd->thd", probs, Vs).reshape(T, nq * hd)
        o = o * torch.sigmoid(gate)
        return self.o_proj.forward(o)


class Qwen3_5MTPHead(BaseOP):
    """Qwen3.5 MTP (next-token-prediction) self-speculation head — a single STANDARD full-attention
    decoder layer (mtp.layers.0) plus the fc fuser and pre-norms; REUSES the target's embed_tokens
    and TIED lm_head (no dedicated embed/head in the checkpoint):

        h_mtp = layer( fc( concat[ pre_fc_norm_embedding(embed(tok)), pre_fc_norm_hidden(last_hidden) ] ) )
        logits = lm_head( mtp.norm(h_mtp) )

    Run K times autoregressively (own short draft chain, no paged KV); see MTPProposer."""

    def __init__(self, config: ModelConfig, layer_id: int, embed: VocabParallelEmbedding,
                 lm_head: ParallelLMHead,
                 mlp_factory: Callable[[ModelConfig], BaseOP] = Qwen3MLP):
        eps = config.rms_norm_eps
        self.pre_fc_norm_embedding = RMSNorm(config.hidden_size, eps=eps, plus_one=True)
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=eps, plus_one=True)
        # REPLICATED (not column-parallel): the fc fuses concat[norm(embed), norm(hidden)] into the
        # FULL hidden seed that feeds the (head-sharded) MTP layer — exactly like GLM's MTP eh_proj /
        # EAGLE3's fc. ColParallel would shard the output to hidden/tp and feed the layer a truncated
        # hidden (latent TP>1 bug; the Qwen MTP was only ever validated at TP=1 where it's a no-op).
        self.fc = LinearReplicated(2 * config.hidden_size, config.hidden_size, has_bias=False)
        self.self_attn = Qwen3_5MTPAttn(config, layer_id)
        # MoE models (qwen3_5_moe, e.g. the 35B) ship a MoE MTP block (mtp.layers.0.mlp.experts.*);
        # dense models a plain MLP. The same mlp_factory the backbone uses builds the right one.
        self.mlp = mlp_factory(config)
        self.input_layernorm = RMSNormFused(size=config.hidden_size, eps=eps, plus_one=True)
        self.post_attention_layernorm = RMSNormFused(size=config.hidden_size, eps=eps, plus_one=True)
        self.norm = RMSNormFused(size=config.hidden_size, eps=eps, plus_one=True)
        # Tied to the TARGET embed + lm_head (hidden, not loaded/saved as MTP weights).
        self._embed = embed
        self._lm_head = lm_head

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._embed.forward(tokens)

    def fuse(self, embed_e: torch.Tensor, last_hidden: torch.Tensor) -> torch.Tensor:
        e = self.pre_fc_norm_embedding.forward(embed_e)
        h = self.pre_fc_norm_hidden.forward(last_hidden)
        return self.fc.forward(torch.cat([e, h], dim=-1))

    def step(
        self, fused: torch.Tensor, positions: torch.Tensor, cache: "list", step: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(fused, None)
        x = self.self_attn.forward_draft(x, positions, cache, step)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        hidden = x + residual
        normed = self.norm.forward(hidden, None)[0]
        # Full-vocab logits via the (tied) lm_head's TP all_gather — identical on every rank.
        logits = self._lm_head.logits_all_rows(normed)
        return logits, hidden


class Qwen3_5ForConditionalGeneration(BaseLLMModel):
    def __init__(
        self, config: ModelConfig, *, mlp_factory: Callable[[ModelConfig], BaseOP] = Qwen3MLP,
        mtp_mlp_factory: "Callable[[ModelConfig], BaseOP] | None" = None,
    ):
        self.model = Qwen3_5Model(config, mlp_factory=mlp_factory)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        # MTP self-speculation head (mtp.* namespace). Built only when the checkpoint ships one
        # (mtp_num_hidden_layers>0); reuses the target embed + tied lm_head (no dedicated tensors). The
        # MTP head may be a different precision than the backbone (e.g. a bf16/fp16 head on a quantized
        # model) — mtp_mlp_factory (built by the quantized subclass from the head's ignore-list status)
        # overrides the backbone factory; falls back to it when unset (dense models / same precision).
        self.mtp = (
            Qwen3_5MTPHead(config, layer_id=config.num_layers,
                           embed=self.model.embed_tokens, lm_head=self.lm_head,
                           mlp_factory=mtp_mlp_factory or mlp_factory)
            if config.mtp_num_hidden_layers > 0
            else None
        )
        super().__init__()

    def forward(self, return_hidden: bool = False):
        input_ids = get_global_ctx().batch.input_ids
        if return_hidden:
            # (post-norm for lm_head, pre-norm residual for the MTP seed, aux). MTP seeds from
            # the PRE-final-norm residual stream (its pre_fc_norm_hidden re-normalizes it).
            final, pre_norm, aux_hidden = self.model.forward(input_ids, return_hidden=True)
            return self.lm_head.forward(final), pre_norm, aux_hidden
        return self.lm_head.forward(self.model.forward(input_ids))

    def set_capture_layers(self, ids: List[int] | None) -> None:
        self.model.set_capture_layers(ids)

    def iter_gdn_layers(self) -> List[GDNLinearAttn]:
        """GDN bridge ops in gdn_layer_id order (engine uses this to size gdn_state + warmup)."""
        out: List[GDNLinearAttn] = [
            layer.linear_attn
            for layer in self.model.layers.op_list
            if isinstance(getattr(layer, "linear_attn", None), GDNLinearAttn)
        ]
        return out


__all__ = ["Qwen3_5ForConditionalGeneration"]
