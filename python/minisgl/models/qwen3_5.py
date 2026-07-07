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
        # Spec-decode aux capture: decoder-layer ids whose output hidden is stashed (None = off).
        self._capture_layer_ids: List[int] | None = None
        self._aux_hidden: List[torch.Tensor] | None = None
        # CAM editable-memory tap (leading "_" hides these from BaseOP's state-dict walk, like
        # _capture_layer_ids). Staged per-request BEFORE forward via stage_cam(); a byte-exact no-op
        # when unstaged (guard below skips apply_tap entirely). See python/minisgl/cam/.
        self._cam = None                     # CAMMemory instance (or None)
        self._cam_bank: torch.Tensor | None = None   # [1,K,mem] single read bank (eager prefill/decode)
        self._cam_conf: torch.Tensor | None = None   # [1] retrieval-strength scalar (or None)
        self._cam_tap_layer: int | None = None       # decoder-layer index to inject after
        # Graph-capture DECODE path (Phase 2): a static per-ROW buffer [max_bs,K,mem] + conf [max_bs].
        # Active only while _cam_use_buf (during capture and its baked-in replay ops); eager forwards use
        # the single-bank path above. Set by stage_cam_buf() from CAMGraphCapture.
        self._cam_bank_buf: torch.Tensor | None = None
        self._cam_conf_buf: torch.Tensor | None = None
        self._cam_use_buf: bool = False

    def set_capture_layers(self, ids: List[int] | None) -> None:
        self._capture_layer_ids = list(ids) if ids else None

    def stage_cam(self, cam, bank: torch.Tensor | None, conf: torch.Tensor | None) -> None:
        """Stage a CAM read bank/conf for the NEXT forward (call before model.forward). bank=None ->
        the tap is skipped (normal serving is unperturbed). `cam` provides apply_tap + tap_layer."""
        self._cam = cam
        self._cam_bank = bank
        self._cam_conf = conf
        self._cam_tap_layer = getattr(cam, "tap_layer", None) if cam is not None else None

    def clear_cam(self) -> None:
        self._cam_bank = None
        self._cam_conf = None

    def stage_cam_buf(self, cam, bank_buf: torch.Tensor | None, conf_buf: torch.Tensor | None,
                      use_buf: bool = True) -> None:
        """Point the tap at a static per-row buffer for graph capture/replay (Phase 2). `use_buf=False`
        (after capture) reverts to the eager single-bank path — the captured graph already baked in the
        buffer-path ops, so replay reads the buffer regardless of this Python flag."""
        self._cam = cam
        self._cam_bank_buf = bank_buf
        self._cam_conf_buf = conf_buf
        self._cam_use_buf = use_buf
        self._cam_tap_layer = getattr(cam, "tap_layer", None) if cam is not None else None

    def forward(
        self, input_ids: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        # Aux capture is OFF unless return_hidden AND layers are programmed: zero cost otherwise.
        cap = self._capture_layer_ids if return_hidden else None
        cap_set = set(cap) if cap else None
        grabbed: dict[int, torch.Tensor] = {}
        cam_bank = self._cam_bank
        for lid, layer in enumerate(self.layers.op_list):
            x, residual = layer.forward(x, residual)
            if cap_set is not None and lid in cap_set:
                # output hidden of layer lid = the residual stream after it (feeds the next layer).
                grabbed[lid] = residual.clone()
            # CAM tap: inject into the residual stream after tap_layer. No-op (skipped) unless a bank is
            # staged for this forward, so normal serving is byte-identical. The tap sees the full
            # post-layer hidden h = x + residual and returns h + upd; fold upd into `residual` so the next
            # layer's input_layernorm(x, residual) sums the injected hidden (mirrors the HF output[0] hook).
            if lid == self._cam_tap_layer and self._cam is not None:
                if cam_bank is not None:
                    # eager single-bank path (prefill / eager decode): one bank broadcast to all rows.
                    h = x + residual
                    # fold the tap's ADDITIVE update onto residual: apply_tap(h)-h == upd (0 at gamma=0,
                    # so byte-exact); the next input_layernorm sums x+residual as usual.
                    residual = residual + (self._cam.apply_tap(h, cam_bank, self._cam_conf) - h)
                elif self._cam_use_buf and self._cam_bank_buf is not None:
                    # graph-capture DECODE path: a static per-ROW buffer (row == batch position). A zero
                    # row is a tap no-op (padding / non-memory / seed-once-placed). Recorded into the
                    # captured graph; replay re-runs it over the in-place-refreshed buffer.
                    h = x + residual
                    n = h.shape[0]
                    residual = residual + (
                        self._cam.apply_tap_rows(h, self._cam_bank_buf[:n], self._cam_conf_buf[:n]) - h)
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
        super().__init__(config, layer_id)
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
        self, config: ModelConfig, *, mlp_factory: Callable[[ModelConfig], BaseOP] = Qwen3MLP
    ):
        self.model = Qwen3_5Model(config, mlp_factory=mlp_factory)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        # MTP self-speculation head (mtp.* namespace). Built only when the checkpoint ships one
        # (mtp_num_hidden_layers>0); reuses the target embed + tied lm_head (no dedicated tensors).
        self.mtp = (
            Qwen3_5MTPHead(config, layer_id=config.num_layers,
                           embed=self.model.embed_tokens, lm_head=self.lm_head,
                           mlp_factory=mlp_factory)
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
