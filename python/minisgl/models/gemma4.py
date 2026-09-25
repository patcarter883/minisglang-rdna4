"""Gemma4 — the shared backbone of `gemma4` (autoregressive) and `diffusion_gemma` (block
diffusion). Both checkpoints carry the SAME 30-layer stack; only the head and the namespace differ
(`model.language_model.*` vs `model.decoder.*`), so everything structural lives here and the heads
stay thin.

Five things about this architecture diverge from every other model in the repo, and four of the five
fail SILENTLY — plausible, grammatical output that is quietly wrong. They are called out at each
site, but collected here because they are what a reviewer should check first:

  1. The two layer types have DIFFERENT head_dim AND kv-head counts — 25 sliding layers at
     (16 q, 8 kv, D=256) and 5 full layers at (16 q, 2 kv, D=512). Not just a per-layer head COUNT
     the way Laguna varies it. ModelConfig carries the full geometry in `head_dim`/`num_kv_heads`
     (they size the main paged pool) and the sliding geometry in `swa_head_dim`/`swa_num_kv_heads`.
  2. The softmax scale is 1.0, NOT 1/sqrt(head_dim) (reference `self.scaling = 1.0`). The
     temperature is folded into the LEARNED k_norm, whose weight is a near-constant 0.1260 on
     sliding / 0.0623 on full layers. Applying head_dim**-0.5 on top roughly doubles or halves the
     logit temperature depending on the layer type.
  3. The full-attention layers ship NO v_proj (`attention_k_eq_v`). V is not an alias of the cached
     K — it is the PRE-norm, PRE-RoPE k_proj output passed through an UNWEIGHTED RMSNorm, while K is
     k_norm'd and RoPE'd. Both must be materialised and cached.
  4. The dense MLP and the routed experts are PARALLEL branches off the same residual, summed, under
     three separate norms (post_feedforward_layernorm_1 on the dense output,
     post_feedforward_layernorm_2 on the MoE output, post_feedforward_layernorm on their sum). The
     router reads the RAW residual, not either branch's normalized input.
  5. `layer_scalar` rescales the ENTIRE residual stream at the end of every layer (~0.7 typical,
     0.0986 at layer 0). Dropping it compounds to ~0.7^30 and destroys the model.

Not applicable to the shipping checkpoints, deliberately not implemented: per-layer-embeddings
(`hidden_size_per_layer_input` is 0) and KV sharing (`num_kv_shared_layers` is 0). Both short-circuit
in the reference. The vision tower is skipped by the loader — this engine is text-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearRowParallel,
    MoELayer,
    OPList,
    ParallelLMHead,
    RMSNorm,
    VocabParallelEmbedding,
    gelu_tanh_and_mul,
)
from minisgl.layers import _tail_hip
from minisgl.layers.norm import RMSNormNoScale
from minisgl.layers.tp_overlap import ar_span, overlap_active, tp_overlap_chunks
from minisgl.distributed import DistributedCommunicator
from minisgl.quant import create_linear_method
from minisgl.utils import div_even, nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from minisgl.quant.config import QuantConfig

    from .config import ModelConfig, RotaryConfig


@dataclass(frozen=True)
class Gemma4LayerPlan:
    """Per-layer attention descriptor, derived purely from ModelConfig — no model-name branch."""

    is_sliding: bool
    num_kv_heads: int
    head_dim: int
    sliding_window: int  # 0 for a full layer
    kv_id: int  # compact index: full -> main pool; sliding -> SWA ring pool
    rotary_config: "RotaryConfig"
    has_v_proj: bool  # False on the full layers (attention_k_eq_v)


def gemma4_layer_plan(config: "ModelConfig", layer_id: int) -> Gemma4LayerPlan:
    """Full layers: 2 kv heads at head_dim 512, the proportional RoPE, the full-context main pool.
    Sliding layers: 8 kv heads at head_dim 256, the default RoPE, a 1024-token ring pool. The two
    compact kv id spaces are disjoint because the two pools are physically separate tensors."""
    assert config.layer_types is not None, "Gemma4 requires a per-layer attention schedule"
    is_sliding = config.layer_types[layer_id] == "sliding_attention"
    if is_sliding:
        return Gemma4LayerPlan(
            is_sliding=True,
            # swa_* are None when a checkpoint gives both layer types one geometry; fall back rather
            # than assume the split, so a future uniform Gemma4 variant still builds.
            num_kv_heads=config.swa_num_kv_heads or config.num_kv_heads,
            head_dim=config.swa_head_dim or config.head_dim,
            sliding_window=config.sliding_window or 0,
            kv_id=config.swa_layer_ids.index(layer_id),
            rotary_config=config.sliding_rotary_config or config.rotary_config,
            has_v_proj=True,
        )
    return Gemma4LayerPlan(
        is_sliding=False,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        sliding_window=0,
        kv_id=config.full_attn_layer_ids.index(layer_id),
        rotary_config=config.rotary_config,
        # `attention_k_eq_v` drops v_proj on the GLOBAL layers only; the sliding layers keep theirs.
        has_v_proj=not config.attention_k_eq_v,
    )


class Gemma4Attention(BaseOP):
    """GQA with per-layer head_dim, per-layer RoPE, per-layer window, and — on the full layers —
    no v_proj at all."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        plan = gemma4_layer_plan(config, layer_id)
        head_dim, nkv = plan.head_dim, plan.num_kv_heads
        nqo = config.num_qo_heads
        q = config.quant
        prefix = f"model.layers.{layer_id}.self_attn"

        def _method(module: str):
            name = f"{prefix}.{module}"
            return create_linear_method(
                q, quantized=q is not None and q.is_module_quantized(name)
            )

        # q, k (and v on the sliding layers) in ONE merged projection: the checkpoint ships them
        # apart, the loader stacks each rank's shards (weight.py `_gemma4_qkv_merge`). One decode
        # GEMV instead of three — the three shared the same input row, so apart they paid three
        # launches and three passes of fixed per-call cost for one input read's worth of work.
        # A full layer ships NO v_proj (`attention_k_eq_v`): its merge is [q | k], and building a v
        # slice would fail the loader's exact-key check — the absence IS the architecture signal.
        quantized = {m: q is not None and q.is_module_quantized(f"{prefix}.{m}")
                     for m in (("q_proj", "k_proj", "v_proj") if plan.has_v_proj else ("q_proj", "k_proj"))}
        if len(set(quantized.values())) != 1:
            raise ValueError(f"{prefix}: q/k/v quantization differs {quantized}; cannot merge them")
        self.qkv_proj = LinearQKVMerged(
            config.hidden_size, head_dim, nqo, nkv, has_bias=False,
            quant_method=_method("q_proj"), has_v=plan.has_v_proj,
        )
        self._has_v = plan.has_v_proj
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        # `with_scale=False` in the reference: no weight tensor exists in the checkpoint, and V is
        # normalized on EVERY layer — including the sliding ones that do have a v_proj.
        self._v_norm = RMSNormNoScale(eps=config.rms_norm_eps)
        self.attn = AttentionLayer(
            layer_id=plan.kv_id,
            head_dim=head_dim,
            num_qo_heads=nqo,
            num_kv_heads=nkv,
            rotary_config=plan.rotary_config,
            q_norm=self.q_norm,  # applied BEFORE RoPE, matching the reference op order
            k_norm=self.k_norm,
            sliding_window=plan.sliding_window,
        )
        self.o_proj = LinearOProj(
            head_dim * nqo, config.hidden_size, has_bias=False, quant_method=_method("o_proj")
        )
        self._head_dim = head_dim
        # Local kv-head count after TP sharding; AttentionLayer already computed it (and may have
        # REPLICATED rather than split when nkv < tp_size, which the full layers' 2 heads can hit).
        self._nkv_local = self.attn.num_kv_heads
        self.plan = plan

    @nvtx_annotate("MHA")
    def forward(
        self,
        x: torch.Tensor,
        x_fp8: torch.Tensor | None = None,
        act_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`x_fp8`/`act_scales`: the fp8 form of `x` computed by the `input_layernorm` that produced
        it (`RMSNorm.forward_quant`). q/k/v are three SEPARATE linears over the SAME rows, so without
        the pair each one re-reads (M, K) and launches its own `compute_act_fp8_and_scales_kernel` —
        the same activation quantized three times per layer. Bit-identical when it fires; `None` (an
        unquantized checkpoint, or a tail_hip predating the op) just restores exactly that."""
        qkv = self.qkv_proj.forward(x, x_fp8=x_fp8, act_scales=act_scales)
        q_dim, kv_dim = self.attn.qo_attn_dim, self.attn.kv_attn_dim
        # On a full layer V reuses k_proj's OUTPUT, not the cached key: the key that reaches the KV
        # pool has since been k_norm'd (a learned gain) and RoPE'd, neither of which V gets. Reading
        # it back off the key would silently rotate the values.
        if self._has_v:
            q, k, v_src = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        else:
            q, k = qkv.split([q_dim, kv_dim], dim=-1)
            v_src = k
        # AttentionLayer runs q/k_norm + RoPE and this layer's scale-less v_norm (one fused launch
        # when tail_hip.qk_norm_rope applies). v_src may BE k's projection output; it is read before
        # anything is written.
        o = self.attn.forward_qkv(q, k, v_src, v_norm_eps=self._v_norm.eps)
        return self.o_proj.forward(o)


class Gemma4Router(BaseOP):
    """Softmax-over-all-experts router with two learned rescalings that are easy to miss.

    `scale` is a per-HIDDEN-DIM vector applied after an unweighted RMSNorm and further multiplied by
    hidden_size**-0.5; `per_expert_scale` is a per-EXPERT vector applied to the top-k weights AFTER
    they are renormalized to sum to 1 — so the final gate weights deliberately do NOT sum to 1.
    Order matters: renormalize first, then scale."""

    def __init__(self, hidden_size: int, num_experts: int, eps: float):
        self.weight = torch.empty(num_experts, hidden_size)  # <- checkpoint `router.proj.weight`
        self.scale = torch.empty(hidden_size)
        self.per_expert_scale = torch.empty(num_experts)
        self._norm = RMSNormNoScale(eps=eps)
        self._scalar_root_size = hidden_size**-0.5
        self._top_k = 0  # set by the block (top_k is a model-level config, not a router weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        from minisgl.layers.minv import minv_linear

        h = self._norm.forward(x)
        h = h * self.scale * self._scalar_root_size
        # M-invariant GEMM, as for every other router in this repo: these logits drive top-k expert
        # SELECTION, so a ~1-ULP M-dependence between a cold forward and a prefix-reused one could
        # flip a near-tie and route to a different expert.
        logits = minv_linear(h, self.weight)
        # softmax is monotonic, so top-k SELECTION is identical whether it runs here or on the raw
        # logits; fp32 only tightens the weights the kernel will combine with.
        probs = torch.softmax(logits.float(), dim=-1)
        w, idx = torch.topk(probs, k=self._top_k, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        return w * self.per_expert_scale.float()[idx], idx

    def post_load(self) -> None:
        # The fused route reads the per-expert scale in the logits' dtype: cast it ONCE here rather
        # than on every call (a cast per layer per step is exactly the launch count being removed).
        self._pes = self.per_expert_scale.to(self.weight.dtype).contiguous()
        self._post_load_done = True

    def route_normed(self, n_router: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`forward` from the router's ALREADY-normalized input (gemma4_attn_tail emits it): the
        M-invariant GEMM, then softmax + top-k + renormalize + per-expert scale in one kernel. The
        same route as `forward` up to exact probability ties, which may be ordered either way."""
        from minisgl.layers.minv import minv_linear

        return _tail_hip.gemma4_route(minv_linear(n_router, self.weight), self._pes, self._top_k)


class Gemma4DenseMLP(BaseOP):
    """The always-on dense branch (intermediate_size 2112). Unquantized in the shipping checkpoints
    — the whole `mlp.*` namespace is in the compressed-tensors ignore list — but gated on the config
    rather than hard-coded, so a future fully-quantized Gemma4 builds correctly."""

    def __init__(
        self,
        config: "ModelConfig",
        layer_id: int | None,
        *,
        quant_module: str | None = None,
    ):
        # `quant_module` names the module whose quantization policy this MLP follows, for the one
        # instance that is NOT layer-scoped: DiffusionGemma's self-conditioning block is the same
        # primitive at the same width and reuses this class rather than forking it.
        q = config.quant
        name = quant_module or f"model.layers.{layer_id}.mlp.gate_proj"
        qm = create_linear_method(
            q, quantized=q is not None and q.is_module_quantized(name)
        )
        inter = config.intermediate_size
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [inter, inter], has_bias=False, quant_method=qm
        )
        self.down_proj = LinearRowParallel(
            inter, config.hidden_size, has_bias=False, quant_method=qm
        )

    def forward(self, x: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        # The checkpoint declares `hidden_activation: gelu_pytorch_tanh`, i.e. the TANH approximation.
        # `gelu_and_mul` is the exact erf gelu — close enough to look right and never to fail, which
        # is precisely why it must not be used here. Same activation as the routed experts.
        return self.down_proj.forward(gelu_tanh_and_mul(self.gate_up_proj.forward(x)), reduce=reduce)


class Gemma4DecoderLayer(BaseOP):
    """Attribute names mirror the checkpoint's module paths 1:1, so the weight loader is a namespace
    rewrite plus expert stacking rather than a mapping table."""

    def __init__(self, config: "ModelConfig", layer_id: int, expert_quant: "QuantConfig | None"):
        eps, hidden = config.rms_norm_eps, config.hidden_size
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.self_attn = Gemma4Attention(config, layer_id)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)

        self.pre_feedforward_layernorm = RMSNorm(hidden, eps=eps)
        self.mlp = Gemma4DenseMLP(config, layer_id)
        self.post_feedforward_layernorm_1 = RMSNorm(hidden, eps=eps)

        self.pre_feedforward_layernorm_2 = RMSNorm(hidden, eps=eps)
        self.router = Gemma4Router(hidden, config.num_experts, eps)
        self.router._top_k = config.num_experts_per_tok
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=hidden,
            intermediate_size=config.moe_intermediate_size,
            # The router already renormalized AND applied per_expert_scale; a second renormalize in
            # the kernel would divide that scale straight back out.
            renormalize=False,
            activation="gelu",
            quant=expert_quant,
        )
        self.post_feedforward_layernorm_2 = RMSNorm(hidden, eps=eps)

        self.post_feedforward_layernorm = RMSNorm(hidden, eps=eps)
        self.layer_scalar = torch.empty(1)
        self._comm = DistributedCommunicator()

    def _ffn(self, residual: torch.Tensor, span: "ar_span") -> tuple:
        """The FFN half of the layer, issued for one row range. Returns the two OUTSTANDING all_reduce
        handles plus the row range's residual; the caller consumes them once every chunk is issued.

        Splitting issue from consumption is the whole point: the dense branch's all_reduce is handed to
        the side stream and the MoE branch — which is independent of it, the two meet only at
        `dense + moe` and each carries its OWN post-norm, so they cannot be fused into one collective —
        computes underneath it on the main stream."""
        dense_partial = self.mlp.forward(
            self.pre_feedforward_layernorm.forward(residual), reduce=False
        )
        ar_dense = span.all_reduce(dense_partial)
        # The router reads the RAW residual, not the norm-2 output the experts consume.
        topk_weights, topk_ids = self.router.forward(residual)
        moe_partial = self.experts.forward(
            hidden_states=self.pre_feedforward_layernorm_2.forward(residual),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            reduce=False,
        )
        ar_moe = span.all_reduce(moe_partial)
        return ar_dense, ar_moe, residual

    def _fused_ok(self, o: torch.Tensor) -> bool:
        """The fused hand-off/route/combine kernels (tail_hip) apply: present in this image, a 16-bit
        activation, and the b128 row path (hidden % 8)."""
        return (hasattr(_tail_hip, "gemma4_attn_tail") and o.dtype in (torch.float16, torch.bfloat16)
                and o.shape[-1] % 8 == 0 and hasattr(self.router, "_pes")
                and self.router.scale.dtype == o.dtype and self.layer_scalar.dtype == o.dtype)

    def _ffn_fused(self, h, n_dense, n_moe, n_router, span: "ar_span") -> tuple:
        """`_ffn` over the three normalized views gemma4_attn_tail produced in one pass."""
        dense_partial = self.mlp.forward(n_dense, reduce=False)
        ar_dense = span.all_reduce(dense_partial)
        topk_weights, topk_ids = self.router.route_normed(n_router)
        moe_partial = self.experts.forward(
            hidden_states=n_moe, topk_weights=topk_weights, topk_ids=topk_ids, reduce=False,
        )
        ar_moe = span.all_reduce(moe_partial)
        return ar_dense, ar_moe, h

    def _combine_fused(self, ar_dense, ar_moe, residual: torch.Tensor) -> torch.Tensor:
        """`_combine` in one kernel: both post-norms, their sum, the outer norm, the residual add and
        the layer scalar — bit-identical to the chain."""
        return _tail_hip.gemma4_ffn_combine(
            ar_dense.wait().contiguous(), ar_moe.wait().contiguous(), residual,
            self.post_feedforward_layernorm_1.weight, self.post_feedforward_layernorm_2.weight,
            self.post_feedforward_layernorm.weight, self.layer_scalar, self.post_feedforward_layernorm.eps,
        )

    def _combine(self, ar_dense, ar_moe, residual: torch.Tensor) -> torch.Tensor:
        dense = self.post_feedforward_layernorm_1.forward(ar_dense.wait())
        moe = self.post_feedforward_layernorm_2.forward(ar_moe.wait())
        h = self.post_feedforward_layernorm.forward(dense + moe)
        h = residual + h
        # Rescales the WHOLE residual stream, not just the FFN branch.
        return h * self.layer_scalar

    @nvtx_annotate("Layer")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # PRODUCER-SIDE act-quant: the norm already holds the row in registers, so the fp8 form of
        # its own output is an epilogue there rather than three separate (M, K) re-reads in
        # q/k/v_proj. `forward_quant` is bit-identical to `forward` and returns a None pair when the
        # native kernel does not apply, so this call site needs no gate on the checkpoint.
        h, h_fp8, h_scales = self.input_layernorm.forward_quant(x)
        h = self.self_attn.forward(h, h_fp8, h_scales)
        if self._fused_ok(h):
            # FUSED (tail_hip): the post-attention norm, the residual add, and all three normalized
            # views the FFN half reads — the dense pre-norm, the MoE pre-norm and the router's scaled
            # unweighted norm — in ONE pass instead of seven launches; and below, routing in one
            # kernel after the router GEMM, and the whole combine in one. A bs=1 decode issued 1712
            # kernels/token against vLLM's 766 on the same cards, at ~3.5 us of graph-replay dead time
            # per kernel boundary. Bit-identical to the chain (tail/tests/test_gemma4_fusions.py)
            # except the ordering of exactly-tied experts.
            h, n_dense, n_moe, n_router = _tail_hip.gemma4_attn_tail(
                h.contiguous(), residual.contiguous(), self.post_attention_layernorm.weight,
                self.pre_feedforward_layernorm.weight, self.pre_feedforward_layernorm_2.weight,
                self.router.scale, self.router._scalar_root_size, self.post_attention_layernorm.eps,
            )
            n = h.shape[0]
            k = tp_overlap_chunks()
            with ar_span(self._comm) as span:
                if k <= 1 or n < 2 * k or not overlap_active(h):
                    return self._combine_fused(*self._ffn_fused(h, n_dense, n_moe, n_router, span))
                bounds = [(n * i) // k for i in range(k + 1)]
                pending = [
                    self._ffn_fused(h[bounds[i]:bounds[i + 1]], n_dense[bounds[i]:bounds[i + 1]],
                                    n_moe[bounds[i]:bounds[i + 1]], n_router[bounds[i]:bounds[i + 1]], span)
                    for i in range(k)
                ]
                return torch.cat([self._combine_fused(*p) for p in pending], dim=0)
        # Post-norm on the attention OUTPUT, then the residual add — not the usual pre-norm order,
        # so the fused rmsnorm+residual-add op does not apply here.
        h = self.post_attention_layernorm.forward(h)
        h = residual + h

        # The FFN half is ROW-INDEPENDENT (norms, router, MLP, MoE and the adds are all per-row), which
        # is what licenses both overlaps below. Attention is not, so the split starts here.
        #
        # chunks == 1 (default): no row split at all. The dense all_reduce simply rides the side stream
        # while the MoE computes, which is bit-exact in the strongest sense — the identical collective
        # on the identical tensor, differing only in stream.
        # chunks > 1: additionally pipeline row chunks, so chunk i's MoE all_reduce (the one with
        # nothing after it to hide behind) overlaps chunk i+1's dense+MoE compute. Disjoint rows keep
        # the COLLECTIVE exact, but they also change M for the MoE grouped GEMM, whose fused gemm2
        # reduction order is M-dependent — so this arm is a MEASURED bit-exactness claim, not a
        # structural one. See tools/tp_overlap_bitexact.py.
        n = h.shape[0]
        k = tp_overlap_chunks()
        with ar_span(self._comm) as span:
            # Split rows only when the collectives will actually be overlapped. Under capture (or below
            # the threshold) every all_reduce is inline, so a split would buy no overlap while still
            # paying the extra launches and changing the MoE grouped-GEMM's M.
            if k <= 1 or n < 2 * k or not overlap_active(h):
                return self._combine(*self._ffn(h, span))
            bounds = [(n * i) // k for i in range(k + 1)]
            pending = [self._ffn(h[bounds[i] : bounds[i + 1]], span) for i in range(k)]
            return torch.cat([self._combine(*p) for p in pending], dim=0)


class Gemma4Model(BaseOP):
    def __init__(self, config: "ModelConfig", expert_quant: "QuantConfig | None"):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Gemma4DecoderLayer(config, i, expert_quant) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._embed_scale = config.embed_scale
        self._embed_scale_cache: torch.Tensor | None = None

    def _scale_tensor(self, like: torch.Tensor) -> torch.Tensor:
        # The reference casts sqrt(hidden) to the WEIGHT dtype before multiplying, so in fp16 the
        # constant is 53.0625 and not 53.0660. Materializing it as a tensor of `like`'s dtype
        # reproduces that rounding; a bare python float would multiply at fp32 precision.
        cache = self._embed_scale_cache
        if cache is None or cache.dtype != like.dtype or cache.device != like.device:
            cache = torch.tensor(self._embed_scale, dtype=like.dtype, device=like.device)
            self._embed_scale_cache = cache
        return cache

    def _embed_scaled(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embedding lookup times sqrt(hidden). Shared with the DiffusionGemma canvas forward, which
        inserts the self-conditioning block between this and the layer loop — the fp16 rounding of
        the scale constant is load-bearing (see `_scale_tensor`), so both paths must go through it."""
        h = self.embed_tokens.forward(input_ids)
        if self._embed_scale is not None:
            h = h * self._scale_tensor(h)
        return h

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self._embed_scaled(input_ids)
        # Image soft tokens: the vision tower's rows replace the placeholder embeddings (the reference
        # masked_scatter). Prepared per batch by Engine._prepare_vision; None on every text-only batch.
        mm = getattr(get_global_ctx().batch, "mm_merge", None)
        if mm is not None:
            h.index_copy_(0, mm[0], mm[1].to(h.dtype))
        for layer in self.layers.op_list:
            h = layer.forward(h)
        return self.norm.forward(h)



def _build_vision(config: "ModelConfig"):
    """The checkpoint's vision tower, when it ships one (ModelConfig.vision); None for text-only."""
    if not getattr(config, "vision", None):
        return None
    from .gemma4_vision import Gemma4VisionTower

    return Gemma4VisionTower(config.vision, config.hidden_size)

class Gemma4ForConditionalGeneration(BaseLLMModel):
    applies_logit_softcap = True  # forward() caps; the sampler must not cap again
    def __init__(self, config: "ModelConfig"):
        # Only the routed experts are quantized; the attention projections, the dense MLP, the
        # router and lm_head are all in the checkpoint's ignore list. The backbone keeps the real
        # quant config so each linear can gate itself via is_module_quantized (no model-name branch),
        # and the experts get it explicitly.
        self.model = Gemma4Model(config, expert_quant=config.quant)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        self._softcap = config.final_logit_softcapping
        self.vision = _build_vision(config)
        super().__init__()

    def forward(self, return_hidden: bool = False):
        hidden = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(hidden)
        if self._softcap is not None:
            # logits = c * tanh(logits / c). Applied to the EMITTED logits only, after lm_head —
            # the reference caps in Gemma4ForCausalLM.forward, not inside the text model. Note the
            # lm_head consumes the UNSCALED embedding matrix; the sqrt(hidden) factor is applied in
            # the embedding forward only, so a tied head must not re-apply it.
            logits = torch.tanh(logits / self._softcap) * self._softcap
        if return_hidden:
            # `hidden` is post-final-norm / pre-lm_head, which is what the draft heads seed from.
            # No aux capture: set_capture_layers is unimplemented for Gemma4, so the base class
            # raises if a proposer ever asks for capture layers rather than silently returning None.
            return logits, hidden, None
        return logits


__all__ = ["Gemma4ForConditionalGeneration", "gemma4_layer_plan", "Gemma4LayerPlan"]
