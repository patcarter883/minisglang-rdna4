"""Phase 3M-1 — Qwen3.5-MoE GDN-hybrid model (text path of Qwen3_5MoeForConditionalGeneration).

Identical to the 4B `qwen3_5` GDN-hybrid decoder (linear_attention/full_attention interleave,
gated partial-rotary attention, (1+w) RMSNorm) EXCEPT the per-layer MLP is a Qwen2-MoE-style
sparse block: a top-k router over `num_experts` grouped experts PLUS an always-on shared expert
weighted by a sigmoid gate (`final = routed(x) + sigmoid(shared_gate(x))·shared(x)`).

In the canonical checkpoint (cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, refs/main = AWQ-gemm g32) ONLY
the routed experts are quantized; the GDN/attention, the shared expert, both gates, and lm_head
are F16 (per the quant `modules_to_not_convert`). So the qwen3_5 BACKBONE is built UNQUANTIZED
(quant=None) and the real AWQ quant is handed only to the MoE experts. Reuses qwen3_5's
decoder/model/head via their `mlp_factory` seam — the MoE block is the sole structural difference.
Serve is gated on TP=2 (the 35B does not fit one 16 GB card); weight map = Phase 3M-3.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    silu_and_mul,
)
from minisgl.layers.moe import (
    get_moe_ar_side_stream,
    moe_async_ar_enabled,
    moe_async_ar_min_tokens,
)
from minisgl.quant import create_linear_method
from minisgl.utils import nvtx_annotate

from .qwen3_5 import Qwen3_5ForConditionalGeneration, _lp_timed

if TYPE_CHECKING:
    from minisgl.quant.config import QuantConfig

    from .config import ModelConfig


class Qwen3_5MoeSharedExpert(BaseOP):
    """Always-on shared expert: an UNQUANTIZED (F16) SwiGLU at shared_expert_intermediate_size."""

    def __init__(self, config: "ModelConfig"):
        inter = config.shared_expert_intermediate_size
        qm = create_linear_method(None)  # F16
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [inter, inter], has_bias=False, quant_method=qm
        )
        self.down_proj = LinearRowParallel(
            inter, config.hidden_size, has_bias=False, quant_method=qm
        )

    def forward(self, x: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        # reduce=False -> return the row-parallel down_proj PARTIAL so the MoE block can fuse it with
        # the routed-expert partial into a single all_reduce.
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)), reduce=reduce)


class Qwen3_5MoeSparseBlock(BaseOP):
    """Router gate + shared gate + shared expert are F16 (in the checkpoint's quant ignore list);
    only the routed experts carry the int4/mxfp4 quant (MoELayer's grouped path). `expert_quant` is
    threaded in separately so the routed-expert precision is explicit here — the surrounding backbone
    config now keeps the real quant (per-module is_module_quantized gates each backbone linear)."""

    def __init__(self, config: "ModelConfig", expert_quant: "QuantConfig | None",
                 force_no_ep: bool = False):
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            # The MTP draft head is built REPLICATED (all experts local) even under EP — it is tiny, and
            # replicating avoids the loader/layer EP-shard inconsistency for a bf16 draft head under a
            # quantized (EP-sharded) backbone. Propose then issues plain-TP all_reduces (deterministic).
            force_no_ep=force_no_ep,
            # ALWAYS renormalize: the Qwen3.5-MoE router (transformers Qwen3_5MoeTopKRouter) divides
            # the top-k softmax probs by their sum UNCONDITIONALLY — there is no norm_topk_prob toggle
            # for this architecture, and the config omits the key (minisgl would default it False).
            # Skipping it leaves the routed weights summing to <1 -> mis-scaled experts -> degenerate.
            renormalize=True,
            quant=expert_quant,
        )
        self.shared_expert = Qwen3_5MoeSharedExpert(config)
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def _fused_partial(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fused shared+routed row-parallel PARTIAL (no all_reduce) over `hidden_states` rows. Both
        down-projections are row-parallel, so each rank holds a partial; the caller reduces once
        (sum_r(routed_r + shared_r) == routed_full + shared_full). Row-independent -> safe to call over
        any disjoint subset of token rows and concatenate (the async-AR chunking below relies on this).
        The shared_expert_gate is replicated (identical per rank), so gating the local partial before
        the reduce is exact: sum_r(g * shared_r) == g * sum_r(shared_r)."""
        experts = self.experts
        shared_out = self.shared_expert.forward(hidden_states, reduce=False)
        shared_out = torch.sigmoid(self.shared_expert_gate.forward(hidden_states)) * shared_out
        router_logits = self.gate.forward(hidden_states)
        routed_out = experts.forward(
            hidden_states=hidden_states, router_logits=router_logits, reduce=False
        )
        return routed_out + shared_out

    def _forward_async_ar(self, hidden_states: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Phase-1 comms/compute overlap: split the fused shared+routed partial into 2 DISJOINT row
        chunks and hide chunk-0's TP all_reduce (side stream, RCCL) behind chunk-1's expert compute
        (main stream). Disjoint rows => BIT-EXACT (each row's all_reduce is an independent 2-rank
        elementwise SUM). Eager prefill only; both TP ranks split at the same deterministic `half`
        (identical `num_tokens` post attention-AR) and submit chunk-0 then chunk-1 all_reduce in program
        order, so RCCL matches the collectives -> no desync/deadlock. Per-chunk events serialize each AR
        against its producer (side waits) and consumer (main waits) -> no half-reduced read."""
        experts = self.experts
        half = num_tokens // 2  # deterministic + identical n on both ranks => identical split
        row_chunks = (hidden_states[:half], hidden_states[half:])
        main = torch.cuda.current_stream()
        side = get_moe_ar_side_stream()
        partials: list[torch.Tensor] = []
        ar_done: list[torch.cuda.Event] = []
        for h in row_chunks:
            combined = self._fused_partial(h)  # main stream (chunk-1 compute overlaps chunk-0's AR)
            ready = torch.cuda.Event()
            ready.record(main)          # combined producer complete on the main stream
            side.wait_event(ready)      # side AR must not start before combined is ready
            with torch.cuda.stream(side):
                combined.record_stream(side)        # allocator: the side stream also uses this tensor
                experts._comm.all_reduce(combined)  # in-place RCCL SUM on the side stream
                done = torch.cuda.Event()
                done.record(side)
            partials.append(combined)
            ar_done.append(done)
        for done in ar_done:
            main.wait_event(done)       # main must not read a chunk before its AR completes
        return torch.cat(partials, dim=0)

    @nvtx_annotate("MoE")
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        experts = self.experts
        # Fuse the shared + routed TP all-reduces into ONE. Both down-projections are row-parallel, so
        # each rank holds a partial; summing the two partials locally and reducing once is exact
        # (sum_r(routed_r + shared_r) == routed_full + shared_full). Saves one all_reduce/layer — 40
        # fewer collectives/step on the 40-layer 35B at TP=2. Only in the pure-TP path: EP reduces the
        # routed experts over a DIFFERENT (dp/EP) group, so there the two must stay separate.
        fuse = experts.tp_size > 1 and not experts.enable_ep
        # Phase-1 comms/compute overlap (eager prefill only): hide chunk-0's fused all_reduce behind
        # chunk-1's expert GEMM. Off by default; never under graph capture (side-stream collectives are
        # graph-unsafe) and only above a token threshold where the overlap beats the doubled launch.
        if (
            fuse
            and moe_async_ar_enabled()
            and num_tokens >= moe_async_ar_min_tokens()
            and not torch.cuda.is_current_stream_capturing()
        ):
            combined = self._forward_async_ar(hidden_states, num_tokens)
            return combined.view(num_tokens, hidden_dim)
        # "shared" sub-bucket of the layer-prof "ffn" total (MINISGL_LAYER_PROF). Summed over all
        # layers, reported per-step -> direct per-step shared-expert cost (Task B #18 attribution).
        shared_out = _lp_timed(
            "shared", lambda h: self.shared_expert.forward(h, reduce=not fuse), hidden_states
        )
        # shared_expert_gate is replicated (identical per rank), so gating the local partial before the
        # fused reduce is exact: sum_r(g * shared_r) == g * sum_r(shared_r).
        shared_out = torch.sigmoid(self.shared_expert_gate.forward(hidden_states)) * shared_out
        router_logits = self.gate.forward(hidden_states)
        routed_out = experts.forward(
            hidden_states=hidden_states, router_logits=router_logits, reduce=not fuse
        )
        combined = routed_out + shared_out
        if fuse:
            combined = experts._comm.all_reduce(combined)
        return combined.view(num_tokens, hidden_dim)


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, config: "ModelConfig"):
        # Config-driven per-module quant across the WHOLE model. The backbone keeps the real quant
        # config; each GDN/attention projection is gated by is_module_quantized (the checkpoint's
        # ignore list), so the AWQ 35B (entire backbone in modules_to_not_convert) stays bf16 while
        # the MXFP4 checkpoint (GDN in_proj_qkv/z/a/b + out_proj quantized; self_attn, conv1d, norm,
        # gates, shared_expert in the ignore list) builds exactly those projections mxfp4. The routed
        # experts still get the quant via the mlp_factory closure. No model-name branch.
        expert_quant = config.quant
        backbone_cfg = config

        def mlp_factory(cfg: "ModelConfig") -> BaseOP:
            return Qwen3_5MoeSparseBlock(cfg, expert_quant)

        # The MTP head follows the checkpoint's precision: quantized only if mtp.* is NOT in the quant
        # ignore list. A bf16/fp16 MTP head on a quantized backbone builds unquantized so its full-
        # precision experts load (QuantConfig.is_module_quantized; universal across quant methods).
        mtp_quant = expert_quant
        if expert_quant is not None and not expert_quant.is_module_quantized(
            "mtp.layers.0.mlp.experts.0.gate_proj"
        ):
            mtp_quant = None

        def mtp_mlp_factory(cfg: "ModelConfig") -> BaseOP:
            return Qwen3_5MoeSparseBlock(cfg, mtp_quant, force_no_ep=True)

        super().__init__(backbone_cfg, mlp_factory=mlp_factory, mtp_mlp_factory=mtp_mlp_factory)


__all__ = ["Qwen3_5MoeForConditionalGeneration"]
