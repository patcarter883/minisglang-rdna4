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
from minisgl.layers.tp_overlap import rowchunked_ar_span
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
        # `forward_swiglu`, NOT silu_and_mul(forward(x)) — see Linear.forward_swiglu. It IS that
        # expression whenever the fused kernel does not apply, so this is never a behaviour
        # change; hand-rolling it is how this MLP stayed unfused while two other models were not.
        return self.down_proj.forward(self.gate_up_proj.forward_swiglu(x), reduce=reduce)


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

    def accepts_producer_actquant(self) -> bool:
        """Can this block consume the feeding RMSNorm's (x_fp8, act_scales) pair?

        Only the ROUTED experts are quantized in this architecture — the router gate, the shared-
        expert gate and the whole shared expert are in the checkpoint's ignore list and stay bf16 —
        so exactly ONE consumer per layer can use the pair, not the two or three a dense block would
        offer. Under EP the pair cannot be used at all (MoELayer refuses it: the all_gather reorders
        rows). Asked rather than assumed so the decoder layer never builds a pair nobody will take:
        the producer's quant epilogue is not free, and paying for it with no consumer is a net loss.
        """
        e = self.experts
        return bool(getattr(e._moe_method, "supports_producer_actquant", False)) and not e.enable_ep

    def post_load(self) -> None:
        super().post_load()
        # The router and the shared-expert gate read the same row and are both replicated bf16:
        # ONE decode GEMV (layers/same_input_gemv.py) instead of a 256-wide and a 1-wide launch.
        from minisgl.layers.same_input_gemv import fuse_same_input

        self._gates_fused = fuse_same_input(
            "qwen3_5_moe.router+shared_expert_gate", (self.gate, self.shared_expert_gate))

    def _gates(self, h: torch.Tensor):
        """(router_logits, shared_expert_gate logit). The routing kernels take raw logits and want
        them contiguous, so the fused path pays one tiny copy for the router slice — still cheaper
        than the N=1 GEMV launch it replaces."""
        fused = getattr(self, "_gates_fused", None)
        if fused is None:
            return self.gate.forward(h), self.shared_expert_gate.forward(h)
        logits, sg = fused.forward(h, lambda x: [self.gate.forward(x), self.shared_expert_gate.forward(x)])
        return logits.contiguous(), sg

    def _fused_partial(self, hidden_states: torch.Tensor,
                       x_fp8: torch.Tensor | None = None,
                       act_scales: torch.Tensor | None = None) -> torch.Tensor:
        """Fused shared+routed row-parallel PARTIAL (no all_reduce) over `hidden_states` rows. Both
        down-projections are row-parallel, so each rank holds a partial; the caller reduces once
        (sum_r(routed_r + shared_r) == routed_full + shared_full). Row-independent -> safe to call over
        any disjoint subset of token rows and concatenate (the async-AR chunking below relies on this).
        The shared_expert_gate is replicated (identical per rank), so gating the local partial before
        the reduce is exact: sum_r(g * shared_r) == g * sum_r(shared_r)."""
        experts = self.experts
        shared_out = self.shared_expert.forward(hidden_states, reduce=False)
        router_logits, shared_gate = self._gates(hidden_states)
        shared_out = torch.sigmoid(shared_gate) * shared_out
        routed_out = experts.forward(
            hidden_states=hidden_states, router_logits=router_logits, reduce=False,
            x_fp8=x_fp8, act_scales=act_scales,
        )
        return routed_out + shared_out

    @nvtx_annotate("MoE")
    def forward(self, hidden_states: torch.Tensor,
                x_fp8: torch.Tensor | None = None,
                act_scales: torch.Tensor | None = None) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        experts = self.experts
        # `hidden_states` was just .view()'d to 2D, which is the shape the pair already describes
        # (the producer ran on the same (T,H) rows). If a caller ever reshapes rather than views,
        # this catches it instead of letting the pair describe a different tensor.
        if x_fp8 is not None and x_fp8.shape[0] != hidden_states.shape[0]:
            raise AssertionError(
                f"producer pair has {x_fp8.shape[0]} rows, hidden_states has {hidden_states.shape[0]}"
            )
        # Fuse the shared + routed TP all-reduces into ONE. Both down-projections are row-parallel, so
        # each rank holds a partial; summing the two partials locally and reducing once is exact
        # (sum_r(routed_r + shared_r) == routed_full + shared_full). Saves one all_reduce/layer — 40
        # fewer collectives/step on the 40-layer 35B at TP=2. Only in the pure-TP path: EP reduces the
        # routed experts over a DIFFERENT (dp/EP) group, so there the two must stay separate.
        fuse = experts.tp_size > 1 and not experts.enable_ep
        # Comms/compute overlap: hide chunk i's fused all_reduce behind chunk i+1's expert GEMM. This
        # used to be a bespoke ~30-line side-stream dance here; it is now `rowchunked_ar_span`, the
        # shared primitive in layers/tp_overlap.py, which every TP model uses. The gate moved in there
        # too — the helper itself falls back to the plain produce+all_reduce under graph capture, below
        # the token threshold, or when overlap is off, so there is no condition to keep in sync here.
        #
        # NOTE [2026-09-08]: this said "now DEFAULT-OFF (MINISGL_TP_AR_CHUNKS=1)" and that is WRONG —
        # tp_overlap.py:157 reads `_env_int("MINISGL_TP_AR_CHUNKS", 2)`, so overlap is ON by default. The
        # row split was documented here as bit-exact "by construction"; it is not, and never was — the
        # argument covers the all_reduce but not the expert GEMM, whose kernel choice depends on M.
        # Measured up to 1.6e-2 on bf16 (tools/tp_overlap_bitexact.py). This block fuses shared+routed
        # into ONE collective and so has no second independent branch to hide it behind, which means a
        # row split is the only overlap available here — hence: opt in, knowing the trade.
        if fuse:
            # A CPU-COMPUTED expert tier must NOT run inside the chunked span. `cpu_submit` does a
            # blocking `.to("cpu", copy=True)` to hand activations to the host pool — a HOST SYNC —
            # and a host sync inside the span's side-stream region deadlocks: chunk i's async
            # all_reduce is still in flight, the collective needs BOTH ranks to keep issuing, and
            # both ranks are parked in the copy instead. Observed 2026-09-07 as a WEDGED card ~3 s
            # into the first forward: one card at 100% util with 0% memory traffic and idle
            # power (a kernel that never returned), the peer idle at 3%/22 W, both CPU-MoE worker
            # threads asleep in `queue.get` — i.e. the tier was never given work, the collective
            # was. It is CPU-tier-exclusive because this is the only path that routes on the host
            # (`_ep_route`) instead of inside the kernel.
            #
            # num_chunks=1 takes `rowchunked_ar_span`'s plain `produce(x) + one all_reduce` arm, on
            # the main stream with no span at all. Scoped to the layer that host-syncs rather than
            # disabling overlap globally (MINISGL_TP_OVERLAP=0), so streamed layers keep it.
            # NOTE: overlap is ON by default — `_CHUNKS = _env_int("MINISGL_TP_AR_CHUNKS", 2)`. The
            # comment above claiming "DEFAULT-OFF (MINISGL_TP_AR_CHUNKS=1)" is stale and is why this
            # looked impossible; corrected there too.
            _cpu_moe = bool(getattr(getattr(experts, "_weight_offload", None),
                                    "computes_on_cpu", False))
            combined = rowchunked_ar_span(
                experts._comm, hidden_states, self._fused_partial,
                row_aligned=(x_fp8, act_scales),
                num_chunks=1 if _cpu_moe else None,
            )
            return combined.view(num_tokens, hidden_dim)
        # "shared" sub-bucket of the layer-prof "ffn" total (MINISGL_LAYER_PROF). Summed over all
        # layers, reported per-step -> direct per-step shared-expert cost (Task B #18 attribution).
        shared_out = _lp_timed(
            "shared", lambda h: self.shared_expert.forward(h, reduce=not fuse), hidden_states
        )
        # shared_expert_gate is replicated (identical per rank), so gating the local partial before the
        # fused reduce is exact: sum_r(g * shared_r) == g * sum_r(shared_r).
        router_logits, shared_gate = self._gates(hidden_states)
        shared_out = torch.sigmoid(shared_gate) * shared_out
        routed_out = experts.forward(
            hidden_states=hidden_states, router_logits=router_logits, reduce=not fuse,
            x_fp8=x_fp8, act_scales=act_scales,
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

        def mlp_factory(cfg: "ModelConfig", name_prefix: str | None = None) -> BaseOP:
            return Qwen3_5MoeSparseBlock(cfg, expert_quant)

        # The MTP head follows the checkpoint's precision: quantized only if mtp.* is NOT in the quant
        # ignore list. A bf16/fp16 MTP head on a quantized backbone builds unquantized so its full-
        # precision experts load (QuantConfig.is_module_quantized; universal across quant methods).
        mtp_quant = expert_quant
        if expert_quant is not None and not expert_quant.is_module_quantized(
            "mtp.layers.0.mlp.experts.0.gate_proj"
        ):
            mtp_quant = None

        def mtp_mlp_factory(cfg: "ModelConfig", name_prefix: str | None = None) -> BaseOP:
            return Qwen3_5MoeSparseBlock(cfg, mtp_quant, force_no_ep=True)

        super().__init__(backbone_cfg, mlp_factory=mlp_factory, mtp_mlp_factory=mtp_mlp_factory)


__all__ = ["Qwen3_5MoeForConditionalGeneration"]
