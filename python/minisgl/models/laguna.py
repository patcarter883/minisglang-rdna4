"""Laguna (poolside/Laguna-XS.2) — sliding-window-attention hybrid MoE. This module carries the
SWA-specific, CONFIG-DRIVEN attention wiring (per-layer QO head count, dual RoPE, per-layer sliding
window, and the compact KV-pool index that routes full layers to the main pool and sliding layers to
the window-bounded ring pool). It is model-name-free: every per-layer choice falls out of ModelConfig
(attn_head_counts, layer_types, sliding_window, sliding_rotary_config) exactly as parsed from the
checkpoint — no branch on model_type.

Scope: this file provides `LagunaAttention` (the gated GQA mixer, ready for the Phase-3 decoder) and
the `laguna_layer_plan` helper that computes each layer's (kind, heads, rope, window, compact-kv-id).
The full `LagunaForCausalLM` (MoE INT4/INT8 experts, dense layer-0, Hadamard R1, untied lm_head) is
the separate 5-phase bring-up in docs/laguna-port/LAGUNA_SERVING_PLAN.md — its MoE body is not built
here. What IS here is the attention subsystem the SWA feature exists to serve, and it is unit-checked
by meta-instantiation (tools/test_laguna_attn.py) against the real Laguna config shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from minisgl.distributed import get_tp_info
from minisgl.layers import AttentionLayer, BaseOP, LinearColParallelMerged, LinearOProj, RMSNorm
from minisgl.quant import create_linear_method
from minisgl.utils import div_even

if TYPE_CHECKING:
    from minisgl.models import ModelConfig, RotaryConfig


@dataclass(frozen=True)
class LayerPlan:
    """The per-layer attention descriptor, derived purely from ModelConfig."""

    is_sliding: bool
    num_qo_heads: int
    sliding_window: int  # 0 for a full layer
    kv_id: int  # compact index: full layers -> main pool; sliding layers -> SWA ring pool
    rotary_config: "RotaryConfig"


def laguna_layer_plan(config: "ModelConfig", layer_id: int) -> LayerPlan:
    """Compute layer `layer_id`'s attention plan. Full layers use rotary_config (yarn) + the full
    main pool (compact id = position among full-attn layers); sliding layers use sliding_rotary_config
    (default) + a `sliding_window`-token ring (compact id = position among sliding layers). The two
    id spaces are disjoint because the two pools are physically separate tensors."""
    assert config.layer_types is not None, "Laguna requires a per-layer attention schedule"
    is_sliding = config.layer_types[layer_id] == "sliding_attention"
    # Per-layer QO head count (Laguna: 48 full / 64 sliding); fall back to the uniform count.
    heads = (
        config.attn_head_counts[layer_id]
        if config.attn_head_counts is not None
        else config.num_qo_heads
    )
    if is_sliding:
        window = config.sliding_window or 0
        kv_id = config.swa_layer_ids.index(layer_id)
        rope = config.sliding_rotary_config or config.rotary_config
    else:
        window = 0
        kv_id = config.full_attn_layer_ids.index(layer_id)
        rope = config.rotary_config
    return LayerPlan(
        is_sliding=is_sliding,
        num_qo_heads=heads,
        sliding_window=window,
        kv_id=kv_id,
        rotary_config=rope,
    )


class LagunaAttention(BaseOP):
    """Gated GQA with per-layer heads + per-layer RoPE + optional sliding window.

    Structural differences from a plain GQA that this class threads generically from the config:
      * `num_qo_heads` is PER LAYER (48 full / 64 sliding) — sizes q_proj / o_proj / g_proj.
      * `rotary_config` is PER LAYER (full → yarn partial-0.5; sliding → default full-rotary).
      * `sliding_window` is PER LAYER (0 full / 512 sliding) → the AttentionLayer masks + routes the
        window-bounded ring pool for sliding layers.
      * a SEPARATE `g_proj` per-head output gate (Laguna: `self_attn.g_proj [heads, hidden]`),
        sigmoid-applied to the attention output before o_proj.
      * QK-norm (per-head RMSNorm on q/k before RoPE).
    The compact KV-pool id (full-main vs swa-ring) is passed to AttentionLayer as layer_id.
    """

    def __init__(self, config: "ModelConfig", layer_id: int):
        plan = laguna_layer_plan(config, layer_id)
        head_dim = config.head_dim
        nqo, nkv = plan.num_qo_heads, config.num_kv_heads
        q = config.quant
        prefix = f"model.layers.{layer_id}"

        def _method(module: str):
            name = f"{prefix}.self_attn.{module}"
            quantized = q is not None and q.is_module_quantized(name)
            return create_linear_method(q, quantized=quantized)

        # q/k/v are separate column-parallel projections (per-layer head count). All attention
        # projections are bf16 in the Laguna checkpoint (self_attn is ignore-listed) — the quant
        # method still falls out of the config, no model-name branch.
        self.q_proj = LinearColParallelMerged(
            config.hidden_size, [nqo * head_dim], has_bias=False, quant_method=_method("q_proj")
        )
        self.k_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False, quant_method=_method("k_proj")
        )
        self.v_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False, quant_method=_method("v_proj")
        )
        # Per-head output gate: g_proj emits one scalar per QO head ([nqo, hidden]).
        self.g_proj = LinearColParallelMerged(
            config.hidden_size, [nqo], has_bias=False, quant_method=_method("g_proj")
        )
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.attn = AttentionLayer(
            layer_id=plan.kv_id,  # compact pool id (full → main pool, sliding → SWA ring pool)
            head_dim=head_dim,
            num_qo_heads=nqo,
            num_kv_heads=nkv,
            rotary_config=plan.rotary_config,  # per-layer rope (full=yarn / sliding=default)
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            sliding_window=plan.sliding_window,  # 0 full / 512 sliding
        )
        self.o_proj = LinearOProj(
            head_dim * nqo, config.hidden_size, has_bias=False, quant_method=_method("o_proj")
        )
        self._head_dim = head_dim
        self._nqo_local = div_even(nqo, get_tp_info().size)
        self.plan = plan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        gate = self.g_proj.forward(x)  # [n, nqo_local]
        qkv = torch.cat([q, k, v], dim=-1)
        o = self.attn.forward(qkv)  # [n, nqo_local*head_dim]
        # per-head sigmoid output gate (Laguna gated attention)
        o = o.view(n, self._nqo_local, self._head_dim) * torch.sigmoid(gate).unsqueeze(-1)
        return self.o_proj.forward(o.reshape(n, self._nqo_local * self._head_dim))


class LagunaForCausalLM(BaseOP):
    """Registry entry point. The attention subsystem (LagunaAttention + SWA ring pool) is complete;
    the MoE body (256-expert INT4/INT8 routed experts, dense layer-0, Hadamard R1, untied lm_head)
    is the Phase 3–4 bring-up in docs/laguna-port/LAGUNA_SERVING_PLAN.md and is not assembled here."""

    def __init__(self, config: "ModelConfig"):
        raise NotImplementedError(
            "LagunaForCausalLM: the SWA attention subsystem (config parse, window-bounded ring KV "
            "pool, kernel masking, per-layer heads + dual RoPE) is implemented and validated, but the "
            "MoE body (INT4/INT8 experts, dense L0, Hadamard R1) is the separate 5-phase bring-up in "
            "docs/laguna-port/LAGUNA_SERVING_PLAN.md. Build it there, wiring attention via "
            "minisgl.models.laguna.LagunaAttention."
        )
