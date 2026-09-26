"""Gemma-4 assistant drafter (`gemma4_assistant`, e.g. google/gemma-4-26B-A4B-it-assistant).

A 4-layer decoder that owns no KV: every layer is Q-only and attends over the TARGET's paged cache —
sliding layers over the target's last sliding layer (the SWA ring pool), the full layer over the
target's last full layer (the main pool). One draft step:

    x = pre_projection(cat(target_embed(tok) * sqrt(backbone), seed))          [n, hidden]
    x = 4 x [q-only attention -> sandwich-normed residual -> gelu MLP -> residual] * layer_scalar
    d = norm(x);  next_tok = argmax(lm_head(d));  next_seed = post_projection(d)  [n, backbone]

Every step queries from the same position (the last position the target processed) over the same
keys, so a draft chain needs no draft-side cache or rollback.

TP: q_proj / gate / up are column-sharded by head / intermediate, o_proj and down_proj are
row-sharded (all_reduce), matching the target's kv-head sharding so each rank's q heads read its own
kv heads. pre/post_projection are replicated; lm_head is vocab-parallel over the drafter's own
(tied) embedding.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from minisgl.distributed import get_tp_info
from minisgl.layers import ParallelLMHead, RMSNorm, gelu_tanh_and_mul
from minisgl.layers.base import BaseOP

from .draft_linear import SHARD_COL, SHARD_NONE, SHARD_ROW, DraftLinear


class _AttnTarget:
    """Where one drafter layer type reads K/V: the target's pool, its compact layer id, its RoPE."""

    def __init__(self, pool, kv_id: int, rotary, head_dim: int, is_fp8: bool):
        self.k_cache = pool.k_cache(kv_id)
        self.v_cache = pool.v_cache(kv_id)
        self.k_descale = pool.k_descale[kv_id] if is_fp8 else None
        self.v_descale = pool.v_descale[kv_id] if is_fp8 else None
        self.rotary = rotary
        self.head_dim = head_dim


class Gemma4AssistantLayer(BaseOP):
    def __init__(self, hidden: int, num_heads: int, head_dim: int, inter: int, eps: float):
        tp = get_tp_info().size
        self.num_heads_local = num_heads // tp
        self.head_dim = head_dim
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.q_proj = DraftLinear(hidden, num_heads * head_dim, SHARD_COL)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.o_proj = DraftLinear(num_heads * head_dim, hidden, SHARD_ROW)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps=eps)
        # gate|up of THIS rank's intermediate slice, stacked, so gelu_tanh_and_mul reads one tensor.
        self.gate_up = DraftLinear(hidden, 2 * (inter // tp), SHARD_NONE)
        self.down_proj = DraftLinear(inter, hidden, SHARD_ROW)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps=eps)
        self.layer_scalar = torch.ones(1)

    def attention(self, x: torch.Tensor, pos: torch.Tensor, tgt: _AttnTarget,
                  block_table: torch.Tensor, ctx_lens: torch.Tensor, decode, scale: float) -> torch.Tensor:
        n = x.shape[0]
        q = self.q_proj.forward(x)
        q = self.q_norm.forward(q.view(-1, self.head_dim)).view(n, -1)
        q = tgt.rotary.forward_one(pos, q).view(n, self.num_heads_local, self.head_dim)
        if tgt.k_descale is not None:
            o = decode[1](q, tgt.k_cache, tgt.v_cache, block_table, ctx_lens, scale,
                          tgt.k_descale, tgt.v_descale, 0)
        else:
            o = decode[0](q, tgt.k_cache, tgt.v_cache, block_table, ctx_lens, scale, 0)
        return self.o_proj.forward(o.reshape(n, -1))

    def forward(self, x, pos, tgt, block_table, ctx_lens, decode, scale) -> torch.Tensor:
        a = self.attention(self.input_layernorm.forward(x), pos, tgt, block_table, ctx_lens,
                           decode, scale)
        x = x + self.post_attention_layernorm.forward(a)
        m = self.down_proj.forward(gelu_tanh_and_mul(self.gate_up.forward(
            self.pre_feedforward_layernorm.forward(x))))
        x = x + self.post_feedforward_layernorm.forward(m)
        return x * self.layer_scalar


class Gemma4AssistantDraft(BaseOP):
    def __init__(self, hf_config, layer_types: List[str]):
        text = hf_config.text_config
        self.hidden = int(text.hidden_size)
        self.backbone = int(hf_config.backbone_hidden_size)
        self.vocab = int(text.vocab_size)
        self.layer_types = list(layer_types)
        eps = float(text.rms_norm_eps)
        nh = int(text.num_attention_heads)
        tp = get_tp_info().size
        assert nh % tp == 0, f"gemma4_assistant: {nh} heads do not split over tp={tp}"
        if getattr(hf_config, "use_ordered_embeddings", False):
            raise NotImplementedError("gemma4_assistant: use_ordered_embeddings (centroid logits)")
        self.pre_projection = DraftLinear(2 * self.backbone, self.hidden, SHARD_NONE)
        self.post_projection = DraftLinear(self.hidden, self.backbone, SHARD_NONE)
        # transformers >= 5.17 carries per-layer geometry in `per_layer_config` (and drops
        # `global_head_dim`); older configs have the two globals.
        per = getattr(text, "per_layer_config", None)

        def head_dim(i, t):
            if per is not None:
                return int(per[i].head_dim)
            return int(text.head_dim if t == "sliding_attention" else text.global_head_dim)

        self.layers = [
            Gemma4AssistantLayer(self.hidden, nh, head_dim(i, t), int(text.intermediate_size), eps)
            for i, t in enumerate(self.layer_types)
        ]
        self.norm = RMSNorm(self.hidden, eps=eps)
        self.lm_head = ParallelLMHead(self.vocab, self.hidden)

    def load(self, sd: dict, device, dtype) -> None:
        """`sd` is the full checkpoint state dict (CPU); every tensor is sliced to this rank here."""
        tp = get_tp_info()

        def take(name):
            return sd.pop(name).to(dtype)

        self.pre_projection.load(take("pre_projection.weight"), device)
        self.post_projection.load(take("post_projection.weight"), device)
        for i, layer in enumerate(self.layers):
            p = f"model.layers.{i}."
            for norm in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                         "post_feedforward_layernorm"):
                getattr(layer, norm).weight = take(p + norm + ".weight").to(device)
            layer.q_norm.weight = take(p + "self_attn.q_norm.weight").to(device)
            layer.q_proj.load(take(p + "self_attn.q_proj.weight"), device)
            layer.o_proj.load(take(p + "self_attn.o_proj.weight"), device)
            gate, up = take(p + "mlp.gate_proj.weight"), take(p + "mlp.up_proj.weight")
            n = gate.shape[0] // tp.size
            sl = slice(tp.rank * n, (tp.rank + 1) * n)
            layer.gate_up.load(torch.cat([gate[sl], up[sl]], dim=0), device)
            layer.down_proj.load(take(p + "mlp.down_proj.weight"), device)
            layer.layer_scalar = take(p + "layer_scalar").to(device)
        self.norm.weight = take("model.norm.weight").to(device)
        emb = take("model.embed_tokens.weight")
        start, count = self.lm_head.vocab_range
        w = torch.zeros(self.lm_head.num_embeddings_tp, self.hidden, dtype=dtype)
        w[:count] = emb[start:start + count]
        self.lm_head.weight = w.to(device)
        sd.pop("lm_head.weight", None)
        if sd:
            raise ValueError(f"gemma4_assistant: unexpected checkpoint tensors {sorted(sd)[:8]}")

    def step(self, embed: torch.Tensor, seed: torch.Tensor, pos: torch.Tensor,
             attn: Tuple[tuple, tuple], decode, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """One draft step. `embed` is the target's scaled embedding of the input token; `attn` is
        ((sliding target, block_table, ctx_lens), (full target, block_table, ctx_lens)).
        Returns (argmax token [n], next seed [n, backbone])."""
        x = self.pre_projection.forward(torch.cat([embed, seed.to(embed.dtype)], dim=-1))
        for t, layer in zip(self.layer_types, self.layers):
            tgt, bt, lens = attn[0] if t == "sliding_attention" else attn[1]
            x = layer.forward(x, pos, tgt, bt, lens, decode, scale)
        d = self.norm.forward(x)
        tok = self.lm_head.logits_all_rows(d).argmax(dim=-1)
        return tok, self.post_projection.forward(d)


__all__ = ["Gemma4AssistantDraft", "_AttnTarget"]
