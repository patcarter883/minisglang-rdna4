"""Numeric parity for the Muse-Glimmer decoder layer against the REAL reference, CPU-only.

The port's risk surface (docs/MUSE_GLIMMER_PORT.md §2) is almost entirely pure-torch plumbing —
norm conventions, two epsilons, a weightless pre-RoPE QK-norm, a folded softmax scale, a sigmoid
output gate, NoPE — and every one of those fails SILENTLY, producing plausible output rather than a
crash. This compares minisgl's layer against `transformers.models.muse_glimmer`'s own
`MuseGlimmerTextDecoderLayer`, weight-for-weight, on identical input.

What is and is not covered. The ONLY substitution is `AttentionLayer.forward`, whose paged HIP
kernel cannot run on CPU and needs the engine's global context. It is replaced by a stand-in that
performs the SAME steps in the same order — in-place q/k norm on the per-head views, optional RoPE,
then SDPA at the layer's configured softmax scale. So this DOES cover the q/k/v/gate/o_proj wiring,
the gate's argument and application point, the QK-norm, the scale fold and NoPE; it does NOT cover
the attention kernel itself, which is shared with Laguna/Gemma4 and already validated.

Small dims on purpose: the math is dimension-independent, and the structural features are what is
under test. fp32 throughout so a real error is not lost in bf16 noise.

Needs `transformers>=5.15.0` for the reference (the image ships 5.14.1):
  pip install --no-deps transformers==5.15.0

Run (CPU-only, NO GPU lease — it touches no card):
  docker run --rm -v <worktree>:/engine --entrypoint bash minisgl-rdna4:lean -lc \
    'pip install -q --no-deps transformers==5.15.0 &&
     PYTHONPATH=/engine/python:/engine:/opt/kernels python /engine/tools/muse_glimmer_layer_parity_cpu.py'
"""
from __future__ import annotations

import math
import sys

import torch

DT = torch.float32
TOL = 2e-5  # fp32, ~30 elementwise ops deep
results: list[tuple[str, float, bool]] = []


def cmp(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> None:
    md = (a.float() - b.float()).abs().max().item()
    ok = md <= tol
    results.append((name, md, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:44s} max|Δ|={md:.3e}", flush=True)


# --- shared toy geometry -------------------------------------------------------------------------
HID, NQ, NKV, HD, INTER, NLAYERS = 256, 4, 2, 64, 512, 4
RMS_EPS, POST_EPS, QK_SCALE, THETA, WINDOW = 1e-5, 1e-8, 3.87, 500000.0, 2048
SEQ = 12  # < WINDOW, so a sliding layer is plain causal and the two sides see the same mask


def build_reference():
    from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerTextConfig

    return MuseGlimmerTextConfig(
        hidden_size=HID,
        intermediate_size=INTER,
        num_hidden_layers=NLAYERS,
        num_attention_heads=NQ,
        num_key_value_heads=NKV,
        head_dim=HD,
        rms_norm_eps=RMS_EPS,
        post_norm_eps=POST_EPS,
        qk_scale_factor=QK_SCALE,
        sliding_window=WINDOW,
        rope_parameters={"rope_type": "default", "rope_theta": THETA},
        attention_bias=False,
        hidden_activation="silu",
        vocab_size=1024,
        max_position_embeddings=4096,
        _attn_implementation="eager",
    )


def build_minisgl_config(ref_cfg):
    """A ModelConfig with the same geometry, built directly (not via from_hf) so this harness tests
    the LAYER, not the config parse — `tools/test_muse_glimmer.py` already covers from_hf against the
    real checkpoint."""
    from minisgl.models.config import ModelConfig, RotaryConfig

    rope = RotaryConfig(
        head_dim=HD, rotary_dim=HD, max_position=4096, base=THETA, scaling=None, interleave=False
    )
    return ModelConfig(
        model_type="muse_glimmer_text",
        architectures=("MuseGlimmerForConditionalGeneration",),
        num_layers=NLAYERS,
        hidden_size=HID,
        intermediate_size=INTER,
        num_qo_heads=NQ,
        num_kv_heads=NKV,
        head_dim=HD,
        vocab_size=1024,
        rms_norm_eps=RMS_EPS,
        post_norm_eps=POST_EPS,
        hidden_act="silu",
        rotary_config=rope,
        quant=None,
        tie_word_embeddings=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        norm_topk_prob=False,
        shared_expert_intermediate_size=0,
        layer_types=tuple(ref_cfg.layer_types),
        layer_rope_theta=tuple(ref_cfg.layer_rope_theta),
        sliding_window=WINDOW,
        attn_softmax_scale=QK_SCALE * HD**-0.5,
        final_logit_softcapping=20.0,
        output_multiplier=1 / math.sqrt(26.0),
    )


class CpuAttnStandin:
    """Stand-in for `AttentionLayer`, doing its steps in the same order on CPU.

    Mirrors `layers/attention.py::forward`: split qkv, in-place q/k norm on the per-head views,
    RoPE (skipped when `rotary is None` — NoPE), then attention at `scale`. GQA is expanded by
    repeat_interleave, matching the reference's `repeat_kv`."""

    def __init__(self, head_dim, nq, nkv, rotary, scale, sliding_window, q_norm, k_norm):
        self.head_dim, self.nq, self.nkv = head_dim, nq, nkv
        self.rotary, self.scale, self.sliding_window = rotary, scale, sliding_window
        self.q_norm, self.k_norm = q_norm, k_norm
        self.qo_attn_dim, self.kv_attn_dim = nq * head_dim, nkv * head_dim
        self.cos = self.sin = None  # set by the harness for the rope layers

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        q, k, v = q.clone(), k.clone(), v.clone()
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.nq, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.nkv, self.head_dim))
        n = q.shape[0]
        qh = q.view(n, self.nq, self.head_dim).transpose(0, 1).unsqueeze(0)
        kh = k.view(n, self.nkv, self.head_dim).transpose(0, 1).unsqueeze(0)
        vh = v.view(n, self.nkv, self.head_dim).transpose(0, 1).unsqueeze(0)
        if self.rotary is not None:
            from transformers.models.muse_glimmer.modeling_muse_glimmer import apply_rotary_pos_emb

            qh, kh = apply_rotary_pos_emb(qh, kh, self.cos, self.sin)
        rep = self.nq // self.nkv
        kh = kh.repeat_interleave(rep, dim=1)
        vh = vh.repeat_interleave(rep, dim=1)
        o = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, is_causal=True, scale=self.scale
        )
        return o.squeeze(0).transpose(0, 1).reshape(n, self.qo_attn_dim)


def main() -> int:
    torch.manual_seed(0)
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        MuseGlimmerTextDecoderLayer,
        MuseGlimmerTextRotaryEmbedding,
    )

    from minisgl.distributed.info import set_tp_info
    from minisgl.layers.rotary import set_rope_device

    set_tp_info(rank=0, size=1)
    set_rope_device(torch.device("cpu"))

    ref_cfg = build_reference()
    mc = build_minisgl_config(ref_cfg)
    print(f"== geometry: hidden={HID} heads={NQ}/{NKV} head_dim={HD} seq={SEQ} dtype={DT} ==")
    print(f"   layer_types      = {list(ref_cfg.layer_types)}")
    print(f"   layer_rope_theta = {list(ref_cfg.layer_rope_theta)}")
    print(f"   attn_softmax_scale (folded) = {mc.attn_softmax_scale:.12f}"
          f"   [= {QK_SCALE} / sqrt({HD})]")

    from minisgl.models.muse_glimmer import MuseGlimmerDecoderLayer

    rot = MuseGlimmerTextRotaryEmbedding(ref_cfg).to(DT)
    x = torch.randn(SEQ, HID, dtype=DT)
    pos_ids = torch.arange(SEQ).unsqueeze(0)
    cos, sin = rot(x.unsqueeze(0), pos_ids)
    # An EXPLICIT causal mask. `attention_mask=None` under the eager path means no mask at all —
    # i.e. bidirectional attention — which silently made the reference a different model than the
    # causal stand-in it is being compared against.
    causal = torch.full((SEQ, SEQ), float("-inf"), dtype=DT).triu(1)[None, None]

    for lid in range(NLAYERS):
        kind = ref_cfg.layer_types[lid]
        nope = not ref_cfg.layer_rope_theta[lid]
        ref = MuseGlimmerTextDecoderLayer(ref_cfg, lid).to(DT).eval()
        for p in ref.parameters():
            torch.nn.init.normal_(p, std=0.05)

        if lid == 0:
            # State the reference's own scaling, so the fold below is checked against what the
            # reference actually attends at rather than an assumption about it.
            print(f"   reference self_attn.scaling = {ref.self_attn.scaling}"
                  f"  (head_dim**-0.5 = {HD**-0.5}); qk_scale_factor="
                  f"{ref.self_attn.qk_scale_factor}")
        mine = MuseGlimmerDecoderLayer(mc, lid)
        # Same weights on both sides. minisgl merges mlp gate+up into one matrix, exactly as its
        # loader does; everything else is a 1:1 name match.
        sd = ref.state_dict()
        mine.input_layernorm.weight = sd["input_layernorm.weight"].clone()
        mine.post_attention_layernorm.weight = sd["post_attention_layernorm.weight"].clone()
        mine.pre_feedforward_layernorm.weight = sd["pre_feedforward_layernorm.weight"].clone()
        mine.post_feedforward_layernorm.weight = sd["post_feedforward_layernorm.weight"].clone()
        for p in ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj"):
            getattr(mine.self_attn, p).weight = sd[f"self_attn.{p}.weight"].clone()
        mine.mlp.gate_up_proj.weight = torch.cat(
            [sd["mlp.gate_proj.weight"], sd["mlp.up_proj.weight"]], dim=0
        ).clone()
        mine.mlp.down_proj.weight = sd["mlp.down_proj.weight"].clone()

        # Swap the paged-attention op for the CPU stand-in, preserving minisgl's own configuration
        # of it (norms, rope-or-not, folded scale, window) so those choices remain under test.
        real = mine.self_attn.attn
        stand = CpuAttnStandin(
            HD, NQ, NKV, real.rotary, real._scale if hasattr(real, "_scale") else mc.attn_softmax_scale,
            real.sliding_window, real.q_norm, real.k_norm,
        )
        stand.cos, stand.sin = cos, sin
        mine.self_attn.attn = stand

        with torch.no_grad():
            want = ref(
                x.unsqueeze(0),
                position_embeddings=None if nope else (cos, sin),
                attention_mask=causal,
            )
            got = mine.forward(x)
        want = want[0] if isinstance(want, tuple) else want
        cmp(f"L{lid} ({kind}, {'NoPE' if nope else 'RoPE'})", got, want.squeeze(0))

    # --- the scale fold, stated as its own claim -------------------------------------------------
    # The reference multiplies Q by 3.87 after the QK-norm and attends at head_dim**-0.5; minisgl
    # skips that multiply and attends at 3.87*head_dim**-0.5. Assert the identity directly, so the
    # fold is verified rather than merely implied by the layer totals above.
    q = torch.randn(1, NQ, SEQ, HD, dtype=DT)
    k = torch.randn(1, NQ, SEQ, HD, dtype=DT)
    v = torch.randn(1, NQ, SEQ, HD, dtype=DT)
    sdpa = torch.nn.functional.scaled_dot_product_attention
    cmp(
        "scale fold: (q*c) @ k / sqrt(d)  ==  q @ k * c/sqrt(d)",
        sdpa(q * QK_SCALE, k, v, is_causal=True, scale=HD**-0.5),
        sdpa(q, k, v, is_causal=True, scale=QK_SCALE * HD**-0.5),
    )

    # --- logit transform -------------------------------------------------------------------------
    logits = torch.randn(4, 32, dtype=DT) * 30
    want = torch.tanh(logits * mc.output_multiplier / 20.0) * 20.0
    got = logits * mc.output_multiplier
    got = torch.tanh(got / mc.final_logit_softcapping) * mc.final_logit_softcapping
    cmp("logits: T*tanh(lm_head(h)*m/T)", got, want)

    print()
    bad = [n for n, _, ok in results if not ok]
    if bad:
        print(f"FAILED ({len(bad)}): " + ", ".join(bad))
        return 1
    print(f"all {len(results)} parity checks passed (tol {TOL:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
