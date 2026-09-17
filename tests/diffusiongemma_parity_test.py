"""DiffusionGemma NUMERICAL parity against the HuggingFace reference — both execution roles.

CPU-only, float32 — needs no GPU and cannot disturb a running serve. Run inside the serve
image (the host torch install is broken, and the image ships `transformers.models.diffusion_gemma`):

    docker run --rm --entrypoint bash -v <worktree>:/wt \
      -v ${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
      minisgl-rdna4:lean -lc 'cd /wt && PYTHONPATH=/wt/python:/opt/kernels \
        python tests/diffusiongemma_parity_test.py'

`diffusiongemma_build_test.py` proves the right tensors arrive on the right modules. This one proves
the forward MATH agrees, which is the only one of the two whose failure mode is silent. It answers
the question the whole port rests on — can ONE instantiated Gemma4 stack serve both the causal
encoder role and the bidirectional canvas role? — by running both through the real minisgl modules
against the real reference:

  [1] the self-conditioning block, on the checkpoint's OWN fp32-widened weights, driven at the
      magnitude a real soft embedding actually has (an embedding row times sqrt(hidden) ~ 53x);
  [2] `soft_embedding`, the [canvas, hidden] state the engine carries between denoising steps in
      place of the reference's [canvas, 262144] logits — must be EXACT, not close, and must not
      depend on the row-chunking that bounds its softmax transient;
  [3] a full 4-layer synthetic stack: minisgl `forward_canvas` vs the reference's whole
      encoder-then-decoder forward, with and without a self-conditioning signal;
  [4] the same stack in its ENCODER role (causal + sliding window) against the reference encoder —
      the two-roles-one-stack claim, measured;
  [5] the semantics that would otherwise be assumed: the decoder really is bidirectional, the
      sliding layers really do NOT window the canvas, and the encoder cache really is truncated to
      sliding_window - 1. Each is stated as a measured DIFFERENCE from the wrong choice, so a future
      change that quietly reintroduces causality or a canvas window fails here.

Everything the reference computes on GPU-only kernels (the routed-expert GEMM) is driven from the
SAME reference module on both sides, so only the surrounding wiring is under test.
"""

from __future__ import annotations

import copy
import dataclasses
import glob
import sys
import types

import torch

MODEL_ID = "cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4"
MODEL_GLOB = f"/root/.cache/huggingface/hub/models--{MODEL_ID.replace('/', '--')}/snapshots/*/"
CKPT = "model.decoder."  # the diffusion_gemma checkpoint namespace

# Synthetic stack for the full-forward checks. head_dim must come from minisgl's supported RoPE set
# {64,128,256,512} (rotary.py asserts it), so the 256/512 split of the real checkpoint is modelled
# as 64/128 — same STRUCTURE (sliding head_dim < full head_dim, more sliding kv heads than full),
# small enough for fp32 CPU. sliding_window 8 against a 40-token prompt puts the encoder cache well
# past the truncation point, which is the case that matters.
SYN = dict(
    hidden_size=128, intermediate_size=64, moe_intermediate_size=32,
    num_attention_heads=4, num_key_value_heads=2, head_dim=64,
    global_head_dim=128, num_global_key_value_heads=1,
    num_hidden_layers=4, vocab_size=128, num_experts=4, top_k_experts=2, sliding_window=8,
)
PROMPT, CANVAS = 40, 6


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.skips = 0

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:46s} {detail}")
        return ok

    def close(self, name: str, got: torch.Tensor, want: torch.Tensor, tol: float) -> bool:
        d = (got.float() - want.float()).abs()
        ref = want.float().abs()
        max_abs = d.max().item()
        scale = ref.max().item()
        rel_fro = (d.norm() / ref.norm().clamp_min(1e-30)).item()
        return self.check(
            name,
            max_abs <= tol * max(scale, 1e-12),
            f"max|abs|={max_abs:.3e}  ref|max|={scale:.3e}  rel_fro={rel_fro:.3e}  tol={tol:.0e}",
        )

    def skip(self, name: str, why: str) -> None:
        self.skips += 1
        print(f"  SKIP {name:46s} {why}")


def _open(folder: str):
    from safetensors import safe_open

    return [
        safe_open(p, framework="pt", device="cpu")
        for p in sorted(glob.glob(folder + "*.safetensors"))
    ]


def _get(handles, key):
    for h in handles:
        if key in h.keys():
            return h.get_tensor(key)
    return None


def _rows(handles, key, ids):
    """A few ROWS of a large tensor without materializing it (embed_tokens is 1.38 GiB)."""
    for h in handles:
        if key in h.keys():
            sl = h.get_slice(key)
            return torch.stack([sl[int(i) : int(i) + 1, :][0] for i in ids])
    return None


# ==============================================================================================
def check_self_conditioning(rep: Report, mc, tc, handles) -> None:
    """[1] the self-conditioning block on the checkpoint's own weights."""
    print("\n[1] self-conditioning block vs DiffusionGemmaSelfConditioning (real weights, fp32)")
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaSelfConditioning as RefSC,
    )

    from minisgl.models.diffusion_gemma import DiffusionGemmaSelfConditioning

    p = CKPT + "self_conditioning."
    gp, up, dp, pn = (
        _get(handles, p + "gate_proj.weight").float(),
        _get(handles, p + "up_proj.weight").float(),
        _get(handles, p + "down_proj.weight").float(),
        _get(handles, p + "pre_norm.weight").float(),
    )
    print(
        f"  (real tensors: gate {tuple(gp.shape)} up {tuple(up.shape)} down {tuple(dp.shape)} "
        f"pre_norm mean={pn.mean():.4f} max={pn.max():.4f}; intermediate_size={mc.intermediate_size}"
        f", moe_intermediate_size={mc.moe_intermediate_size})"
    )

    cfgu = dataclasses.replace(mc, quant=None)
    mine = DiffusionGemmaSelfConditioning(cfgu)
    mine.gate_up_proj.weight = torch.cat([gp, up], dim=0)
    mine.down_proj.weight = dp
    mine.pre_norm.weight = pn.clone()

    ref = RefSC(tc).float()
    with torch.no_grad():
        ref.gate_proj.weight.copy_(gp)
        ref.up_proj.weight.copy_(up)
        ref.down_proj.weight.copy_(dp)
        ref.pre_norm.weight.copy_(pn)

    # Drive it at the magnitude a REAL signal has. The self-conditioning signal is
    # softmax(logits) @ embed_tokens * sqrt(hidden); in the confident limit that is one embedding
    # ROW times 53.06, so real rows off the checkpoint are the honest probe. A unit-normal probe
    # would sit two orders of magnitude low and hide a scale error entirely.
    torch.manual_seed(21)
    ids = torch.randint(0, mc.vocab_size, (7,))
    rows = _rows(handles, CKPT + "embed_tokens.weight", ids).float()
    signal = rows * mc.embed_scale
    embeds = _rows(handles, CKPT + "embed_tokens.weight", torch.randint(0, mc.vocab_size, (7,)))
    embeds = embeds.float() * mc.embed_scale
    print(
        f"  (probe: |signal|max={signal.abs().max():.3f} rms={signal.pow(2).mean().sqrt():.4f}; "
        f"a unit-normal probe would be rms=1.0)"
    )
    rep.close("self-conditioning forward", mine.forward(embeds, signal), ref(embeds, signal), 1e-5)

    # The zero-signal first step: minisgl short-circuits the MLP, the reference runs it on zeros.
    # These must be EXACTLY equal (RMSNorm(0)=0, gelu(0)*0=0, down_proj has no bias) -- and both
    # must still differ from the raw embedding, because post_norm applies unconditionally.
    got0 = mine.forward(embeds, None)
    want0 = ref(embeds, torch.zeros_like(embeds))
    rep.check(
        "zero-signal short-circuit is EXACT",
        torch.equal(got0, want0),
        f"max|abs|={(got0 - want0).abs().max().item():.3e} (must be exactly 0)",
    )
    rep.check(
        "post_norm applies on step 1 too",
        (got0 - embeds).abs().max().item() > 1e-3,
        f"|post_norm(e) - e|max={(got0 - embeds).abs().max().item():.3e} — skipping the block on "
        f"the first denoising step would feed layer 0 a differently-scaled input than every "
        f"other step",
    )


# ==============================================================================================
def check_soft_embedding(rep: Report, mc, handles) -> None:
    """[2] the [canvas, hidden] state carried between denoising steps."""
    print("\n[2] soft_embedding — the carried state, against the reference's logits@E")
    import minisgl.models.diffusion_gemma as dg_mod
    from minisgl.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion

    # num_layers=0 keeps embed_tokens + the real vocab on the real code path at CPU cost.
    vocab = 4096
    cfg = dataclasses.replace(
        mc, quant=None, num_layers=0, vocab_size=vocab, tie_word_embeddings=True
    )
    torch.set_default_dtype(torch.float32)
    model = DiffusionGemmaForBlockDiffusion(cfg)
    torch.manual_seed(31)
    # Real embedding rows, so the output magnitude is the real one.
    ids = torch.randint(0, mc.vocab_size, (vocab,))
    model.model.embed_tokens.weight = _rows(handles, CKPT + "embed_tokens.weight", ids).float()

    # `soft_embedding` now consumes the SAMPLER'S softmax rather than building its own, so the input
    # under test is `normalized_probs(l)[1]` — the Categorical-shifted softmax — while the reference
    # quantity is still `softmax(l) @ E`. Proving those two agree IS the test: it is the substitution
    # the whole saving rests on, and it is the one place a "mathematically identical" claim about an
    # fp32 shift can be checked instead of asserted.
    from minisgl.diffusion import normalized_probs

    rows = 128
    logits = torch.randn(rows, vocab) * 4.0
    probs = normalized_probs(logits)[1]
    got = model.soft_embedding(probs)
    want = (logits.softmax(dim=-1, dtype=torch.float32) @ model.model.embed_tokens.weight) * (
        model.model._scale_tensor(logits)
    )
    rep.close("soft_embedding(sampler probs) vs softmax(l) @ E * sqrt(h)", got, want, 1e-6)
    print(f"       (|out|max={got.abs().max().item():.3f} over {rows} rows, "
          f"vocab chunk={dg_mod._SOFT_EMBED_VOCAB_CHUNK or 'whole shard'})")

    # THE CHUNK AXIS IS VOCAB, NOT ROWS, and that is the entire fix: a row chunk re-streams the whole
    # embedding shard per chunk (8x/step measured, 5.9 GB against a 0.74 GB floor), a vocab chunk
    # re-streams nothing because the slices are disjoint. Blocking K changes only the fp32 summation
    # order, so the two must agree to rounding — checked here, because a wrong slice offset would
    # still produce plausible embeddings.
    saved = dg_mod._SOFT_EMBED_VOCAB_CHUNK
    try:
        dg_mod._SOFT_EMBED_VOCAB_CHUNK = vocab // 8
        blocked = model.soft_embedding(probs)
    finally:
        dg_mod._SOFT_EMBED_VOCAB_CHUNK = saved
    rep.close(f"vocab-chunked ({vocab // 8} cols) == one GEMM", blocked, got, 1e-6)

    # A caller may hand over ONE RANK'S SHARD instead of the full-vocab row (that is what the
    # vocab-parallel canvas tail does under TP). At tp_size=1 the shard IS the whole vocab, so the
    # discrimination the served path relies on — width == count means "already local" — is exercised
    # by the sharded arithmetic itself: sum over disjoint column blocks must reproduce the whole.
    start, count = model.model.embed_tokens.vocab_range
    rep.check(
        "single-rank shard covers the whole vocab (no silent slice)",
        (start, count) == (0, vocab),
        f"vocab_range={(start, count)} at tp_size=1; if this were a proper sub-range the width test "
        f"in soft_embedding would mis-route a full-vocab row",
    )
    canvas_length = 256

    # Carrying the SOFT EMBEDDING rather than the logits is the whole point: quantify what it saves.
    n_logits = canvas_length * mc.vocab_size
    n_soft = canvas_length * mc.hidden_size
    rep.check(
        "carried state is [canvas, hidden], not [canvas, vocab]",
        tuple(got.shape) == (rows, mc.hidden_size),
        f"{tuple(got.shape)}; at the real canvas_length {canvas_length} that is "
        f"{n_soft * 2 / 2**20:.1f} MiB carried instead of {n_logits * 2 / 2**20:.0f} MiB "
        f"({n_logits / n_soft:.0f}x)",
    )


# ==============================================================================================
def synthetic_config(path: str):
    """A small DiffusionGemmaConfig with the real one's SHAPE.

    `per_layer_config` is set to None deliberately. In transformers 5.14.1 (the shipped version) it
    is already None on the real config and both attention classes read `config.global_head_dim` /
    `config.head_dim` directly; populating it instead marks the config HETEROGENEOUS and every
    `config.head_dim` read then raises AmbiguousGlobalPerLayerAttributeError. (The port's own spec
    doc describes the opposite gotcha, from an older reference copy — see the note in the summary.)
    """
    from transformers import AutoConfig

    cfg = copy.deepcopy(AutoConfig.from_pretrained(path))
    tc = cfg.text_config
    tc.per_layer_config = None
    for key, value in SYN.items():
        setattr(tc, key, value)
    tc.layer_types = ["sliding_attention"] * (SYN["num_hidden_layers"] - 1) + ["full_attention"]
    cfg.canvas_length = CANVAS
    for c in (cfg, tc):
        if hasattr(c, "quantization_config"):
            delattr(c, "quantization_config")  # the synthetic stack is dense fp32
    return cfg


class _EagerAttention:
    """The attention KERNEL, in torch, so the comparison isolates the MODEL.

    Serves both roles off one flag. Canvas: the queries attend over [encoder cache | canvas] with NO
    mask at all — not causal, and NOT windowed on the sliding layers (the window is enforced upstream
    by what the encoder cache retained, never by a mask; see the reference's own "DiT module has to
    attend fully" comment). Encoder: causal, and windowed to the last `sliding_window` keys
    inclusive of self on the sliding layers. Softmax scale is 1.0 on both — the temperature lives in
    the learned k_norm."""

    def __init__(self, layer_of, cache=None, causal: bool = False):
        self.layer_of = layer_of
        self.cache = cache
        self.causal = causal

    def forward(self, q, k, v, layer_id, batch, sliding_window=0):
        n, heads, dim = q.shape
        kk, vv = k.view(n, -1, dim), v.view(n, -1, dim)
        prefix = 0
        if self.cache is not None:
            layer = self.cache.layers[self.layer_of[(sliding_window > 0, layer_id)]]
            prefix = layer.keys.shape[2]
            kk = torch.cat([layer.keys[0].permute(1, 0, 2), kk], dim=0)
            vv = torch.cat([layer.values[0].permute(1, 0, 2), vv], dim=0)
        rep = heads // kk.shape[1]
        scores = torch.einsum("qhd,khd->hqk", q, kk.repeat_interleave(rep, dim=1))
        if self.causal:
            qi = torch.arange(n).unsqueeze(1)
            ki = torch.arange(prefix + n).unsqueeze(0) - prefix
            allowed = ki <= qi
            if sliding_window:
                allowed &= qi - ki < sliding_window
            scores = scores.masked_fill(~allowed, float("-inf"))
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.einsum("hqk,khd->qhd", attn, vv.repeat_interleave(rep, dim=1)).contiguous()


def _build_minisgl(mc, hf_model):
    """minisgl model holding the reference's weights; the routed experts ARE the reference's."""
    import types as _types

    from minisgl.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion
    from minisgl.models.gemma4 import gemma4_layer_plan

    model = DiffusionGemmaForBlockDiffusion(mc)
    layer_of = {}
    for i in range(mc.num_layers):
        plan = gemma4_layer_plan(mc, i)
        layer_of[(plan.is_sliding, plan.kv_id)] = i
    for i, layer in enumerate(model.model.layers.op_list):
        ref_experts = hf_model.model.decoder.layers[i].experts
        layer.experts = _types.SimpleNamespace(
            forward=lambda hidden_states, topk_weights, topk_ids, reduce=True, _e=ref_experts: _e(
                hidden_states, topk_ids, topk_weights
            )
        )
    sd = {}
    for key, tensor in hf_model.state_dict().items():
        if not key.startswith(CKPT):
            continue
        native = "model." + key.removeprefix(CKPT)
        if ".experts." in native:
            continue
        if native.endswith(".router.proj.weight"):
            native = native.replace(".router.proj.weight", ".router.weight")
        sd[native] = tensor.clone()
    for prefix in ["model.self_conditioning"] + [
        f"model.layers.{i}.mlp" for i in range(mc.num_layers)
    ]:
        gate, up = sd.pop(f"{prefix}.gate_proj.weight"), sd.pop(f"{prefix}.up_proj.weight")
        sd[f"{prefix}.gate_up_proj.weight"] = torch.cat([gate, up], dim=0)
    model.load_state_dict(sd)  # raises on ANY leftover or missing key
    return model, layer_of


def _run_minisgl(model, fn, positions, backend):
    import minisgl.layers.attention as attn_mod

    ctx = types.SimpleNamespace(
        batch=types.SimpleNamespace(positions=positions), attn_backend=backend
    )
    saved = attn_mod.get_global_ctx
    attn_mod.get_global_ctx = lambda: ctx
    try:
        return fn()
    finally:
        attn_mod.get_global_ctx = saved


def check_full_stack(rep: Report, path: str) -> None:
    """[3][4][5] the whole stack, both roles, against the whole reference."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaForBlockDiffusion as HFDG,
    )

    from minisgl.models.config import ModelConfig

    cfg = synthetic_config(path)
    tc = cfg.text_config
    hf = HFDG(cfg)
    torch.manual_seed(0)
    for p in hf.parameters():
        p.normal_(0, 0.3)
    for name, buf in hf.named_buffers():
        if name.endswith("layer_scalar"):
            buf.uniform_(0.4, 0.9)  # the real ones run ~0.1-0.7; never leave them at 1.0
    # `named_buffers` hands back the encoder's and the decoder's layer_scalar SEPARATELY — they are
    # nn.Buffers, which HF's tying machinery does not tie, which is the entire reason the checkpoint
    # ships 30 encoder tensors and the loader asserts them equal. Randomizing them independently
    # would build a model whose two roles are genuinely different, and check [4] below would then be
    # measuring that instead of the port. Tie them here, exactly as the checkpoint does.
    for i, layer in enumerate(hf.model.encoder.language_model.layers):
        layer.layer_scalar.copy_(hf.model.decoder.layers[i].layer_scalar)
    hf.eval()

    mc = dataclasses.replace(
        ModelConfig.from_hf(cfg, spec_algorithm="none"), quant=None
    )
    print(
        f"\n[3] full stack, CANVAS role — synthetic {mc.num_layers}-layer "
        f"(head_dim {mc.swa_head_dim} sliding / {mc.head_dim} full, "
        f"kv {mc.swa_num_kv_heads}/{mc.num_kv_heads}, k_eq_v={mc.attention_k_eq_v}, "
        f"window={mc.sliding_window}, scale={mc.attn_softmax_scale})"
    )
    model, layer_of = _build_minisgl(mc, hf)

    torch.manual_seed(3)
    prompt = torch.randint(0, tc.vocab_size, (1, PROMPT))
    canvas = torch.randint(0, tc.vocab_size, (1, CANVAS))
    cpos = torch.arange(PROMPT, PROMPT + CANVAS)

    def hf_canvas(cv, sc_logits):
        cache = DynamicCache(config=tc)
        out = hf(
            input_ids=prompt,
            decoder_input_ids=cv,
            past_key_values=cache,
            self_conditioning_logits=sc_logits,
            decoder_position_ids=cpos.unsqueeze(0),
        )
        return out.logits[0], cache

    for label, sc_logits in (
        ("no signal (denoising step 1)", None),
        ("with signal (steps 2..N)", torch.randn(1, CANVAS, tc.vocab_size, generator=(
            torch.Generator().manual_seed(7))) * 2.0),
    ):
        want, cache = hf_canvas(canvas, sc_logits)
        # HF takes the raw self-conditioning LOGITS and softmaxes them inside; this engine's
        # `soft_embedding` takes the softmax, because on the served path the sampler already built
        # it. Softmaxing here is not a shortcut around the parity claim — it is the same op HF is
        # about to do, hoisted to the call site.
        soft = (None if sc_logits is None
                else model.soft_embedding(sc_logits[0].softmax(dim=-1, dtype=torch.float32)))
        got = _run_minisgl(
            model,
            lambda: model.forward_canvas(canvas[0], soft),
            cpos,
            _EagerAttention(layer_of, cache=cache, causal=False),
        )
        rep.close(f"canvas logits, {label}", got, want, 2e-6)

    # ---- [4] the SAME stack in the encoder role -------------------------------------------
    print("\n[4] full stack, ENCODER role — the same instantiated stack, causal + windowed")
    _, cache = hf_canvas(canvas, None)
    enc_hidden = hf.model.encoder(
        input_ids=prompt, past_key_values=DynamicCache(config=tc)
    ).last_hidden_state[0]
    got_enc = _run_minisgl(
        model,
        lambda: model.model.forward(prompt[0]),
        torch.arange(PROMPT),
        _EagerAttention(layer_of, cache=None, causal=True),
    )
    rep.close("encoder final hidden (causal + window)", got_enc, enc_hidden, 2e-6)

    # ---- [5] the semantics that would otherwise be assumed ---------------------------------
    print("\n[5] the decoder semantics, each stated as a measured difference from the wrong choice")
    want, cache = hf_canvas(canvas, None)

    # Bidirectional: perturbing the LAST canvas token must move canvas position 0. A causal decoder
    # would leave every earlier position bit-identical.
    perturbed = canvas.clone()
    perturbed[0, -1] = (perturbed[0, -1] + 1) % tc.vocab_size
    want_p, _ = hf_canvas(perturbed, None)
    d0 = (want_p[0] - want[0]).abs().max().item()
    dl = (want_p[-1] - want[-1]).abs().max().item()
    rep.check(
        "reference decoder is BIDIRECTIONAL",
        d0 > 1e-4,
        f"changing canvas[{CANVAS - 1}] moves canvas[0] logits by {d0:.3e} "
        f"(and canvas[{CANVAS - 1}] by {dl:.3e}); a causal decoder would give exactly 0",
    )

    # Causality is what a naive port would inherit from the AR path: price it.
    got_causal = _run_minisgl(
        model,
        lambda: model.forward_canvas(canvas[0], None),
        cpos,
        _EagerAttention(layer_of, cache=cache, causal=True),
    )
    rep.check(
        "serving the canvas CAUSALLY is a different model",
        (got_causal - want).abs().max().item() > 1e-2,
        f"causal=1 on the canvas moves the logits by max|abs|="
        f"{(got_causal - want).abs().max().item():.3e} (ref|max|={want.abs().max().item():.3e})",
    )

    # The sliding layers must NOT window the canvas. The window lives in what the encoder cache
    # retained; re-applying it as a mask over [prefix|canvas] is a DIFFERENT model.
    class _Windowed(_EagerAttention):
        def forward(self, q, k, v, layer_id, batch, sliding_window=0):
            n, heads, dim = q.shape
            layer = self.cache.layers[self.layer_of[(sliding_window > 0, layer_id)]]
            prefix = layer.keys.shape[2]
            kk = torch.cat([layer.keys[0].permute(1, 0, 2), k.view(n, -1, dim)], dim=0)
            vv = torch.cat([layer.values[0].permute(1, 0, 2), v.view(n, -1, dim)], dim=0)
            rep_ = heads // kk.shape[1]
            scores = torch.einsum("qhd,khd->hqk", q, kk.repeat_interleave(rep_, dim=1))
            if sliding_window:
                qi = torch.arange(n).unsqueeze(1) + prefix
                ki = torch.arange(prefix + n).unsqueeze(0)
                scores = scores.masked_fill((qi - ki).abs() >= sliding_window, float("-inf"))
            attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            return torch.einsum(
                "hqk,khd->qhd", attn, vv.repeat_interleave(rep_, dim=1)
            ).contiguous()

    got_win = _run_minisgl(
        model,
        lambda: model.forward_canvas(canvas[0], None),
        cpos,
        _Windowed(layer_of, cache=cache, causal=False),
    )
    rep.check(
        "a symmetric canvas window is a different model",
        (got_win - want).abs().max().item() > 1e-3,
        f"a +/-{mc.sliding_window} window on the 3 sliding layers moves the logits by max|abs|="
        f"{(got_win - want).abs().max().item():.3e} — the window is a property of what the encoder "
        f"cache RETAINED, never a mask over the canvas",
    )

    # The cache truncation itself: sliding_window - 1, not sliding_window.
    kept = [cache.layers[i].keys.shape[2] for i in range(mc.num_layers)]
    rep.check(
        "encoder cache keeps sliding_window - 1 prefix keys",
        kept[:-1] == [mc.sliding_window - 1] * (mc.num_layers - 1) and kept[-1] == PROMPT,
        f"prompt={PROMPT} window={mc.sliding_window}: sliding layers keep {kept[:-1]} "
        f"(= window-1), the full layer keeps {kept[-1]}; at the real window that is 1023, not 1024",
    )
    rep.check(
        "cache position bookkeeping survives truncation",
        cache.get_seq_length(layer_idx=0) == PROMPT,
        f"get_seq_length(sliding layer 0) = {cache.get_seq_length(layer_idx=0)} (the CUMULATIVE "
        f"length, not the {kept[0]} keys physically held) — so the canvas positions start at "
        f"cur_len, and the reference's default decoder_position_ids are right",
    )


# ==============================================================================================
def main() -> int:
    matches = glob.glob(MODEL_GLOB)
    if not matches:
        print(f"SKIP: {MODEL_ID} not cached under {MODEL_GLOB}")
        return 0
    path = matches[0]

    from minisgl.distributed import set_tp_info

    set_tp_info(0, 1)
    import minisgl.layers.rotary as rotary_mod
    from transformers import AutoConfig

    from minisgl.models.config import ModelConfig

    rotary_mod.set_rope_device(torch.device("cpu"))
    torch.set_default_dtype(torch.float32)
    torch.set_grad_enabled(False)

    hf = AutoConfig.from_pretrained(path)
    mc = ModelConfig.from_hf(hf, spec_algorithm="none")
    handles = _open(path)

    print(
        f"[parity] {MODEL_ID}  hidden={mc.hidden_size} vocab={mc.vocab_size} "
        f"embed_scale={mc.embed_scale:.6f} softcap={mc.final_logit_softcapping}"
    )

    rep = Report()
    check_self_conditioning(rep, mc, hf.text_config, handles)
    check_soft_embedding(rep, mc, handles)
    check_full_stack(rep, path)

    print(
        f"\n{'PASS' if rep.failures == 0 else f'FAIL ({rep.failures} checks)'}"
        f"{f'  [{rep.skips} skipped]' if rep.skips else ''}"
    )
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
