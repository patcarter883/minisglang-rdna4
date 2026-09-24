"""Parity gate for the DFlash drafter's O(window) eager propose (`denoise_cached`).

WHAT IS BEING CLAIMED, and what is NOT.

The eager propose (a) window-slices the prefix to the rows the drafter's causal + sliding-window mask
can reach, and (b) runs the block attention on the HIP paged flash kernel
(attn_prefill_paged.flash_prefill_paged, models/dflash.py `drafter_attend`), not in torch.

  * (a) is EXACT in exact arithmetic: every dropped row is -inf for every query, so it contributes
    exp(-inf - max) == 0.0. That is checked STRUCTURALLY below (asserts), not numerically.
  * (b) is NOT bit-identical to the torch einsum/softmax it replaced, and is not required to be. A
    drafted token only changes ACCEPTANCE — the target verifies every draft, so a drafter's numerics
    are a speed question, never a correctness one. (The previous version of this file required
    drafted-token identity with the torch path and used a ~1-ULP argmax flip to justify keeping torch
    attention; that requirement is withdrawn.)

THE CONTRACT this gate enforces: the shipped path is numerically CLOSE to an fp32-attention
reference — the pre-change full-prefix `repeat_interleave` formulation with q/K/V upcast to fp32 —
and no further from it than the bf16 torch path it replaced:
  (1) max|delta logit| (shipped vs fp32 ref) <= the torch-bf16 path's own max|delta logit| x 1.5,
  (2) drafted-token argmax agreement with the fp32 ref >= the torch-bf16 path's agreement - 1%.
Both are REPORTED per prefix length spanning P < W, P == W and P >> W. Random weights (near-uniform
logits, tiny top-1 margins) make this an adversarial stand-in for a trained drafter; the real-weight
numbers are in tools/dflash_drafter_attn_bench.py.

Run inside the serve image on one leased card:
  gpu-lease -n 1 -- docker run --rm <ROCm device flags> -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES \
      -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES -v <worktree>:/engine \
      -e PYTHONPATH=/engine/python:/opt/kernels minisgl-rdna4:<tag> \
      python /engine/tools/dflash_window_parity.py
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/engine/python")

from minisgl.distributed import set_tp_info  # noqa: E402

set_tp_info(0, 1)  # standalone (no engine): the HIP-engage logger reads tp info on first op

from minisgl.models.dflash import DFlashDraftModel  # noqa: E402

# Laguna-XS-2.1-DFlash geometry (config.json of poolside/Laguna-XS-2.1-DFlash-NVFP4), trunk only.
HIDDEN = 2048
INTER = 8192
LAYERS = 5
HEADS = 64
KV_HEADS = 8
HEAD_DIM = 128
NUM_AUX = 5
WINDOW = 512
BLOCK = 16
EPS = 1e-6
ROPE_THETA = 500000.0


def _ref_attend_block(layer, hidden, block_pos, k_ctx, v_ctx, attn_mask, fp32=False):
    """The PRE-CHANGE `_DFlashLayer.attend_block` body: full prefix, `repeat_interleave` GQA, torch
    einsum/softmax. `fp32=True` upcasts q/K/V for the attention only — the REFERENCE both the shipped
    HIP path and the old bf16 torch path are scored against."""
    B = hidden.shape[0]
    H, Hkv, hd = layer.num_heads, layer.num_kv_heads, layer.head_dim
    residual = hidden
    x = layer.input_layernorm.forward(hidden)
    q = layer.q_proj.forward(x).view(B, H, hd)
    k_noise = layer.k_proj.forward(x).view(B, Hkv, hd)
    v_noise = layer.v_proj.forward(x).view(B, Hkv, hd)
    layer.q_norm.forward_inplace(q)
    layer.k_norm.forward_inplace(k_noise)
    q_flat, kn_flat = layer._rotary.forward(
        block_pos, q.reshape(B, H * hd).contiguous(), k_noise.reshape(B, Hkv * hd).contiguous()
    )
    q = q_flat.view(B, H, hd)
    k_noise = kn_flat.view(B, Hkv, hd)
    K = torch.cat([k_ctx, k_noise], dim=0)
    V = torch.cat([v_ctx, v_noise], dim=0)
    group = H // Hkv
    K = K.repeat_interleave(group, dim=1)
    V = V.repeat_interleave(group, dim=1)
    dt = V.dtype
    if fp32:
        q, K, V = q.float(), K.float(), V.float()
    scores = torch.einsum("bhd,shd->bhs", q, K) * layer.scale
    if attn_mask is not None:
        scores = scores + attn_mask.unsqueeze(1)
    probs = scores.softmax(dim=-1).to(V.dtype)
    attn = torch.einsum("bhs,shd->bhd", probs, V).to(dt)
    if layer.gated:
        gate = torch.nn.functional.softplus(layer.g_proj.forward(x).float()).to(attn.dtype)
        attn = attn * gate.unsqueeze(-1)
    attn_out = layer.o_proj.forward(attn.reshape(B, H * hd))
    hidden = residual + attn_out
    residual = hidden
    normed = layer.post_attention_layernorm.forward(hidden)
    return residual + layer._mlp(normed)


def _ref_project_ctx(layer, target_hidden, ctx_pos):
    """The PRE-CHANGE `project_ctx`: two rope launches, two staging copies, query result discarded."""
    m = target_hidden.shape[0]
    Hkv, hd = layer.num_kv_heads, layer.head_dim
    k_ctx = layer.k_proj.forward(target_hidden).view(m, Hkv, hd)
    v_ctx = layer.v_proj.forward(target_hidden).view(m, Hkv, hd)
    layer.k_norm.forward_inplace(k_ctx)
    _, kc_flat = layer._rotary.forward(
        ctx_pos, k_ctx.reshape(m, Hkv * hd).contiguous(), k_ctx.reshape(m, Hkv * hd).contiguous()
    )
    return kc_flat.view(m, Hkv, hd), v_ctx


@torch.inference_mode()
def _ref_denoise_cached(model, noise_embed, prefix_kv, block_pos, *, fp32=False):
    """The PRE-CHANGE `denoise_cached` over the FULL (unsliced) prefix, torch attention."""
    hidden = noise_embed
    P = prefix_kv[0][0].shape[0]
    masks = model.layer_masks(P, noise_embed.shape[0], noise_embed.device)
    for layer, (k_ctx, v_ctx), mask in zip(model.layers, prefix_kv, masks):
        hidden = _ref_attend_block(layer, hidden, block_pos, k_ctx, v_ctx, mask, fp32=fp32)
    return model.norm.forward(hidden)


def _randomize(model, dtype, device, gen):
    def r(shape, scale):
        return (torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
                * scale).to(dtype)

    for mod, shape in [(model.fc, (HIDDEN, NUM_AUX * HIDDEN))]:
        mod.weight = r(shape, HIDDEN ** -0.5)
    model.hidden_norm.weight = r((HIDDEN,), 0.1) + 1.0
    model.norm.weight = r((HIDDEN,), 0.1) + 1.0
    for an in model.aux_hidden_norms:
        an.weight = r((HIDDEN,), 0.1) + 1.0
    for layer in model.layers:
        layer.input_layernorm.weight = r((HIDDEN,), 0.1) + 1.0
        layer.post_attention_layernorm.weight = r((HIDDEN,), 0.1) + 1.0
        layer.q_norm.weight = r((HEAD_DIM,), 0.1) + 1.0
        layer.k_norm.weight = r((HEAD_DIM,), 0.1) + 1.0
        layer.q_proj.weight = r((HEADS * HEAD_DIM, HIDDEN), HIDDEN ** -0.5)
        layer.k_proj.weight = r((KV_HEADS * HEAD_DIM, HIDDEN), HIDDEN ** -0.5)
        layer.v_proj.weight = r((KV_HEADS * HEAD_DIM, HIDDEN), HIDDEN ** -0.5)
        layer.o_proj.weight = r((HIDDEN, HEADS * HEAD_DIM), (HEADS * HEAD_DIM) ** -0.5)
        layer.g_proj.weight = r((HEADS, HIDDEN), HIDDEN ** -0.5)
        layer.gate_proj.weight = r((INTER, HIDDEN), HIDDEN ** -0.5)
        layer.up_proj.weight = r((INTER, HIDDEN), HIDDEN ** -0.5)
        layer.down_proj.weight = r((HIDDEN, INTER), INTER ** -0.5)


@torch.inference_mode()
def main(seed: int = 20260801):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    gen = torch.Generator(device=device).manual_seed(seed)

    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = DFlashDraftModel(
                hidden_size=HIDDEN, intermediate_size=INTER, num_layers=LAYERS,
                num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM,
                num_aux_layers=NUM_AUX, rms_norm_eps=EPS, rope_theta=ROPE_THETA,
                max_position=262144, decoder_layer_type="laguna_xs",
                sliding_window=WINDOW, causal=True, per_aux_norm=True,
            )
    finally:
        torch.set_default_dtype(prev)
    _randomize(model, dtype, device, gen)
    # Stand-in for the borrowed target lm_head: a fixed random [vocab, hidden]. Identical on every
    # leg, so ANY fixed linear map exercises the same argmax question with far less memory.
    vocab = 8192
    head_w = (torch.randn((vocab, HIDDEN), generator=gen, device=device, dtype=torch.float32)
              * HIDDEN ** -0.5).to(dtype)

    tally = dict(new=0, old=0, n=0, dnew=0.0, dold=0.0)
    print(f"{'P':>7} {'rows':>5} {'|logit|':>9} {'new dmax':>10} {'new agree':>10} "
          f"{'old dmax':>10} {'old agree':>10}")
    for P in (64, 256, 511, 512, 513, 1024, 4096, 16384):
        base = 100000  # absolute position of the block; prefix occupies [base-P, base-1]
        aux = (torch.randn((P, NUM_AUX, HIDDEN), generator=gen, device=device,
                           dtype=torch.float32) * 0.5).to(dtype)
        ctx_pos = torch.arange(base - P, base, dtype=torch.int32, device=device)
        block_pos = torch.arange(base, base + BLOCK, dtype=torch.int32, device=device)
        noise = (torch.randn((BLOCK, HIDDEN), generator=gen, device=device,
                             dtype=torch.float32) * 0.5).to(dtype)

        # The prefix K/V exactly as the proposer caches it: projected ONCE at true absolute positions.
        target_hidden = model.fuse_aux(aux)
        prefix_ref = [_ref_project_ctx(l, target_hidden, ctx_pos) for l in model.layers]
        prefix_new = [l.project_ctx(target_hidden, ctx_pos) for l in model.layers]
        # single-rope project_ctx must be BIT-identical (same kernel, same input).
        assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                   for a, b in zip(prefix_ref, prefix_new)), (
            f"project_ctx single-rope is not bit-identical at P={P}")

        def _logits(h):
            return (h.float() @ head_w.float().T)[1:]

        lg_ref = _logits(_ref_denoise_cached(model, noise, prefix_ref, block_pos, fp32=True))
        lg_old = _logits(_ref_denoise_cached(model, noise, prefix_ref, block_pos))
        lg_new = _logits(model.denoise_cached(noise, prefix_new, block_pos))
        ref_am = lg_ref.argmax(-1)

        def _cmp(lg):
            return ((lg_ref - lg).abs().max().item(), int((lg.argmax(-1) == ref_am).sum()))

        (dn, an), (do, ao) = _cmp(lg_new), _cmp(lg_old)
        rows_new = model.window_prefix(P)
        nk = lg_ref.shape[0]

        # STRUCTURAL check (exact): every prefix row the window slice drops is -inf for EVERY query
        # in the block, and the surviving mask equals the mask the sliced call builds (no
        # position-base shift). Laguna is uniform (every layer causal @ WINDOW).
        full_mask = model._block_mask(P, BLOCK, device, True, WINDOW)
        dropped = P - rows_new
        if dropped > 0:
            assert bool(torch.isinf(full_mask[:, :dropped]).all()), (
                f"P={P}: window slice would drop a LIVE key")
        sliced_mask = model._block_mask(rows_new, BLOCK, device, True, WINDOW)
        assert torch.equal(full_mask[:, dropped:], sliced_mask), (
            f"P={P}: sliced mask != full mask tail -- the position base IS shifted")

        print(f"{P:>7} {rows_new:>5} {lg_ref.abs().max().item():>9.2f} "
              f"{dn:>10.3e} {str(an) + '/' + str(nk):>10} {do:>10.3e} {str(ao) + '/' + str(nk):>10}")
        tally["new"] += an; tally["old"] += ao; tally["n"] += nk
        tally["dnew"] = max(tally["dnew"], dn); tally["dold"] = max(tally["dold"], do)
    return tally


if __name__ == "__main__":
    # EXIT CODE = the structural asserts in main() (hard) + the closeness contract (see docstring).
    tot = dict(new=0, old=0, n=0, dnew=0.0, dold=0.0)
    for s in (20260801, 11, 12345, 987654321, 424242):
        t = main(s)
        for k in ("new", "old", "n"):
            tot[k] += t[k]
        tot["dnew"] = max(tot["dnew"], t["dnew"]); tot["dold"] = max(tot["dold"], t["dold"])
    n = max(1, tot["n"])
    a_new, a_old = tot["new"] / n, tot["old"] / n
    ok_d = tot["dnew"] <= 1.5 * tot["dold"]
    ok_a = a_new >= a_old - 0.01
    print("STRUCTURAL GATE: PASS (all masked-row / mask-tail / single-rope asserts held)")
    print(f"CLOSENESS vs fp32-attention reference: shipped HIP max|dlogit| {tot['dnew']:.3e} "
          f"(torch-bf16 {tot['dold']:.3e}) -> {'PASS' if ok_d else 'FAIL'}; drafted-token agreement "
          f"shipped {100 * a_new:.2f}% vs torch-bf16 {100 * a_old:.2f}% -> {'PASS' if ok_a else 'FAIL'}")
    raise SystemExit(0 if (ok_d and ok_a) else 1)
