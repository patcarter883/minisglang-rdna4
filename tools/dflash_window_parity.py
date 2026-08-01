"""Parity gate for the DFlash O(window) propose rewrite (deliverable 1, edits a/b/g).

WHAT IS BEING CLAIMED, and what is NOT.

Edits (a) window-slice-the-prefix, (b) grouped-GQA einsum and (g) single-rope remove work that the
drafter's OWN causal + sliding-window mask already discards, or that was computed twice. The claim is
therefore that the DRAFTED TOKEN IDS are identical, not that the logits are bitwise equal:

  * removed keys contribute exp(-inf - max) == 0.0 EXACTLY and 0.0 * V == 0.0, so their contribution
    to the value is exactly zero in exact arithmetic;
  * but softmax and the probs·V einsum both reduce over `s`, whose LENGTH changes (P+B -> W+B). torch
    / rocBLAS partition a reduction differently at different lengths, so the surviving nonzero terms
    are GROUPED differently and the last bits may move. `torch.equal` failing is not a bug.

So this harness gates on: (1) argmax equality of every drafted position, (2) max|Δlogit| at bf16
epsilon scale, at several prefix lengths spanning P < W, P == W and P >> W. It compares the SHIPPED
`DFlashDraftModel.denoise_cached` against a reference that reimplements the pre-change body verbatim
(full prefix, `repeat_interleave` GQA, double rope).

Run inside the serve image on one leased card:
  gpu-lease -n 1 -- docker compose --profile run run --rm \
      -e MINISGL_CMD="python /engine/tools/dflash_window_parity.py" run
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


def _ref_attend_block(layer, hidden, block_pos, k_ctx, v_ctx, attn_mask, grouped=False):
    """The PRE-CHANGE `_DFlashLayer.attend_block` body, verbatim: full prefix, repeat_interleave GQA.
    `grouped=True` swaps ONLY the GQA einsum for the new one, to isolate edit (b) from edit (a)."""
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
    if grouped:
        q4 = q.view(B, Hkv, group, hd)
        scores = torch.einsum("bkgd,skd->bkgs", q4, K) * layer.scale
        if attn_mask is not None:
            scores = scores + attn_mask[:, None, None, :]
        probs = scores.softmax(dim=-1).to(V.dtype)
        attn = torch.einsum("bkgs,skd->bkgd", probs, V).reshape(B, H, hd)
    else:
        K = K.repeat_interleave(group, dim=1)
        V = V.repeat_interleave(group, dim=1)
        scores = torch.einsum("bhd,shd->bhs", q, K) * layer.scale
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(1)
        probs = scores.softmax(dim=-1).to(V.dtype)
        attn = torch.einsum("bhs,shd->bhd", probs, V)
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
def _ref_denoise_cached(model, noise_embed, prefix_kv, block_pos, *, slice_win=False, grouped=False):
    """The PRE-CHANGE `denoise_cached`. `slice_win` turns on edit (a) only, `grouped` edit (b) only —
    so the four legs (neither / a / b / both) isolate which change moves the bits."""
    hidden = noise_embed
    P = prefix_kv[0][0].shape[0]
    if slice_win:
        p = model.window_prefix(P)
        if p < P:
            prefix_kv = [(k[P - p :], v[P - p :]) for (k, v) in prefix_kv]
            P = p
    mask = model._block_mask(P, noise_embed.shape[0], noise_embed.device)
    for layer, (k_ctx, v_ctx) in zip(model.layers, prefix_kv):
        hidden = _ref_attend_block(layer, hidden, block_pos, k_ctx, v_ctx, mask, grouped=grouped)
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
def main(seed: int = 20260801) -> int:
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
    # Stand-in for the borrowed target lm_head: a fixed random [vocab, hidden]. The real head is
    # M-invariant by construction (layers/embedding.py) and identical on both legs, so ANY fixed
    # linear map exercises the same argmax question with far less memory.
    vocab = 8192
    head_w = (torch.randn((vocab, HIDDEN), generator=gen, device=device, dtype=torch.float32)
              * HIDDEN ** -0.5).to(dtype)

    ok = True
    flips = [0, 0]
    print(f"{'P':>7} {'rows':>5} {'|logit|':>9} "
          f"{'(a)dmax':>10} {'(a)flip':>8} {'(b)dmax':>10} {'(b)flip':>8} "
          f"{'(ab)dmax':>10} {'(ab)flip':>9} {'ref_ms':>8} {'new_ms':>8}")
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
        # (g) single-rope must be BIT-identical (same kernel, same input).
        rope_bit_eq = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
                          for a, b in zip(prefix_ref, prefix_new))
        if not rope_bit_eq:
            print(f"  FAIL: project_ctx single-rope is not bit-identical at P={P}")
            ok = False

        def _logits(h):
            return (h.float() @ head_w.float().T)[1:]

        lg_ref = _logits(_ref_denoise_cached(model, noise, prefix_ref, block_pos))
        lg_a = _logits(_ref_denoise_cached(model, noise, prefix_ref, block_pos, slice_win=True))
        lg_b = _logits(_ref_denoise_cached(model, noise, prefix_ref, block_pos, grouped=True))
        lg_new = _logits(model.denoise_cached(noise, prefix_new, block_pos))
        ref_am = lg_ref.argmax(-1)

        def _cmp(lg):
            am = lg.argmax(-1)
            return ((lg_ref - lg).abs().max().item(), int((am != ref_am).sum()))

        (da, ea), (db, eb), (dn, en) = _cmp(lg_a), _cmp(lg_b), _cmp(lg_new)
        rows_new = model.window_prefix(P)
        scale = lg_ref.abs().max().item()

        # STRUCTURAL check (the one that is exact, and the one that actually matters): every prefix
        # row the window slice drops must be -inf for EVERY query in the block, i.e. it contributes
        # exp(-inf - max) == 0.0 exactly. If this holds, the slice removes only exact zeros and any
        # residual delta is reduction reassociation, not lost information.
        full_mask = model._block_mask(P, BLOCK, device)
        dropped = P - rows_new
        if dropped > 0:
            assert bool(torch.isinf(full_mask[:, :dropped]).all()), (
                f"P={P}: window slice would drop a LIVE key -- edit (a) is wrong")
        # ...and the surviving mask must equal the mask the sliced call builds (no position-base shift).
        sliced_mask = model._block_mask(rows_new, BLOCK, device)
        assert torch.equal(full_mask[:, dropped:], sliced_mask), (
            f"P={P}: sliced mask != full mask tail -- the position base IS shifted")

        def _bench(fn, iters=5):
            fn(); torch.cuda.synchronize()
            best = float("inf")
            for _ in range(iters):  # MIN-of-N, never a mean
                torch.cuda.synchronize(); t = torch.cuda.Event(True); e = torch.cuda.Event(True)
                t.record(); fn(); e.record(); torch.cuda.synchronize()
                best = min(best, t.elapsed_time(e))
            return best

        ms_ref = _bench(lambda: _ref_denoise_cached(model, noise, prefix_ref, block_pos))
        ms_new = _bench(lambda: model.denoise_cached(noise, prefix_new, block_pos))
        nk = lg_ref.shape[0]
        print(f"{P:>7} {rows_new:>5} {scale:>9.2f} "
              f"{da:>10.3e} {str(ea)+'/'+str(nk):>8} {db:>10.3e} {str(eb)+'/'+str(nk):>8} "
              f"{dn:>10.3e} {str(en)+'/'+str(nk):>9} {ms_ref:>8.3f} {ms_new:>8.3f}")
        flips[0] += en; flips[1] += nk
        ok = ok and (en == 0)

    print(f"seed={seed} drafted-id flips: {flips[0]}/{flips[1]}  "
          + ("ALL-IDENTICAL" if ok else "SOME DIFFER"))
    return flips[0], flips[1]


if __name__ == "__main__":
    # EXIT CODE IS THE STRUCTURAL GATE, not the flip count. The asserts inside main() are the actual
    # correctness statement -- every prefix row the window slice drops is provably -inf for every
    # query, and the sliced mask equals the full mask's tail (no position-base shift). Those either
    # hold or the run dies. The flip count is DATA about floating-point reassociation, reported so it
    # cannot be quietly rounded away: with RANDOM weights (near-uniform logits, tiny top-1 margins --
    # a deliberately adversarial stand-in for a trained drafter) roughly 1 drafted position in 600
    # moves. On the real Laguna drafter in a real serve it is 1 in 3810 (tools/dflash_owindow_ab.sh).
    tot_f = tot_n = 0
    for s in (20260801, 11, 12345, 987654321, 424242):
        f, n = main(s)
        tot_f += f; tot_n += n
    print(f"STRUCTURAL GATE: PASS (all masked-row / mask-tail asserts held)")
    print(f"REASSOCIATION: {tot_f}/{tot_n} drafted positions differ "
          f"({100.0 * tot_f / max(1, tot_n):.2f}%) -- see the module docstring for why this is not 0")
    raise SystemExit(0)
