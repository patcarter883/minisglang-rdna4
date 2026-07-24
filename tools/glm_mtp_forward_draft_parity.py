#!/usr/bin/env python
"""Parity test for the GLM MLA MTP draft-attention (GLMMTPAttention.forward_draft ->
forward_draft_masked). The two paths share the q/k/v projection + RoPE + per-head materialization; the
ONLY divergent logic is the attention core:

  * eager (forward_draft:373-379): stacks a growing per-step Python list -> [S,T,H,*] and einsums
    `scores[t,h,s] = q[t,h]·k[s,t,h]` over its dynamic length S (causal, current incl.).
  * capturable (forward_draft_masked): writes k/v into a FIXED [max_slots,max_ctx,H,*] buffer at a
    per-row cursor and einsums over the WHOLE max_ctx window with an additive -inf mask beyond the
    cursor: `scores[b,h,s] = q[b,h]·k_buf[b,s,h] + mask`.

GLM MLA materializes full multi-head K/V (H q == H k, NO GQA) with an ASYMMETRIC k-dim (qk = nope+rope)
vs v-dim (vhd) — the reason GLM needs its own core, distinct from Qwen's grouped path. This asserts the
mask math is byte-exact vs the sliced stack, PER ROW with DIFFERENT context lengths (softmax(-inf)=0 is
the only thing dropped). Pure einsum/softmax — no real weights, runs on CPU.

Run:  python tools/glm_mtp_forward_draft_parity.py   (CPU ok; add DEV=cuda under a lease for the GPU dtype)
"""
import os

import torch

torch.manual_seed(0)
DEV = os.environ.get("DEV", "cpu")
DT = torch.float32 if DEV == "cpu" else torch.bfloat16


def core_stack(q_full, k_list, v_list, scale):
    """Eager core (mirrors forward_draft): stack the per-step list [S,H,*], attend causally over S."""
    Ks = torch.stack(k_list, dim=0)  # [S,H,qk]
    Vs = torch.stack(v_list, dim=0)  # [S,H,vhd]
    scores = torch.einsum("hd,shd->hs", q_full, Ks) * scale  # [H,S]
    probs = scores.softmax(dim=-1).to(Vs.dtype)
    return torch.einsum("hs,shd->hd", probs, Vs)  # [H,vhd]


def core_masked(q_full, k_buf_row, v_buf_row, mask_bias_row, scale):
    """Capturable core (mirrors forward_draft_masked): full-window attend over the buffer + -inf mask."""
    scores = torch.einsum("hd,shd->hs", q_full, k_buf_row) * scale  # [H,max_ctx]
    scores = scores + mask_bias_row.view(1, -1)
    probs = scores.softmax(dim=-1).to(v_buf_row.dtype)
    return torch.einsum("hs,shd->hd", probs, v_buf_row)  # [H,vhd]


def main():
    H, qk, vhd = 96, 192, 128          # GLM-4.7-Flash-ish MLA head geometry (per-head materialized)
    max_ctx = 64
    scale = float(qk) ** -0.5
    # A batch of rows with DIFFERENT committed context lengths (the whole reason a sliced tensor can't
    # serve the batch — each row masks a different amount of the shared max_ctx window).
    ctx_lens = [1, 5, 16, 31, 64]
    mism = 0
    for row, L in enumerate(ctx_lens):
        # random per-step materialized k/v (as forward_draft would append) + a query at the last step.
        k_list = [torch.randn(H, qk, device=DEV, dtype=DT) for _ in range(L)]
        v_list = [torch.randn(H, vhd, device=DEV, dtype=DT) for _ in range(L)]
        q_full = torch.randn(H, qk, device=DEV, dtype=DT)

        ref = core_stack(q_full, k_list, v_list, scale)

        # Buffered twin: write the same k/v into a padded [max_ctx,H,*] buffer, mask cols >= L.
        k_buf = torch.zeros(max_ctx, H, qk, device=DEV, dtype=DT)
        v_buf = torch.zeros(max_ctx, H, vhd, device=DEV, dtype=DT)
        for s in range(L):
            k_buf[s] = k_list[s]
            v_buf[s] = v_list[s]
        col = torch.arange(max_ctx, device=DEV)
        write_col = L - 1  # last written column
        mask_bias = torch.where(col <= write_col, 0.0, float("-inf")).to(torch.float32)
        got = core_masked(q_full, k_buf, v_buf, mask_bias, scale)

        # DISCRIMINATOR: the SAME masked-core math but sliced to exactly L cols (no zero-pad, no -inf).
        # If THIS is bit-identical to eager, the mask LOGIC (write_col, -inf beyond, axis/order) is
        # exactly right and the only residual vs the full-window path is float reduction ORDER over the
        # zero-padded tail — benign (and irrelevant to output: spec verify makes propose lossless).
        sliced = core_masked(q_full, k_buf[:L], v_buf[:L], mask_bias[:L], scale)
        assert torch.equal(ref, sliced), (
            f"row {row} ctx={L}: masked LOGIC diverges from eager even without padding — real bug")

        md = (ref - got).abs().max().item()
        tol = 1e-5 if DT == torch.float32 else 5e-3
        ok = torch.allclose(ref, got, atol=tol, rtol=tol)
        print(f"[parity] row {row} (ctx={L}) sliced=BIT-EXACT  full-window max|Δ|={md:.2e} "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            mism += 1

    if mism:
        raise SystemExit(f"FAIL: {mism}/{len(ctx_lens)} rows exceed tol — masked core diverges from eager")
    print(f"[parity] GLM MLA forward_draft stack vs masked-buffer: mask LOGIC bit-exact (sliced), "
          f"full-window equal within float reduction-order rounding across ctx {ctx_lens} "
          f"(H={H}, qk={qk}, vhd={vhd}). Spec verify makes propose lossless regardless.")


if __name__ == "__main__":
    main()
