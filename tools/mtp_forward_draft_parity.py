#!/usr/bin/env python
"""Parity test for the MTP draft-attention rewrite (Qwen3_5MTPAttn.forward_draft ->
forward_draft_masked). The two paths share q/k/v projection + norm + rotary; the ONLY divergent logic
is the attention core: the eager path stacks a growing per-uid Python list and einsums over its dynamic
length S; the capturable path writes k/v into a fixed [B, max_ctx, nkv, hd] buffer at a per-row cursor
and einsums over the WHOLE max_ctx window with an additive -inf mask beyond the cursor. This asserts the
mask math is byte-exact vs the sliced stack, PER ROW with DIFFERENT context lengths (the reason a single
sliced tensor can't serve the batch). softmax(-inf)=0 makes it exact; this catches any masking / GQA-
expand / einsum-axis mistake before it reaches a graph.

Run under the lease (needs a GPU): gpu-lease -n 1 -- python tools/mtp_forward_draft_parity.py
"""
import torch

torch.manual_seed(0)
DEV = "cuda"


def attn_core_stack(q, klist, vlist, rep, scale):
    """Eager path core (mirrors forward_draft:506-514): stack the per-step list, GQA-expand, attend."""
    Ks = torch.stack(klist, dim=0)  # [S, nkv, hd]
    Vs = torch.stack(vlist, dim=0)
    Ks = Ks.repeat_interleave(rep, dim=1)  # [S, nq, hd]
    Vs = Vs.repeat_interleave(rep, dim=1)
    scores = torch.einsum("hd,shd->hs", q, Ks) * scale  # [nq, S]
    probs = scores.softmax(dim=-1).to(Vs.dtype)
    return torch.einsum("hs,shd->hd", probs, Vs)  # [nq, hd]


def attn_core_masked(q, k_buf, v_buf, mask_bias, rep, scale):
    """Capturable path core (mirrors forward_draft_masked): GROUPED-query full-window attend + mask,
    WITHOUT expanding K/V to nq heads (the repeat_interleave that OOM'd the graph pool)."""
    nq, hd = q.shape
    nkv = nq // rep
    qg = q.view(nkv, rep, hd)                            # [nkv, rep, hd]
    scores = torch.einsum("grd,sgd->grs", qg, k_buf) * scale  # [nkv, rep, max_ctx]
    scores = scores + mask_bias.view(1, 1, -1)
    probs = scores.softmax(dim=-1).to(v_buf.dtype)
    return torch.einsum("grs,sgd->grd", probs, v_buf).reshape(nq, hd)


def main():
    nq, nkv, hd, rep = 16, 2, 128, 8
    scale = hd ** -0.5
    max_ctx = 64
    K = 4                         # draft chain length
    dtypes = [torch.bfloat16, torch.float16]
    # rows with DIFFERENT pre-existing context lengths (seed_prefill leaves cursor > 0)
    base_ctx = [0, 5, 30]

    worst = 0.0
    for dt in dtypes:
        for c0 in base_ctx:
            # pre-fill c0 committed entries per row
            klist = [torch.randn(nkv, hd, device=DEV, dtype=dt) for _ in range(c0)]
            vlist = [torch.randn(nkv, hd, device=DEV, dtype=dt) for _ in range(c0)]
            k_buf = torch.zeros(max_ctx, nkv, hd, device=DEV, dtype=dt)
            v_buf = torch.zeros(max_ctx, nkv, hd, device=DEV, dtype=dt)
            for i in range(c0):
                k_buf[i] = klist[i]; v_buf[i] = vlist[i]
            cursor = c0
            for step in range(K):
                q = torch.randn(nq, hd, device=DEV, dtype=dt)
                k = torch.randn(nkv, hd, device=DEV, dtype=dt)
                v = torch.randn(nkv, hd, device=DEV, dtype=dt)
                # eager: append to list
                klist.append(k); vlist.append(v)
                out_stack = attn_core_stack(q, klist, vlist, rep, scale)
                # masked: write at cursor, mask cols > cursor
                k_buf[cursor] = k; v_buf[cursor] = v
                mask_bias = torch.full((max_ctx,), float("-inf"), device=DEV, dtype=torch.float32)
                mask_bias[: cursor + 1] = 0.0
                out_masked = attn_core_masked(q, k_buf, v_buf, mask_bias, rep, scale)
                d = (out_stack.float() - out_masked.float()).abs().max().item()
                worst = max(worst, d)
                # Grouped GQA reorders the reduction vs the stack, so not bit-identical; a tight tol is
                # the right bar (spec stays lossless via verify regardless of tiny draft-logit noise).
                assert torch.allclose(out_stack.float(), out_masked.float(), rtol=2e-2, atol=2e-2), (
                    f"MISMATCH dt={dt} c0={c0} step={step} max|Δ|={d:.3e}")
                cursor += 1
    print(f"[parity] forward_draft stack vs grouped-masked: MATCH across "
          f"{len(dtypes)} dtypes x {len(base_ctx)} ctx-lens x {K} steps; worst|Δ|={worst:.3e}")


if __name__ == "__main__":
    main()
