"""CPU unit test for the ported TiDAR fused-forward mask/layout (C.3). No GPU.

Run:  PYTHONPATH=python python tests/tidar_mask_test.py

Pins the fused-forward math before it drives any kernel: (1) the vectorised allow matrix equals the
human-readable per-pair predicate; (2) the mask predicate matches the TiDAR structural spec (S causal,
R_r sees prefix + first-r drafts + own replica); (3) fused position ids follow the §7.6 convention;
(4) the logit split (verify slice + replica drafts) round-trips; (5) replica selection maps accept->r.
"""
from __future__ import annotations

import torch

from minisgl.spec.tidar_mask import (
    MaskDescriptor,
    _allow_pair,
    build_allow_matrix,
    additive_bias,
    square_additive_bias,
    select_next_drafts_row_range,
    fused_forward_position_ids,
    parse_fused_forward_logits,
    fused_paged_layout,
    fused_paged_layout_segmented,
)


def test_vectorised_matches_predicate() -> None:
    print("vectorised allow matrix == per-pair predicate:")
    for P, B in [(0, 2), (3, 2), (5, 4), (7, 3)]:
        d = MaskDescriptor(prefix_len=P, block_len=B)
        allow = build_allow_matrix(d)
        assert allow.shape == (d.q_len, d.kv_len), allow.shape
        for q in range(d.q_len):
            for k in range(d.kv_len):
                ref = _allow_pair(d, q, k)
                assert bool(allow[q, k]) == ref, (P, B, q, k, bool(allow[q, k]), ref)
    print("  ok  matches over P,B in {(0,2),(3,2),(5,4),(7,3)}")


def test_predicate_semantics() -> None:
    print("predicate semantics (S causal, R_r sees prefix+first-r-drafts+own replica):")
    P, B = 4, 3
    d = MaskDescriptor(prefix_len=P, block_len=B)
    A = build_allow_matrix(d)  # [q_len, kv_len]
    # S[i] attends all prefix
    for i in range(B):
        assert A[i, :P].all(), i
    # S causal within S; S never attends any replica
    for i in range(B):
        for j in range(B):
            assert bool(A[i, P + j]) == (j <= i), (i, j)
        assert not A[i, P + B:].any(), i  # no replica cols
    # R_r[m]: sees prefix, first-r drafts of S, and own replica only
    for r in range(B):
        for m in range(B):
            q = d.replica_start(r) + m  # new-region q index of R_r[m]
            assert A[q, :P].all(), (r, m)                     # prefix
            for j in range(B):
                assert bool(A[q, P + j]) == (j < r), (r, m, j)  # first-r drafts
            for rr in range(B):
                for mm in range(B):
                    kk = P + d.replica_start(rr) + mm
                    assert bool(A[q, kk]) == (rr == r), (r, m, rr, mm)  # own replica only
    print("  ok  S causal + no-replica; R_r sees prefix + first-r drafts + own replica (bidir)")


def test_additive_bias_values() -> None:
    print("additive bias = 0/-inf mirror of allow:")
    d = MaskDescriptor(prefix_len=5, block_len=4)
    A = build_allow_matrix(d)
    bias = additive_bias(d, dtype=torch.float32)
    assert torch.equal(bias == 0.0, A)
    assert torch.equal(torch.isneginf(bias), ~A)
    sq = square_additive_bias(d)
    L = d.prefix_len + d.q_len
    assert sq.shape == (L, L), sq.shape
    # prefix rows are plain causal
    idx = torch.arange(d.prefix_len)
    assert torch.equal((sq[:d.prefix_len, :d.prefix_len] == 0.0), idx[:, None] >= idx[None, :])
    print("  ok  additive/square bias mirror allow; prefix rows causal")


def test_position_ids() -> None:
    print("fused position ids (§7.6): S contiguous, R_r at prefix+replica_drafts(r)+m:")
    P, B = 6, 4
    d = MaskDescriptor(prefix_len=P, block_len=B)
    pos = fused_forward_position_ids(d)
    assert len(pos) == d.q_len
    for j in range(B):
        assert pos[j] == P + j, (j, pos[j])                    # S
    for r in range(B):
        for m in range(B):
            q = d.replica_start(r) + m
            assert pos[q] == P + d.replica_drafts(r) + m, (r, m, pos[q])
    print(f"  ok  positions[:B]={pos[:B]}  R_0={pos[d.replica_start(0):d.replica_start(0)+B]}")


def test_logit_split_and_select() -> None:
    print("parse_fused_forward_logits round-trip + replica selection:")
    P, B, V = 5, 3, 32
    d = MaskDescriptor(prefix_len=P, block_len=B)
    torch.manual_seed(0)
    # prefix_queried=True : logits over the whole [P + q_len] sequence
    full = torch.randn(d.kv_len, V)
    p_ar, rep = parse_fused_forward_logits(full, d, prefix_queried=True)
    assert p_ar.shape == (B + 1, V) and rep.shape == (B, B, V)
    assert torch.equal(p_ar, full[P - 1:P + B])
    assert torch.equal(rep.reshape(-1, V), full[P + B:])
    # prefix_queried=False : new-region-only logits [q_len], carry the prefix-tail row
    newonly = torch.randn(d.q_len, V)
    tail = torch.randn(V)
    p_ar2, rep2 = parse_fused_forward_logits(newonly, d, prefix_queried=False, prefix_tail_logit=tail)
    assert p_ar2.shape == (B + 1, V) and rep2.shape == (B, B, V)
    assert torch.equal(p_ar2[0], tail) and torch.equal(p_ar2[1:], newonly[:B])
    assert torch.equal(rep2.reshape(-1, V), newonly[B:])
    # replica selection: accept k -> replica r=k (replica_offset 0), clamped to B-1
    for k in range(0, B + 2):
        s, e = select_next_drafts_row_range(d, k)
        r = max(0, min(B - 1, k))
        assert (s, e) == (d.replica_start(r), d.replica_start(r) + B), (k, s, e)
    print("  ok  logit split (both layouts) + replica selection accept->r")


def test_fused_paged_layout() -> None:
    print("fused_paged_layout ([confirmed | S | R*] over cached prefix):")
    C, B = 10, 4  # cached_len, block_len
    positions, mask, n_query, b = fused_paged_layout(C, B)
    q_new = B + B * B
    context_len = C + 1 + q_new
    assert n_query == 1 + q_new and b == B
    assert len(positions) == n_query
    assert mask.shape == (n_query, context_len), mask.shape
    # positions: confirmed@C; S_j@C+1+j; R_r[m]@C+1+r+m
    assert positions[0] == C
    for j in range(B):
        assert positions[1 + j] == C + 1 + j, (j, positions[1 + j])
    for r in range(B):
        for m in range(B):
            qi = 1 + B + r * B + m
            assert positions[qi] == C + 1 + r + m, (r, m, positions[qi])
    allow = mask == 0.0
    # confirmed row: causal over prefix + itself -> keys 0..C
    assert allow[0, : C + 1].all() and not allow[0, C + 1:].any()
    # S_i (query row 1+i): sees cached[0..C-1] + confirmed@C + S_{<=i}; NOT replicas
    for i in range(B):
        row = 1 + i
        assert allow[row, : C + 1].all(), i                       # prefix + confirmed
        for j in range(B):
            kcol = C + 1 + j                                      # S_j key column
            assert bool(allow[row, kcol]) == (j <= i), (i, j)     # causal within S
        assert not allow[row, C + 1 + B:].any(), i                # no replica keys
    # R_r[m] (query row 1+B+r*B+m): sees prefix+confirmed + first-r drafts + own replica only
    for r in range(B):
        for m in range(B):
            row = 1 + B + r * B + m
            assert allow[row, : C + 1].all(), (r, m)              # prefix + confirmed
            for j in range(B):
                assert bool(allow[row, C + 1 + j]) == (j < r), (r, m, j)  # first-r drafts
            for rr in range(B):
                for mm in range(B):
                    kcol = C + 1 + B + rr * B + mm
                    assert bool(allow[row, kcol]) == (rr == r), (r, m, rr, mm)  # own replica
    print(f"  ok  n_query={n_query} context_len={context_len}; confirmed/S/R keys correct")


def test_fused_paged_layout_segmented() -> None:
    print("fused_paged_layout_segmented (ctx tokens feed conv, masked from attention):")
    C, B, TP = 10, 4, 2
    r = fused_paged_layout_segmented(C, B, TP)
    n_query = 1 + B + B * (TP + B)
    assert r["n_query"] == n_query, (r["n_query"], n_query)
    assert len(r["positions"]) == n_query
    rows = r["rows"]
    allow = r["mask"] == 0.0
    context_len = C + n_query
    # confirmed (row 0): prefix + self only
    assert allow[0, :C].all() and allow[0, C] and not allow[0, C + 1:].any()
    # S rows (1..B): prefix + confirmed + S[:i] causal; NOT ctx, NOT R
    for i in range(B):
        qi = 1 + i
        assert allow[qi, :C].all() and allow[qi, C], i               # prefix + confirmed
        for kj, (kk, kr, kl, _) in enumerate(rows):
            kcol = C + kj
            if kk == "S":
                assert bool(allow[qi, kcol]) == (kl <= i), (i, kj)   # causal within S
            elif kk in ("ctx", "R"):
                assert not allow[qi, kcol], (i, kk, kj)              # never ctx/R
    # ctx rows: causal over committed+confirmed+S up to qabs (col index == position) + self
    for qi, (k, _r, _loc, qabs) in enumerate(rows):
        if k != "ctx":
            continue
        for kp in range(context_len):
            expect = (kp < qabs) or (kp == C + qi)
            assert bool(allow[qi, kp]) == expect, (qi, kp, qabs)
    # ctx_src: abs positions = last TP of [committed|confirmed|drafts[:r]]
    off = 0
    for r_idx in range(B):
        for t in range(TP):
            pr, abs_pos = r["ctx_src"][off]; off += 1
            assert abs_pos == max(0, C + 1 + r_idx - TP + t), (r_idx, t, abs_pos)
            assert rows[pr][0] == "ctx"
    # R_r rows: prefix + confirmed + S[:r] + own replica block; NOT ctx, NOT other R
    for (rr, prs) in r["replica_rows"]:
        assert len(prs) == B
        for qi in prs:
            assert allow[qi, :C].all() and allow[qi, C], rr          # prefix + confirmed
            for kj, (kk, kr, kl, _) in enumerate(rows):
                kcol = C + kj
                if kk == "S":
                    assert bool(allow[qi, kcol]) == (kl < rr), (rr, kj)
                elif kk == "ctx":
                    assert not allow[qi, kcol], (rr, kj)
                elif kk == "R":
                    assert bool(allow[qi, kcol]) == (kr == rr), (rr, kj)
    print(f"  ok  n_query={n_query} ctx feeds conv only (self-attn), R_r sees S[:r]+own block")


def main() -> None:
    test_vectorised_matches_predicate()
    test_predicate_semantics()
    test_additive_bias_values()
    test_position_ids()
    test_logit_split_and_select()
    test_fused_paged_layout()
    test_fused_paged_layout_segmented()
    print("\nALL TiDAR mask/layout tests passed.")


if __name__ == "__main__":
    main()
