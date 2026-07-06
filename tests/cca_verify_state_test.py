"""CPU unit test for the CCA spec-decode verify-state reconstruction (no GPU, no HIP kernel).

Run:  PYTHONPATH=python python tests/cca_verify_state_test.py

The lossless-verify story for a CCA-hybrid (ZAYA) target rests on ONE structural claim: the CCA
``conv_state`` is a rolling window of RAW ``qk_new`` columns (cca_kernel.hip: ``window[c,TP]=qk_new``;
``conv_states[slot,c,i]=window[c,i+1]`` — roll left, new token at tail), so the conv window AFTER
each verify token can be reconstructed in torch WITHOUT a new HIP kernel. ``prev_hs`` after a token
is simply that token's input hidden state.

This test validates that reconstruction (``capture_cca_verify_state``) against a reference roll-left
recurrence — the documented kernel semantics — bit-exactly, plus the gather-install
(``CCAStateCache.install_verify_state``). The full end-to-end lossless gate (verify+install == N
single-token decodes through the real HIP kernels) runs on the GPU box; this pins the math first.
"""
from __future__ import annotations

import torch

from minisgl.cca.metadata import CCAMetadata, capture_cca_verify_state
from minisgl.kvcache.cca_state import CCAStateCache


def _ref_roll(init_win: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reference conv-window recurrence == the kernel's documented roll-left/append.

    init_win: [C, TP] starting window (col 0 oldest .. col TP-1 newest).
    x:        [L, C] per-token raw qk_new columns.
    returns:  [L, C, TP] — the window AFTER each of the L tokens.
    """
    C, TP = init_win.shape
    win = init_win.clone()
    out = torch.empty(x.shape[0], C, TP, dtype=init_win.dtype)
    for t in range(x.shape[0]):
        # roll left, append new token at the tail (== cca_kernel.hip conv-state update)
        win = torch.cat([win[:, 1:], x[t].unsqueeze(1)], dim=1)
        out[t] = win
    return out


def test_single_seq_reconstruction() -> None:
    print("single-seq conv-window reconstruction (unfold == roll-left recurrence):")
    torch.manual_seed(0)
    C, TP, L = 1280, 2, 5
    init = torch.randn(C, TP, dtype=torch.float32)
    x = torch.randn(L, C, dtype=torch.float32)  # per-token qk_new columns

    ref = _ref_roll(init, x)  # [L, C, TP]

    # mimic the helper's per-seq unfold path
    stream = torch.cat([init.transpose(0, 1), x], dim=0)  # [TP+L, C]
    windows = stream.unfold(0, TP, 1)  # [L+1, C, TP]
    got = windows[1 : 1 + L]  # window AFTER token j == win[j+1]

    assert torch.equal(got, ref), "unfold reconstruction != roll-left recurrence"
    print(f"  ok  bit-exact over L={L}, C={C}, TP={TP}")


def test_batched_helper_ragged() -> None:
    print("capture_cca_verify_state (batched, ragged seqs):")
    torch.manual_seed(1)
    C, TP, hidden = 1280, 2, 2048
    seg_lens = [4, 2, 3]  # ragged extend_lens (K+1 per seq); e.g. accept-truncated tails
    N = sum(seg_lens)
    Q = max(seg_lens)
    num_seqs = len(seg_lens)

    qk_new = torch.randn(N, C, dtype=torch.float32)
    hs = torch.randn(N, hidden, dtype=torch.float32)
    init_states = torch.randn(num_seqs, C, TP, dtype=torch.float32)

    md = CCAMetadata(
        is_prefill=True,
        num_seqs=num_seqs,
        query_start_loc=torch.tensor([0, 4, 6, 9], dtype=torch.int32),
        state_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
        capture_verify_state=True,
        verify_max_qlen=Q,
        seg_lens=seg_lens,
    )
    capture_cca_verify_state(md, cca_layer_id=0, qk_new=qk_new, init_states=init_states, hs=hs)

    conv_scr = md.conv_scratch[0]  # [Q, N=num_seqs, C, TP]
    prev_scr = md.prev_scratch[0]  # [Q, N=num_seqs, hidden]
    assert conv_scr.shape == (Q, num_seqs, C, TP), conv_scr.shape
    assert prev_scr.shape == (Q, num_seqs, hidden), prev_scr.shape

    off = 0
    for i, Li in enumerate(seg_lens):
        x = qk_new[off : off + Li]
        ref = _ref_roll(init_states[i], x)  # [Li, C, TP]
        assert torch.equal(conv_scr[:Li, i], ref), f"seq {i} conv scratch mismatch"
        assert torch.equal(prev_scr[:Li, i], hs[off : off + Li]), f"seq {i} prev scratch mismatch"
        off += Li
    print(f"  ok  conv+prev scratch match per-seq reference over seg_lens={seg_lens}")


def test_install_gather() -> None:
    print("CCAStateCache.install_verify_state (gather accepted-prefix state):")
    torch.manual_seed(2)
    num_cca_layers, num_slots = 2, 8
    C, TP, hidden = 1280, 2, 2048
    Q, N = 4, 3  # verify_max_qlen, num_seqs

    cache = CCAStateCache(
        num_cca_layers=num_cca_layers, num_slots=num_slots, conv_dim=C, conv_kernel=TP,
        hidden_size=hidden, dtype=torch.float32, device=torch.device("cpu"),
    )
    conv_scratch = {lid: torch.randn(Q, N, C, TP) for lid in range(num_cca_layers)}
    prev_scratch = {lid: torch.randn(Q, N, hidden) for lid in range(num_cca_layers)}
    slots = torch.tensor([1, 4, 7], dtype=torch.long)  # CCA slot per seq
    t_index = torch.tensor([3, 0, 2], dtype=torch.long)  # accepted_count-1 per seq

    cache.install_verify_state(conv_scratch, prev_scratch, slots, t_index)

    for lid in range(num_cca_layers):
        for i in range(N):
            s = int(slots[i]); t = int(t_index[i])
            assert torch.equal(cache.conv_states[lid, s], conv_scratch[lid][t, i]), (lid, i, "conv")
            assert torch.equal(cache.prev_hs[lid, s], prev_scratch[lid][t, i]), (lid, i, "prev")
    print("  ok  installed slots hold scratch[t_index, seq] for every layer")


def test_first_token_window() -> None:
    print("first-token window edge case (init_col1 dropped-in correctly):")
    C, TP = 4, 2
    init = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]])  # [C,TP]
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # [L=1, C]
    stream = torch.cat([init.transpose(0, 1), x], dim=0)  # [TP+1, C]
    got = stream.unfold(0, TP, 1)[1:2][0]  # window after token 0: [C, TP]
    # expected: roll-left of init + append x0 -> [init_col1, x0]
    want = torch.stack([init[:, 1], x[0]], dim=1)  # [C, TP]
    assert torch.equal(got, want), (got, want)
    print("  ok  window after token0 == [init[:,1], x0]")


def main() -> None:
    test_single_seq_reconstruction()
    test_batched_helper_ragged()
    test_install_gather()
    test_first_token_window()
    print("\nALL CCA verify-state reconstruction tests passed.")


if __name__ == "__main__":
    main()
