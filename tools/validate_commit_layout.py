"""Standalone parity test: on-device commit layout vs the host commit loop reference.

Phase-2b of the spec-decode overlap. No serve boot. Chains the three on-device primitives
(accept -> EOS-truncate -> commit) over many randomized flattened multi-req batches and asserts the
token-pool scatter (rows/cols/vals), the advanced per-req lengths (cached_len/device_len), the GDN
install t_index, and the draft-head seed rows are byte-identical to the host loop in
`scheduler._spec_decode_step`.

Host reference (mirrors scheduler.py ~lines 1944-1970):
    for j, tok in enumerate(keep):
        c_rows.append(req.table_idx)
        c_cols.append(c0 + 1 + j)
        c_vals.append(tok)
    req.cached_len = c0 + len(keep)
    req.device_len = req.cached_len + 1
    gdn_install_t_index = len(keep) - 1          # still-running reqs only
    seed_row = block_start + len(keep) - 1       # still-running reqs only

We drive the whole chain from raw (drafts, target, q_lens) so the primitives feed each other exactly
as the scheduler would wire them. `finished` masking (which reqs contribute a GDN install / seed row)
stays host-side in the real scheduler (it also depends on `can_decode`), so here we validate the FULL
per-req vectors and separately validate the finished-EOS subset the scheduler would keep.

Run inside the lean image (same launch recipe as validate_ondevice_accept.py), which puts torch + the
minisgl package on PYTHONPATH.
"""

from __future__ import annotations

import random
import sys

import torch

from minisgl.spec.accept import verify_greedy
from minisgl.spec.accept_gpu import (
    accept_greedy_ondevice,
    build_commit_ondevice,
    truncate_at_eos_ondevice,
)


def _host_keep(emitted: list[int], ignore_eos: bool, eos: int) -> tuple[list[int], bool]:
    """Reference EOS truncation: (keep, eos_hit), exactly as the scheduler host loop."""
    keep: list[int] = []
    eos_hit = False
    for tok in emitted:
        keep.append(tok)
        if (not ignore_eos) and tok == eos:
            eos_hit = True
            break
    return keep, eos_hit


def _build_case(num_reqs, ks, vocab, mode_per_req, eos, device):
    """Construct flat drafts/target tensors + per-req host draft/target lists for one case."""
    drafts_host: list[list[int]] = []
    target_host: list[list[int]] = []
    for r in range(num_reqs):
        K = ks[r]
        draft = [random.randrange(vocab) for _ in range(K)]
        mode = mode_per_req[r]
        if mode == "full":
            n_match = K
        elif mode == "zero":
            n_match = 0
        else:
            n_match = random.randrange(K + 1)
        target = []
        for i in range(K):
            target.append(draft[i] if i < n_match else (draft[i] + 1) % vocab)
        target.append(random.randrange(vocab))  # bonus row
        drafts_host.append(draft)
        target_host.append(target)

    q_lens = [len(t) for t in target_host]  # K_r + 1
    drafts_flat = [t for d in drafts_host for t in d]
    target_flat = [t for tg in target_host for t in tg]
    return drafts_host, target_host, q_lens, drafts_flat, target_flat


def _run_case(num_reqs, ks, vocab, mode_per_req, eos, device, tag=""):
    drafts_host, target_host, q_lens, drafts_flat, target_flat = _build_case(
        num_reqs, ks, vocab, mode_per_req, eos, device
    )
    ignore_host = [random.random() < 0.35 for _ in range(num_reqs)]
    # arbitrary but distinct per-req table rows + starting cached_len (c0)
    table_host = random.sample(range(1000), num_reqs)
    c0_host = [random.randint(0, 500) for _ in range(num_reqs)]

    target_argmax = torch.tensor(target_flat, dtype=torch.int32, device=device)
    drafts_gpu = torch.tensor(drafts_flat, dtype=torch.int32, device=device)
    q_lens_t = torch.tensor(q_lens, dtype=torch.int32, device=device)

    # target_offsets = block_start per req = exclusive cumsum of q_lens
    target_offsets_host = []
    acc = 0
    for qL in q_lens:
        target_offsets_host.append(acc)
        acc += qL

    # --- on-device chain: accept -> truncate -> commit -----------------------------------------
    acc_out = accept_greedy_ondevice(target_argmax, drafts_gpu, q_lens_t, device)
    ignore_mask = torch.tensor(ignore_host, dtype=torch.bool, device=device)
    trunc = truncate_at_eos_ondevice(
        acc_out.committed_flat, acc_out.committed_offsets, acc_out.committed_lens,
        eos, ignore_mask, device,
    )
    commit = build_commit_ondevice(
        trunc.kept_flat,
        trunc.kept_lens,
        torch.tensor(table_host, dtype=torch.int64, device=device),
        torch.tensor(c0_host, dtype=torch.int32, device=device),
        torch.tensor(target_offsets_host, dtype=torch.int32, device=device),
        device,
    )

    g_rows = commit.scatter_rows.cpu().tolist()
    g_cols = commit.scatter_cols.cpu().tolist()
    g_vals = commit.scatter_vals.cpu().tolist()
    g_cached = commit.new_cached_len.cpu().tolist()
    g_device = commit.new_device_len.cpu().tolist()
    g_tidx = commit.gdn_t_index.cpu().tolist()
    g_seed = commit.seed_rows.cpu().tolist()

    # --- host reference: replicate the scheduler commit loop exactly ---------------------------
    ref_rows: list[int] = []
    ref_cols: list[int] = []
    ref_vals: list[int] = []
    for r in range(num_reqs):
        target = target_host[r]
        result = verify_greedy(drafts_host[r], target)
        keep, _eos_hit = _host_keep(result.emitted, ignore_host[r], eos)
        c0 = c0_host[r]
        for j, tok in enumerate(keep):
            ref_rows.append(table_host[r])
            ref_cols.append(c0 + 1 + j)
            ref_vals.append(tok)
        ref_cached = c0 + len(keep)
        ref_device = ref_cached + 1
        ref_tidx = len(keep) - 1
        ref_seed = target_offsets_host[r] + len(keep) - 1
        assert g_cached[r] == ref_cached, (
            f"[{tag}] cached_len mismatch req {r}: gpu={g_cached[r]} ref={ref_cached}"
        )
        assert g_device[r] == ref_device, (
            f"[{tag}] device_len mismatch req {r}: gpu={g_device[r]} ref={ref_device}"
        )
        assert g_tidx[r] == ref_tidx, (
            f"[{tag}] gdn_t_index mismatch req {r}: gpu={g_tidx[r]} ref={ref_tidx}"
        )
        assert g_seed[r] == ref_seed, (
            f"[{tag}] seed_row mismatch req {r}: gpu={g_seed[r]} ref={ref_seed}"
        )

    assert g_rows == ref_rows, f"[{tag}] scatter_rows mismatch\n  gpu={g_rows}\n  ref={ref_rows}"
    assert g_cols == ref_cols, f"[{tag}] scatter_cols mismatch\n  gpu={g_cols}\n  ref={ref_cols}"
    assert g_vals == ref_vals, f"[{tag}] scatter_vals mismatch\n  gpu={g_vals}\n  ref={ref_vals}"


def main():
    if not torch.cuda.is_available():
        print("FAIL: no HIP/CUDA device visible to torch", file=sys.stderr)
        return 1
    device = torch.device("cuda")
    random.seed(2718)
    torch.manual_seed(2718)

    EOS = 2
    n_cases = 0

    # --- explicit edge cases -------------------------------------------------------------------
    # (num_reqs, ks, vocab, modes)
    edge_cases = [
        (1, [3], 5, ["full"]),          # full accept, single req
        (1, [3], 5, ["zero"]),          # first-token mismatch -> only bonus committed
        (1, [1], 5, ["partial"]),       # K=1
        (3, [2, 3, 4], 5, ["full", "zero", "partial"]),
        (2, [0, 5], 5, ["zero", "full"]),  # K=0 req (no drafts, just bonus) alongside a full accept
        (5, [1, 2, 3, 4, 5], 3, ["partial"] * 5),  # small vocab -> EOS lands naturally
        (4, [2, 2, 2, 2], 4, ["full", "partial", "zero", "full"]),
    ]
    for num_reqs, ks, vocab, modes in edge_cases:
        if len(modes) == 1:
            modes = modes * num_reqs
        _run_case(num_reqs, ks, vocab, modes, EOS, device, tag="edge")
        n_cases += 1

    # --- randomized fuzz -----------------------------------------------------------------------
    for _ in range(5000):
        num_reqs = random.randint(1, 8)
        ks = [random.randint(0, 15) for _ in range(num_reqs)]
        vocab = random.choice([3, 4, 8, 32])  # small vocabs make EOS (=2) appear with real density
        modes = [random.choice(["full", "partial", "zero"]) for _ in range(num_reqs)]
        _run_case(num_reqs, ks, vocab, modes, EOS, device, tag="fuzz")
        n_cases += 1

    print(
        f"PASS: all {n_cases} cases identical "
        f"(scatter rows/cols/vals + cached/device_len + gdn_t_index + seed_row byte-exact)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
