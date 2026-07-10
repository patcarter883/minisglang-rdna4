"""Standalone parity test: on-device greedy acceptance vs the host reference `verify_greedy`.

No serve boot. Generates many randomized flattened multi-req batches (non-uniform K_r), runs BOTH
`accept_greedy_ondevice` (GPU) and the per-req host `verify_greedy`, and asserts num_accepted and
committed token ids are byte-identical for every case, including edge cases.

Run inside the lean image (see the launch recipe in the task / CLAUDE.md), which puts torch + the
minisgl package on PYTHONPATH.
"""

from __future__ import annotations

import random
import sys

import torch

from minisgl.spec.accept import verify_greedy
from minisgl.spec.accept_gpu import accept_greedy_ondevice


def _build_case(num_reqs, ks, vocab, mode_per_req, device):
    """Construct flat drafts/target tensors + the per-req host lists for one case.

    mode_per_req[r] in {"full", "partial", "zero"} controls where the first mismatch lands so we
    exercise full-accept / partial / first-token-mismatch deliberately (on top of random noise).
    """
    drafts_host: list[list[int]] = []
    target_host: list[list[int]] = []  # per req, length K_r + 1
    for r in range(num_reqs):
        K = ks[r]
        draft = [random.randrange(vocab) for _ in range(K)]
        # target[i] == draft[i] up to some accept boundary, then differs; plus a random bonus row.
        mode = mode_per_req[r]
        if mode == "full":
            n_match = K
        elif mode == "zero":
            n_match = 0
        else:  # partial: random boundary in [0, K]
            n_match = random.randrange(K + 1)
        target = []
        for i in range(K):
            if i < n_match:
                target.append(draft[i])
            else:
                # force a mismatch at position i
                bad = random.randrange(vocab)
                while bad == draft[i]:
                    bad = random.randrange(vocab)
                target.append(bad)
        target.append(random.randrange(vocab))  # bonus/correction row
        drafts_host.append(draft)
        target_host.append(target)

    drafts_flat = [t for d in drafts_host for t in d]
    target_flat = [t for row in target_host for t in row]
    q_lens = [ks[r] + 1 for r in range(num_reqs)]

    drafts_gpu = torch.tensor(drafts_flat, dtype=torch.int32, device=device)
    target_gpu = torch.tensor(target_flat, dtype=torch.int32, device=device)
    q_lens_gpu = torch.tensor(q_lens, dtype=torch.int32, device=device)
    return drafts_host, target_host, drafts_gpu, target_gpu, q_lens_gpu


def _run_case(drafts_host, target_host, drafts_gpu, target_gpu, q_lens_gpu, device):
    out = accept_greedy_ondevice(target_gpu, drafts_gpu, q_lens_gpu, device)
    gpu_na = out.num_accepted.cpu().tolist()
    gpu_lens = out.committed_lens.cpu().tolist()
    gpu_offsets = out.committed_offsets.cpu().tolist()
    gpu_flat = out.committed_flat.cpu().tolist()

    for r, (d, t) in enumerate(zip(drafts_host, target_host)):
        ref = verify_greedy(d, t)
        # num_accepted parity
        assert gpu_na[r] == ref.num_accepted, (
            f"num_accepted mismatch req {r}: gpu={gpu_na[r]} ref={ref.num_accepted}\n"
            f"  draft={d}\n  target={t}"
        )
        # committed ids parity (slice the flat buffer by the returned offsets/lens)
        start = gpu_offsets[r]
        length = gpu_lens[r]
        gpu_committed = gpu_flat[start : start + length]
        assert gpu_committed == list(ref.emitted), (
            f"committed ids mismatch req {r}: gpu={gpu_committed} ref={list(ref.emitted)}\n"
            f"  draft={d}\n  target={t}"
        )
    return gpu_na


def main():
    if not torch.cuda.is_available():
        print("FAIL: no HIP/CUDA device visible to torch", file=sys.stderr)
        return 1
    device = torch.device("cuda")
    random.seed(1234)
    torch.manual_seed(1234)

    n_cases = 0

    # --- explicit edge cases ---------------------------------------------------------------
    edge_specs = [
        # (ks, modes, small vocab to force collisions)
        ([1], ["full"], 4),  # K=1, all-match
        ([1], ["zero"], 4),  # K=1, first-token mismatch
        ([5], ["full"], 8),  # single req all-match
        ([5], ["zero"], 8),  # single req first-token mismatch
        ([15], ["partial"], 6),  # max K partial
        ([1, 15, 3, 8], ["zero", "full", "partial", "partial"], 5),  # non-uniform K flattened
        ([2, 2, 2], ["full", "zero", "partial"], 3),  # tiny vocab, lots of accidental matches
        ([10, 1], ["partial", "full"], 7),
    ]
    for ks, modes, vocab in edge_specs:
        case = _build_case(len(ks), ks, vocab, modes, device)
        _run_case(*case, device)
        n_cases += 1

    # --- randomized fuzz --------------------------------------------------------------------
    modes_pool = ["full", "partial", "zero"]
    for _ in range(4000):
        num_reqs = random.randint(1, 8)
        ks = [random.randint(1, 15) for _ in range(num_reqs)]
        modes = [random.choice(modes_pool) for _ in range(num_reqs)]
        # small vocab some of the time to create accidental matches past the forced boundary,
        # which stresses that num_accepted is the LEADING run (not total matches).
        vocab = random.choice([3, 5, 16, 1000])
        case = _build_case(num_reqs, ks, vocab, modes, device)
        _run_case(*case, device)
        n_cases += 1

    print(f"PASS: all {n_cases} cases identical (num_accepted + committed ids byte-exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
