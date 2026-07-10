"""Standalone parity test: on-device EOS truncation vs the host `keep`-loop reference.

Phase-2a of the spec-decode overlap. No serve boot. Generates many randomized flattened multi-req
committed buffers (non-uniform committed_lens, EOS injected at arbitrary positions, per-req
ignore_eos mix), runs `truncate_at_eos_ondevice` (GPU) and the per-req host truncation loop from
`scheduler._spec_decode_step`, and asserts kept_len, finished-on-EOS, and the truncated ids are
byte-identical for every case, including edge cases.

Host reference (mirrors scheduler.py ~line 1936-1942):
    keep = []
    eos_hit = False
    for tok in committed:
        keep.append(tok)
        if (not ignore_eos) and tok == eos_token_id:
            eos_hit = True
            break

Run inside the lean image (same launch recipe as validate_ondevice_accept.py), which puts torch +
the minisgl package on PYTHONPATH.
"""

from __future__ import annotations

import random
import sys

import torch

from minisgl.spec.accept_gpu import truncate_at_eos_ondevice


def _host_truncate(committed: list[int], ignore_eos: bool, eos: int) -> tuple[list[int], bool]:
    """Reference: return (keep, eos_hit) exactly as the scheduler host loop would."""
    keep: list[int] = []
    eos_hit = False
    for tok in committed:
        keep.append(tok)
        if (not ignore_eos) and tok == eos:
            eos_hit = True
            break
    return keep, eos_hit


def _build_committed(committed_host: list[list[int]], device):
    """Flatten per-req committed id lists into (committed_flat, committed_offsets, committed_lens)."""
    lens = [len(c) for c in committed_host]
    flat = [t for c in committed_host for t in c]
    offsets = []
    acc = 0
    for L in lens:
        offsets.append(acc)
        acc += L
    committed_flat = torch.tensor(flat, dtype=torch.int32, device=device)
    committed_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    committed_lens = torch.tensor(lens, dtype=torch.int32, device=device)
    return committed_flat, committed_offsets, committed_lens


def _run_case(committed_host, ignore_eos_host, eos, device, tag=""):
    committed_flat, committed_offsets, committed_lens = _build_committed(committed_host, device)
    ignore_mask = torch.tensor(ignore_eos_host, dtype=torch.bool, device=device)

    out = truncate_at_eos_ondevice(
        committed_flat, committed_offsets, committed_lens, eos, ignore_mask, device
    )
    gpu_lens = out.kept_lens.cpu().tolist()
    gpu_fin = out.kept_finished_eos.cpu().tolist()
    gpu_offsets = out.kept_offsets.cpu().tolist()
    gpu_flat = out.kept_flat.cpu().tolist()

    for r, (committed, ig) in enumerate(zip(committed_host, ignore_eos_host)):
        keep, eos_hit = _host_truncate(committed, ig, eos)
        assert gpu_lens[r] == len(keep), (
            f"[{tag}] kept_len mismatch req {r}: gpu={gpu_lens[r]} ref={len(keep)}\n"
            f"  committed={committed} ignore_eos={ig} eos={eos}"
        )
        assert bool(gpu_fin[r]) == eos_hit, (
            f"[{tag}] finished_eos mismatch req {r}: gpu={bool(gpu_fin[r])} ref={eos_hit}\n"
            f"  committed={committed} ignore_eos={ig} eos={eos}"
        )
        start = gpu_offsets[r]
        length = gpu_lens[r]
        gpu_kept = gpu_flat[start : start + length]
        assert gpu_kept == keep, (
            f"[{tag}] kept ids mismatch req {r}: gpu={gpu_kept} ref={keep}\n"
            f"  committed={committed} ignore_eos={ig} eos={eos}"
        )


def main():
    if not torch.cuda.is_available():
        print("FAIL: no HIP/CUDA device visible to torch", file=sys.stderr)
        return 1
    device = torch.device("cuda")
    random.seed(4321)
    torch.manual_seed(4321)

    EOS = 2  # fixed sentinel eos id for all cases; non-eos tokens are drawn to avoid/hit it
    n_cases = 0

    # --- explicit edge cases -------------------------------------------------------------------
    # committed lists use ids in a small vocab that EXCLUDES/INCLUDES EOS (=2) deliberately.
    edge_cases = [
        # (committed_host, ignore_eos_host, note)
        ([[2]], [False], "single req, EOS at index 0 -> kept_len=1, finished"),
        ([[5, 2, 7]], [False], "EOS mid-slice -> kept through index 1, finished"),
        ([[5, 7, 3]], [False], "no EOS -> keep all, not finished"),
        ([[5, 2, 7]], [True], "ignore_eos with EOS present -> keep all, not finished"),
        ([[5, 7, 2]], [False], "EOS at LAST position -> keep all (len unchanged), finished"),
        ([[2], [2], [2]], [False, False, False], "all-reqs EOS at 0"),
        ([[2, 9, 9], [5, 6, 7, 2], [8]], [False, False, False], "non-uniform lens, mixed EOS pos"),
        # ignore_eos per-req mix with EOS present in each: only non-ignore reqs truncate.
        ([[2, 9], [2, 9], [4, 2, 8]], [True, False, True], "ignore_eos per-req mix"),
        ([[4, 4, 4, 4, 4, 4]], [False], "long no-EOS slice, keep all"),
        ([[2, 2, 2]], [False], "leading EOS run -> stop at FIRST (kept_len=1)"),
        # a req whose ONLY eos is guarded by ignore_eos, alongside a truncating req
        ([[9, 2, 9, 2], [9, 2, 9, 2]], [True, False], "same ids, ignore flips the outcome"),
    ]
    for committed_host, ignore_host, note in edge_cases:
        _run_case(committed_host, ignore_host, EOS, device, tag=note)
        n_cases += 1

    # --- randomized fuzz -----------------------------------------------------------------------
    for _ in range(6000):
        num_reqs = random.randint(1, 8)
        committed_host: list[list[int]] = []
        ignore_host: list[bool] = []
        for _ in range(num_reqs):
            L = random.randint(1, 16)
            # Small vocab that INCLUDES EOS (=2) so EOS lands at random positions with real density;
            # sometimes force a no-EOS slice, sometimes force EOS at position 0 or the last position.
            vocab = random.choice([3, 4, 8])  # 2 is a valid id -> EOS appears naturally
            row = [random.randrange(vocab) for _ in range(L)]
            flavor = random.random()
            if flavor < 0.15:  # force NO eos
                row = [t if t != EOS else (EOS + 1) % vocab or 1 for t in row]
                # guarantee none equal EOS
                row = [t if t != EOS else 1 for t in row]
            elif flavor < 0.30:  # force EOS at position 0
                row[0] = EOS
            elif flavor < 0.40:  # force EOS only at the last position
                row = [t if t != EOS else 1 for t in row]
                row[-1] = EOS
            committed_host.append(row)
            ignore_host.append(random.random() < 0.35)
        _run_case(committed_host, ignore_host, EOS, device, tag="fuzz")
        n_cases += 1

    print(f"PASS: all {n_cases} cases identical (kept_len + finished_eos + kept ids byte-exact)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
