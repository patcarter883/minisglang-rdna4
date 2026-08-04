"""Compare two fp8-KV scale sidecars — the gate for a calibrator change.

The TP=2 calibrator's whole claim is that sharding the FORWARD does not change the SCALES: each rank
sees only its own KV heads, so the per-head amax rows are gathered into global rows before any scale
is computed. That claim is checkable — run the same model on the same fixture at TP=1 and TP=2 and
diff the tables.

Agreement is CLOSE, not bit-exact, and expecting exactness would be wrong: TP changes the reduction
order inside the forward (the QKV projection is column-sharded, attention output row-sharded and
all-reduced), so the activations themselves differ in the last bits and a max-observer can land on a
different token. What must hold is that no head's scale MOVES MATERIALLY — a scale is a range
promise, and a few 1e-3 of relative drift is far inside the range e4m3 resolves.

    python tools/kv_scale_compare.py A.safetensors B.safetensors [--tol 0.02]

Prints max/mean/median relative difference and the worst keys; exits non-zero if the max exceeds
`--tol`. numpy-only, so it runs on the host without booting the ROCm image.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from safetensors.numpy import load_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=0.02, help="max allowed relative difference")
    args = ap.parse_args()

    A, B = load_file(args.a), load_file(args.b)
    if set(A) != set(B):
        only_a, only_b = sorted(set(A) - set(B))[:5], sorted(set(B) - set(A))[:5]
        print(f"FAIL: key sets differ ({len(A)} vs {len(B)}); only-A {only_a} only-B {only_b}")
        return 1

    worst = []
    all_rel = []
    for k in sorted(A):
        a, b = A[k].astype(np.float64).ravel(), B[k].astype(np.float64).ravel()
        if a.shape != b.shape:
            print(f"FAIL: {k} shape {a.shape} vs {b.shape}")
            return 1
        rel = np.abs(b - a) / np.maximum(np.abs(a), 1e-12)
        all_rel.append(rel)
        worst.append((rel.max(), k, int(rel.argmax())))
    r = np.concatenate(all_rel)
    worst.sort(reverse=True)

    print(f"{len(A)} tensors, {r.size} scale entries")
    print(f"rel diff: max {r.max():.3e}  mean {r.mean():.3e}  median {np.median(r):.3e}")
    print(f"fraction over 1%: {float((r > 0.01).mean()):.4f}   over 5%: {float((r > 0.05).mean()):.4f}")
    print("worst 5:")
    for m, k, i in worst[:5]:
        print(f"  {m:.3e}  {k}[head {i}]  {A[k].ravel()[i]:.6g} -> {B[k].ravel()[i]:.6g}")
    if r.max() > args.tol:
        print(f"FAIL: max rel diff {r.max():.3e} > tol {args.tol}")
        return 1
    print(f"PASS: max rel diff {r.max():.3e} <= tol {args.tol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
