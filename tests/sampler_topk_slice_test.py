"""`_apply_top_k_top_p` (topk slice) == `_apply_top_k` + `_apply_top_p` (two full-vocab sorts).

Run on a real card:
    gpu-lease -n 1 -- docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --ipc host --shm-size 16gb \
      -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
      -v <worktree>:/engine -e PYTHONPATH=/opt/kernels:/engine/python \
      --entrypoint bash minisgl-rdna4:lean -lc \
      'source /opt/venv/bin/activate && python /engine/tests/sampler_topk_slice_test.py'

WHY THE FIXTURES LOOK LIKE THIS. A tie is the ONE input where the two implementations are permitted
to disagree: topk and sort may order equal values differently, so a different member of a tied group
survives. It is tempting to assume a softmax-of-randn fixture is tie-free and assert bit equality on
it — it is NOT. At this vocab most of the distribution underflows in fp32 and the tail is riddled
with exact duplicates (asserting whole-row distinctness fails immediately). Those tail ties are
harmless, because top-k discards them either way; only a tie STRADDLING the k boundary can change
the answer. So the equality cases assert distinctness of the top k+1 values only, and ties get their
own case asserting the INVARIANTS that hold regardless of which tied element is kept: the number of
survivors, and the total retained mass.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/engine/python")
from minisgl.engine.sample import (  # noqa: E402
    _apply_top_k,
    _apply_top_k_top_p,
    _apply_top_p,
)

VOCAB = 248_320  # the q4e / Qwen3.8 padded vocab — the shape that motivated this
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAIL
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        FAIL += 1


def reference(probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor | None) -> torch.Tensor:
    out = _apply_top_k(probs, top_k)
    return _apply_top_p(out, top_p) if top_p is not None else out


def rand_probs(bs: int, vocab: int, dev: torch.device, *, head: int = 0) -> torch.Tensor:
    """Softmax over randn. Ties in the TAIL are unavoidable in fp32 at this vocab (most of the mass
    underflows to a handful of values) and are harmless — they are all discarded by top-k either
    way. What would make the two implementations legitimately disagree is a tie STRADDLING the
    boundary, so that is what `head` guards: the top `head` values must be mutually distinct, which
    pins which elements survive."""
    logits = torch.randn(bs, vocab, device=dev, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    if head:
        top = torch.topk(probs, head + 1, dim=-1).values
        for r in range(bs):
            assert torch.unique(top[r]).numel() == head + 1, (
                f"fixture row {r} has a tie inside its top {head + 1} — regenerate the seed")
    return probs


def main() -> int:
    if not torch.cuda.is_available():
        print("no HIP device — this test must run on a card")
        return 2
    dev = torch.device("cuda")
    torch.manual_seed(0)

    print("equality vs the two-sort reference (distinct values):")
    for bs, k, p in ((1, 20, 0.95), (2, 20, 0.95), (4, 1, 0.95), (2, 50, 1.0), (2, 20, None)):
        probs = rand_probs(bs, VOCAB, dev, head=k)
        tk = torch.full((bs,), k, device=dev, dtype=torch.long)
        tp = None if p is None else torch.full((bs,), p, device=dev, dtype=torch.float32)
        got = _apply_top_k_top_p(probs.clone(), tk, tp)
        exp = reference(probs.clone(), tk, tp)
        check(f"bs={bs} k={k} p={p}", torch.equal(got, exp),
              f"max|Δ|={(got - exp).abs().max().item():.3e}")

    print("per-row MIXED k (the batch case a single kmax must not flatten):")
    bs = 3
    probs = rand_probs(bs, VOCAB, dev, head=200)
    tk = torch.tensor([1, 20, 200], device=dev, dtype=torch.long)
    tp = torch.full((bs,), 0.95, device=dev, dtype=torch.float32)
    got, exp = _apply_top_k_top_p(probs.clone(), tk, tp), reference(probs.clone(), tk, tp)
    check("k=[1,20,200] p=0.95", torch.equal(got, exp))
    check("row k=1 keeps exactly one", int((got[0] > 0).sum()) == 1,
          f"kept {int((got[0] > 0).sum())}")

    print("top-k DISABLED for a row (kmax >= vocab -> reference fallback):")
    tk = torch.tensor([20, VOCAB], device=dev, dtype=torch.long)
    tp = torch.full((2,), 0.95, device=dev, dtype=torch.float32)
    probs = rand_probs(2, VOCAB, dev, head=20)
    got, exp = _apply_top_k_top_p(probs.clone(), tk, tp), reference(probs.clone(), tk, tp)
    check("k=[20, vocab]", torch.equal(got, exp))

    print("TIES — order may differ, invariants may not:")
    probs = torch.zeros(1, VOCAB, device=dev, dtype=torch.float32)
    probs[0, :100] = 1.0 / 100  # 100 exactly-equal candidates
    tk = torch.tensor([20], device=dev, dtype=torch.long)
    got = _apply_top_k_top_p(probs.clone(), tk, None)
    exp = reference(probs.clone(), tk, None)
    check("tie: same number of survivors",
          int((got > 0).sum()) == int((exp > 0).sum()),
          f"{int((got > 0).sum())} vs {int((exp > 0).sum())}")
    check("tie: same retained mass",
          torch.allclose(got.sum(), exp.sum()),
          f"{got.sum().item():.6f} vs {exp.sum().item():.6f}")

    print("timing (bs=2, the served decode shape):")
    probs = rand_probs(2, VOCAB, dev, head=20)
    tk = torch.full((2,), 20, device=dev, dtype=torch.long)
    tp = torch.full((2,), 0.95, device=dev, dtype=torch.float32)

    def bench(fn, n=50):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    ms_ref = bench(lambda: reference(probs.clone(), tk, tp))
    ms_new = bench(lambda: _apply_top_k_top_p(probs.clone(), tk, tp))
    print(f"    two full-vocab sorts : {ms_ref:7.3f} ms/step")
    print(f"    topk slice           : {ms_new:7.3f} ms/step   ({ms_ref / ms_new:.1f}x faster)")
    check("topk slice is not slower", ms_new < ms_ref, f"{ms_new:.3f} vs {ms_ref:.3f} ms")

    print(f"\n{'ALL PASS' if not FAIL else f'{FAIL} FAILURE(S)'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
