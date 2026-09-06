"""The prefill-chunk group-alignment gate — CPU only, no GPU, no model.

WHAT THIS PROVES, AND WHY IT IS NOT A GPU TEST
----------------------------------------------
`QSAPlan.build` refuses a prefill chunk whose `cached_len` is not a multiple of
`indexer_compress_ratio` (r = 4): the group straddling that boundary has members this forward does
not hold, so there is nothing to average into its compressed key. That refusal is what capped
`tools/serve.sh`'s qwen4exp arm at CONC=1.

`docs/measurements/QSA_INDEXER.md` §4c attributed it to PREFIX REUSE. That is wrong, and the wrong
fix (rounding the radix match down to a multiple of r) is a NO-OP on this model: qwen4_exp forces
the NAIVE prefix cache (`engine/config.py::resolve_prefix_cache` — the PLE recurrent state is not
covered by the snapshot store), so `handle.cached_len` is always 0. And a radix match could not
produce it either: every radix `cached_len` is `align_down(..., page_size)` and QSA already demands
`page_size % r == 0`.

The real source is chunk PACKING, and it is pure host arithmetic — which is why this gate needs no
GPU. `token_budget` is per STEP, not per request; a request's FINAL chunk is `remain_len`, an
arbitrary number; the LEFTOVER budget is then handed to the next request in the same batch and
becomes its first chunk. Both recorded reproductions fall straight out of that:

    16382 = 15*1024 + 1022  ->  leftover 2  ->  second prompt's next chunk starts at cached_len=2
     4087 =  3*1024 + 1015  ->  leftover 9  ->  cached_len=9

This test drives the REAL `PrefillManager`/`PrefillAdder` (not a re-implementation) over the two
recorded configurations and asserts:

  A. with `chunk_gran = 1` the defect reproduces — the exact recorded cached_len values. A fix
     whose "before" leg does not fail is measuring nothing.
  B. with `chunk_gran = 4` every non-final chunk ends on a group boundary, so no `cached_len` a
     forward ever sees is unaligned; and every prompt still completes, with its tokens covered
     exactly once and in order (the alignment must not drop or duplicate a token).

RUN (no devices needed):
    docker run --rm -v <worktree>:/engine --entrypoint bash minisgl-rdna4:<tag> -lc \
      'PYTHONPATH=/engine/python python /engine/tests/qwen4exp_qsa_chunk_align_test.py'
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from minisgl.scheduler.prefill import ChunkedReq, PrefillManager  # noqa: E402
from minisgl.scheduler.utils import PendingReq  # noqa: E402

# `_add_one_req` stages the chunk's token ids through pinned host memory, which needs a CUDA
# context this CPU-only gate deliberately does not have. The staging is not what is under test —
# the CHUNK BOUNDARIES are — so make it a no-op copy. (Harness-side only; nothing in
# python/minisgl is patched.)
torch.Tensor.pin_memory = lambda self, *a, **k: self  # type: ignore[assignment]

_failures: list = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)
    if not cond:
        _failures.append(name)
    return cond


class _FakeRow:
    """`token_pool[idx]` / `page_table[idx]` stand-in: slicing and copy_ are no-ops here — this gate
    is about the SCHEDULING arithmetic, and a real device copy would need a GPU."""

    def __getitem__(self, _k):
        return self

    def copy_(self, *_a, **_kw):
        return self


class _FakeHandle:
    cached_len = 0

    def get_matched_indices(self):
        return _FakeRow()


class _FakeMatch:
    cuda_handle = _FakeHandle()


class _FakeCacheManager:
    """Naive prefix cache (what qwen4_exp forces): no match, no eviction, unlimited space."""

    page_size = 16
    is_recurrent_radix = False
    available_size = 1 << 30

    def match_req(self, _req):
        return _FakeMatch()

    def lock(self, _handle):
        pass

    def unlock(self, _handle):
        return None


class _FakeTableManager:
    def __init__(self) -> None:
        self._free = list(range(8))
        self.token_pool = _FakeRow()
        self.page_table = _FakeRow()

    @property
    def available_size(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        return self._free.pop()

    def free(self, slot: int) -> None:
        self._free.append(slot)


class _FakeDecodeManager:
    inflight_tokens = 0


def _pending(uid: int, n: int) -> PendingReq:
    from minisgl.core import SamplingParams

    return PendingReq(uid, torch.arange(n, dtype=torch.int64), SamplingParams())


def _drive(prompt_len: int, width: int, budget: int, gran: int, max_steps: int = 4000):
    """Run the real prefill scheduler over `width` identical-length prompts until all finish.

    Returns (list of every (uid, cached_len, chunk_len) a forward would have seen, steps)."""
    pm = PrefillManager(_FakeCacheManager(), _FakeTableManager(), _FakeDecodeManager())
    pm.chunk_gran = gran
    for i in range(width):
        pm.pending_list.append(_pending(i, prompt_len))
    seen, steps = [], 0
    while pm.runnable and steps < max_steps:
        batch = pm.schedule_next_batch(budget)
        steps += 1
        if batch is None:
            # No request could be admitted this step. With a real engine a decode step would run;
            # here it means the budget was exhausted, which must not be a livelock.
            if steps > 4 and not seen:
                raise AssertionError("prefill made no progress at all")
            continue
        for req in batch.reqs:
            seen.append((req.uid, req.cached_len, req.extend_len))
            if isinstance(req, ChunkedReq):
                req.complete_one()   # what Engine.forward_batch does to every row
    assert steps < max_steps, "prefill did not terminate"
    return seen, steps


def _coverage_ok(seen, width: int, prompt_len: int) -> bool:
    """Every prompt's tokens covered exactly once, contiguously, in order."""
    for uid in range(width):
        spans = [(c, c + e) for u, c, e in seen if u == uid]
        pos = 0
        for lo, hi in spans:
            if lo != pos or hi <= lo:
                return False
            pos = hi
        if pos != prompt_len:
            return False
    return True


def main() -> int:
    r = 4
    budget = 1024
    # The two configurations recorded in QSA_INDEXER.md §4c, at their ACTUAL prompt lengths.
    cases = [
        ("48L TP=2 r2b.log", 16382, 2, 2),   # (name, prompt_len, conc, recorded cached_len)
        ("4L  TP=1 r3.log", 4087, 2, 9),
    ]

    print("A. chunk_gran = 1 (the shipped behaviour) — the defect must REPRODUCE")
    for name, n, width, recorded in cases:
        seen, _ = _drive(n, width, budget, gran=1)
        bad = sorted({c for _u, c, _e in seen if c % r != 0})
        check(f"{name}: unaligned cached_len appears", bool(bad), f"unaligned={bad[:6]}")
        check(f"{name}: reproduces the RECORDED cached_len={recorded}", recorded in bad,
              f"unaligned={bad[:6]}")
        check(f"{name}: token coverage is exact", _coverage_ok(seen, width, n))

    print("B. chunk_gran = 4 (the fix) — every chunk a forward sees is group-aligned")
    for name, n, width, _recorded in cases:
        seen, steps = _drive(n, width, budget, gran=r)
        bad = sorted({c for _u, c, _e in seen if c % r != 0})
        check(f"{name}: no unaligned cached_len", not bad, f"unaligned={bad[:6]} steps={steps}")
        check(f"{name}: token coverage is exact", _coverage_ok(seen, width, n),
              f"chunks={len(seen)}")
        print(f"      {name}: {len(seen)} chunks over {steps} scheduler steps")

    print("C. a single request is untouched (the fix must not perturb CONC=1)")
    for gran in (1, r):
        seen, _ = _drive(16382, 1, budget, gran=gran)
        globals().setdefault("_single", {})[gran] = seen
    check("CONC=1 chunk sequence identical with and without the alignment",
          globals()["_single"][1] == globals()["_single"][r],
          f"{len(globals()['_single'][1])} chunks")

    print("D. a chunk_gran that cannot fit the budget must not livelock")
    seen, steps = _drive(4087, 2, budget=6, gran=r)
    check("budget=6 still completes both prompts", _coverage_ok(seen, 2, 4087), f"steps={steps}")

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} — {_failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
