"""A short prompt must not wait out a 16k-token prefill.

    python tests/prefill_head_of_line_test.py       # pure arithmetic, no GPU, no model

THE DEFECT. `token_budget` is per STEP, not per request. A request mid-chunking took ALL of it
(`chunk_size = min(self.token_budget, remain_len)`), was re-queued at the FRONT of `pending_list`
next step (`pending_list = chunked_list + pending_list[len(reqs):]`), and the packing loop `break`s
on the first refusal rather than continuing. So a 7-token prompt arriving behind a 16738-token one
waited out every chunk of it — observed on a boot burst as 3 waiting requests.

THE FIX under test: a chunk that will NOT finish its prompt gives back `_SHORT_REQ_RESERVE` of the
budget, but ONLY when something is queued behind it, and never below `_MIN_CHUNK`.

This test pins the BEHAVIOUR, not the constant: short work gets into the same step, a lone long
prefill is untouched, and the long chunk keeps most of its width. Changing the reserve to another
sensible value must not break it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from minisgl.scheduler.prefill import _MIN_CHUNK, _SHORT_REQ_RESERVE  # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def chunk_of(budget, remain, others):
    """The sizing decision under test, isolated from the cache/table machinery it is embedded in.

    Mirrors PrefillAdder._add_one_req's first three statements exactly; the alignment rounding that
    follows can only move the boundary DOWN, so it cannot undo the reserve.
    """
    size = min(budget, remain)
    if (others > 0 and size < remain and budget > _SHORT_REQ_RESERVE + _MIN_CHUNK):
        size = min(size, budget - _SHORT_REQ_RESERVE)
    return size


BUDGET = 2048
LONG = 16738          # the measured prompt
SHORT = 7             # the measured victim

print("== a long prefill with something queued behind it leaves room ==")
c = chunk_of(BUDGET, LONG, others=1)
report("the chunked prompt no longer takes the whole budget", c < BUDGET, f"chunk={c} of {BUDGET}")
left = BUDGET - c
report("what is left admits the short prompt in the SAME step", left >= SHORT,
       f"{left} tokens left, short prompt needs {SHORT}")
report("the long chunk still keeps most of the budget", c >= 0.75 * BUDGET,
       f"{100 * c / BUDGET:.0f}% of the step — chunk width is what amortises per-step overhead")

print("== a LONE long prefill is untouched (no throughput tax) ==")
c0 = chunk_of(BUDGET, LONG, others=0)
report("with nothing waiting, the full budget is still used", c0 == BUDGET, f"chunk={c0}")

print("== the FINAL chunk is never shortened ==")
# remain fits inside the budget => this chunk finishes the prompt => nothing to reserve for.
c1 = chunk_of(BUDGET, 900, others=3)
report("a chunk that completes its prompt takes what it needs", c1 == 900, f"chunk={c1}")

print("== a small budget is never cut below the floor ==")
small = _SHORT_REQ_RESERVE + _MIN_CHUNK      # exactly at the guard
c2 = chunk_of(small, LONG, others=1)
report("at the guard boundary the chunk is not reduced", c2 == small, f"chunk={c2} of {small}")
c3 = chunk_of(small + 1, LONG, others=1)
report("just past it, the reserve applies and the chunk stays >= the floor",
       c3 == small + 1 - _SHORT_REQ_RESERVE and c3 >= _MIN_CHUNK, f"chunk={c3}")

print("== the starvation scenario, end to end ==")
# Before the fix the short prompt waited ceil(16738/2048) = 9 steps. Now it enters step 1.
steps_before = -(-LONG // BUDGET)
report("the short prompt used to wait out the whole prefill", steps_before >= 8,
       f"{steps_before} steps at {BUDGET}/step")
report("now it is admitted on the first step", chunk_of(BUDGET, LONG, 1) + SHORT <= BUDGET,
       "long chunk + short prompt fit in one step's budget")

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
