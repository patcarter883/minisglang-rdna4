"""One-time engagement log for the custom HIP kernels, plus the per-call COUNTS behind it.

Verifies that each custom HIP kernel actually FIRES on the serve path — and, critically, keeps
firing under cudagraph capture (a graph-unsafe kernel would fail capture or silently fall back to a
non-HIP reference). Each kernel calls ``engaged(name)`` at its entry; the first call per name emits
``[hip-engage] <name>`` to the rank-0 log. Under cudagraph the entry runs during CAPTURE, so a
missing line after capture = that kernel did not run (fell back / gated off). Silence the LOG with
``MINISGL_HIP_ENGAGE_LOG=0``.

WHY THE LOG ALONE CANNOT ANSWER AN A/B, AND WHAT ``counts()`` IS FOR
--------------------------------------------------------------------
This repo's standing rule is "diff the engaged() ledgers per leg — a vanished arm is a silent
dispatch regression benches are blind to". The ledger as written is a SET, and a set SATURATES: it
records that an arm fired once, ever. In a two-leg A/B inside ONE process the set is already full
by the time leg B starts, so the set-diff is empty by construction for both legs — it cannot
distinguish "both legs dispatched the same arms" from "leg B dispatched nothing at all". Across two
PROCESSES it is only marginally better: it still cannot see an arm that fired 12 times on one leg
and once on the other.

``COUNTS`` is the same ledger keyed the same way, but as a tally, so ``counts()`` before and after a
leg gives that leg's own dispatch profile and two legs can actually be differenced. Costs one dict
update per call, which is the same order as the set lookup it sits beside.

ONE RESULT HERE IS EASY TO MISREAD, so it is written down rather than discovered: **under cudagraph
REPLAY these counts do not move.** ``engaged()`` is host Python inside the op wrappers; a replay
re-executes recorded kernel launches and never re-enters Python. So a captured leg's per-step deltas
are ZERO except on the steps that ran eagerly, and its *capture-time* counts are one pass' worth.
Frozen counts across a captured leg are the positive signal that it replayed — not evidence the arm
stopped working. Compare a captured leg against the eager leg's PER-STEP profile divided through,
or compare capture-time counts to one eager step; never compare raw totals and call the difference a
regression. (``weights/moe_interpose.py::RESOLVE_COUNTS`` carries the identical caveat for the
offload seam, for the identical reason.)
"""
from __future__ import annotations

import os

from minisgl.utils import init_logger

_logger = init_logger("hip-engage")
_seen: set[str] = set()
_ON = os.environ.get("MINISGL_HIP_ENGAGE_LOG", "1") != "0"

# name -> how many times that arm has been dispatched from HOST python. See the module docstring for
# why this exists alongside `_seen` and for what it does under graph replay.
COUNTS: dict[str, int] = {}


def engaged(name: str) -> None:
    COUNTS[name] = COUNTS.get(name, 0) + 1
    if _ON and name not in _seen:
        _seen.add(name)
        _logger.info_rank0(f"[hip-engage] {name}")


def counts() -> dict[str, int]:
    """A SNAPSHOT of the dispatch tally. Copy, not the live dict, so a caller holding a "before"
    cannot have it mutated out from under them by the leg it is measuring."""
    return dict(COUNTS)


def counts_delta(before: dict[str, int]) -> dict[str, int]:
    """`counts() - before`, keeping only the arms that actually moved.

    Arms present in `before` and ABSENT from the delta are the finding this is built for: an arm
    that dispatched on the earlier leg and dispatched zero times on this one.
    """
    now = counts()
    keys = set(now) | set(before)
    return {k: now.get(k, 0) - before.get(k, 0) for k in sorted(keys)
            if now.get(k, 0) != before.get(k, 0)}
