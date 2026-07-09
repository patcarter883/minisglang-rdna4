from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    page_size: int
    # Running decode set keyed by uid: a dict gives O(1) add/remove/membership by uid AND, because
    # dicts are ordered, a stable base for a lazily-rebuilt uid-sorted VIEW (`_ordered`). The whole
    # point is that the per-decode-step scheduler reads an ALREADY-sorted list instead of re-sorting
    # the whole concurrent running set every iteration (was `sorted(running_reqs, key=uid)` each step,
    # O(n log n) though membership barely changes). The sort is recomputed only when membership
    # changes (admit / finish / abort) — steady-state decode steps pay nothing.
    running_reqs: Dict[int, Req] = field(default_factory=dict)
    _ordered: List[Req] | None = field(default=None, init=False, repr=False)

    def _ordered_reqs(self) -> List[Req]:
        if self._ordered is None:
            self._ordered = sorted(self.running_reqs.values(), key=lambda req: req.uid)
        return self._ordered

    @property
    def ordered_reqs(self) -> List[Req]:
        """uid-sorted running reqs as a fresh list — identical ordering to the old per-step
        `sorted(running_reqs, key=uid)`, but the sort is cached until membership changes."""
        return list(self._ordered_reqs())

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        # Incrementally fold the just-forwarded batch into the running set: newly-decodable reqs join,
        # reqs that can no longer decode (device_len hit max_device_len) drop. A req only becomes
        # non-decodable after a forward advances its device_len (engine.forward_batch -> complete_one),
        # and that SAME batch is what's handed here, so testing can_decode over `reqs` alone reproduces
        # the old `{r for r in running.union(reqs) if r.can_decode}` exactly — at cost O(len(reqs)) per
        # step instead of O(len(running)) rebuild of the whole set.
        for req in reqs:
            uid = req.uid
            if req.can_decode:
                if uid not in self.running_reqs:
                    self.running_reqs[uid] = req
                    self._ordered = None
            elif self.running_reqs.pop(uid, None) is not None:
                self._ordered = None

    def remove_req(self, req: Req) -> None:
        if self.running_reqs.pop(req.uid, None) is not None:
            self._ordered = None

    def abort_req(self, uid: int) -> Req | None:
        req = self.running_reqs.pop(uid, None)
        if req is not None:
            self._ordered = None
        return req

    @property
    def inflight_tokens(self) -> int:
        reqs = self.running_reqs
        tokens_reserved = (self.page_size - 1) * len(reqs)  # 1 page reserved
        return sum(req.remain_len for req in reqs.values()) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=list(self._ordered_reqs()), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
