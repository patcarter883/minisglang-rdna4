"""ONE CUDA-graph capture/replay path for every model-based spec proposer's PROPOSE forward.

WHY THIS FILE EXISTS (and why it is not four files)
---------------------------------------------------
Verify has been captured for a long time (``engine/graph.py``). PROPOSE was not: MTP grew a private
capture (a ``_run_chain`` on ``MTPProposer``) and DFlash / EAGLE3 stayed fully eager, which is the
same debt ``KERNEL_CORE_POLICY.md`` forbids on the kernel side — four bodies that differ only in
"which drafter" but each re-derive slot bookkeeping, bucket padding, pool sharing, EP gating and
teardown. A fix to one silently misses the others (MTP's capture, for instance, is invisible to
``GraphRunner.destroy_cuda_graphs`` and captures lazily mid-serve at unbounded graph count).

So: the *mechanism* lives here exactly once, and a proposer supplies only the parts that are
genuinely drafter-specific, as four hooks:

  ``init_propose_capture(engine)``   allocate every persistent buffer ONCE (draft KV, cursors, I/O)
  ``stage_propose(...) -> Staged``   HOST side: pick active rows, fill the static input buffers
  ``propose_body(bs)``               THE CAPTURED BODY. static buffers in, static buffers out.
  ``read_drafts(reqs, staged)``      ONE device->host sync per STEP, after replay

``propose()`` on this class is the WHOLE captured step: stage -> replay-or-eager -> read. The body is
the same callable in both cases, which is what makes "replay == eager" a testable claim rather than
an assertion (tools/propose_capture_ab.sh runs exactly that comparison, per proposer). One subclass
overrides it — ``DFlashProposer.propose`` — and only to DISPATCH ON THE DRAFTER: a causal+windowed
checkpoint delegates straight back here via ``super().propose``, an unwindowed/CCA one takes its own
eager path (there is no fixed-capacity shape to capture). Nothing overrides the stage/replay/read
sequence itself.

THE RULES THE BODY MUST OBEY (each one is a real failure this repo has hit)
--------------------------------------------------------------------------
 1. No host sync inside the body — no ``.item()/.cpu()/.tolist()``, no ``int(tensor)``, no Python
    branch on a device value. Every per-row scalar arrives as a row of a static index tensor.
 2. No data-dependent shapes. A growing ``torch.stack(cache)`` is the canonical violation; the fix is
    a fixed window + an additive ``-inf`` mask (``softmax(-inf) == 0``).
 3. No Python control flow that varies per step. Fixed trip counts only.
 4. Index tensors are PERSISTENT and refreshed IN PLACE — a graph records kernel argument POINTERS.
 5. Allocation inside the body is charged to the graph's private pool permanently, per graph. Keep
    transients small (this is why every drafter here uses a GROUPED-query einsum instead of
    ``repeat_interleave`` — the expanded K/V is rep x larger and it OOM'd the pool).
 6. ``torch.inference_mode()`` around BOTH warmup and capture (grad-active in-place-on-view ops trip
    the autograd view guard and break capture).
 7. Warm up on a side stream (twice) before capture, then capture with a SHARED pool.
 8. Collectives are capturable only in lockstep with identical shapes on every rank — fine for
    plain TP / EP-over-TP, NOT for DP+EP (an idle replica self-agrees a different N eagerly).
 9. Padded rows must be harmless: they point at a reserved NULL slot nobody reads.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch

from minisgl.utils import init_logger

from .base import Proposer, ProposeCaptureStats

if TYPE_CHECKING:
    from minisgl.core import Req

logger = init_logger(__name__)

__all__ = ["CapturableProposer", "StagedPropose"]


class StagedPropose:
    """What ``stage_propose`` computed on the HOST for one propose step.

    ``bs`` is how many ACTIVE rows were written into the static input buffers; ``rows[j]`` is the
    index into the caller's ``reqs`` list that static row ``j`` corresponds to (rows that draft
    nothing are simply absent); ``budget[j]`` is that row's per-request draft budget, applied on the
    HOST after the sync (drafting a token the scheduler then drops is lossless — verify gates every
    emitted token — and it keeps the body's row count static, which capture needs)."""

    __slots__ = ("bs", "rows", "budget")

    def __init__(self, bs: int, rows: List[int], budget: List[int]) -> None:
        self.bs = bs
        self.rows = rows
        self.budget = budget


class CapturableProposer(Proposer):
    """Base for a proposer whose propose forward is CUDA-graph captured.

    Subclasses implement the four hooks below and call ``init_propose_capture(engine)`` at the end of
    ``__init__``. Everything else — bucket selection, pad rows, warmup/capture, replay, the EP gate,
    the engagement counters and teardown — is inherited and exists once.
    """

    # Declared capturable. The scheduler asks for this before calling capture_propose_graphs.
    propose_capturable: bool = True

    # ---------------------------------------------------------------- hooks (subclass implements)
    def init_propose_capture(self, engine) -> None:
        """Allocate every persistent tensor the body reads/writes: the drafter's KV pool, its
        per-slot cursors, and the static propose I/O buffers. Called once, from __init__."""
        raise NotImplementedError

    def stage_propose(self, reqs: List["Req"], num_draft: int, ctx, **kw) -> Optional[StagedPropose]:
        """HOST side, OUTSIDE the graph. Select the rows that will draft, reset slots whose owner
        changed, and refresh the static input buffers in place (``copy_``). Return None when no row
        drafts (the step degenerates to a plain decode). Must not read device values back."""
        raise NotImplementedError

    def propose_body(self, bs: int) -> None:
        """THE CAPTURED BODY. Reads only the static buffers' first ``bs`` rows, writes only static
        outputs and the persistent draft KV, in place. Obeys rules 1-6 above."""
        raise NotImplementedError

    def read_drafts(self, reqs: List["Req"], staged: StagedPropose, **kw) -> List[List[int]]:
        """ONE device->host sync for the whole step, after the replay. Returns per-req draft lists."""
        raise NotImplementedError

    def pad_propose_rows(self, bs: int, bucket: int) -> None:
        """Point static rows [bs, bucket) at the reserved NULL slot with an empty cursor, so the
        padded rows compute garbage into a buffer row nobody reads. Default: nothing to do."""

    def fill_propose_capture_rows(self, bucket: int) -> None:
        """Fill the static input buffers with a self-consistent DUMMY batch of ``bucket`` rows for
        warmup+capture (all rows on the NULL slot). Default: pad every row."""
        self.pad_propose_rows(0, bucket)

    @property
    def propose_max_rows(self) -> int:
        """How many rows the static propose buffers hold — the ceiling on a captured bucket."""
        return self._g_out.shape[0]

    # ------------------------------------------------------------------------ shared machinery
    def init_propose_capture_state(self, engine, *, tag: str) -> None:
        """Shared half of ``init_propose_capture`` — call it from the subclass hook."""
        from minisgl.distributed import is_ep_over_tp

        self._pc_tag = tag
        self._pc_graphs: Dict[int, "torch.cuda.CUDAGraph"] = {}
        self._pc_pool = None
        self._pc_bs_list: List[int] = []
        self._pc_replays = 0
        self._pc_eager = 0
        self._pc_warned: set = set()
        # DP+EP is the ONE configuration where a captured propose is wrong rather than slow: the
        # in-graph MoE all_gather pins a fixed N while an idle replica self-agrees its own N eagerly.
        # EP-over-TP is fine (the draft head is built replicated, so propose issues no EP collective).
        self._pc_allowed = (not bool(getattr(engine, "enable_ep", False))) or is_ep_over_tp()
        if not self._pc_allowed:
            logger.info_rank0(
                f"spec-decode: {tag} propose capture DISABLED under DP+EP (the in-graph MoE "
                "all_gather cannot match an idle replica's self-agreed N) — propose runs eager")

    @torch.inference_mode()
    def capture_propose_graphs(self, bs_list: Sequence[int]) -> None:
        """Capture one propose graph per batch-size BUCKET, at BOOT, on the caller's stream.

        Buckets (not one graph per exact batch size, which is what MTP's private capture did) bound
        the graph count and the pool memory, and move the capture cost off the serving path — a
        lazily captured graph is a multi-hundred-ms stall in the middle of a request. Padding is
        cheap because the low buckets are fine-grained ([1,2,4,8,12,16,...]), so bs=1 pads to 1."""
        if not self._pc_allowed:
            return
        # A bucket can never exceed the static buffers' row count: those buffers ARE the propose
        # batch's address space, and a proposer may have sized them below max_running_req (DFlash
        # caps its prefix ring against free VRAM). Rows past the cap are already refused in staging,
        # so bs can never reach such a bucket — capturing one would just waste memory.
        bs_list = sorted({int(b) for b in bs_list if 1 <= int(b) <= self.propose_max_rows})
        if not bs_list:
            return logger.info_rank0(
                f"spec-decode: {self._pc_tag} propose capture skipped (no graph batch sizes)")
        from minisgl.engine.graph import get_free_memory, mem_GB

        dev = self._device
        torch.cuda.synchronize(dev)
        free0 = get_free_memory(dev)
        logger.info_rank0(
            f"Capturing {self._pc_tag} PROPOSE CUDA graphs sizes={bs_list}; free {mem_GB(free0)}")
        # Warm on a SIDE stream (allocates rocBLAS workspaces / autotune outside the graph), then
        # capture into a pool shared by every bucket.
        side = torch.cuda.Stream()
        for bs in sorted(bs_list, reverse=True):
            self.fill_propose_capture_rows(bs)
            side.wait_stream(torch.cuda.current_stream())   # each iteration: side after the last capture
            with torch.cuda.stream(side):
                self.propose_body(bs)
                self.propose_body(bs)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pc_pool):
                self.propose_body(bs)
            if self._pc_pool is None:
                self._pc_pool = graph.pool()
            self._pc_graphs[bs] = graph
        self._pc_bs_list = sorted(self._pc_graphs)
        self.reset_propose_state()
        logger.info_rank0(
            f"spec-decode: {self._pc_tag} PROPOSE graphs CAPTURED buckets={self._pc_bs_list}; "
            f"free {mem_GB(get_free_memory(dev))} (was {mem_GB(free0)})")

    def reset_propose_state(self) -> None:
        """Drop any per-request state the warmup/capture dummy batch dirtied. Default: nothing."""

    def _propose_bucket(self, bs: int) -> Optional[int]:
        for b in self._pc_bs_list:
            if b >= bs:
                return b
        return None

    def _pc_warn_once(self, key: str, msg: str) -> None:
        if key not in self._pc_warned:
            self._pc_warned.add(key)
            logger.warning_rank0(f"spec-decode: {self._pc_tag} propose falling back to EAGER — {msg}")

    def run_propose_body(self, bs: int) -> None:
        """Replay the captured graph for this batch size, or run the SAME body eagerly.

        A captured path that silently degrades to eager is the failure mode this whole exercise is
        about, so every fallback reason is logged once and both counters are reported."""
        bucket = self._propose_bucket(bs) if self._pc_graphs else None
        if bucket is None:
            self._pc_eager += 1
            if not self._pc_allowed:
                pass  # already logged at init; DP+EP is a deliberate, permanent eager path
            elif not self._pc_graphs:
                self._pc_warn_once("nograph", "no propose graphs were captured (graphs disabled?)")
            else:
                self._pc_warn_once(
                    "bs", f"batch size {bs} exceeds the largest captured bucket "
                          f"{self._pc_bs_list[-1]}")
            self.propose_body(bs)
            return
        if bucket > bs:
            self.pad_propose_rows(bs, bucket)
        self._pc_graphs[bucket].replay()
        self._pc_replays += 1

    def propose_capture_stats(self) -> "ProposeCaptureStats":
        """The engagement evidence, read by the scheduler's spec-timing line every 50 steps.

        MUST NOT ASSUME `init_propose_capture_state` RAN. Inheriting this class declares that the
        proposer CAN be captured; whether it IS depends on the checkpoint (DFlash only allocates its
        capture state for a causal+windowed drafter — a z-lab / CCA drafter, or
        MINISGL_DFLASH_PERSIST_KV=0, keeps the eager per-uid path and never calls the init hook).
        Dereferencing `self._pc_replays` unconditionally therefore took the scheduler worker down
        mid-serve with an AttributeError on exactly those configurations, under exactly the
        diagnostic (MINISGL_SPEC_TIMING=1) that exists to prove capture is engaged. So: read through
        `getattr`, and report the honest mode rather than a 0/0 that looks like "no eager fallbacks"."""
        if not getattr(self, "propose_capturable", False) or not hasattr(self, "_pc_replays"):
            return ProposeCaptureStats(
                0, 0, [], "never",
                getattr(self, "propose_uncapturable_reason",
                        "capture state was never initialised for this checkpoint/config"))
        if not self._pc_allowed:
            return ProposeCaptureStats(
                0, self._pc_eager, [], "never",
                "DP+EP — an in-graph MoE all_gather cannot match an idle replica's self-agreed N")
        if not self._pc_bs_list:
            return ProposeCaptureStats(
                self._pc_replays, self._pc_eager, [], "failed",
                "declared capturable but NO propose graphs exist (graphs disabled, or capture OOM'd)")
        return ProposeCaptureStats(
            self._pc_replays, self._pc_eager, list(self._pc_bs_list), "captured")

    def destroy_propose_graphs(self) -> None:
        """Release the captured graphs. Must run BEFORE NCCL teardown or shutdown can hang — the
        GraphRunner calls this for every proposer it knows about."""
        self._pc_graphs = {}
        self._pc_bs_list = []
        self._pc_pool = None

    # ------------------------------------------------------------------------------ the template
    @torch.inference_mode()
    def propose(self, reqs: List["Req"], num_draft: int, ctx, **kw) -> List[List[int]]:
        staged = self.stage_propose(reqs, num_draft, ctx, **kw)
        if staged is None or staged.bs == 0:
            return [[] for _ in reqs]
        self.run_propose_body(staged.bs)
        return self.read_drafts(reqs, staged, **kw)
