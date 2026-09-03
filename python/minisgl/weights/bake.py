"""Stage A of the weight-offload load path: the ordered window, the VRAM ledger, and the seal.

WHAT STAGE A IS (`docs/WEIGHT_OFFLOAD_PLAN.md` §5.2). The model is built on the meta device and then
loaded normally, at full VRAM cost, exactly as today. Only AFTER `post_load()` has finished every
quantized repack are the host-placed containers moved into the pinned arena and their device
originals dropped. Peak VRAM is therefore the un-offloaded model: Stage A serves any model that FITS
at load but that you would rather not keep resident. Stage B (per-container chunked load, plan §5.2)
is what makes a model that does NOT fit at load bootable, and it is not in this file.

WHAT IS HERE AND WHAT IS NOT. The actual moving of bytes belongs to `moe_interpose.MoEWeightSeam`
(discovery, `alloc_like`, `copy_`, the A1.4 granule read-back, alias-preserving rebind) and the
placement decision belongs to `plan.resolve_weight_plan` / `placement.plan_layer_granular`. This
module owns the three things neither of them can: the ORDER those steps run in relative to
`Engine.__init__`, the byte LEDGER that decides whether the KV pool may trust the result
(`accounting.py`), and the SEAL that closes the mapping window.

THE WINDOW, AND WHY EACH BOUNDARY IS WHERE IT IS. `Engine.__init__` runs:

          build on meta device          <- (1) resolve the plan FROM the meta model, right after
          load_state_dict               <- (2) attach the HOST arena BEFORE this
          post_load()                   <- (3) the loaded, repacked originals exist here
                                        <- (4) bind the seams: move, verify, rebind, drop
          _determine_num_pages          <- (5) KV sizing reads the sealed ledger

  (1) The plan is resolved from the META-BUILT MODEL, not from `ModelConfig` alone, and that is a
      GENERALITY requirement rather than a convenience. Given a model, `resolve_weight_plan` reads
      the layer set, this rank's EP/TP sharding and the per-expert byte count off objects that
      already exist, through the one format-agnostic granule walker — so a new model family or quant
      format implements NOTHING. Given only config it TRANSCRIBES seven builder files and nine
      container `__init__`s, and that transcription is already wrong in this repo:
      `models/utils.py`'s `MoEMLP` builds its `MoELayer` with no `quant=` at all, so the config path
      sizes bf16 experts as int4 and under-reserves the arena ~4x, in the direction that makes an
      infeasible plan look FEASIBLE.

      Resolving after the meta build costs nothing the fail-fast ordering cared about: the meta
      build allocates ZERO device bytes, and the capacity abort still lands at `attach()` before
      `load_state_dict`. What the ordering forbids is resolving after the WEIGHTS are read.

      The inputs must still be rank-identical: the budget must NOT be derived from a live
      `mem_get_info` delta — that differs between the two rank processes, and two ranks resolving
      different plans emit plausible wrong text with no error anywhere. The meta model is not such
      an input; it is the same shapes on every rank, sharded by the same rank-aware constructors.
  (2) The arena is pinned before the checkpoint is read. Pinning is the expensive, capacity-bound
      step (P3b: 4.88 GB/s, and 62 of 68 GiB was the ceiling for two ranks on an IDLE box), so doing
      it first turns "this box does not have the RAM" into a failure in seconds instead of after a
      full load. It costs ZERO device bytes — asserted at `seal()`, never assumed.
  (3)/(4) The bind happens strictly between `post_load()` and `_determine_num_pages`, so everything
      it moves is inside the `device_used = old_free - new_free` window the KV sizing measures. That
      is what lets the device tier be billed by the EXISTING `model_memory` term with no sixth
      subtrahend (see `accounting.py`). It also has to be after `post_load()` for a second reason:
      `derive_granule_spec` refuses meta tensors, because every meta tensor reports `data_ptr() == 0`
      and aliasing and expert-invariance are then undetectable.
  (5) **Nothing may map after the seal.** A late mapping is invisible to
      `Scheduler._prefill_budget_now`'s `reserved - allocated` correction (`scheduler.py:2289`) and
      silently collapses the prefill budget, while the warning at `:2302` misdirects the operator to
      lower `--memory-ratio`. `seal()` reaches `hipmem.freeze()`, which latches that shut
      process-wide (rule R1); the latch lives in `hipmem` rather than here because it must also bind
      arena code that never sees a session.

ONLY HOST-PLACED LAYERS MOVE. Under layer-granular placement a device-resident layer is already
exactly where it needs to be, so `MoEWeightSeam.bind(StackKind.DEVICE)` moves nothing. That is not an
optimisation, it is a capacity requirement: allocating a fresh device stack and copying into it would
hold both copies live at once, and at f=0.20 on a 16 GB card that transient is several GB the card
does not have. It also makes device layers bit-identical to a non-offloaded serve by construction,
which removes them from every numerics gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional, Protocol, Tuple

from .accounting import WeightArenaAccounting, gib

__all__ = [
    "MemProbe",
    "StageADriver",
    "StageAPhase",
    "StageARuntime",
    "StageASession",
    "UnconfiguredDeviceTierError",
    "weight_arena_torch_slack_bytes",
]


MemProbe = Callable[[], Tuple[int, int, int]]
"""`() -> (free, allocated, reserved)` in bytes. Injected by the engine, so the session never has to
pick a device or decide when to synchronize."""


# Arena bytes torch has reserved but not allocated, published at `seal()` for
# `Scheduler._prefill_budget_now`. Module-level rather than read off the engine because the scheduler
# lives in a different object graph and must not import the engine. 0 on every serve that does not
# offload, which is what keeps that guard byte-identical today.
_TORCH_SLACK = 0


def weight_arena_torch_slack_bytes() -> int:
    """See `accounting.WeightArenaAccounting.torch_slack_bytes`. 0 until an enabled session seals."""
    return _TORCH_SLACK


def _reset_for_tests() -> None:
    """Clear the published slack. Tests only; `hipmem.teardown_window` handles the R1 latch."""
    global _TORCH_SLACK
    _TORCH_SLACK = 0


# =============================================================================================
# Phases and the driver contract
# =============================================================================================


class StageAPhase(IntEnum):
    """Monotonic. Every transition asserts its predecessor, so a caller cannot bind before the
    weights are loaded, attach the arena after KV sizing, or seal twice."""

    INIT = 0
    PLANNED = 1
    ATTACHED = 2
    LOADED = 3
    BOUND = 4
    SEALED = 5


class StageADriver(Protocol):
    """Everything Stage A needs that is not ordering or accounting.

    `StageARuntime` below is the production implementation; tests supply their own. Deliberately
    tiny — the session owns WHEN, the driver owns WHAT.
    """

    def plan_bytes(self) -> Tuple[int, int, int]:
        """`(host_resident, device_resident, total_offloadable)` for THIS rank, from the plan."""

    def attach_host_arena(self) -> None:
        """Reserve, pin and self-test the arena. Called before `load_state_dict`."""

    def attach_notes(self) -> Tuple[str, ...]:
        """Advisories the reservation produced (packing waste, grown chunk, unbounded headroom).

        Optional — the session reaches it through `getattr`. It exists because every one of those
        advisories is set on the arena and, until this hook, none of them reached a log."""

    def bind(self, model: Any) -> Any:
        """Discover the seams and execute the plan over them. Returns a `BindOutcome`."""

    def moved_bytes(self) -> int:
        """Bytes actually copied into the arena by `bind()`."""

    def arena_torch_bytes(self) -> int:
        """Arena bytes served through torch's allocator (0 if the rows bypass it).

        MUST be a measurement of TORCH, not of how many bytes the caller handed out. It is the term
        that tells `model_memory_correction()` how much of `memory_allocated()` is really host RAM,
        so a value that stays at the full host tier when the `MemPool` silently stopped routing
        removes bytes from the model term that were never in it — over-stating `available_memory` and
        sizing a KV pool that OOMs on the first forward. See `StageARuntime.arena_torch_bytes`."""

    def observed_device_bytes(self) -> Optional[int]:
        """Device-tier bytes MEASURED off the live post-load containers, or None if unavailable.

        Independent of the plan on purpose. `offloadable_total - copied` is not a measurement: with
        `total = host + device` it reduces algebraically to `host - copied`, which is the copied-
        bytes check restated, so it can never catch a plan whose byte model disagrees with what
        `post_load()` actually produced."""

    def arena_activity(self) -> Tuple[int, int]:
        """`(alloc_events, served_bytes)` off the arena's own MemPool — a capture-time witness.

        Deliberately the POOL's counters, not the stack allocator's byte total: torch calls the C ABI
        alloc callback directly, so an allocation made during graph capture never passes through
        `TorchStackAllocator` and would not move `arena_torch_bytes()`. See
        `StageASession.verify_after_capture`."""

    def assert_arena_clean(self) -> None:
        """Raise if any arena row silently landed in VRAM (`ArenaMemPool` hipMalloc fallback)."""

    def verify_arena_layout(self) -> Any:
        """Raise if the carved layout is not the reserved one. Optional; reached by `getattr`.

        Distinct from `assert_arena_clean`, which answers "did anything escape to VRAM". This
        answers "did what stayed in the arena land where the plan put it" — the drift that produces
        plausible text with no crash, and the one the reservation's exactness now makes checkable."""

    def freeze(self) -> None:
        """`arena.mark_populated()` + `arena.freeze()`; rule R1 closes here."""

    def describe(self) -> str: ...


def _noop_log(msg: str) -> None:
    """Default sink for a session built without a logger (tests, and the disabled path)."""


# =============================================================================================
# The session the engine drives
# =============================================================================================


@dataclass
class StageASession:
    """The Stage-A window as an explicit object, so its ordering cannot be got wrong by editing.

    A DISABLED session (no driver, or a plan with nothing host-resident) supports the identical call
    sequence and does nothing. That is on purpose, per plan §6.2: the path runs on EVERY serve, so it
    cannot rot into a branch nobody exercises, and it costs zero device bytes when the model fits.
    """

    driver: Optional[StageADriver] = None
    probe: Optional[MemProbe] = None
    accounting: WeightArenaAccounting = field(default_factory=WeightArenaAccounting)
    phase: StageAPhase = StageAPhase.INIT
    outcome: Any = None
    log: Callable[[str], None] = _noop_log
    # The TP CPU (gloo) group, kept — not dropped after `_resolve_driver` — so that every step of the
    # window can be made SYMMETRIC across ranks. See `_run_staged`.
    agreement_group: Any = None
    # Test injection point for the cross-rank barrier: any callable taking this rank's status and
    # returning one status per rank. Mirrors `WeightPlanResolution.assert_rank_agreement(gather=...)`.
    gather: Any = None
    # The arena MemPool's `(alloc_events, served_bytes)` as of `seal()`. `verify_after_capture()`
    # compares against it: an arena allocation served DURING graph capture moves these, and its
    # address is by then baked into a graph that will replay for the life of the process.
    _sealed_arena_activity: Tuple[int, int] = (0, 0)

    # -- construction -------------------------------------------------------------------------

    @classmethod
    def begin(
        cls,
        config: Any,
        *,
        probe: Optional[MemProbe] = None,
        log: Optional[Callable[[str], None]] = None,
        driver: Optional[StageADriver] = None,
        device_budget_bytes: Optional[int] = None,
        budget_is_derived: bool = False,
        agreement_group: Any = None,
        gather: Any = None,
        model: Any = None,
    ) -> "StageASession":
        """Resolve the plan — call on the META-BUILT model, before `load_state_dict`.

        `model` IS THE GENERALITY ARGUMENT, and omitting it is the difference between "a new model
        family or quant format implements NOTHING" and "a new model family or quant format edits a
        transcription table". With a model in hand `resolve_weight_plan` READS what it needs off
        objects that already exist: the layer set from `moe_interpose.discover_moe_layers`, this
        rank's sharding from `MoELayer.local_num_experts/enable_ep/ep_size`, and the per-expert byte
        count from the one format-agnostic granule walker. Without it the resolver falls back to
        `plan.build_planned_layers`, which GUESSES all three from config — and that guess has a live
        counterexample in this repo: `models/utils.py:145`'s `MoEMLP` constructs `MoELayer(...)` with
        no `quant=` argument at all, so its experts are bf16 whatever `config.quant` says, the config
        path sizes that stack as int4, and the arena is under-reserved ~4x IN THE DIRECTION THAT
        MAKES AN INFEASIBLE PLAN LOOK FEASIBLE. No config field distinguishes it.

        Resolving after the meta build costs nothing the "fail fast" ordering cared about: a meta
        build allocates ZERO bytes on either card, and the capacity abort still lands at `attach()`
        (`raise_if_infeasible()`, then the first pinned page), still before `load_state_dict`. What
        the ordering forbids is resolving after the weights have been READ, not after shapes exist.

        `driver` is an injection point for tests; production passes None and the plan resolver is
        consulted, in which case `device_budget_bytes` is REQUIRED — see `_resolve_driver`.

        `budget_is_derived` says whether that budget was an operator choice or a fallback the engine
        computed from the card. It is not cosmetic: the fallback is `total_memory * memory_ratio`,
        i.e. the whole KV budget, and a non-empty plan resolved under it is arithmetically guaranteed
        to fail `_determine_num_pages`'s `assert num_pages > 1` — but only AFTER the arena is pinned
        and the checkpoint is loaded. Passing the flag is what turns that into a refusal from
        integers. See `_refuse_derived_budget_that_leaves_no_kv`.

        `agreement_group` is the TP CPU (gloo) group. Pass it whenever tp_size > 1: the plan is only
        rank-identical because it is a pure integer function of config, and `device_budget_bytes` is
        an input this module cannot vet. See `WeightPlanResolution.assert_rank_agreement`. It is also
        RETAINED on the session and used by `_run_staged` to make every later step symmetric — the
        plan being identical does not make the STEPS identical, because pinning host pages and
        measuring free VRAM are per-rank, racy and timing-dependent."""
        drv = (
            driver
            if driver is not None
            else _resolve_driver(
                config,
                device_budget_bytes,
                agreement_group,
                gather=gather,
                model=model,
                log=log or _noop_log,
                # Whether the number above was CHOSEN by the operator or derived from the card. The
                # resolver cannot tell, and the two have opposite meanings: a chosen tier is a
                # decision, a derived one is `total_memory * memory_ratio` — the entire KV budget —
                # and a non-empty plan under it can never boot. See
                # `_refuse_derived_budget_that_leaves_no_kv`.
                budget_is_derived=budget_is_derived,
            )
        )
        host, device, offloadable = drv.plan_bytes() if drv is not None else (0, 0, 0)
        s = cls(
            driver=drv,
            probe=probe,
            accounting=WeightArenaAccounting(
                host_bytes=host, device_bytes=device, offloadable_bytes=offloadable
            ),
            phase=StageAPhase.PLANNED,
            log=log or _noop_log,
            agreement_group=agreement_group,
            gather=gather,
        )
        if s.enabled:
            s.log(f"weight offload: {drv.describe()}")
        return s

    @classmethod
    def disabled(cls, *, log: Optional[Callable[[str], None]] = None) -> "StageASession":
        """An inert session that still accepts the whole call sequence.

        Explicit rather than `begin(config, driver=None)`, because `begin` treats a None driver as
        "ask the resolver" — the two must not be spelled the same way, or a caller who meant "off"
        would silently get whatever the resolver decided."""
        return cls(driver=None, phase=StageAPhase.PLANNED, log=log or _noop_log)

    @property
    def enabled(self) -> bool:
        """A plan with at least one host-resident layer. An all-device plan is a no-op by design."""
        return self.driver is not None and self.accounting.host_bytes > 0

    # -- phase machine ------------------------------------------------------------------------

    def _advance(self, frm: StageAPhase, to: StageAPhase) -> None:
        if self.phase != frm:
            raise RuntimeError(
                f"weight offload: expected phase {frm.name} but the session is in {self.phase.name}. "
                "The Stage-A window is ordered plan -> attach -> load -> bind -> seal, and every "
                "boundary is load-bearing for either capacity-failure latency or KV accounting "
                "(see this module's docstring)."
            )
        self.phase = to

    def _sample(self, tag: str) -> None:
        if self.probe is None:
            return
        free, allocated, reserved = self.probe()
        self.accounting.sample(tag, free=free, allocated=allocated, reserved=reserved)

    # -- cross-rank symmetry --------------------------------------------------------------------

    def _run_staged(self, stage: str, body: Callable[[], Any]) -> Any:
        """Run one step of the window and make its SUCCESS OR FAILURE identical on every rank.

        WHY THIS IS NOT PARANOIA. `assert_rank_agreement` proves the ranks resolved the same PLAN,
        which is a pure integer function of config. It says nothing about whether the same plan
        SUCCEEDS on both ranks, and every step after it depends on a per-rank, racy, timing-dependent
        resource:

          * `attach()` pins host pages. P3b measured the ceiling at 62 GiB *across two ranks* against
            a 68.8 GiB target on an IDLE box, so which rank gets its pages is decided by which one
            calls `hipHostMalloc` first. `raise_if_infeasible()` reads a live `MemAvailable` for the
            same reason. Rank 0 can succeed and rank 1 fail on the same config, on the same boot.
          * `seal()` gates on MEASURED free-VRAM deltas, on the pool's `hipMalloc` fallback count and
            on allocator deltas. All three are per-rank readings of two different physical cards.

        Without this, the loser raises out of `Engine.__init__` and its process dies, while the
        winner walks on to `_determine_num_pages` -> `_sync_get_memory()` -> `all_reduce` on the TP
        gloo group, which `_init_communication` deliberately builds with `timeout=timedelta(days=7)`
        so an idle serve does not die. So the observable failure is a serve that HANGS FOR A WEEK
        with the real error printed only in the dead rank's log.

        The barrier itself must never be the desync, so it follows the pattern
        `WeightPlanResolution.assert_rank_agreement` documents: the local exception is CAUGHT (which
        is what keeps the failing rank alive long enough to participate), every rank enters the
        gather unconditionally, and the comparison happens strictly after it returns. A rank whose
        session is disabled still participates — "one rank thinks offload is off" is precisely a
        divergence this has to survive rather than deadlock on.
        """
        local: Optional[BaseException] = None
        result: Any = None
        try:
            result = body()
        except BaseException as exc:  # noqa: BLE001 - re-raised below, after the barrier
            local = exc
        statuses = self._gather_status(stage, local)
        if statuses is not None:
            bad = [(i, m) for i, m in enumerate(statuses) if m]
            if bad and local is None:
                raise RuntimeError(
                    f"weight offload: Stage-A '{stage}' failed on {len(bad)} of "
                    f"{len(statuses)} rank(s) but succeeded here. Raising in lockstep so the peer "
                    "does not block on the next collective (the TP gloo group has a 7-day timeout, "
                    "so a one-sided abort presents as a hung serve, not as an error).\n"
                    + "\n".join(f"  rank{i}: {m}" for i, m in bad)
                )
        if local is not None:
            raise local
        return result

    def _gather_status(self, stage: str, local: Optional[BaseException]) -> Optional[Tuple[str, ...]]:
        """One status string per rank ("" == ok), or None when there is nothing to gather over."""
        mine = "" if local is None else f"{type(local).__name__}: {local}"
        if self.gather is not None:
            return tuple(str(s or "") for s in self.gather(mine))
        if self.agreement_group is None:
            return None
        try:
            import torch.distributed as dist
        except Exception:  # pragma: no cover - torch-free host
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None
        if dist.get_world_size(self.agreement_group) <= 1:
            return None
        box: list = [None] * dist.get_world_size(self.agreement_group)
        dist.all_gather_object(box, mine, group=self.agreement_group)
        return tuple(str(s or "") for s in box)

    # -- the four steps -----------------------------------------------------------------------

    def attach(self) -> None:
        """Pin the host arena. Call BEFORE `load_state_dict` (`engine.py:239`)."""
        self._advance(StageAPhase.PLANNED, StageAPhase.ATTACHED)
        self._run_staged("attach", self._attach_body)

    def _attach_body(self) -> None:
        if not self.enabled:
            return
        self._sample("pre_attach")
        self.driver.attach_host_arena()
        self._sample("post_attach")
        self.log(
            f"weight offload: host arena attached, {gib(self.accounting.host_bytes)} planned "
            f"(device cost {gib(self.accounting.host_arena_device_cost)}, must be ~0)"
        )
        # The reservation's advisories. Computed by `PinnedWeightArena.reserve` and, until this, read
        # by nobody — including the chunk-growth one, which means the plan's feasibility answer and
        # the arena's actual reservation were charged against different chunk sizes.
        notes = getattr(self.driver, "attach_notes", None)
        for note in (notes() if callable(notes) else ()):
            self.log(f"weight offload: {note}")

    def note_loaded(self) -> None:
        """Call immediately after `post_load()` (`engine.py:241`)."""
        self._advance(StageAPhase.ATTACHED, StageAPhase.LOADED)

    def bind(self, model: Any) -> Any:
        """Execute the plan over the live model: move, verify, rebind, drop.

        Call after `post_load()` and BEFORE `_determine_num_pages` (`engine.py:245`)."""
        self._advance(StageAPhase.LOADED, StageAPhase.BOUND)
        return self._run_staged("bind", lambda: self._bind_body(model))

    def _bind_body(self, model: Any) -> Any:
        if not self.enabled:
            return None
        self._sample("pre_bake")
        self.outcome = self.driver.bind(model)
        self.accounting.copied_bytes = int(self.driver.moved_bytes())
        self.accounting.arena_torch_bytes = int(self.driver.arena_torch_bytes() or 0)
        # MEASURED off the live post-load containers, never inferred from the plan. Without it the
        # ledger's "device tier == plan.device_resident_bytes" check is algebraically the copied-
        # bytes check restated (`total - copied` == `host - copied` when `total = host + device`)
        # and cannot catch a plan whose byte model disagrees with post_load's real output.
        observed_fn = getattr(self.driver, "observed_device_bytes", None)
        observed = observed_fn() if callable(observed_fn) else None
        self.accounting.observed_device_bytes = None if observed is None else int(observed)
        self._sample("post_bake")
        describe = getattr(self.outcome, "describe", None)
        self.log(f"weight offload: {describe() if callable(describe) else self.outcome}")
        return self.outcome

    def seal(self) -> None:
        """Close the mapping window, then gate the accounting.

        Raises if the arena and the plan disagree about bytes, or if the host arena took VRAM. Call
        immediately before `_determine_num_pages`.

        THREE GATES, IN THIS ORDER, ALL BEFORE THE KV POOL IS SIZED:

          1. `assert_arena_clean()` — no arena row fell back to `hipMalloc`. A fallback is a DOUBLE
             error against the KV budget, not a slowdown: the bytes really are in VRAM, and
             `model_memory_correction` then subtracts those same bytes from the model term as though
             they were host RAM, so the pool is oversized by twice the fallback. `ArenaMemPool`'s own
             docstring says "assert it is zero, do not merely log it" — this is where that happens.
          2. `assert_device_accounting()` — plan §5.3's mandated boot assertion, fed the MEASURED
             device-resident bytes. This is what makes "no sixth subtrahend" safe: the device tier is
             billed only by `model_memory`, so if the plan and the live containers disagree about how
             big it is, nothing else would ever notice.
          3. the byte ledger (`accounting.report()`).
        """
        self._advance(StageAPhase.BOUND, StageAPhase.SEALED)
        self._run_staged("seal", self._seal_body)

    def _seal_body(self) -> None:
        global _TORCH_SLACK
        if not self.enabled:
            return
        clean = getattr(self.driver, "assert_arena_clean", None)
        if callable(clean):
            clean()
        # Then the LAYOUT. `assert_arena_clean` answers "did anything escape to VRAM"; this answers
        # "did what stayed land where it was reserved". Both before the KV pool is sized, because a
        # drifted layout dequantizes one expert against another's scale — plausible text, no crash,
        # and nothing downstream can see it.
        layout = getattr(self.driver, "verify_arena_layout", None)
        if callable(layout):
            v = layout()
            what = getattr(v, "describe", None)
            self.log(f"weight offload: arena layout verified — {what() if callable(what) else v}")
        self._assert_device_accounting()
        self._require_complete_ledger()
        self.driver.freeze()  # arena.mark_populated() + arena.freeze() -> hipmem.freeze() (rule R1)
        _TORCH_SLACK = self.accounting.torch_slack_bytes
        self._sealed_arena_activity = self._arena_activity()
        rep = self.accounting.report()
        if not rep.ok:
            raise RuntimeError(
                "weight offload: the arena and the plan disagree, so the KV pool would be sized "
                "against a number that is not true.\n" + rep.render()
            )
        self.log(rep.render())

    def verify_after_capture(self) -> None:
        """Re-gate the arena AFTER every HIP graph has been captured. Call once, from the engine.

        WHY `seal()` CANNOT DO THIS. `ArenaMemPool.assert_clean()` has a dedicated failure branch for
        "N allocation(s) hit the arena callback DURING HIP graph capture", and `ArenaMemPool._alloc`
        raises `alloc_during_capture` only while `torch.cuda.is_current_stream_capturing()` is true.
        But `seal()` runs at `engine.py:284` and every capture — `GraphRunner` (`:578`) and
        `_capture_canvas_graphs` (`:597`) — runs after it. At the only point the counter was ever
        read it is provably zero, so the check could never fire and the failure it names had no
        detector at all. This is that detector.

        WHAT GOES WRONG IF NOBODY LOOKS. Two distinct failures, both silent:

          * An allocation SERVED from the arena mid-capture succeeds (the bump allocator hands out a
            pointer, and pointer-wise a graph is happy to bake it in), but the arena never frees, so
            those bytes are gone from the host tier for the life of the process while the capacity
            plan still counts them as available. The address is now inside a captured graph, so it is
            also the one allocation that can never be moved or reclaimed.
          * An allocation that MISSES the arena mid-capture cannot fall back — `hipMalloc` is illegal
            during capture, so `_fallback` deliberately returns NULL, which torch dereferences. That
            is a segfault whose only breadcrumb is one `[weight-arena]` line on stderr.

        The `arena_activity()` comparison catches the first case even on a torch that never reports
        capture state (`_capturing()` answers False on an old torch and `alloc_during_capture` stays
        0), because it is arithmetic on the arena's own counters rather than a query. It must read
        the POOL's `(alloc_events, served_bytes)` and not the stack allocator's `bytes_used`: a
        capture-time allocation arrives as torch calling the C ABI callback directly, so it never
        touches `TorchStackAllocator` and would leave that total untouched.

        Symmetric across ranks for the same reason every other step is: capture is per-rank, and a
        one-sided raise here happens strictly BEFORE the first forward, i.e. before the first
        in-graph collective, so the peer would block on it forever.
        """
        if self.phase < StageAPhase.SEALED:
            raise RuntimeError(
                "weight offload: verify_after_capture() ran before seal(); it is the POST-capture "
                f"gate and the session is in {self.phase.name}."
            )
        self._run_staged("post_capture", self._verify_after_capture_body)

    def _arena_activity(self) -> Tuple[int, int]:
        """`(alloc_events, served_bytes)` off the arena's own MemPool, or `(0, 0)` without one."""
        fn = getattr(self.driver, "arena_activity", None)
        if not callable(fn):
            return (0, 0)
        a, b = fn()
        return (int(a), int(b))

    def _verify_after_capture_body(self) -> None:
        if not self.enabled:
            return
        now = self._arena_activity()
        if now != self._sealed_arena_activity:
            raise RuntimeError(
                "weight offload: the host arena was ALLOCATED FROM during HIP graph capture — "
                f"(alloc_events, served_bytes) was {self._sealed_arena_activity} at seal() and is "
                f"{now} after capture. A capture-time arena allocation is never freed (the arena is "
                "a forward-only bump allocator with a no-op free) and its address is now baked into "
                "a graph that replays for the life of the process, so the bytes are permanently "
                "lost from the host tier while the capacity plan still counts them as available. "
                "Only weight materialisation may run inside ArenaMemPool.use()."
            )
        clean = getattr(self.driver, "assert_arena_clean", None)
        if callable(clean):
            clean()
        self.log("weight offload: arena clean after graph capture")

    def _require_complete_ledger(self) -> None:
        """Every sample the gate differences MUST be present, or the gate is not a gate.

        `WeightArenaAccounting._delta` returns 0 for a missing sample — deliberately, so a DISABLED
        session that never sampled anything stays usable instead of raising. The cost of that
        choice is that on an ENABLED session a missing sample makes the checks pass VACUOUSLY:
        `host_arena_device_cost` reads 0 ("the host arena took no VRAM") and
        `originals_released` reads 0 against a `copied_bytes` of 0, so the single highest-value
        assertion in the feature — the one that catches Phase 0's `location=Host` silently handing
        back VRAM — is silently switched off rather than failed.

        That is reachable: a session constructed with `probe=None` but a live driver takes no
        samples at all, and `_sample` is a no-op rather than an error. This turns "the ledger was
        never populated" into a refusal instead of a green report over an empty table.
        """
        missing = [t for t in self.accounting.ORDER if not self.accounting.has(t)]
        if missing:
            raise RuntimeError(
                "weight offload: the VRAM ledger is incomplete — no "
                f"{', '.join(missing)} sample(s) were taken, so the seal gate would pass "
                "vacuously (every check is a difference between two samples and a missing one "
                "reads as a delta of 0). An enabled Stage-A session MUST be constructed with a "
                "memory probe; without one there is no evidence the host arena cost zero device "
                "bytes, which is the assertion Phase 0 exists to force."
            )

    def _assert_device_accounting(self) -> None:
        """Plan §5.3's boot assertion, against the measured device tier. No-op without a resolution.

        Delegated to `WeightPlanResolution.assert_device_accounting` rather than reimplemented, so
        the tolerance and the failure text live with the plan that is being asserted about.
        """
        observed = self.accounting.observed_device_bytes
        if observed is None:
            return
        resolution = getattr(self.driver, "resolution", None)
        assert_fn = getattr(resolution, "assert_device_accounting", None)
        if callable(assert_fn):
            assert_fn(observed, tolerance_bytes=self.accounting.tol_bytes)

    # -- what the engine reads afterwards -------------------------------------------------------

    def model_memory_correction(self) -> Tuple[int, int]:
        """`(allocated, reserved)` bytes to remove from `_determine_num_pages`'s model term.

        `(0, 0)` unless an enabled session has SEALED — reading it earlier would apply a correction
        derived from an incomplete ledger, and reading it on a non-offloading serve must be free."""
        if not self.enabled or self.phase < StageAPhase.SEALED:
            return 0, 0
        return self.accounting.model_memory_correction()

    def kv_annotation(self) -> str:
        """The `weight-arena=...` suffix for the single `KV sizing:` line.

        Delegates to `WeightPlanResolution.summary_line()` when the driver carries one, so the boot
        banner and the KV-sizing line can never quote different numbers for the same plan. Empty
        string when nothing is offloaded, which keeps that log line byte-identical today."""
        if not self.enabled:
            return ""
        line = getattr(self.driver, "summary_line", None)
        return " " + line() if callable(line) else ""

    def kv_budget_failure_hint(self) -> str:
        """Extra clause for `Engine._determine_num_pages`'s "Not enough memory for KV cache" assert.

        Empty on every serve that does not offload, so that assert's text is unchanged today.

        WHY IT IS NEEDED. The device tier is billed by `model_memory` and nothing else — that is the
        plan's "no sixth subtrahend" rule, and it is correct. The consequence is that a device tier
        which is too large does not fail as "the tier is too large"; it fails as a KV pool of zero
        pages, blamed on recurrent state, the draft model or graph capture. The remedies the assert
        names all make it WORSE: lowering `--memory-ratio` shrinks the same budget the tier already
        consumed, and `--num-pages` overrides the sizing without freeing anything. The one lever
        that actually moves is the tier, so it has to be in the message.
        """
        if not self.enabled:
            return ""
        dev = gib(self.accounting.device_bytes)
        host = gib(self.accounting.host_bytes)
        return (
            f". WEIGHT OFFLOAD IS ACTIVE: {dev} per rank of resident expert tier is billed inside "
            f"`model` above (by design — the device tier has no separate reservation), with {host} "
            f"streamed from the pinned host arena. If `model` is close to the whole budget, the "
            f"lever is --weight-offload-device-gb (LOWER it: every GiB returned to the pool is "
            f"~200k KV tokens), not --memory-ratio, which shrinks the same budget the tier already "
            f"took"
        )


# =============================================================================================
# The production driver: plan + arena + torch pool + seams
# =============================================================================================


class StageARuntime:
    """Composes the resolved plan with the arena, the torch pool, the stack allocator and the seams.

    This is the only object that knows all five exist. Everything it does is a call into a sibling
    module — the value is that the composition happens once, in the order `StageASession` enforces,
    instead of being spelled out at the engine call site where the ordering rules are invisible.

    `torch` and the HIP binding are imported lazily inside the methods that need them, so
    constructing a runtime (and printing its plan) works on a machine with neither.
    """

    def __init__(
        self,
        resolution: Any,
        *,
        device_index: int,
        rank: int = 0,
        local_ranks: int = 1,
        label: str = "weights",
    ) -> None:
        from .config import create_pinned_weight_arena, resolve_arena_settings

        self.resolution = resolution
        self.settings = resolve_arena_settings()
        self.device_index = int(device_index)
        self.arena = create_pinned_weight_arena(
            device_index, rank=rank, local_ranks=local_ranks, label=label, settings=self.settings
        )
        self.pool = None
        self.allocator = None
        self.outcome = None
        self.seams: Tuple[Any, ...] = ()

    # -- StageADriver ---------------------------------------------------------------------------

    @property
    def plan(self):
        return self.resolution.plan

    def plan_bytes(self) -> Tuple[int, int, int]:
        p = self.plan
        return (p.host_resident_bytes, p.device_resident_bytes, p.total_resident_bytes)

    def host_row_bound_bytes(self) -> int:
        """Proven upper bound on ONE allocation the torch `MemPool` will ask the arena for.

        SINCE M1-B THIS IS THE FALLBACK, not the reservation: `host_row_requests()` enumerates the
        rows and `attach_host_arena` reserves those, which is exact rather than bounded. This is
        what a plan that could not break its containers down still gets, and it must stay a
        GUARANTEE — see `chunk_plan.headroom_chunks`.

        The arena's bump allocator refuses to straddle a chunk, so `headroom_chunks` can only
        guarantee placement for `chunk - m` bytes per chunk, where `m` bounds a single region. With
        `m` unknown it falls back to `ceil(extra/chunk)`, which under-reserves by the abandoned tails
        and pushes the last rows into `hipMalloc` VRAM (see `attach_host_arena`).

        A row is one tensor of one host-placed layer's containers, and those tensors sum to that
        layer's `resident_bytes`, so the per-layer figure bounds every row. Read off the PLAN, not
        off the model: this runs before `load_state_dict`, which is the entire point of doing the
        capacity work here.

        DELEGATED TO `OffloadPlan.max_host_row_bytes` RATHER THAN RECOMPUTED HERE, and that is not
        tidiness. `plan.arena_reservation_bytes` — which is what `raise_if_infeasible()` gates on,
        two lines above the `reserve()` this feeds — charges chunks against the SAME bound. If the
        two are computed in two places they can differ, and a bound here that is LARGER than the one
        the resolver used produces the worst available outcome: the resolver says FEASIBLE, prints a
        "raise the device tier to >= X" figure derived from the smaller bound, and then `reserve()`'s
        capacity gate refuses the boot anyway — so the operator is handed a tier that still does not
        work. One property, one definition.

        The plan's figure is also TIGHTER: it falls back to the per-layer `resident_bytes` only when
        nothing better is known (`LayerWeights.row_bound`), and uses the largest single COMPONENT
        when the sizing path could derive one. A row really is one component's stacked slab, never a
        whole container, so the layer figure over-reserves by ~2x on every quantized checkpoint —
        over-reserving is charged against the live `MemAvailable` gate and refuses boots that would
        have fitted, which is the same class of wrong answer as under-reserving, just the polite one.
        """
        return self.plan.max_host_row_bytes

    def attach_notes(self) -> Tuple[str, ...]:
        """Everything the reservation learned that a log has to carry.

        `PinnedWeightArena.reserve` sets three advisories and `allocate_raw` records a shortfall
        diagnosis, and until this hook not one of them reached an operator — they were computed,
        stored on the arena, and read by nobody. The chunk-growth one in particular is a change to
        the granularity the PLAN charged capacity against (`resolve_arena_settings().chunk_bytes`),
        so a silent growth means the plan's feasibility answer and the arena's actual reservation
        were computed against different chunk sizes.
        """
        notes = [
            n
            for n in (
                getattr(self.arena, "chunk_growth_advisory", None),
                getattr(self.arena, "headroom_advisory", None),
                getattr(self.arena, "packing_advisory", None),
            )
            if n
        ]
        plan = getattr(self.arena, "plan", None)
        if plan is not None:
            notes.append(f"arena reservation: {plan.describe()}")
        return tuple(notes)

    def host_row_requests(self) -> Tuple[Any, ...]:
        """The host tier as ENUMERATED, ordered arena rows. `()` when the plan could not break the
        containers down and the anonymous-headroom bound has to be used instead.

        Delegated to `OffloadPlan.host_row_requests` for the same reason `host_row_bound_bytes` is
        delegated: `WeightPlanResolution.host_reservation_bytes_per_rank` — the number
        `raise_if_infeasible()` gates on, two lines above the `reserve()` this feeds — is computed
        from the identical list. Two derivations of the same row set can differ, and a reservation
        larger than the one the resolver called feasible refuses the boot AFTER telling the operator
        the tier they granted was enough."""
        return self.plan.host_row_requests()

    def attach_host_arena(self) -> None:
        """Reserve the rows, pin, self-test, then stand up the torch pool over the arena.

        `raise_if_infeasible()` runs FIRST so an impossible plan fails before a single page is
        pinned.

        THE ROWS ARE ENUMERATED, NOT GUESSED, AND THAT IS WORTH ~5 GiB/rank OF VRAM. This used to
        reserve the whole tier as anonymous HEADROOM — `reserve([], extra_bytes=host_resident_bytes,
        extra_max_region_bytes=max_host_row_bytes)` — on the reasoning that the rows are carved by
        the torch `MemPool` callback at bind time and their sizes are a property of the `post_load()`
        containers, which do not exist yet. The second half of that is false: `sizing.meta_gemm_spec`
        builds the real container under `torch.device("meta")` before `load_state_dict` and
        `sizing.post_load_delta_bytes` carries the two shipped containers that are not byte-invariant
        across `post_load`, so the row SIZES are knowable here even though the row IDENTITIES are
        not.

        The difference is not marginal. `chunk_plan.headroom_chunks` can only guarantee `chunk - m`
        placeable bytes per chunk, i.e. it prices the worst row landing at the worst offset in EVERY
        chunk; on the target shape (six rows per layer summing to 666 MiB, 2 GiB chunks) that is 20
        chunks where the real next-fit allocator fits three whole layers per chunk and uses 16 —
        +28% of the pinned tier, which at a 55.80 GiB node ceiling is 5+ GiB/rank of device tier the
        operator has to surrender to compensate, on a 16 GiB card.

        So the rows go in as FORECAST `RegionRequest`s (`chunk_plan.RegionRequest.forecast`): they
        are laid out by the same allocator that will carve them, which makes `reserve()`'s chunk
        count the real one, gives `ChunkPlan.digest()` something to hash besides the chunk count, and
        gives `verify_matches_plan()` something to compare. They do NOT claim the carve will use
        their names — the `MemPool` C ABI passes a size and no identity — so reconciliation is by
        envelope and by attribution; see `PinnedWeightArena.verify_matches_plan`.

        `extra_bytes` / `extra_max_region_bytes` stay on the call and are the FALLBACK, not
        vestigial: a plan whose containers could not be broken down (a test double, a future
        descriptor, a format whose spec exposes no component list) still gets the bounded
        anonymous reservation rather than a silent `ceil(payload/chunk)`. Dropping the bound is a
        one-line edit that imports, boots, and puts the tail of the host tier in VRAM.
        """
        import torch

        from .stacks import TorchStackAllocator
        from .torch_pool import ArenaMemPool

        self.resolution.raise_if_infeasible()
        rows = self.host_row_requests()
        self.arena.reserve(
            rows,
            extra_bytes=0 if rows else self.plan.host_resident_bytes,
            extra_max_region_bytes=0 if rows else self.host_row_bound_bytes(),
        )
        self.arena.attach(selftest=self.settings.selftest, first_touch=self.settings.first_touch)
        self.pool = ArenaMemPool(self.arena)
        self.allocator = TorchStackAllocator(
            device=torch.device("cuda", self.device_index),
            host_alloc=lambda shape, dtype: self.pool.empty(*shape, dtype=dtype),
        )

    def bind(self, model: Any) -> Any:
        """Attach a seam to every `MoELayer`, then execute the plan over them.

        `freeze=False`: the seams are frozen by `StageASession.seal()`, together with the arena and
        the process-wide `hipmem` latch, so there is exactly one moment at which the mapping window
        closes rather than three."""
        from .moe_interpose import attach_seams, bind_plan

        self.seams = attach_seams(model)
        self.outcome = bind_plan(self.seams, self.plan, self.allocator, freeze=False)
        return self.outcome

    def moved_bytes(self) -> int:
        return int(getattr(self.outcome, "moved_bytes", 0) or 0)

    def arena_torch_bytes(self) -> int:
        """Arena bytes that ACTUALLY went through torch's allocator — the pool's own counter.

        This number's only job is to say how much of `torch.cuda.memory_allocated()` is host RAM, so
        that `model_memory_correction()` can remove it. It must therefore be a measurement OF TORCH,
        and it used to be `TorchStackAllocator.bytes_used(HOST)` — the bytes this module handed out,
        which by construction equals `moved_bytes` whatever torch did with them. Two consequences,
        both in the unsafe direction:

          * its own contract ("0 if the rows bypass it") could never be met. Every documented way the
            `MemPool` stops routing — installed on the wrong device at TP=2, a torch that changes
            `use_mem_pool` semantics, an older torch without the `device=` kwarg — leaves the stack
            allocator's total at the full host tier while `memory_allocated()` never counted a byte
            of it. The correction then subtracts tens of GB from a reading that does not contain
            them, `model_memory` under-bills the model, `available_memory` is over-stated and the KV
            pool is sized against VRAM that is gone. That failure does not appear at boot; it appears
            on the first forward.
          * it counted `hipMalloc` FALLBACK rows, which are real VRAM, as arena bytes to be removed
            from the model term.

        `ArenaMemPool.served_bytes` is incremented inside the C ABI alloc callback, and only when the
        arena actually satisfied the request, so both cases collapse to 0/short by construction and
        the ledger's "torch released the originals" check then reads correctly instead of failing
        with a message about a surviving alias. `assert_arena_clean()` still refuses any fallback at
        all; this makes the arithmetic right even before that gate is consulted.

        Slightly OVER-states by the arena's per-row alignment padding (`served_bytes` counts the
        carved region, torch counts its own rounding). Both are far below the 64 MiB ledger
        tolerance, and over-stating is clamped by `alloc_correction`'s `min(copied_bytes, ...)`.
        """
        return 0 if self.pool is None else int(self.pool.served_bytes)

    def observed_device_bytes(self) -> Optional[int]:
        """Device-resident expert bytes measured off the live post-load containers."""
        return None if self.outcome is None else int(
            getattr(self.outcome, "device_resident_bytes", 0) or 0
        )

    def arena_activity(self) -> Tuple[int, int]:
        """`(alloc_events, served_bytes)` straight off the `ArenaMemPool`.

        These are the counters the C ABI callback increments, so they move for ANY allocation torch
        routes to the arena — including one made from inside HIP graph capture, which never touches
        `TorchStackAllocator` and so leaves `arena_torch_bytes()` unchanged. That is the whole point:
        `StageASession.verify_after_capture` compares this pair across the capture window."""
        if self.pool is None:
            return (0, 0)
        return (int(self.pool.alloc_events), int(self.pool.served_bytes))

    def assert_arena_clean(self) -> None:
        """No arena row may have fallen back to `hipMalloc`.

        The fallback exists because returning NULL from a ctypes allocator callback is a segfault
        with no diagnostic, so a bad allocation becomes a COUNTED correctness problem instead. This
        is where the count is read. Two things go wrong at once when it is non-zero and nobody looks:
        bytes budgeted as host-resident are sitting in VRAM the capacity plan says is free, AND
        `model_memory_correction()` subtracts those same bytes from the model term as if they were
        host RAM — so the KV pool is oversized by twice the fallback and the failure surfaces much
        later as an unrelated OOM.

        `expect_served_bytes` is passed, and `ArenaMemPool.assert_clean`'s own docstring says why it
        has to be: the zero checks are the ones a fallback count cannot see. Every documented way the
        pool stops routing entirely — installed on the wrong device at TP=2, a torch that changes
        `use_mem_pool` — yields `torch_fallbacks == 0` over a serve that put the whole host tier in
        VRAM, i.e. a perfectly clean-looking ledger. Called with no expectation the gate only refuses
        the "nothing at all was served" shape.

        The expectation is `moved_bytes()`, NOT `plan.host_resident_bytes`. `served_bytes` counts the
        carved region (aligned up), so it is `>= copied` by construction and the comparison is exact
        rather than tolerance-bearing — whereas the plan figure is only equal to the copied bytes
        within the ledger's 64 MiB tolerance, so gating on it here would turn a boot the accounting
        report accepts into a hard refusal from a strict `<`. Whether the bake moved what the plan
        priced is check 2's job, with the tolerance that check owns.
        """
        if self.pool is not None:
            self.pool.assert_clean(expect_served_bytes=self.moved_bytes())

    def verify_arena_layout(self) -> Any:
        """The carve followed the reservation. THE FIRST PRODUCTION CALLER of this check.

        `PinnedWeightArena.verify_matches_plan()` has existed since M1-A with no caller worth
        having, because on the shipping shape it had nothing to compare: the tier was reserved as
        anonymous headroom, so the planned table was empty and it passed vacuously. Now the rows go
        in enumerated, so the call is evidence — and `require_coverage` makes a reservation that
        somehow reverted to headroom fail HERE, loudly, instead of being reported as a pass.

        Runs inside `seal()`, before the byte ledger and before `_determine_num_pages`: a layout
        drift means one expert's bytes are at another's address, which produces plausible text with
        no crash and is undetectable from any later vantage point.
        """
        return self.arena.verify_matches_plan()

    def freeze(self) -> None:
        """One moment closes the whole mapping window: the seams, the arena, and the HIP binding.

        `bind_plan` was called with `freeze=False` precisely so this is not spread across three
        places — rule R1 is a statement about a point in the process's life, and a reader has to be
        able to find that point."""
        for seam in self.seams:
            seam.freeze()
        self.arena.mark_populated()
        self.arena.freeze()  # -> hipmem.freeze(): every mapping entry point raises from here

    def summary_line(self) -> str:
        return self.resolution.summary_line()

    def describe(self) -> str:
        return self.resolution.summary_line()


def _warn_if_offload_was_requested(
    config: Any, resolution: Any, log: Callable[[str], None]
) -> None:
    """Say so, loudly, when the operator asked for offload and the resolver planned nothing.

    An empty plan is the NORMAL, correct outcome on a model that fits — that is the whole reason
    this path can run on every serve. But it is also what a DENSE model produces, and what a model
    family whose sparse block is not a `MoELayer` produces, and in those cases `--weight-offload-gb`
    / `--weight-offload-device-gb` are accepted, echoed back in `[serve]`, and then do exactly
    nothing. The operator's next observation is an OOM at load with the flags apparently set, and
    nothing anywhere connects the two.

    The reason it can be dense-shaped and still silent is a real gap, not a hypothetical:
    `plan.observed_planned_layers` walks `moe_interpose.discover_moe_layers`, and
    `moe_interpose.attach_seams` binds `MoELayer` only. `layers/linear._LinearTPImpl` already
    carries the granule declarations (`_granule_dense`, `_granule_policy`) so the WALKER handles
    dense — but no planner enumerates dense linears and no seam binds them, so dense weights,
    `lm_head` and the embeddings are never offloadable. That is the repo's "MoE and dense land
    together" rule outstanding, and until it is discharged an operator has to be told rather than
    left to infer it from an OOM.

    A log line, not a raise: refusing to boot because a knob had no effect would turn "you asked for
    something this build cannot do" into an outage, and the knobs are documented as clamps on an
    automatic decision rather than as switches.
    """
    asked = max(
        float(getattr(config, "weight_offload_device_gb", 0.0) or 0.0),
        float(getattr(config, "weight_offload_gb", 0.0) or 0.0),
    )
    if asked <= 0:
        return
    diag = getattr(resolution, "diagnostics", None) or {}
    n_layers = diag.get("n_offloadable_layers")
    skipped = tuple(diag.get("skipped", ()) or ())
    why = (
        "no offloadable MoE layer was found at all — this build offloads `MoELayer` expert stacks "
        "only, so a DENSE model, or a family whose sparse block is not a MoELayer, has nothing to "
        "plan (dense linears / lm_head / embeddings are not yet bindable; see "
        "weights/moe_interpose.attach_seams)"
        if n_layers == 0
        else "every offloadable layer fits the device budget, so nothing needs to be host-resident"
    )
    log(
        "weight offload: --weight-offload-device-gb/--weight-offload-gb were set but the resolved "
        f"plan is EMPTY and NO weights will be host-resident. Reason: {why}."
        + (f" Skipped: {'; '.join(skipped)}." if skipped else "")
    )


class UnconfiguredDeviceTierError(RuntimeError):
    """The device tier was DERIVED, and the derived value cannot leave room for a KV pool.

    Its own class so the engine (and a test) can distinguish "the operator has not chosen a tier"
    from a capacity failure, a plan desync or an arena fault. See
    `_refuse_derived_budget_that_leaves_no_kv`.
    """


def _refuse_derived_budget_that_leaves_no_kv(resolution: Any, budget_is_derived: bool) -> None:
    """Abort in MILLISECONDS on a plan that `_determine_num_pages` is guaranteed to reject.

    `Engine._weight_offload_device_budget`, with no `--weight-offload-device-gb`, derives the tier as
    `total_memory * memory_ratio`. That is not a conservative over-grant; it is the WHOLE KV BUDGET.
    `_determine_num_pages` computes

        available = memory_ratio * old_free - model - state - draft - graph - snap

    and the device tier is billed inside `model` — deliberately, because the plan forbids a sixth
    subtrahend. So whenever the expert stack does NOT fit (which is exactly when the plan is
    non-empty), the greedy fill takes the tier all the way up to the budget, `model >= tier ==
    memory_ratio * total >= memory_ratio * old_free`, and `available` is negative before a single
    other reservation is counted. `assert num_pages > 1` cannot not fire.

    Left alone the operator pays for that arithmetic the slowest possible way: `attach()` pins tens
    of GiB of unevictable host RAM (P3b: 4.88 GB/s, and the two-rank ceiling was already swapping),
    the whole checkpoint is read and repacked, the bake copies the host tier, and only THEN does the
    KV assert fire. That is minutes with the box in swap, to reach a conclusion available from
    integers before anything was allocated — the exact latency the Stage-A ordering exists to remove.

    So refuse here, and name the number. This is not an on/off switch and does not reintroduce one
    (plan §6.2): the plan is still derived on every serve, a model that FITS still resolves to an
    empty plan and this is never reached, and an operator-supplied tier passes through untouched.
    The only case that refuses is the one that could never have booted, and it refuses with the
    resolver's own sweep so the tier chosen next is a measured point rather than a guess.
    """
    if not budget_is_derived:
        return
    raise UnconfiguredDeviceTierError(
        "weight offload: this checkpoint does not fit the card, so the resolver planned a host tier "
        "— but no --weight-offload-device-gb was given, so the device tier was DERIVED as the "
        "card's whole KV budget (total_memory x memory_ratio). The device tier is billed inside "
        "`model` in _determine_num_pages (by design: the plan forbids a sixth subtrahend), so a tier "
        "equal to the budget leaves the KV pool exactly nothing and the boot would fail at "
        "`assert num_pages > 1` — after pinning the host arena and loading the whole checkpoint. "
        "Refusing now instead, from integers, before a page is pinned.\n"
        f"  resolved plan: {resolution.summary_line()}\n"
        "  FIX: set --weight-offload-device-gb (GiB per rank) to a tier that leaves room for the KV "
        "pool, the dense weights and the graph buffers. Each GiB returned to the pool is ~200k KV "
        "tokens; the sweep below prices the trade.\n"
        + str(getattr(resolution, "sweep_text", "") or "")
    )


def _resolve_driver(
    config: Any,
    device_budget_bytes: Optional[int],
    agreement_group: Any = None,
    *,
    gather: Any = None,
    model: Any = None,
    log: Callable[[str], None] = _noop_log,
    budget_is_derived: bool = False,
) -> Optional[StageADriver]:
    """Consult the single plan resolver (`weights/plan.py`, plan §6.2) and wrap it in a runtime.

    `resolve_weight_plan` is the only place that decides what is offloaded; `--weight-offload-gb`
    only CLAMPS its automatic decision and is never an on/off switch. When the whole stack fits the
    device budget the resolution is empty, `enabled` is False, and Stage A costs nothing — which is
    what keeps this path on every serve so it cannot rot.

    `model` is the meta-built model and is forwarded verbatim. It is what keeps this resolver
    format- and family-agnostic: see `StageASession.begin`. It is optional only so a caller with no
    model (a sizing tool, a test) still works, never because production may omit it.

    Returns None (an inert session) when the plan is empty, so a non-offloading serve never
    constructs an arena, never imports the HIP binding, and behaves exactly as it does today.

    `device_budget_bytes` is MANDATORY here. The resolver's own fallback is
    `config.weight_offload_device_gb`, which is 0.0 on every serve nobody has configured, and the
    resolver reads a zero device budget as "the expert tier may occupy no VRAM at all" — an ALL-HOST
    plan for every MoE model, fitting or not. That is a silent 10x regression, not a safe default, so
    the one caller that reaches the resolver has to name a number and this raises if it does not.
    """
    from .plan import resolve_weight_plan

    if device_budget_bytes is None:
        raise ValueError(
            "weight offload: StageASession.begin() reached the plan resolver without a device "
            "budget. A missing budget resolves to 0 bytes of VRAM for the expert tier, i.e. every "
            "MoE layer streamed from host RAM on a serve that never asked for offload. Pass "
            "device_budget_bytes (Engine._weight_offload_device_budget derives it from the card's "
            "total memory), or inject a driver in tests."
        )

    # The planner charges host capacity in WHOLE ARENA CHUNKS, because `PinnedWeightArena.attach`
    # hipHostMallocs and first-touches every chunk at full `chunk_bytes`. That charge has to be made
    # against the chunk size THIS process will actually use, or an operator who moves
    # MINISGL_WEIGHT_ARENA_CHUNK_MIB gets a plan that says FITS and an arena that aborts mid-pin.
    from .config import resolve_arena_settings

    resolution = resolve_weight_plan(
        config,
        device_budget_bytes=device_budget_bytes,
        arena_chunk_bytes=resolve_arena_settings().chunk_bytes,
        # THE GENERALITY HINGE. With a model the resolver reads the layer set, this rank's EP/TP
        # sharding and the per-expert byte count off the live objects (`observed_planned_layers`);
        # without one it falls back to `build_planned_layers`, which transcribes seven builder files
        # and nine container `__init__`s and is already wrong for `models/utils.py`'s unquantized
        # `MoEMLP`. Forwarded, never re-derived here.
        model=model,
    )
    # BEFORE the `enabled` early return, and that ordering is load-bearing. Every rank must enter the
    # gather or the gather itself desyncs: if one rank resolved an empty plan and its peer did not,
    # putting this after the branch would leave the offloading rank blocked in all_gather_object
    # forever while the other walked on. Checking here means the FIRST symptom of a divergence is a
    # clear error on both ranks instead of a hang, or — worse — two ranks quietly sizing different KV
    # pools from different `model_memory_correction()` values (`num_pages` is not cross-rank reduced).
    resolution.assert_rank_agreement(agreement_group, gather=gather)
    if not resolution.enabled:
        _warn_if_offload_was_requested(config, resolution, log)
        return None
    _refuse_derived_budget_that_leaves_no_kv(resolution, budget_is_derived)
    return StageARuntime(
        resolution,
        device_index=int(getattr(config, "device_index", 0) or 0),
        rank=int(getattr(getattr(config, "tp_info", None), "rank", 0) or 0),
        local_ranks=int(getattr(resolution, "local_ranks", 1) or 1),
    )
