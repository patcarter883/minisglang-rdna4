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
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple

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

    def prove_seam(self, model: Any) -> Any:
        """Prove the offload arm is in the serving path of `model`. Returns a `SeamResidencyProof`.

        Optional — reached through `getattr`, so a test double that implements only the byte ledger
        keeps working. Every OTHER method on this protocol reports on the arena or the plan, and all
        of them can answer correctly about a model the engine will not run; this is the only one
        that asks the live module tree."""

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


def _engaged(name: str) -> None:
    """One line in the process-wide engagement ledger (`minisgl._hip_engage`).

    LAZY, and it swallows ImportError, for the layering reason this package's `__init__` documents:
    `minisgl._hip_engage` reaches `minisgl.utils`, which imports torch through `utils.arch`, and
    `bake` must stay importable (and unit-testable) on a machine where `import torch` fails. The
    ledger is diagnostic, so losing it on such a machine is correct; losing the ability to plan is
    not.
    """
    try:
        from minisgl._hip_engage import engaged
    except Exception:  # pragma: no cover - torch-free host
        return
    engaged(name)


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
    # The model `bind()` was executed over, kept so `seal()` and `verify_after_capture()` can ASSERT
    # THE SEAM against the object the engine actually serves rather than against the seam list the
    # binder handed back. The engine holds this model for the life of the process anyway, so the
    # reference costs nothing; what it buys is that the two gates cannot be pointed at a different
    # tree from the forward. `None` on a disabled session and on every `bind()` that never ran.
    _model: Any = None
    #: The last `SeamResidencyProof`. Read by the engine's boot banner and by the harness.
    seam_proof: Any = None
    #: How many times `verify_after_capture()` reached its BODY with the session enabled. 0 on every
    #: non-offloading serve, and 0 is also what a silently-disabled offload looks like — which is
    #: why the harness asserts on this rather than on the fact that the call site exists.
    verify_after_capture_fired: int = 0
    #: The `stream_tier.ExpertStreamTier` this window adopted layers into, or None. The THIRD tier:
    #: layers whose experts are re-read from the checkpoint per forward instead of living in VRAM or
    #: in the pinned arena. Held here because `install_stream_hooks()` must run after `seal()` — the
    #: hook is a forward-path mutation and the window's rule is that nothing about residency changes
    #: until the mapping window has closed.
    stream: Any = None

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
        stream: Any = None,
        stream_layers: "Sequence[int] | None" = None,
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

        `stream` / `stream_layers` hand this window the THIRD tier. The named layers are EXCLUDED
        from the plan — `resolve_weight_plan(layer_indices=...)` never sees them — so the arena is
        reserved, the capacity gate is asked, and the KV pool is sized for the layers that actually
        occupy a tier. That exclusion is the whole point: on the target checkpoint {device, pinned
        host} is short by ~15.5 GiB and `HostArenaCapacityError` refuses correctly; the stream tier
        is what removes bytes from the question rather than what waves it through. See
        `weights/stream_tier.py`.

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
                stream=stream,
                stream_layers=stream_layers,
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
            stream=stream if drv is not None else None,
        )
        if s.enabled:
            s.log(f"weight offload: {drv.describe()}")
        elif stream is not None:
            # A stream tier with no host tier means the plan came back empty AFTER the stream layers
            # were excluded — i.e. everything left fits VRAM. Saying so is not cosmetic: the caller
            # built a tier, and an inert session would never adopt a layer into it, so the boot would
            # try to keep all of them resident and OOM with no line explaining why.
            (log or _noop_log)(
                "weight offload: a stream tier was supplied but the plan is empty (the non-streamed "
                "layers fit the device budget). Nothing will be streamed."
            )
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

    def chunked_sink(self) -> Any:
        """The Stage-B sink for this window, or None when there is nothing to offload.

        Call BETWEEN `attach()` and `note_loaded()` — i.e. inside the load — and hand the result to
        `stage_b.ChunkedWeightLoader(sink=...)`. Returning None (disabled session) is not an error
        and the caller must handle it: a serve with no host tier still wants the chunked loader's
        raised LOAD ceiling, it just has nothing to bake, and `ChunkedWeightLoader` defaults to
        `DeviceLayerSink` in exactly that case.

        The phase is asserted rather than assumed because the sink closes over the arena allocator:
        asked before `attach()` there is no arena, and asked after `bind()` the seams it would create
        are ones no gate will ever look at.
        """
        if self.phase is not StageAPhase.ATTACHED:
            raise RuntimeError(
                f"weight offload: chunked_sink() must be called between attach() and note_loaded(); "
                f"the session is in {self.phase.name}."
            )
        if not self.enabled:
            return None
        factory = getattr(self.driver, "chunked_sink", None)
        if not callable(factory):
            raise RuntimeError(
                "weight offload: this Stage-A driver has no chunked_sink(), so a chunked load would "
                "leave every host-placed layer in VRAM while the plan claims it is host-resident."
            )
        return factory()

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
        self._model = model
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
        # THEN assert the SEAM — after freeze, so what is proved is the final, capture-ready state
        # the engine is about to serve, and against the live model rather than the binder's own
        # return value. Every gate above this line interrogates the arena or the plan; all of them
        # pass just as happily over a model whose MoE layers the forward never routes through the
        # seam. See `moe_interpose.prove_seam_residency`.
        self._prove_seam("seal")
        # THE BOOT HALF OF THE ENGAGED LEDGER. `moe_interpose.resolve` publishes the FORWARD half
        # (`weight_offload.moe_resolve[host|device]`); this publishes that a Stage-A window actually
        # ran to completion on this rank. Diffing the two legs of an A/B then shows the difference
        # between "offload was configured" and "offload sealed and the forward used it" — which are
        # the two things a serve-level bench cannot tell apart.
        _engaged("weight_offload.stage_a_sealed")
        _TORCH_SLACK = self.accounting.torch_slack_bytes
        self._sealed_arena_activity = self._arena_activity()
        rep = self.accounting.report()
        if not rep.ok:
            raise RuntimeError(
                "weight offload: the arena and the plan disagree, so the KV pool would be sized "
                "against a number that is not true.\n" + rep.render()
            )
        self.log(rep.render())

    def install_stream_hooks(self) -> None:
        """Arm the THIRD tier's per-forward gather. Call AFTER `seal()`, once, from the engine.

        After seal, and that ordering is the same rule the rest of this window follows: seal is the
        single moment residency stops changing, and a hook installed before it would be a forward-path
        mutation inside the window the gates are still measuring. The hook itself moves no weights and
        touches no arena page — it wraps `MoELayer.forward` so the layer's routed rows are read from
        the checkpoint immediately before its kernel runs.

        NOT CAPTURE-SAFE. The hook body does file I/O, allocates and syncs, all of which are illegal
        under HIP graph capture. `verify_after_capture()` would not catch it (it watches the ARENA,
        and this tier does not touch the arena), so it is stated here and in `stream_tier.py` rather
        than left to be discovered: serve with `--cuda-graph-max-bs 0` until a capturable design
        exists. This is the one undischarged merge requirement the third tier adds.
        """
        if self.phase is not StageAPhase.SEALED:
            raise RuntimeError(
                f"weight offload: install_stream_hooks() must be called after seal(); the session is "
                f"in {self.phase.name}."
            )
        if self.stream is None:
            return
        self.stream.install_hooks()
        self.log(f"weight offload: {self.stream.describe()}")
        _engaged("weight_offload.stream_tier")

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
        # COUNTED, not merely called. This gate has never fired in this repo's history: until the
        # engine booted with a non-empty plan, `enabled` was False on every serve and the body
        # returned immediately, so "we call verify_after_capture()" and "verify_after_capture() ran"
        # were different statements with no way to tell them apart from a log. The counter is what a
        # harness asserts on, and it counts only the calls that reached the body with the gate live.
        self._run_staged("post_capture", self._verify_after_capture_body)

    def _prove_seam(self, where: str) -> None:
        """Run the driver's seam proof, if it has one, and publish the result.

        `getattr` rather than a hard call because `StageADriver` is an injection point: the tests'
        doubles implement the byte-ledger half of the protocol and have no model walk. A driver that
        cannot prove its seam simply does not, and `seam_proof` stays None — which the engine's
        banner renders as "unproven" rather than as a pass.
        """
        fn = getattr(self.driver, "prove_seam", None)
        if not callable(fn) or self._model is None:
            return
        self.seam_proof = fn(self._model)
        self.log(f"weight offload [{where}]: {self.seam_proof.describe()}")

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
        self.verify_after_capture_fired += 1
        _engaged("weight_offload.verify_after_capture")
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
        # Re-prove the seam on the far side of capture. `assert_clean()` above answers "did anything
        # allocate from the arena during capture"; this answers the other half — "is the seam the
        # captured graphs were built over still the one the model holds". A container rebound
        # between seal() and here (a late post_load, a second bake, a reload path) leaves every
        # captured graph replaying against an address no seam describes, and `resolve()`'s identity
        # check would only report it from inside a replay, i.e. after the pointer is already baked
        # in. Cheap: it is a module walk plus two pointer-range tests per tensor, once per boot.
        self._prove_seam("post-capture")
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
        stream: Any = None,
        stream_layers: "Sequence[int] | None" = None,
    ) -> None:
        from .config import create_pinned_weight_arena, resolve_arena_settings

        self.stream = stream
        self.stream_layers = tuple(sorted(int(i) for i in (stream_layers or ())))
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
        #: The CPU-COMPUTE tier's executor, created lazily by `chunked_sink()`'s factory when the
        #: first CPU-placed layer arrives (its shapes are read off the live containers, not the
        #: plan). None on every serve with `--weight-offload-cpu-layers 0`, which is all of them
        #: unless it was asked for.
        self.cpu_worker = None
        self.rank = int(rank)
        self.local_ranks = int(local_ranks)
        #: Set by `chunked_sink()` when the caller loads via Stage B. Its presence is what makes
        #: `bind()` ADOPT rather than execute — see there.
        self.sink = None

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

    def chunked_sink(self) -> Any:
        """A `stage_b.LayerSink` that bakes each MoE layer AS THE CHUNKED LOAD FINALIZES IT.

        This is the Stage-B entry point into the Stage-A window, and it exists because the two paths
        disagree about *when* a layer is bakeable, not about *how*. The one-shot path can only bake
        after `post_load()` has finished the whole model, which requires the whole checkpoint to be
        resident at once — the thing the target checkpoint cannot do. The chunked path knows a layer
        is final the moment its own chunk closes, so it bakes there and the live set stays one chunk.

        `SeamLayerSink` calls `moe_interpose.bind_seam` — the same function `bind_plan` calls per
        layer — so the validate-before-copy order, the bitwise read-back, the weakref leak proof and
        the byte accounting are shared code and cannot drift between the two paths.

        Requires `attach_host_arena()` to have run: the allocator it hands the sink IS the arena. A
        sink built before attach would silently take `allocator=None` and `bind_seam(HOST, None)`
        would place the "host" tier in VRAM.
        """
        from .stage_b import SeamLayerSink

        if self.allocator is None:
            raise RuntimeError(
                "weight offload: chunked_sink() was asked for before attach_host_arena(), so there "
                "is no arena allocator to hand it. The host tier would be bound with a None "
                "allocator, which places it in VRAM under a host budget."
            )
        # `selftest=` is deliberately NOT passed: `settings.selftest` is the ARENA's per-chunk word
        # probe count, while `SeamLayerSink`'s is `moe_interpose.SELFTEST_SAMPLE`, the bake's
        # read-back row sample. They are different quantities with the same name, and crossing them
        # silently weakens the bake's own verification.
        self.sink = SeamLayerSink(
            self.plan,
            self.allocator,
            stream=self.stream,
            stream_layers=self.stream_layers,
            cpu_worker_factory=self._make_cpu_worker,
        )
        return self.sink

    def _make_cpu_worker(self, *, hidden: int, inter: int, top_k: int) -> Any:
        """Open the AVX-512 pool for this rank and start its dispatcher. Called ONCE, by the sink.

        The core list is PHYSICAL and node-wide disjoint across ranks — `cpu_native.
        default_core_list` carries the two measurements that force that (an SMT sibling of a busy
        core costs ~50%; only core 0 boosts, so it is left to the engine). The thread count is a
        REFUSAL, not a clamp: `cpu_tier.CoreBudget.assert_fits` raises when the node-wide total
        exceeds what a live TP=2 serve leaves free, because §1.5 measured the pool falling off a
        cliff (a fixed 6.0 ms/layer, ~12x its budget) rather than degrading in proportion when
        starved.
        """
        import os

        from .cpu_native import NativeVnniBackend, default_core_list
        from .cpu_tier import CORE_BUDGET
        from .cpu_worker import CpuMoEWorker

        threads = int(os.environ.get("MINISGL_CPU_MOE_THREADS", "2"))
        CORE_BUDGET.assert_fits(
            threads * self.local_ranks,
            what=f"the CPU MoE tier at {threads} thread(s)/rank x {self.local_ranks} rank(s)",
        )
        cores = default_core_list(self.rank, self.local_ranks, threads)
        backend = NativeVnniBackend(hidden, inter, top_k, threads, cores)
        self.cpu_worker = CpuMoEWorker(backend, name=f"cpu-moe-r{self.rank}").start()
        return self.cpu_worker

    def bind(self, model: Any) -> Any:
        """Attach a seam to every `MoELayer`, then execute the plan over them.

        `freeze=False`: the seams are frozen by `StageASession.seal()`, together with the arena and
        the process-wide `hipmem` latch, so there is exactly one moment at which the mapping window
        closes rather than three.

        WHEN THE LOAD WAS CHUNKED THIS ADOPTS INSTEAD OF EXECUTING. `SeamLayerSink` already attached
        and bound every layer's seam during the load, and re-running `attach_seams`/`bind_plan` over
        them would attach a SECOND seam to containers whose device originals have already been
        released — i.e. it would try to copy freed memory into arena rows that are already carved.
        The session's phase machine is unchanged: `bind()` is still the step that produces the
        outcome and the seam list every later gate reads, it just reads them off the sink."""
        from .moe_interpose import attach_seams, bind_plan

        if self.stream is not None and self.sink is None:
            raise RuntimeError(
                "weight offload: a stream tier was configured but the load was ONE-SHOT, so no layer "
                "was ever adopted into it. Every streamed layer is still a full 1.465 GiB expert "
                "stack in VRAM under a plan that does not bill it. The stream tier requires Stage B "
                "(`Engine._load_weight_chunked`), because adoption has to happen as each layer is "
                "finalized or the peak is the whole tier."
            )
        if self.sink is not None:
            self.seams = tuple(self.sink.seams)
            self.outcome = self.sink.outcome
            if not self.seams:
                raise RuntimeError(
                    "weight offload: the chunked load's sink bound no seams at all. Every MoE layer "
                    "the plan named was supposed to pass through it; an empty seam list means the "
                    "chunk enumeration finalized nothing and the arena holds no weights."
                )
            return self.outcome
        self.seams = attach_seams(model)
        self.outcome = bind_plan(self.seams, self.plan, self.allocator, freeze=False,
                                 cpu_worker=self.cpu_worker)
        return self.outcome

    def prove_seam(self, model: Any) -> Any:
        """ASSERT THE SEAM: the offload arm is in the path of the model the engine will serve.

        The arena is asked for `owns_pointer`, which is what turns "the seam says host" into "the
        tensor the kernel will read is inside a pinned chunk". Everything else `seal()` checks is a
        question about the arena or the plan and can pass over a model nobody runs — see
        `moe_interpose.prove_seam_residency` for the failure mode this repo has already paid for.
        """
        from .moe_interpose import prove_seam_residency

        return prove_seam_residency(
            model,
            self.seams,
            owns_pointer=None if self.arena is None else self.arena.owns_pointer,
        )

    def moved_bytes(self) -> int:
        """ARENA bytes, per the protocol contract — NOT `outcome.moved_bytes` verbatim.

        `BindOutcome.moved_bytes` totals every non-device placement, and since the CPU-COMPUTE tier
        that includes copies into pageable `torch.empty(device="cpu")` which never reach the arena,
        the torch pool, or `memory_allocated()`. All three consumers of this method are arena
        claims — `StageAAccounting.copied_bytes` (compared against `plan.host_resident_bytes`), the
        `model_memory_correction()` cap, and `TorchStackPool.assert_clean(expect_served_bytes=)` —
        so the CPU tier's bytes come off first or each one is wrong by the whole CPU tier, in the
        direction that over-corrects the model term and over-sizes the KV pool. Identical to
        `outcome.moved_bytes` on any plan with no CPU layers.
        """
        moved = int(getattr(self.outcome, "moved_bytes", 0) or 0)
        return moved - int(getattr(self.outcome, "cpu_resident_bytes", 0) or 0)

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
    stream: Any = None,
    stream_layers: "Sequence[int] | None" = None,
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

    # The STREAM tier's layers are removed from the question entirely. `layer_indices` is the
    # resolver's own restriction knob and it names what IS planned, so the complement is passed.
    # Derived from the model's own MoE walk, never from a layer count: the MTP draft head owns a
    # `MoELayer` too, and an index-arithmetic complement would silently plan or unplan it.
    layer_indices = None
    if stream_layers:
        from .plan import _path_layer_index
        from .moe_interpose import discover_moe_layers

        streamed = {int(i) for i in stream_layers}
        if model is None:
            raise ValueError(
                "weight offload: a stream tier was configured with no model to enumerate. The "
                "complement of the streamed layers cannot be derived from a layer count without "
                "guessing whether the MTP draft head is one of them."
            )
        all_idx = [_path_layer_index(p) for p, _ in discover_moe_layers(model)]
        unknown = streamed - {i for i in all_idx if i is not None}
        if unknown:
            raise ValueError(
                f"weight offload: stream layers {sorted(unknown)} are not MoE layers of this model "
                f"(it has {sorted(i for i in all_idx if i is not None)}). A stream tier pointed at "
                f"a layer that does not exist would leave the real one device-resident."
            )
        layer_indices = sorted(i for i in all_idx if i is not None and i not in streamed)

    resolution = resolve_weight_plan(
        config,
        device_budget_bytes=device_budget_bytes,
        layer_indices=layer_indices,
        arena_chunk_bytes=resolve_arena_settings().chunk_bytes,
        # THE CPU-COMPUTE TIER, opt-in from the engine config. `resolve_weight_plan` RAISES on a
        # refused request rather than silently downgrading to 0, which is what makes
        # `--weight-offload-cpu-layers` a statement about the served configuration instead of a
        # hint. Read by `getattr` for the same reason the two byte budgets are: a config object
        # from an older serve, or a test double, does not carry the field.
        num_cpu_layers=int(getattr(config, "weight_offload_cpu_layers", 0) or 0),
        # `cpu_repacked=False` is a FACT about this build, not a conservatism. The e4m3 layout
        # (0.9x the bytes) needs a bake-time re-read of the raw checkpoint scales; `cpu_native`
        # serves the fp16-folded scales the engine already holds, so the plan may not claim the
        # 10%. See `weights/cpu_native.py`.
        cpu_repacked=False,
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
        stream=stream,
        stream_layers=stream_layers,
    )
