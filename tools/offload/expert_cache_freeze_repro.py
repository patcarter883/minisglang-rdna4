#!/usr/bin/env python3
"""CPU reproduction of the expert-residency-cache replacement FREEZE.

WHAT THIS IS. A deterministic, GPU-free driver for the REAL
`minisgl.weights.expert_cache.ExpertResidencyCache` — the class is imported and driven, never
reimplemented — under a faithful stub of the only things it needs a GPU for. It reproduces the
production signature measured on qwen4exp MXFP4 (2056 slots, LOW_WATER=25, TP=2):

    free=0, to_retract pinned at exactly _low_water, inflight pinned at exactly _low_water,
    throttled FROZEN, promotions == evictions advancing ~100x slower than deferred.

WHY IT REPRODUCES, i.e. what the harness gets right that a naive driver does not. The references
do NOT arrive one step at a time. `route_trace.maybe_install(observe_only=True)` sets
`drain_every = min(64, ring) = 64` (route_trace.py:577), and `RouteTracer.begin_forward` drains
only when `n_since_drain >= drain_every` (route_trace.py:307). So the scheduler hands the cache
**64 steps x 48 layers of route records in ONE burst**, inside ONE `_step_boundary`
(scheduler.py:3040-3045), and then nothing at all for the next 63 steps.

The manager thread absorbs that whole burst in milliseconds, while the only thing that can turn a
retracted slot back into a free one — `apply_pending()` — runs at the END of that single step.
`_free` therefore gets topped up ONCE per burst, by at most `_low_water` slots, and the other
~2500 admitted misses in the burst hit `free == 0` and count `deferred`. The knob named after the
free pool bounds installs per BURST, not per step:

    installs per 64 steps <= _low_water    =>    <= 25/64 = 0.39 per tick

This harness models the burst explicitly (`--drain-every`, default 64, matching the shipped
observe-only default) and shows the freeze vanish at `--drain-every 1`, which is the control arm.

WHAT IS STUBBED — the DEVICE ONLY. `_FakeTorch` proxies the real `torch` module and overrides
exactly: `torch.full`/`torch.empty` (map the fake cuda device to CPU) and `torch.cuda`
(Event/Stream/current_stream/stream/set_device/is_current_stream_capturing). Streams carry a
`ready_at` tick, events land `--copy-latency` ticks after the stream they are recorded on is free,
and `Event.query()` is a pure comparison against the harness tick counter. That is the real
two-phase handshake's timing, modelled in ticks. The cache's own logic — policy, free list,
retract queue, inflight reaper, admission filter, manager loop, threading — is untouched.

DETERMINISM WITH THE REAL MANAGER THREAD. The real manager thread is started (`cache.start()`),
because `_promote`'s `threaded` branch is the code under test. Between the observe burst and the
`apply_pending()` tick, the harness waits for the manager to have consumed every queued record
(`manager_batches` delta + empty `_q`). That is not a cheat: in production one decode step is
~50 ms and the manager drains 3072 records in ~ms, so "the burst is fully absorbed before the next
tick" is what the hardware does. It just makes it repeatable.

MEASURED, 2026-09-22, in minisgl-rdna4:lean, no GPU. `--drain-every 180 --steps 24000` against the
live production counters (qwen4exp MXFP4, 2056 slots, LOW_WATER=25, TP=2):

    counter            production        this harness
    free                       0                   0     CONSTANT in every sample
    to_retract                25                  25     CONSTANT, == _low_water
    inflight                  25                  25     CONSTANT, == _low_water
    policy                  2031                2031     == slots - _low_water
    resident / fill    2031 / 0.988        2031 / 0.988
    throttled             FROZEN              FROZEN
    promotions      +0.139 /tick        +0.139 /tick
    evictions       +0.139 /tick        +0.140 /tick     1:1 with promotions in both
    deferred          +40   /tick        +50.7  /tick
    observed_h            0.3256              0.3783     (still falling at 24k steps)

THE RATE LAW, confirmed at five points (`--drain-every` 32/64/128/180 + the control at 1):

    promotions per tick  ==  _low_water / (ticks between observe bursts)

    drain 32 -> 0.781   (25/32)      drain 128 -> 0.195  (25/128)
    drain 64 -> 0.391   (25/64)      drain 180 -> 0.139  (25/180)  == production

THE ONE NUMBER THIS HARNESS CANNOT PIN FROM THE CPU. Production's 0.139/tick implies ~180 ticks
between observe bursts; the shipped observe-only default is 64, which predicts 0.391/tick. The
rate law above is exact at every point tested, so the residual factor of 2.8 is the BURST INTERVAL
itself, not the mechanism. Settle it by reading the serve's own boot line -- `[route-trace] ARMED:
... drain=N` (route_trace.py:593) -- and re-running this with `--drain-every N`; a fixture dir in
MINISGL_MOE_ROUTE_TRACE alone moves that default from 64 to 512 (route_trace.py:577).

THE CONTROL ARM, which is what rules out the harness itself. `--drain-every 1` — identical code,
identical `_low_water`, identical policy, identical stubs, the burst removed and nothing else:

    deferred 0/tick (never once runs out of a free slot), free 17-25, promotions +3.28/tick,
    observed_h 0.8918 — above the SLRU oracle's 0.864.

So the freeze is not `_low_water`, not `_max_inflight`, not the policy and not the device stub.
It is that `_free` is replenished once per TICK while references arrive once per 64+ ticks.

Usage:
    PYTHONPATH=<repo>/python python3 tools/offload/expert_cache_freeze_repro.py
    PYTHONPATH=<repo>/python python3 tools/offload/expert_cache_freeze_repro.py --drain-every 1
    PYTHONPATH=<repo>/python python3 tools/offload/expert_cache_freeze_repro.py \
        --drain-every 180 --steps 24000

FIXED 2026-09-22 (d2c3ac43, `expert_cache._service`) — THIS FILE IS NOW A REGRESSION HARNESS, not
an open diagnosis. Against the fixed cache its own verdict block reports the rate law broken:
promotions 3.103/tick against the freeze's `_low_water/drain_every` = 0.391/tick.

SEE ALSO `expert_cache_replay.py`, which answers the questions this one cannot and vice versa.
That one replays the CAPTURED 22,001-step route fixture instead of a generated stream, scores the
hit rate off the DEVICE-VISIBLE `slot_of` tables rather than the policy's `observed_h` (the gap
between those two is where the silent wrong-bytes bug was found: 13.05% of expert reads), and
needs no torch at all, so it runs on the bare host in 20 s. This one needs no fixture and can
sweep the reference stream's own properties, which a replay of one capture cannot.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import os
import random
import sys
import threading
import time

import torch

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------------------------
# The DEVICE stub. Nothing below touches the cache's own logic.
# ---------------------------------------------------------------------------------------------
class FakeDevice:
    """A torch.device look-alike whose `.type` is "cuda" so the cache takes its production paths.

    `apply_pending()` returns early on a non-cuda device and `start()` refuses to spawn the
    manager, so a plain CPU device would silently test a different program.
    """

    type = "cuda"
    index = 0

    def __repr__(self) -> str:                      # pragma: no cover - debug only
        return "device(type='cuda', index=0)"


class Clock:
    """The harness tick counter. Advanced once per scheduler step, read by every fake event."""

    __slots__ = ("t",)

    def __init__(self) -> None:
        self.t = 0


class FakeStream:
    """A stream is a DEPENDENCY BARRIER: the earliest tick newly-enqueued work may start.

    `barrier` moves only on a real cross-stream dependency (`wait_event`/`wait_stream`), NOT per
    enqueued copy. A promotion is a 1.36 MiB DMA — microseconds against a ~50 ms step — so many
    copies share one tick, which is what the production counters show (inflight 25, throttled
    frozen, i.e. nothing is queueing up behind anything). Charging each copy a whole tick would
    invent a serialisation the hardware does not have.
    """

    def __init__(self, clock: Clock, *, is_compute: bool = False) -> None:
        self.clock = clock
        self.is_compute = is_compute
        self._barrier = 0

    @property
    def barrier(self) -> int:
        return max(self._barrier, self.clock.t)

    @barrier.setter
    def barrier(self, v: int) -> None:
        self._barrier = v

    @property
    def queued_work_done_at(self) -> int:
        """When work already on this stream has completed.

        For the COMPUTE stream that is the end of the current step: it carries this step's model
        launches, which is exactly what a retraction event must be ordered after.
        """
        return self.clock.t + 1 if self.is_compute else self.barrier

    def wait_event(self, ev: "FakeEvent") -> None:
        self.barrier = max(self.barrier, ev.land_at)

    def wait_stream(self, other: "FakeStream") -> None:
        self.barrier = max(self.barrier, other.queued_work_done_at)

    def synchronize(self) -> None:                  # pragma: no cover - unused by the cache
        pass


class FakeEvent:
    """Completes when the work queued ahead of it on its stream does. `query()` is exact.

    On the COPY stream that is `barrier + copy_latency` (the publish latency the two-phase
    handshake exists to hide); on the COMPUTE stream it is the end of the current step, which is
    when the retraction of `slot_of[v] = -1` has become visible to every launch that could still
    have been reading the old expert.
    """

    def __init__(self, clock: Clock, latency: int) -> None:
        self.clock = clock
        self.latency = latency
        self.land_at = 1 << 60                      # never, until recorded

    def record(self, stream: "FakeStream" = None) -> None:
        if stream is None:
            self.land_at = self.clock.t + self.latency
            return
        if stream.is_compute:
            self.land_at = stream.queued_work_done_at
        else:
            self.land_at = stream.barrier + self.latency

    def query(self) -> bool:
        return self.clock.t >= self.land_at

    def synchronize(self) -> None:                  # pragma: no cover - the cache never calls this
        raise AssertionError("apply_pending must never synchronize()")


class _FakeCuda:
    def __init__(self, clock: Clock, copy_latency: int) -> None:
        self._clock = clock
        self._copy_latency = copy_latency
        self._compute = FakeStream(clock, is_compute=True)

    def Event(self, *a, **kw) -> FakeEvent:
        return FakeEvent(self._clock, self._copy_latency)

    def Stream(self, *a, **kw) -> FakeStream:
        return FakeStream(self._clock)

    def current_stream(self, device=None) -> FakeStream:
        return self._compute

    def set_device(self, device) -> None:
        pass

    def is_current_stream_capturing(self) -> bool:
        return False

    @contextlib.contextmanager
    def stream(self, s):
        yield


class _FakeTorch:
    """Proxies the real torch; overrides allocation device mapping and the cuda namespace only."""

    def __init__(self, real, cuda: _FakeCuda) -> None:
        self._real = real
        self.cuda = cuda

    def __getattr__(self, name):
        return getattr(self._real, name)

    @staticmethod
    def _fix(kw):
        if isinstance(kw.get("device"), FakeDevice):
            kw = dict(kw, device=CPU)
        return kw

    def full(self, *a, **kw):
        return self._real.full(*a, **self._fix(kw))

    def empty(self, *a, **kw):
        return self._real.empty(*a, **self._fix(kw))

    def zeros(self, *a, **kw):
        return self._real.zeros(*a, **self._fix(kw))


# ---------------------------------------------------------------------------------------------
# The reference stream: near-uniform marginals, strong autocorrelation.
# ---------------------------------------------------------------------------------------------
class RouteGen:
    """Per-layer sliding hot set.

    Near-uniform MARGINALS: every expert enters a hot set uniformly at random, so over a long run
    each expert is referenced about equally often. Strong AUTOCORRELATION: `p_hot` of references
    come from the layer's current hot set, which turns over slowly. Those are the two properties
    the decision doc measured (Gini 0.61 / near-flat marginals, but
    frac(stack distance <= cache size) = 0.8404) and they are what this harness has to carry.
    """

    def __init__(self, *, layers: int, experts: int, top_k: int, hot: int, p_hot: float,
                 p_promote: float, rng: random.Random) -> None:
        self.layers, self.experts, self.top_k = layers, experts, top_k
        self.p_hot, self.p_promote, self.rng = p_hot, p_promote, rng
        self.hot = [rng.sample(range(experts), hot) for _ in range(layers)]
        self.ptr = [0] * layers
        self.hot_n = hot

    def route(self, lid: int):
        rng, ids = self.rng, set()
        hot = self.hot[lid]
        guard = 0
        while len(ids) < self.top_k and guard < 400:
            guard += 1
            if rng.random() < self.p_hot:
                ids.add(hot[rng.randrange(self.hot_n)])
            else:
                # A cold reference. Most are ONE-TOUCH (a passing route); only `p_promote` of them
                # join the hot set. Decoupling the miss rate from the TURNOVER rate is what makes
                # the hot set "stable over hundreds of steps", which is the property the decision
                # doc measured (per-layer stack distance P50 29.9 / P90 138.4) and the property a
                # frozen cache's residual hit rate is entirely determined by.
                e = rng.randrange(self.experts)
                ids.add(e)
                if rng.random() < self.p_promote:
                    hot[self.ptr[lid] % self.hot_n] = e
                    self.ptr[lid] += 1
        return sorted(ids)


def stack_distance_frac(stream, cap: int) -> float:
    """frac(stack distance <= cap): exactly the LRU hit condition, measured on the real stream."""
    lru: "collections.OrderedDict[int, None]" = collections.OrderedDict()
    hits = n = 0
    for key in stream:
        n += 1
        if key in lru:
            hits += 1
            lru.move_to_end(key)
        else:
            lru[key] = None
            if len(lru) > cap:
                lru.popitem(last=False)
    return hits / max(1, n)


# ---------------------------------------------------------------------------------------------
def build_cache(ec, args, clock):
    dev = FakeDevice()
    cache = ec.ExpertResidencyCache(
        num_experts=args.experts,
        expert_bytes=1,                 # slots = budget_bytes // expert_bytes, decoupled from the
        budget_bytes=args.slots,        # toy tensor shapes below so the run stays small on CPU
        device=dev,
    )
    row = args.row_bytes
    for lid in range(args.layers):
        def plane():
            w = torch.arange(args.experts * row, dtype=torch.uint8).reshape(args.experts, row)
            s = torch.ones((args.experts, 2), dtype=torch.float32)
            return (w, s, None)
        cache.register_layer(lid, plane(), plane())
    return cache


def wait_quiescent(cache, expect_batches: int, deadline: float = 20.0) -> None:
    """Block until the manager has consumed every queued record. See the module docstring."""
    t0 = time.time()
    while time.time() - t0 < deadline:
        with cache._lock:
            qlen = len(cache._q)
        if qlen == 0 and cache.stats["manager_batches"] >= expect_batches:
            # one more short settle so the post-item _reclaim for the last record has run
            time.sleep(0.002)
            with cache._lock:
                if len(cache._q) == 0:
                    return
        time.sleep(0.0005)
    raise SystemExit(
        f"manager did not drain: q={len(cache._q)} batches={cache.stats['manager_batches']} "
        f"expected>={expect_batches} thread_alive={cache._thread and cache._thread.is_alive()}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=6000, help="scheduler steps (= apply_pending ticks)")
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--experts", type=int, default=512)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--slots", type=int, default=2056)
    p.add_argument("--low-water", type=int, default=25)
    p.add_argument("--max-inflight", type=int, default=512,
                   help="serve.sh exports 512; the 64 default is NOT what production runs")
    p.add_argument("--drain-every", type=int, default=64,
                   help="route_trace observe-only default; 1 = the control arm (per-step feed)")
    p.add_argument("--route-every", type=int, default=3,
                   help="ticks per route-recording step; >1 models spec draft steps that tick the "
                        "cache but produce no route record. 3 reproduces the measured 157 refs/tick")
    p.add_argument("--copy-latency", type=int, default=1, help="ticks until an event completes")
    p.add_argument("--hot", type=int, default=36, help="per-layer hot-set size")
    p.add_argument("--p-hot", type=float, default=0.90)
    p.add_argument("--p-promote", type=float, default=0.02,
                   help="fraction of COLD references that join the hot set (turnover rate)")
    p.add_argument("--row-bytes", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--report-every", type=int, default=200,
                   help="manager batches between summaries (production default). The manager's own "
                        "summary line is the SAME sampling point production was read at: mid-burst, "
                        "immediately after _reclaim")
    p.add_argument("--sample-at", type=int, nargs=2, default=None,
                   help="two tick numbers to print counters at (default: 40%% and 100%% of steps)")
    args = p.parse_args()

    # Env knobs are read in __init__, so they must be set before the class is imported/constructed.
    os.environ["MINISGL_EXPERT_CACHE_LOW_WATER"] = str(args.low_water)
    os.environ["MINISGL_EXPERT_CACHE_MAX_INFLIGHT"] = str(args.max_inflight)
    os.environ["MINISGL_EXPERT_CACHE_REPORT"] = str(args.report_every)

    clock = Clock()
    fake_cuda = _FakeCuda(clock, args.copy_latency)

    from minisgl.weights import expert_cache as ec
    ec.torch = _FakeTorch(torch, fake_cuda)         # DEVICE stub only; ec's own logic untouched

    # CAPTURE THE MANAGER'S OWN SUMMARY LINE, at the exact point production was sampled at: from
    # the manager thread, mid-burst, immediately after `_reclaim`. Shadowing the module's `print`
    # only redirects its output — it changes nothing the cache does.
    mgr_lines = []

    def _capture(*a, **kw):
        s = " ".join(str(x) for x in a)
        if s.startswith("[expert-cache] slots="):
            mgr_lines.append((clock.t, s))
        else:
            print(*a, **kw)

    ec.print = _capture

    rng = random.Random(args.seed)
    gen = RouteGen(layers=args.layers, experts=args.experts, top_k=args.top_k,
                   hot=args.hot, p_hot=args.p_hot, p_promote=args.p_promote, rng=rng)

    cache = build_cache(ec, args, clock)
    print(f"[repro] slots={cache.slots} low_water={cache._low_water} "
          f"max_inflight={cache._max_inflight} refill_batch={cache._refill_batch} "
          f"candidate_cap={cache._candidate_cap} admit_second={cache._admit_second_ref} "
          f"inflight_max_age={cache._inflight_max_age} q_maxlen={cache._q.maxlen}", flush=True)
    print(f"[repro] drain_every={args.drain_every} route_every={args.route_every} "
          f"copy_latency={args.copy_latency} steps={args.steps}", flush=True)

    cache.start()
    if cache._thread is None:
        raise SystemExit("manager thread did not start — the device stub is wrong")

    sample_at = args.sample_at or [int(args.steps * 0.4), args.steps]
    samples = []
    all_keys = []                                   # for the stack-distance property check
    pending = []                                    # records the tracer is holding in its ring

    for step in range(1, args.steps + 1):
        clock.t = step
        # --- _step_boundary: route_trace.begin_forward() drains FIRST, then tick_expert_cache() ---
        if pending and (step % args.drain_every == 0):
            expect = cache.stats["manager_batches"] + len(pending)
            for lid, ids in pending:
                cache.observe(lid, ids)
            pending = []
            wait_quiescent(cache, expect)
        cache.apply_pending()
        # --- the forward itself: this step's routes land in the tracer's ring, undrained ---
        if step % args.route_every == 0:
            for lid in range(args.layers):
                ids = gen.route(lid)
                pending.append((lid, ids))
                base = lid * args.experts
                all_keys.extend(base + e for e in ids)
        if step in sample_at:
            samples.append((step, cache.summary()))

    cache.stop()

    def parse(s):
        out = {}
        for tok in s.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
        return out

    def deltas(label, a_t, a, b_t, b):
        dt = b_t - a_t
        print(f"\n[repro] {label}: per-tick deltas over {dt} ticks")
        for k in ("refs", "deferred", "promotions", "evictions", "throttled", "admit_deferred",
                  "abandoned", "dropped_refs", "stale_pub"):
            if k in a:
                print(f"  {k:<15} {(b[k] - a[k]) / dt:+9.3f}/tick   ({a[k]:.0f} -> {b[k]:.0f})")
        for k in ("free", "to_retract", "inflight", "policy", "resident"):
            if k in a:
                tag = "   (CONSTANT)" if a[k] == b[k] else ""
                print(f"  {k:<15} {a[k]:.0f} -> {b[k]:.0f}{tag}")
        dp, dd = b["promotions"] - a["promotions"], b["deferred"] - a["deferred"]
        print(f"  deferred/promotions ratio = {dd / max(1.0, dp):.1f}x")

    # ---- (1) the MANAGER-THREAD sample: the exact phase production was read at -----------------
    warm = [(t, s) for t, s in mgr_lines if t >= args.steps * 0.4]
    if len(warm) >= 2:
        print("\n=== MANAGER-THREAD SAMPLES (same print site as the production counters) ===")
        (t0, s0), (t1, s1) = warm[0], warm[-1]
        print(f"[tick {t0}] {s0}")
        print(f"[tick {t1}] {s1}")
        deltas("manager-thread sample", t0, parse(s0), t1, parse(s1))
    else:
        print(f"\n[repro] only {len(mgr_lines)} manager summary lines captured", flush=True)

    # ---- (2) the TICK-BOUNDARY sample: same counters, opposite phase of the handshake ---------
    if len(samples) == 2:
        print("\n=== TICK-BOUNDARY SAMPLES (after apply_pending, manager idle) ===")
        for step, s in samples:
            print(f"[tick {step}] {s}")
        (t0, s0), (t1, s1) = samples
        deltas("tick-boundary sample", t0, parse(s0), t1, parse(s1))

    sd = stack_distance_frac(all_keys, args.slots)
    print(f"\n[repro] reference stream: {len(all_keys)} refs, "
          f"frac(stack distance <= {args.slots}) = {sd:.4f} "
          f"(production trace measured 0.8404); distinct keys touched = {len(set(all_keys))} "
          f"of {args.layers * args.experts}")

    # ---- the verdict: does this arm carry the production signature? ---------------------------
    if warm:
        f = parse(warm[-1][1])
        checks = [
            ("free == 0", f["free"] == 0),
            (f"to_retract == low_water ({cache._low_water})", f["to_retract"] == cache._low_water),
            (f"inflight == low_water ({cache._low_water})", f["inflight"] == cache._low_water),
            (f"policy == slots - low_water ({cache.slots - cache._low_water})",
             f["policy"] == cache.slots - cache._low_water),
            ("throttled frozen", parse(warm[0][1])["throttled"] == f["throttled"]),
            ("promotions == evictions", abs((f["promotions"] - parse(warm[0][1])["promotions"])
                                            - (f["evictions"] - parse(warm[0][1])["evictions"]))
             <= max(2.0, 0.05 * abs(f["promotions"] - parse(warm[0][1])["promotions"]))),
        ]
        dt = warm[-1][0] - warm[0][0]
        dp = (f["promotions"] - parse(warm[0][1])["promotions"]) / max(1, dt)
        dd = (f["deferred"] - parse(warm[0][1])["deferred"]) / max(1, dt)
        checks.append(("deferred >= 25x promotions", dd >= 25 * dp))
        print("\n=== VERDICT: production freeze signature ===")
        for name, ok in checks:
            print(f"  [{'MATCH' if ok else ' -- '}] {name}")
        print(f"  rate law: promotions {dp:.3f}/tick vs _low_water/drain_every = "
              f"{cache._low_water / args.drain_every:.3f}/tick")
        print("  FREEZE REPRODUCED" if all(ok for _, ok in checks)
              else "  not the full production signature in this arm "
                   "(expected when the burst is removed, or when the run is too short for the "
                   "hit rate to decay)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
