"""HOT-PATH COST of the capture-safe route ring: OLD vs NEW `record()`, alternated in ONE process.

    # CPU only. No GPU, no lease, nothing touches /dev/kfd.
    git show 5e231871:python/minisgl/weights/route_trace.py > /tmp/baseline/route_trace.py
    docker run --rm --entrypoint bash \
      -v /home/pat/code/minisgl-rdna4-captsafe:/engine:ro -v /tmp/baseline:/baseline:ro \
      minisgl-gate:pytest -lc 'PYTHONPATH=/engine/python:/opt/kernels \
        python3 /engine/tests/route_trace_hot_path_bench.py --old /baseline/route_trace.py'

WHY THIS FILE EXISTS. `RouteTracer.record` is called once per MoE layer per forward -- 48x per
decode step on the shipped qwen4exp arm -- so a change to it is a change to the decode path's launch
overhead, and "a parity test proves values, not cost" (CLAUDE.md). The capture-safety fix rewrote
`record`'s ring path: the ring write became a STAGE write, the per-row stale-tail clear moved out to
a once-per-step `harvest()`, the per-layer `meta[slot][lid]` dict store went away, and a row-count
compare came in. Every one of those is a claim about cost that has to be measured, not asserted.

HOW IT IS MEASURED, AND WHY IT IS SHAPED LIKE THIS. This repo has a standing rule that a hot card
and separate processes fake 2-3% swings (`gpu-ab-needs-control-arm-and-long-windows`). The same is
true of a CPU under a shared box, so:

  * ONE PROCESS. Both modules are imported into the same interpreter -- the OLD one straight off a
    pre-fix checkout of the very same file -- and the same fixtures are used for both.
  * ARMS ALTERNATED, round-robin, with the arm ORDER ROTATED each round, so drift lands on every arm
    equally instead of on whichever ran last.
  * A CONTROL ARM. `ctl` is a second, independent instance of the NEW tracer running the identical
    body as `new`. Any `new` vs `old` delta smaller than the `new` vs `ctl` spread is noise, and the
    report says so in those words.
  * >= 200,000 record() calls per arm (48 layers x 500 steps x 10 rounds = 240,000).

WHAT A CPU FIXTURE CAN AND CANNOT SAY. It cannot price a HIP dispatch: on the GPU each slice-assign
in `record` is a launch (single-digit microseconds of host time), which dwarfs everything measured
here. So this file reports TWO numbers and they answer different questions:

  * the ATEN OP CENSUS (`--census`, and asserted as a budget in
    `tests/route_trace_hot_path_cost_test.py`) is the launch-count answer, and it is
    architecture-independent: N fewer ops per layer is N fewer launches per layer on the card.
  * the ns/call timing is the PYTHON+DISPATCH answer, and it is the part that is identical on any
    device, because it is interpreter work.

RESULT, 2026-09-23, minisgl-gate:pytest (torch 2.15.0.dev20260827+rocm7.2), CPU, 48 layers /
512 experts / top_k 10, 30 rounds x 500 steps = 720,000 record() calls per arm, three independent
runs agreeing to within 0.5% on the mins:

    old@1  (as shipped pre-fix)   4315.6 ns/call min    NOT the arm to beat: at ring_rows=1 only a
    new@2  (as shipped post-fix)  4214.3 ns/call min     ONE-row forward could take the ring at all
    old@2  (width-matched)        7348.5 ns/call min
    new@1  (width-matched)        3744.4 ns/call min
    ctl    (NEW again)            4228.4 ns/call min    control delta 0.33% on mins

    new@2 vs old@1   -2.35%   FASTER   (7x the control delta; repeatable across three runs)
    new@2 vs old@2  -42.65%   FASTER   (the removed per-row tail clear: 10 aten ops -> 5)
    harvest()           44 ns/step = +0.9 ns/call amortized over 48 = 0.00007% of a 60 ms step

    NO REGRESSION. The eager decode path is 2.35% cheaper per record() and the once-per-step harvest
    does not measurably exist.

Two things are stubbed, identically for both arms, and neither touches the compared work:
  * `torch.cuda.is_current_stream_capturing()` raises AcceleratorError with no ROCm device. Both
    arms call it exactly ONCE per `record`, so the stub cancels in the delta -- but it means the
    ABSOLUTE ns/call here excludes that one real C call. Stated, not hidden.
  * the OLD tracer allocates `ids_host` with `pin_memory=True` unconditionally, which raises with no
    GPU (the NEW one conditions it on the device). `pin_memory` is dropped for the OLD tracer's
    CONSTRUCTION only; `record` never touches `ids_host`.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import statistics
import sys
from time import perf_counter_ns

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

# Shipped qwen4exp shapes: 48 MoE layers, 512 experts, top_k 10, --max-running-requests 2.
LAYERS, EXPERTS, TOP_K = 48, 512, 10
CPU = torch.device("cpu")


def load_old(path: str):
    spec = importlib.util.spec_from_file_location("route_trace_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["route_trace_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


def build(mod, ring_rows: int, ring_steps: int = 16):
    """Construct a tracer with NO file and NO blockmap sync, on cpu."""
    real_empty = torch.empty

    def empty_nopin(*a, **kw):                  # the OLD module pins unconditionally; see docstring
        kw.pop("pin_memory", None)
        return real_empty(*a, **kw)

    torch.empty = empty_nopin
    try:
        kw = dict(model_slug="bench", num_layers=LAYERS, num_experts=EXPERTS, top_k=TOP_K,
                  tp_rank=0, dp_rank=0, expert_bytes=1, ring_steps=ring_steps,
                  ring_rows=ring_rows, drain_every=ring_steps, max_steps=1 << 40,
                  record_prefill=False, blockmap_checks=0, device=CPU)
        return mod.RouteTracer(None, **kw)
    finally:
        torch.empty = real_empty


def arm(mod, tracer, rows: int):
    """Return a closure running ONE decode step: the per-layer `record` calls, nothing else.

    `_CUR_CHUNK.clear()` is the step boundary's own work and is inside the timed body because
    without it `record` sees chunk>0 on the second step and falls to the HOST path -- i.e. it is
    load-bearing for measuring the RING path at all. It is one dict clear per 48 records, identical
    in both arms.
    """
    route = torch.randint(0, EXPERTS, (rows, TOP_K), dtype=torch.int32, device=CPU)
    chunk = mod._CUR_CHUNK
    rec = tracer.record
    try:
        tracer.begin_forward(False, 1, num_rows=rows)        # NEW signature
    except TypeError:
        tracer.begin_forward(False, 1)                       # OLD signature
    assert tracer.slot >= 0

    def step(_range=range(LAYERS)):
        chunk.clear()
        for lid in _range:
            mod._CUR_LID = lid
            rec(route, rows)

    # Prove the arm is exercising the RING path, not the host path (which would .tolist() per layer
    # and make the comparison meaningless). `oversize` is where a host-path record lands.
    step()
    if tracer.oversize:
        raise SystemExit(
            f"ABORT: this arm fell to the HOST path ({len(tracer.oversize)} oversize records). "
            f"rows={rows} ring_rows={tracer.ring_rows} -- the timing would compare a device write "
            f"against 48 blocking .tolist() calls."
        )
    return step


def census(mod, tracer, rows: int) -> dict:
    """Count the aten ops ONE `record()` issues on the ring path. This is the launch-count answer."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class C(TorchDispatchMode):
        def __init__(self):
            self.hist = {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            k = str(func)
            self.hist[k] = self.hist.get(k, 0) + 1
            return func(*args, **(kwargs or {}))

    route = torch.randint(0, EXPERTS, (rows, TOP_K), dtype=torch.int32, device=CPU)
    try:
        tracer.begin_forward(False, 1, num_rows=rows)
    except TypeError:
        tracer.begin_forward(False, 1)
    mod._CUR_CHUNK.clear()
    mod._CUR_LID = 0
    c = C()
    with c:
        tracer.record(route, rows)
    return c.hist


def price_harvest(tracer, iters: int) -> tuple:
    """`harvest()` in isolation: once per STEP, against 48 `record` calls.

    Idempotent via `_harvested`, so the flag has to be reset each iteration; the same reset is timed
    ALONE as the baseline and subtracted, so the reported number is the harvest's own work.
    """
    gc.disable()
    try:
        t0 = perf_counter_ns()
        for _ in range(iters):
            tracer._harvested = False
        base = perf_counter_ns() - t0
        t0 = perf_counter_ns()
        for _ in range(iters):
            tracer._harvested = False
            tracer.harvest()
        full = perf_counter_ns() - t0
    finally:
        gc.enable()
    return (full - base) / iters, base / iters


def run(arms: dict, rounds: int, steps: int) -> dict:
    """Round-robin the arms, rotating the order each round."""
    names = list(arms)
    per_round = {n: [] for n in names}
    gc.disable()
    try:
        for r in range(rounds):
            order = names[r % len(names):] + names[:r % len(names)]
            for n in order:
                step = arms[n]
                t0 = perf_counter_ns()
                for _ in range(steps):
                    step()
                dt = perf_counter_ns() - t0
                per_round[n].append(dt / (steps * LAYERS))
    finally:
        gc.enable()
    return per_round


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=os.environ.get("MINISGL_ROUTE_TRACE_BASELINE",
                                                    "/baseline/route_trace.py"),
                    help="pre-fix route_trace.py (git show <base>:python/minisgl/weights/route_trace.py)")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--census", action="store_true")
    a = ap.parse_args()

    from minisgl.weights import route_trace as NEW

    # Both arms pay this stub exactly once per record(); see the docstring.
    torch.cuda.is_current_stream_capturing = lambda: False

    if not os.path.exists(a.old):
        print(f"SKIP: no baseline module at {a.old!r}; pass --old <pre-fix route_trace.py>")
        return 2
    OLD = load_old(a.old)
    assert not hasattr(OLD.RouteTracer, "harvest"), (
        f"{a.old} already has harvest() -- that is the FIXED module, not the baseline"
    )
    assert hasattr(NEW.RouteTracer, "harvest"), "the repo module has no harvest(); wrong tree?"
    print(f"baseline: {a.old}\n"
          f"shapes:   layers={LAYERS} experts={EXPERTS} top_k={TOP_K}\n"
          f"calls:    {a.rounds} rounds x {a.steps} steps x {LAYERS} layers = "
          f"{a.rounds * a.steps * LAYERS} record() calls per arm\n")

    # --- ATEN OP CENSUS -------------------------------------------------------------------------
    print("== aten ops issued by ONE record() on the ring path (= launches per MoE layer) ==")
    for label, mod, rows_cfg, rows in (
        ("old  ring_rows=1  M=1   (AS SHIPPED pre-fix, non-spec)", OLD, 1, 1),
        ("new  ring_rows=2  M=1   (AS SHIPPED post-fix, mrr=2)", NEW, 2, 1),
        ("old  ring_rows=2  M=1   (width-matched: isolates the code change)", OLD, 2, 1),
        ("new  ring_rows=1  M=1   (width-matched)", NEW, 1, 1),
        ("old  ring_rows=2  M=2   (concurrent decode -- pre-fix this was ring_rows=1)", OLD, 2, 2),
        ("new  ring_rows=2  M=2", NEW, 2, 2),
    ):
        h = census(mod, build(mod, rows_cfg), rows)
        n = sum(h.values())
        short = ", ".join(f"{k.split('.')[-2]}.{k.split('.')[-1]}={v}" if k.count(".") >= 2
                          else f"{k}={v}" for k, v in sorted(h.items()))
        print(f"  {n:2d}  {label}\n      {short}")

    t = build(NEW, 2)
    t.begin_forward(False, 1, num_rows=1)
    NEW._CUR_CHUNK.clear()
    NEW._CUR_LID = 0
    t.record(torch.randint(0, EXPERTS, (1, TOP_K), dtype=torch.int32), 1)
    t._harvested = False
    from torch.utils._python_dispatch import TorchDispatchMode

    class C(TorchDispatchMode):
        def __init__(self):
            self.hist = {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.hist[str(func)] = self.hist.get(str(func), 0) + 1
            return func(*args, **(kwargs or {}))

    c = C()
    with c:
        t.harvest()
    print(f"  {sum(c.hist.values()):2d}  new  harvest()  ONCE PER STEP (not per layer)\n"
          f"      " + ", ".join(f"{k.split('.')[-2]}.{k.split('.')[-1]}={v}"
                                for k, v in sorted(c.hist.items())))
    print(f"\n  per 48-layer decode step, AS SHIPPED: "
          f"old {48 * sum(census(OLD, build(OLD, 1), 1).values())} ops, "
          f"new {48 * sum(census(NEW, build(NEW, 2), 1).values()) + sum(c.hist.values())} ops "
          f"(48 x record + 1 x harvest)")
    if a.census:
        return 0

    # --- TIMING ---------------------------------------------------------------------------------
    arms = {
        "old@1  (as shipped pre-fix)": arm(OLD, build(OLD, 1), 1),
        "new@2  (as shipped post-fix)": arm(NEW, build(NEW, 2), 1),
        "old@2  (width-matched)": arm(OLD, build(OLD, 2), 1),
        "new@1  (width-matched)": arm(NEW, build(NEW, 1), 1),
        "ctl    (NEW again = noise floor)": arm(NEW, build(NEW, 2), 1),
    }
    per_round = run(arms, a.rounds, a.steps)

    print("\n== record() ns/call, one process, arms alternated with the order rotated ==")
    print("   (MIN is the estimator to trust on a box shared with other agents -- interference only "
          "ever\n    ADDS time, so the minimum round is the least-contaminated sample. Median is "
          "shown beside it;\n    the two agree here, which is itself the check.)")
    med = {}
    for n, vals in per_round.items():
        med[n] = statistics.median(vals)
        print(f"  {n:34s} min {min(vals):7.1f}  median {med[n]:7.1f}  max {max(vals):7.1f}  "
              f"spread {(max(vals) - min(vals)) / med[n] * 100:5.1f}%")

    new_k = "new@2  (as shipped post-fix)"
    ctl_k = "ctl    (NEW again = noise floor)"
    ctl_delta = abs(med[new_k] - med[ctl_k]) / med[new_k] * 100
    worst_spread = max((max(v) - min(v)) / statistics.median(v) * 100 for v in per_round.values())
    # TWO noise estimates, and the BAND is the larger. The control-arm delta alone flatters the
    # result (two medians of the same code cancel drift almost perfectly); the per-arm round spread
    # is what actually bounds a single measurement on a box shared with other agents.
    noise = max(ctl_delta, worst_spread / 2)
    print(f"\n  CONTROL: two identical NEW arms differ by {ctl_delta:.2f}%; worst per-arm round "
          f"spread {worst_spread:.1f}%.\n  NOISE BAND = {noise:.2f}% -- a delta inside it is not a "
          f"result.")
    mins = {n: min(v) for n, v in per_round.items()}
    ctl_min = abs(mins[new_k] - mins[ctl_k]) / mins[new_k] * 100
    print(f"  on MINS, the two identical NEW arms differ by {ctl_min:.2f}%.")
    for base in ("old@1  (as shipped pre-fix)", "old@2  (width-matched)"):
        d = (med[new_k] - med[base]) / med[base] * 100
        dm = (mins[new_k] - mins[base]) / mins[base] * 100
        verdict = ("NOISE" if abs(dm) <= max(ctl_min, 1.0) else "FASTER" if dm < 0 else "SLOWER")
        print(f"  new@2 vs {base:30s} {dm:+6.2f}% on mins, {d:+6.2f}% on medians   {verdict}")

    # --- DEFECT (A): the same step with TWO requests -------------------------------------------
    # This is what `ring_rows = 1` actually cost. The pre-fix ring rejected M=2 (M <= ring_rows
    # fails), so every record on a 2-request decode step took the HOST path: `topk_ids.tolist()`
    # plus a Python set+sort, 48 times per step -- and on the real card that .tolist() is a BLOCKING
    # D2H, which a CPU fixture cannot price at all. So the number below is a FLOOR on the pre-fix
    # cost, not an estimate of it: it counts only the Python work and none of the sync.
    print("\n== concurrent decode, M=2 rows -- the shape defect (A) was about ==")
    old_host = build(OLD, 1)
    new_ring = build(NEW, 2)
    old_host.begin_forward(False, 1)
    new_ring.begin_forward(False, 1, num_rows=2)
    route2 = torch.randint(0, EXPERTS, (2, TOP_K), dtype=torch.int32, device=CPU)

    def host_step(_r=range(LAYERS)):
        OLD._CUR_CHUNK.clear()
        old_host.oversize.clear()      # the real code accumulates until the drain; this is a FLOOR
        for lid in _r:
            OLD._CUR_LID = lid
            old_host.record(route2, 2)

    def ring_step(_r=range(LAYERS)):
        NEW._CUR_CHUNK.clear()
        for lid in _r:
            NEW._CUR_LID = lid
            new_ring.record(route2, 2)

    host_step()
    ring_step()
    assert len(old_host.oversize) == LAYERS, "old@1 should have taken the HOST path for all 48"
    assert not new_ring.oversize, "new@2 should have taken the RING path"
    m2 = run({"old@1 M=2 (HOST path, .tolist() x48)": host_step,
              "new@2 M=2 (ring path)": ring_step}, a.rounds, a.steps)
    for n, vals in m2.items():
        print(f"  {n:38s} median {statistics.median(vals):8.1f} ns/call")
    hm, rm = (statistics.median(m2[k]) for k in m2)
    print(f"  READ THIS THE RIGHT WAY ROUND: on a CPU the pre-fix HOST path measures "
          f"{'CHEAPER' if hm < rm else 'dearer'} ({hm:.0f} vs {rm:.0f} ns/call) because `.tolist()` "
          f"on a cpu tensor is a memcpy, not a sync. On the card it is a BLOCKING D2H per MoE layer "
          f"-- 48 per step -- and this fixture cannot price that at all. What the fixture DOES\n"
          f"  settle is the part that is device-independent: the host-path record lands in "
          f"`oversize`, which drain() writes to the trace file and never forwards to the observer, "
          f"so pre-fix those 48 records reached the expert cache NOT AT ALL.")

    # --- THE DRAIN, which `ring_rows` also scales -----------------------------------------------
    # `drain()` does `ids_host.copy_(ids_ring)` over the WHOLE ring, not the filled part, and it is
    # `non_blocking=False` on the scheduler thread. The ring row is `top_k * ring_rows` wide, so
    # widening ring_rows from 1 to max_running_req multiplies that D2H by max_running_req. The BYTES
    # are device-independent and are the number that matters; the wall time below is a cpu->cpu
    # memcpy and is a FLOOR on the real PCIe transfer.
    print("\n== drain() D2H volume, which ring_rows multiplies ==")
    for label, rows, ring in (("pre-fix, non-spec (ring_rows=1)", 1, 1024),
                              ("post-fix, this arm (mrr=2)", 2, 1024),
                              ("post-fix, default mrr=256", 256, 133)):
        by = ring * LAYERS * TOP_K * rows * 4
        print(f"  {label:34s} {by / (1 << 20):8.2f} MiB per drain "
              f"(ring_steps={ring}, width={TOP_K * rows})")
    print("  (the 133 is what maybe_install's 64 MiB budget shrinks 1024 to -- but see "
          "route_trace_hot_path_cost_test.py::test_wide_ring_cannot_crash_the_boot_on_the_budget_path)")

    h_ns, reset_ns = price_harvest(build(NEW, 2), 200_000)
    print(f"\n== harvest(), priced alone, 200,000 iterations ==\n"
          f"  {h_ns:.0f} ns/step  (the `_harvested = False` reset alone is {reset_ns:.0f} ns and is "
          f"subtracted)\n"
          f"  amortized over 48 record() calls: {h_ns / LAYERS:+.1f} ns/call, i.e. "
          f"{h_ns / LAYERS / med[new_k] * 100:.1f}% of one record()")
    print(f"  against a ~60 ms decode step: {h_ns / 60e6 * 100:.5f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
