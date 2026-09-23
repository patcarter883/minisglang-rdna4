"""GATE for defect (A): `ring_rows` must cover the WIDEST decode forward, or every concurrent
decode step is invisible to the expert cache.

    # CPU only. No GPU, no lease, nothing touches /dev/kfd.
    docker run --rm --entrypoint bash \
      -v /home/pat/code/minisgl-rdna4-captsafe:/engine:ro \
      -v /home/pat/code/minisgl-rdna4:/baseline:ro \
      minisgl-rdna4:lean -lc 'PYTHONPATH=/engine/python:/opt/kernels \
        python3 /engine/tests/route_trace_ring_rows_gate_test.py'

    MINISGL_GATE_FALSIFY=1 ...same...        # MUST fail: runs the gate against the UNFIXED value

WHAT IS BEING GATED. `RouteTracer.record` takes its sync-free device-RING path only when
`M <= self.ring_rows` (route_trace.py). Anything wider falls to the HOST path, whose records land in
`self.oversize`, and `drain()` packs `oversize` into the trace FILE but never forwards it to the
observer. The expert cache IS that observer (`engine.py` -> `_tracer.set_observer(_cache.observe)`)
and the ring is its only input. So `ring_rows` is not a fixture-sizing knob: it is the width of the
cache's eyes.

Before this change `engine.py` computed

    ring_rows = max_running_req * (1 + spec_num_draft) if spec_config is not None else 1
                                                                                    ^^^^^

i.e. **1** on every non-spec serve. The shipped qwen4exp arm is exactly that config
(`tools/serve.sh:1126` `: "${GRAPH_BS:=0}"`, `serve.sh:858` caps `CONC=2`, arm `spec_default=none`),
so `max_running_req == 2` and `ring_rows == 1`: every 2-request decode step went to the host path and
the cache saw NOTHING of it. Only single-request steps ever reached the policy.

HOW THIS GATE AVOIDS BEING NEW-vs-ITSELF. Three arms over one byte-identical workload:
  * `new`      — the real `engine._route_trace_ring_rows(config)` value, real `RouteTracer`;
  * `old-value` — the real `RouteTracer`, ring_rows = the value the OLD engine expression yields.
                  The old expression is not paraphrased: its source text is pinned in
                  `BASELINE_RING_ROWS_SRC` below, asserted byte-present in the baseline tree's
                  `engine.py`, and `eval`'d;
  * `old-code`  — the ACTUAL pre-fix `weights/route_trace.py`, loaded by path out of a read-only
                  baseline tree whose sha256 is pinned, with the old ring_rows value. This is the arm
                  that makes the comparison old-CODE rather than old-semantics-as-I-remember-them.
`old-value` and `old-code` must agree exactly; if they diverge the gate fails and says so, because
then the parametrised arm is not a faithful stand-in and no conclusion from it is worth anything.

TWO HOST PATCHES, both off the path under test, both stated so a reader can price them:
  1. `torch.cuda.is_current_stream_capturing()` raises `AcceleratorError` on a GPU-less host, so it
     is replaced by a flag-reader. Both arms get the same patch and every arm here runs it as False
     (this gate is about the EAGER dispatch decision; capture is gated elsewhere).
  2. `torch.empty(..., pin_memory=True)` raises with no GPU, and the pre-fix `__init__` pins
     unconditionally. `torch.empty` is wrapped to drop `pin_memory` for the duration of the OLD
     tracer's construction only. Pinning affects the speed of one D2H, never which path a record
     takes.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from minisgl.engine.config import EngineConfig                       # noqa: E402
from minisgl.engine.engine import _route_trace_ring_rows             # noqa: E402
from minisgl.engine.graph import _determine_cuda_graph_bs            # noqa: E402
from minisgl.distributed.info import DistributedInfo                 # noqa: E402
from minisgl.weights import route_trace as RT                        # noqa: E402

DEV = torch.device("cpu")
FALSIFY = os.environ.get("MINISGL_GATE_FALSIFY", "") not in ("", "0")
BASELINE_TREE = os.environ.get("MINISGL_BASELINE_TREE", "/baseline")

# --- pinned provenance of the pre-fix code ----------------------------------------------------
# `git show 5e231871:<path> | sha256sum` on the merge-base this branch was cut from. A baseline tree
# whose bytes differ is REFUSED rather than used: the shared worktree is mutable and another agent
# editing it mid-run would otherwise silently become "the old code".
BASELINE_SHA = {
    "python/minisgl/weights/route_trace.py":
        "4e787bdba5e56fd8d30e1823bd0dc6a39682fd388c90e78421ee05ce8a92d6fd",
    "python/minisgl/engine/engine.py":
        "f02e2285e14db7a1cfc73bd47e23de24e03f0131fecf2ecc2c426e8d41fb3ca5",
}
# The pre-fix ring_rows expression, VERBATIM from engine.py:401-404 @ 5e231871. Asserted present in
# the baseline file before it is trusted (see check_provenance).
BASELINE_RING_ROWS_SRC = (
    "                config.max_running_req * (1 + config.spec_num_draft)\n"
    "                if config.spec_config is not None else 1"
)

FAILS: list[str] = []
GAPS: list[str] = []
NOTES: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def gap(name: str, closed: bool, detail: str = "") -> None:
    """A defect the fix leaves OPEN. Separate from FAILS so the exit code distinguishes "defect (A)
    is not fixed" (1) from "defect (A) is fixed for the configs gated here, and here is what is
    still open" (2). Never silent: a gate that reports a real hole and exits 0 is the failure mode
    this whole file exists to catch."""
    print(f"  {'OK  ' if closed else 'GAP '}  {name}{('  ' + detail) if detail else ''}")
    if not closed:
        GAPS.append(name)


def note(msg: str) -> None:
    NOTES.append(msg)
    print(f"  note  {msg}")


# --- host patches (see module docstring) -------------------------------------------------------
_CAPTURING = False
torch.cuda.is_current_stream_capturing = lambda: _CAPTURING       # type: ignore[assignment]

_real_empty = torch.empty


def _empty_nopin(*a, **kw):
    kw.pop("pin_memory", None)
    return _real_empty(*a, **kw)


# --- configs -----------------------------------------------------------------------------------
def cfg(*, mrr: int, graph: "int | None", spec: str = "none", k: int = 4) -> EngineConfig:
    return EngineConfig(
        model_path="gate", tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        max_running_req=mrr, cuda_graph_max_bs=graph, spec_algorithm=spec, spec_num_draft=k,
    )


def baseline_ring_rows(config: EngineConfig) -> int:
    """The pre-fix value, by EXECUTING the pinned old source text (not a paraphrase of it)."""
    return int(eval(compile("(\n" + BASELINE_RING_ROWS_SRC + "\n)", "<engine.py@5e231871>", "eval"),
                    {}, {"config": config}))


def derive_ring_rows(config: EngineConfig) -> int:
    """The value under test. MINISGL_GATE_FALSIFY swaps in the UNFIXED derivation; every assertion
    in this file is written against this function, so the falsify run exercises the identical gate
    against the pre-fix code and must fail."""
    return baseline_ring_rows(config) if FALSIFY else _route_trace_ring_rows(config)


# --- provenance --------------------------------------------------------------------------------
def check_provenance() -> "object | None":
    """Returns the pre-fix route_trace MODULE if a trustworthy baseline tree is mounted, else None."""
    print("\n[1] PROVENANCE of the pre-fix code")
    if not os.path.isdir(BASELINE_TREE):
        report("baseline tree mounted", False,
               f"{BASELINE_TREE!r} absent -> the old-CODE arm cannot run. Mount it read-only "
               f"(-v /home/pat/code/minisgl-rdna4:/baseline:ro) or the A/B is value-only.")
        return None
    ok_all = True
    for rel, want in BASELINE_SHA.items():
        p = os.path.join(BASELINE_TREE, rel)
        if not os.path.isfile(p):
            report(f"baseline {rel}", False, "missing")
            ok_all = False
            continue
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        report(f"baseline {rel} sha256", got == want, f"{got[:16]} (want {want[:16]})")
        ok_all &= got == want
    if not ok_all:
        note("baseline tree bytes do not match 5e231871 -> old-CODE arm DISABLED (refusing to "
             "measure against an unknown tree)")
        return None
    eng = open(os.path.join(BASELINE_TREE, "python/minisgl/engine/engine.py")).read()
    report("pinned old ring_rows expression is byte-present in baseline engine.py",
           BASELINE_RING_ROWS_SRC in eng)
    # `M <= ring_rows` and the observer-less oversize loop are the two lines that make ring_rows
    # load-bearing. Assert they are the OLD file's text too, so "ring_rows is the only difference"
    # is checked, not assumed.
    rt_src = open(os.path.join(BASELINE_TREE, "python/minisgl/weights/route_trace.py")).read()
    report("old record() dispatches on `M <= self.ring_rows`",
           "and M <= self.ring_rows:" in rt_src)
    report("old drain() oversize loop never calls the observer",
           re.search(r"for \(step, uid, lid, kind, chunk, ntok, ids\) in self\.oversize:"
                     r"(?:(?!_observer).)*?self\.oversize\.clear\(\)", rt_src, re.S) is not None)
    spec = importlib.util.spec_from_file_location(
        "route_trace_baseline_5e231871",
        os.path.join(BASELINE_TREE, "python/minisgl/weights/route_trace.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report("pre-fix route_trace module imported from baseline tree", True, mod.__file__)
    return mod


# --- the derivation ----------------------------------------------------------------------------
def gate_derivation() -> None:
    print("\n[2] DERIVATION: does ring_rows cover max_running_req AND cuda_graph_max_bs?")
    print(f"  {'config':<44} {'old':>6} {'new':>6} {'floor':>6} {'widest capt. bucket':>21}")
    matrix = [
        # (label, mrr, cuda_graph_max_bs, spec, k)
        ("SHIPPED qwen4exp: mrr=2 graph=0 spec=none", 2, 0, "none", 4),
        ("mrr=2  graph=2  spec=none", 2, 2, "none", 4),
        ("mrr=4  graph=auto(None) spec=none", 4, None, "none", 4),
        ("mrr=1  graph=0  spec=none", 1, 0, "none", 4),
        ("mrr=8  graph=32 spec=none", 8, 32, "none", 4),
        ("mrr=256 graph=auto(None) spec=none", 256, None, "none", 4),
        ("mrr=2  graph=0  spec=mtp k=4", 2, 0, "mtp", 4),
        ("mrr=8  graph=32 spec=mtp k=3", 8, 32, "mtp", 3),
    ]
    for label, mrr, graph, spec, k in matrix:
        c = cfg(mrr=mrr, graph=graph, spec=spec, k=k)
        old = baseline_ring_rows(c)
        new = derive_ring_rows(c)
        # The widest forward the engine can produce, from the engine's OWN bucket list:
        #  - eager decode: one row per admitted request                 -> mrr
        #  - captured decode: the padded BUCKET width                   -> max(_determine_cuda_graph_bs)
        #  - spec verify: (K+1) token-rows per request
        buckets = _determine_cuda_graph_bs(c.cuda_graph_bs, c.cuda_graph_max_bs,
                                           free_memory=8 << 30, max_running_req=c.max_running_req)
        widest_bucket = max(buckets) if buckets else 0
        # FLOOR, not an equality: the two row counts that must fit are the eager decode/verify
        # (`mrr`, times K+1 under spec) and the widest padded capture bucket. Asserting an exact
        # value would just restate the implementation's own formula.
        eager_rows = mrr * ((1 + k) if spec != "none" else 1)
        floor = max(1, eager_rows, widest_bucket)
        print(f"  {label:<44} {old:>6} {new:>6} {floor:>6} {widest_bucket:>21}")
        report(f"[{label}] ring_rows >= eager decode/verify rows ({eager_rows})", new >= eager_rows)
        report(f"[{label}] ring_rows >= widest capturable bucket ({widest_bucket})",
               new >= widest_bucket)
        if new > floor:
            note(f"[{label}] over-provisioned {new} vs floor {floor} (x{new/floor:.0f}): spec "
                 f"multiplies the CAPTURE term too, though the verify bucket list is already "
                 f"clamped to max_running_req (scheduler `verify_bs`). Safe direction — costs "
                 f"ring bytes, never a dropped record.")

    c = cfg(mrr=2, graph=0)
    report("SHIPPED arm: the old derivation really was 1", baseline_ring_rows(c) == 1,
           f"old={baseline_ring_rows(c)}")
    report("SHIPPED arm: the new derivation is max_running_req", derive_ring_rows(c) == 2,
           f"new={derive_ring_rows(c)}")

    # The one hole in the derivation: an EXPLICIT cuda_graph_bs list bypasses cuda_graph_max_bs
    # entirely (`_determine_cuda_graph_bs` returns it verbatim), so a caller can capture a bucket
    # wider than ring_rows. Not CLI-reachable (`server/args.py` exposes only --cuda-graph-max-bs) but
    # reachable programmatically. Gate that it is FAIL-LOUD rather than a silently truncated write.
    print("\n[2b] the residual hole: an explicit cuda_graph_bs list is NOT covered")
    c2 = cfg(mrr=2, graph=0)
    object.__setattr__(c2, "cuda_graph_bs", [1, 2, 64])
    b = _determine_cuda_graph_bs(c2.cuda_graph_bs, c2.cuda_graph_max_bs, 8 << 30, c2.max_running_req)
    note(f"cuda_graph_bs={c2.cuda_graph_bs} -> buckets {b}, but ring_rows={derive_ring_rows(c2)}: "
         f"the derivation does NOT read cuda_graph_bs")
    tr = RT.RouteTracer(None, model_slug="g", num_layers=1, num_experts=64, top_k=10, tp_rank=0,
                        dp_rank=0, expert_bytes=1, ring_steps=4, ring_rows=2, drain_every=4,
                        max_steps=10 ** 9, record_prefill=False, blockmap_checks=0, device=DEV)
    RT._CUR_LID = 0
    try:
        tr._record_captured(torch.zeros((64, 10), dtype=torch.int32))
        report("an over-wide captured bucket raises instead of baking a truncated write", False,
               "no exception")
    except RT.RouteTraceError as e:
        report("an over-wide captured bucket raises instead of baking a truncated write", True,
               str(e).split(".")[0][:70])
    RT._CUR_LID = None


def gate_residual_holes() -> None:
    """Where defect (A) is STILL live after the fix, and whether it is visible when it is."""
    print("\n[2c] RESIDUAL: DP+EP still makes M exceed ring_rows, and nothing counts it")
    from minisgl.distributed.info import DpInfo
    # `MoELayer._ep_dispatch` (layers/moe.py) all_gathers every replica's rows: `g_ids` is
    # (dp*N, top_k) and that is the tensor the grouped kernel — and therefore `record` via
    # quant/kernels.py:676 — receives. So under DP+EP a decode forward carries dp_size * bs rows.
    # (EP-over-TP is exempt: it returns before the gather, rows stay = bs.)
    a = cfg(mrr=2, graph=0)
    b = cfg(mrr=2, graph=0)
    object.__setattr__(b, "dp_info", DpInfo(0, 4))
    object.__setattr__(b, "enable_ep", True)
    gap("ring_rows scales with dp_size under EP",
        derive_ring_rows(b) > derive_ring_rows(a),
        f"dp=1 -> {derive_ring_rows(a)}, dp=4+enable_ep -> {derive_ring_rows(b)}: the derivation "
        f"reads neither dp_info nor enable_ep, so a DP+EP decode (dp*bs = 8 rows) still exceeds it")

    # Drive the real tracer with exactly that shape and see what a reviewer would have to look at.
    rr = derive_ring_rows(a)
    tr = RT.RouteTracer(None, model_slug="g", num_layers=1, num_experts=NUM_EXPERTS, top_k=TOP_K,
                        tp_rank=0, dp_rank=0, expert_bytes=1, ring_steps=8, ring_rows=rr,
                        drain_every=8, max_steps=10 ** 9, record_prefill=False, blockmap_checks=0,
                        device=DEV)
    seen = []
    tr.set_observer(lambda lid, ids: seen.append((lid, tuple(ids))))
    wide = torch.tensor(np.arange(4 * TOP_K).reshape(4, TOP_K) % NUM_EXPERTS,
                        dtype=torch.int32, device=DEV)
    tr.begin_forward(False, 1, is_verify=False, num_rows=4)     # 4 rows, ring holds `rr`
    RT._CUR_LID = 0
    tr.record(wide, num_tokens=4)
    RT._CUR_LID = None
    tr.drain()
    report(f"a decode wider than ring_rows ({4} > {rr}) still reaches NO observer", not seen,
           f"{len(seen)} calls")
    st = tr.stats()
    counters = {k: v for k, v in st.items() if k.startswith("route_trace_rows")
                or k.endswith("capture_unwrapped")}
    gap("some counter fires when a too-wide DECODE is dropped",
        any(v for v in counters.values()),
        f"{counters} -- `record`'s host path increments rows_dropped only for KIND_VERIFY, so the "
        f"decode case that defect (A) IS goes uncounted; close()'s 'every way the union can be "
        f"wrong, in one line' does not cover it")
    gap("record()'s rows_mismatch comment is reachable for EP as it claims",
        st["route_trace_rows_mismatch"] > 0,
        "rows_mismatch is inside the RING path, but an EP-widened forward falls out of the ring "
        "before it — so the inline comment naming EP as a legitimate cause of rows_mismatch "
        "describes a state that cannot occur")
    note("EP also changes WHAT is routed, not just how many rows: `_ep_dispatch` passes the kernel "
         "`local_ids = where(is_local, ep_i - local_expert_offset, 0)`, i.e. LOCAL expert indices "
         "with every non-local slot masked to 0. Anything the tracer recorded under DP+EP would be "
         "in the wrong index space and would over-count expert 0. Pre-existing and out of scope "
         "here, but it means 'widen ring_rows by dp_size' is NOT the whole fix for EP.")


# --- the workload ------------------------------------------------------------------------------
NUM_LAYERS = 48          # qwen4exp: 48 MoE layers
NUM_EXPERTS = 512        # qwen4exp: 512 experts/layer
TOP_K = 10               # qwen4exp: top_k 10
BLOCK = 2 * TOP_K        # experts per STEP, so a step is recoverable from any id it routed to
NUM_STEPS = 24           # 24 * 20 = 480 <= 512, so every step owns a disjoint expert block


def build_workload(plan: "list[int]", seed: int = 1234) -> "list[list[torch.Tensor]]":
    """ids[step][lid] : (M, top_k) int32. Step s, row r owns experts [s*20+r*10, +10), shuffled
    within the block so the tensor is unsorted (the dedupe/sort in drain is exercised) while every
    id still identifies its step and row uniquely."""
    rng = np.random.default_rng(seed)
    out = []
    for s, m in enumerate(plan):
        per_layer = []
        for _ in range(NUM_LAYERS):
            rows = []
            for r in range(m):
                blk = np.arange(s * BLOCK + r * TOP_K, s * BLOCK + r * TOP_K + TOP_K)
                rng.shuffle(blk)
                rows.append(blk)
            per_layer.append(torch.tensor(np.stack(rows), dtype=torch.int32, device=DEV))
        out.append(per_layer)
    return out


def run_arm(mod, ring_rows: int, plan: "list[int]", ids, *, old_style: bool):
    """Drive one tracer over the workload; return (observed refs, per-step observed layer count)."""
    ctor = dict(model_slug="gate", num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS, top_k=TOP_K,
                tp_rank=0, dp_rank=0, expert_bytes=1, ring_steps=16, ring_rows=ring_rows,
                drain_every=8, max_steps=10 ** 9, record_prefill=False, blockmap_checks=0,
                device=DEV)
    if old_style:
        torch.empty = _empty_nopin          # pre-fix __init__ pins unconditionally
        try:
            tr = mod.RouteTracer(None, **ctor)
        finally:
            torch.empty = _real_empty
    else:
        tr = mod.RouteTracer(None, **ctor)

    refs: set[tuple[int, int, int]] = set()
    steps_seen: dict[int, int] = {}
    bad_calls: list[str] = []

    def obs(lid, idlist):
        blocks = {int(e) // BLOCK for e in idlist}
        if len(blocks) != 1:
            bad_calls.append(f"lid={lid} ids span {len(blocks)} step-blocks")
            return
        s = blocks.pop()
        steps_seen[s] = steps_seen.get(s, 0) + 1
        for e in idlist:
            refs.add((s, int(lid), int(e)))

    tr.set_observer(obs)
    supports_rows = "num_rows" in inspect.signature(tr.begin_forward).parameters
    for s, m in enumerate(plan):
        kw = {"num_rows": m} if supports_rows else {}
        tr.begin_forward(False, 1, is_verify=False, **kw)
        for lid in range(NUM_LAYERS):
            mod._CUR_LID = lid
            tr.record(ids[s][lid], num_tokens=m)
        mod._CUR_LID = None
    tr.close()
    return refs, steps_seen, bad_calls


def expected_refs(plan, ids) -> "set[tuple[int, int, int]]":
    want = set()
    for s, m in enumerate(plan):
        for lid in range(NUM_LAYERS):
            for e in ids[s][lid].reshape(-1).tolist():
                want.add((s, lid, int(e)))
    return want


# --- end-to-end --------------------------------------------------------------------------------
def gate_end_to_end(old_mod) -> None:
    print("\n[3] END-TO-END: does a 2-row decode step with NO spec reach the observer?")
    c = cfg(mrr=2, graph=0)                            # the shipped qwen4exp arm
    plan = [1 if s % 2 == 0 else 2 for s in range(NUM_STEPS)]      # alternating M=1 / M=2
    ids = build_workload(plan)
    want = expected_refs(plan, ids)
    m1_steps = {s for s, m in enumerate(plan) if m == 1}
    m2_steps = {s for s, m in enumerate(plan) if m == 2}

    arms = {"new(ring_rows=%d)" % derive_ring_rows(c): (RT, derive_ring_rows(c), False),
            "old-value(ring_rows=%d)" % baseline_ring_rows(c): (RT, baseline_ring_rows(c), True)}
    if old_mod is not None:
        arms["old-CODE(ring_rows=%d)" % baseline_ring_rows(c)] = (old_mod, baseline_ring_rows(c), True)

    results = {}
    for name, (mod, rr, old_style) in arms.items():
        refs, steps_seen, bad = run_arm(mod, rr, plan, ids, old_style=old_style)
        results[name] = (refs, steps_seen)
        report(f"[{name}] no observer call mixed two steps' experts", not bad, "; ".join(bad[:2]))
        print(f"    {name:<26} refs {len(refs):>5}/{len(want):<5} "
              f"steps observed {len(steps_seen):>2}/{len(plan)}  "
              f"(M=1 {len(steps_seen.keys() & m1_steps)}/{len(m1_steps)}, "
              f"M=2 {len(steps_seen.keys() & m2_steps)}/{len(m2_steps)})")

    new_name = next(n for n in results if n.startswith("new"))
    newrefs, newsteps = results[new_name]
    report("FIXED: every M=2 decode step reaches the observer",
           set(newsteps) >= m2_steps, f"missing {sorted(m2_steps - set(newsteps))}")
    report("FIXED: every layer of every step reaches the observer",
           all(v == NUM_LAYERS for v in newsteps.values()),
           f"min layers/step={min(newsteps.values()) if newsteps else 0}")
    report("FIXED: the observed expert set is EXACTLY the routed set (no loss, no padding)",
           newrefs == want, f"missing {len(want - newrefs)}, extra {len(newrefs - want)}")

    for name in results:
        if name.startswith("new"):
            continue
        refs, steps = results[name]
        report(f"UNFIXED[{name}]: NOT ONE M=2 step reached the observer",
               not (set(steps) & m2_steps), f"leaked {sorted(set(steps) & m2_steps)[:4]}")
        report(f"UNFIXED[{name}]: M=1 steps still did (so the loss is row-count-specific, "
               f"not a dead tracer)", set(steps) >= m1_steps)
        lost = len(want - refs) / len(want)
        print(f"    {name:<26} LOST {len(want)-len(refs):>5}/{len(want)} routing references "
              f"= {lost*100:.1f}%")

    if old_mod is not None:
        ov = results[next(n for n in results if n.startswith("old-value"))]
        oc = results[next(n for n in results if n.startswith("old-CODE"))]
        report("old-value arm is byte-faithful to old-CODE (so ring_rows IS the only difference)",
               ov[0] == oc[0] and ov[1] == oc[1],
               f"refs differ by {len(ov[0] ^ oc[0])}, steps {set(ov[1]) ^ set(oc[1])}")


# --- quantification ----------------------------------------------------------------------------
def gate_quantification(old_mod) -> None:
    print("\n[4] QUANTIFICATION at max_running_req=2: what fraction never reached the cache?")
    print("    A reference = one (step, layer, expert) routing the policy could have learned from.")
    print("    Closed form for the pre-fix code: lost = 2f/(1+f), f = fraction of steps at M=2")
    print("    (an M=2 step contributes 2*top_k refs per layer and ALL of them were dropped; an")
    print("     M=1 step contributes top_k and all of them survived).")
    c = cfg(mrr=2, graph=0)
    rr_old, rr_new = baseline_ring_rows(c), derive_ring_rows(c)
    print(f"  {'f (frac steps M=2)':>19} {'steps M=2':>10} {'refs total':>11} "
          f"{'refs lost OLD':>14} {'% lost OLD':>11} {'2f/(1+f)':>10} {'% lost NEW':>11}")
    for n2 in (0, 6, 12, 18, 24):
        f = n2 / NUM_STEPS
        plan = [2] * n2 + [1] * (NUM_STEPS - n2)
        ids = build_workload(plan)
        want = expected_refs(plan, ids)
        old_refs, _, _ = run_arm(RT, rr_old, plan, ids, old_style=False)
        new_refs, _, _ = run_arm(RT, rr_new, plan, ids, old_style=False)
        lost_old = (len(want) - len(old_refs)) / len(want)
        lost_new = (len(want) - len(new_refs)) / len(want)
        closed = 2 * f / (1 + f)
        print(f"  {f:>19.3f} {n2:>10} {len(want):>11} {len(want)-len(old_refs):>14} "
              f"{lost_old*100:>10.1f}% {closed:>10.3f} {lost_new*100:>10.1f}%")
        report(f"f={f:.2f}: measured OLD loss matches the closed form 2f/(1+f)",
               abs(lost_old - closed) < 1e-9, f"measured {lost_old:.6f} vs {closed:.6f}")
        report(f"f={f:.2f}: FIXED loses nothing", lost_new == 0.0, f"{lost_new:.6f}")
    note("f is WORKLOAD-dependent and was NOT measured on hardware here. Bounds for the shipped "
         "arm (max_running_req=2, so M is 1 or 2): at f=0.5 the pre-fix code dropped 66.7% of "
         "decode routing references; at f=1.0 (saturated 2-request decode) it dropped 100% and the "
         "cache learned from decode not at all. The measured h=0.4067 was nonzero, so f<1 on that "
         "trace -- the cache was learning from the M=1 steps only.")


# --- the second consequence of defect (A) ------------------------------------------------------
def gate_host_sync() -> None:
    print("\n[5] SECOND CONSEQUENCE: the pre-fix host path also put a BLOCKING .tolist() per MoE "
          "layer on the concurrent-decode path")
    src = inspect.getsource(RT.RouteTracer.record)
    report("the host path still ends in .tolist() (the thing the module's own docstring forbids "
           "on decode)", "topk_ids.tolist()" in src)
    note(f"with ring_rows=1 a 2-request decode step took that path for ALL {NUM_LAYERS} MoE "
         f"layers, i.e. {NUM_LAYERS} device->host syncs per step. This repo's measured price for "
         f"ONE such sync per index layer (commit ffa1d8c6, py-spy 36% of scheduler samples) was "
         f"131 -> 85 ms/token. Not measurable here (CPU has no sync to pay); reported as a "
         f"structural finding, not a number.")
    t = torch.tensor(np.arange(2 * TOP_K).reshape(2, TOP_K), dtype=torch.int32)
    import time
    t0 = time.perf_counter()
    for _ in range(NUM_LAYERS * 100):
        sorted({int(e) for row in t.tolist() for e in row})
    dt = (time.perf_counter() - t0) / 100
    note(f"host-side CPU cost of that path alone, {NUM_LAYERS} layers x (tolist+set+sort) at the "
         f"real (2,10) shape: {dt*1e3:.3f} ms/step, on the scheduler thread, BEFORE any GPU sync.")


def gate_cost() -> None:
    """What widening the ring actually costs, measured off the real allocations."""
    print("\n[6] COST of the fix: ring/stage bytes and the per-drain D2H")
    src = inspect.getsource(RT.maybe_install)
    ring_default = int(re.search(r'MINISGL_MOE_ROUTE_TRACE_RING", (\d+)\)', src).group(1))
    budget_mb = int(re.search(r'MINISGL_MOE_ROUTE_TRACE_MAX_MB", (\d+)\)', src).group(1))
    report("read the real ring/budget defaults out of maybe_install", True,
           f"ring={ring_default} budget={budget_mb} MiB")
    print(f"  {'max_running_req':>16} {'ring_rows':>10} {'KiB/step':>10} {'ring_steps':>11} "
          f"{'ring MiB':>9} {'D2H MiB/drain':>14} {'drain_every':>12}")
    for mrr in (2, 8, 64, 256):
        c = cfg(mrr=mrr, graph=0)
        rr = derive_ring_rows(c)
        per_step = NUM_LAYERS * TOP_K * rr * 4
        steps = ring_default
        if per_step * steps > budget_mb * (1 << 20):
            steps = max(8, (budget_mb * (1 << 20)) // per_step)
        drain_every = min(64, steps)
        ring_mib = per_step * steps / (1 << 20)
        print(f"  {mrr:>16} {rr:>10} {per_step/1024:>10.1f} {steps:>11} {ring_mib:>9.1f} "
              f"{ring_mib:>14.1f} {drain_every:>12}")
    # The shipped arm, off the real allocation rather than arithmetic.
    c = cfg(mrr=2, graph=0)
    tr = RT.RouteTracer(None, model_slug="g", num_layers=NUM_LAYERS, num_experts=NUM_EXPERTS,
                        top_k=TOP_K, tp_rank=0, dp_rank=0, expert_bytes=1, ring_steps=1024,
                        ring_rows=derive_ring_rows(c), drain_every=64, max_steps=10 ** 9,
                        record_prefill=False, blockmap_checks=0, device=DEV)
    ring_b = tr.ids_ring.numel() * 4
    stage_b = tr.stage.numel() * 4 if hasattr(tr, "stage") else 0
    report("SHIPPED arm ring stays under 8 MiB after widening", ring_b < (8 << 20),
           f"ring {ring_b/(1<<20):.2f} MiB + stage {stage_b/1024:.1f} KiB")
    note("`drain()` copies the WHOLE ring D2H every drain, not just the new steps -- pre-existing, "
         "but defect (A)'s fix multiplies it by ring_rows. At max_running_req=2 that is 1.9 -> 3.8 "
         "MiB per 64-step drain (~60 KiB/step); at max_running_req=256 it goes 0.25 -> 63.8 MiB "
         "per drain, since the 64 MiB budget caps the "
         "ring and the D2H becomes ~64 MiB per 64 steps (~1 MiB/step). NOT measured on hardware. "
         "The task brief's 'cost is trivial' holds for the shipped arm and NOT for a large-batch "
         "config; a bs-aware partial copy (`ids_host[first:last]`) would remove it.")


def main() -> int:
    print(f"route_trace ring_rows gate  (FALSIFY={'ON' if FALSIFY else 'off'}, "
          f"device=cpu, torch={torch.__version__})")
    if FALSIFY:
        print("  !! FALSIFY MODE: derive_ring_rows() returns the PRE-FIX value. This run MUST FAIL.")
    old_mod = check_provenance()
    gate_derivation()
    gate_residual_holes()
    gate_end_to_end(old_mod)
    gate_quantification(old_mod)
    gate_host_sync()
    gate_cost()
    print(f"\n{'=' * 78}")
    if FAILS:
        print(f"GATE FAILED ({len(FAILS)}): defect (A) is NOT fixed for the configs below")
        for f in FAILS:
            print(f"  - {f}")
    else:
        print("GATE PASSES: defect (A) is fixed — a 2-row non-spec decode step reaches the "
              "observer, and the derivation covers max_running_req and cuda_graph_max_bs.")
    if GAPS:
        print(f"\nRESIDUAL GAPS ({len(GAPS)}) — real, still open, NOT what the gate above asserts:")
        for g in GAPS:
            print(f"  - {g}")
    return 1 if FAILS else (2 if GAPS else 0)


_RESULT: "int | None" = None


def _run_once() -> int:
    global _RESULT
    if _RESULT is None:
        _RESULT = main()
    return _RESULT


def test_defect_a_is_fixed():
    """The gate proper: rc 1 means the ring_rows fix does not hold."""
    assert _run_once() != 1, FAILS


def test_no_residual_gaps():
    """Separate on purpose: this one is EXPECTED red until the DP+EP / counter holes are closed.
    It must not be folded into test_defect_a_is_fixed, or closing one would mask the other."""
    assert _run_once() != 2, GAPS


if __name__ == "__main__":
    sys.exit(_run_once())
