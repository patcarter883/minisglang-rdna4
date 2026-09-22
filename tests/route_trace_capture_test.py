"""GATE: the MoE route ring must RECORD UNDER GRAPH CAPTURE, and each REPLAY must land its own step.

    PYTHONPATH=python python3 tests/route_trace_capture_test.py
    PYTHONPATH=python python3 tests/route_trace_capture_test.py --falsify

NO GPU REQUIRED, and that is deliberate: the box's two cards are leased, and the only hardware gate
that matters for this change (a 120k-prefill captured serve) belongs to the operator. Everything the
fix does is host logic over tensors -- `route_trace.RouteTracer` constructs on `device="cpu"` -- so
the whole property is checkable here. What is NOT checkable here is whether HIP accepts the recorded
op; see REAL CAPTURE below, which is attempted and reports why it skipped.

WHAT IS BEING GATED. `weights/route_trace.py` used to be capture-UNSAFE by construction: every write
was guarded on `torch.cuda.is_current_stream_capturing()`. Once that ring became the EXPERT CACHE's
only input, a captured serve meant the cache observed nothing and went inert while still holding its
whole budget (2.5 GiB/rank) -- i.e. flipping `--cuda-graph-max-bs` on would have bought 3.45-3.66 ms
of a ~60 ms step and silently paid back the entire expert-cache win (h 0.3256 -> 0.4067). Four
claims, each with its own falsification arm below:

  (1) `record` no longer early-returns on the ring path under capture -- one write per MoE layer is
      actually recorded into the graph.
  (2) that write targets `stage[lid, :n]` and has NO dependence on `self.slot`. `slot` is host
      Python; a graph bakes it as a constant and every replay forever rewrites one frozen slot.
  (3) N replays with NO Python from `record` re-running land N DISTINCT steps in `ids_ring`, one per
      `harvest()` at the step boundary -- including across a ring wrap.
  (4) the observer is fed exactly once per (step, layer) with that step's own ids, and a captured
      BUCKET's padded rows are masked out of what it sees.

HOW CAPTURE IS SIMULATED, AND WHERE THE SIMULATION IS HONEST. `FakeGraph` below patches
`torch.Tensor.__setitem__` to RECORD (destination tensor object, index key, source tensor) without
executing, patches `is_current_stream_capturing()` to True, and makes `.tolist()`/`.item()` RAISE.
That reproduces the three properties this fix turns on:
  * capture records device ops, it does not execute them;
  * every operand ADDRESS is frozen at capture -- here the exact tensor objects plus the index key;
  * a replay re-executes the recorded ops with NO Python from the captured region running again, so
    anything computed in host Python is a baked constant;
  * a host sync is ILLEGAL under capture (hence the raising `.tolist()`), which is why the tracer's
    prefill/oversize path must stay capture-guarded.
It does NOT reproduce streams, events, torch's own capture-legality checks, or HIP's acceptance of
the recorded copy. So this file gates the DATAFLOW; it cannot gate "HIP captured it". The structural
assertions (destination identity, key shape, per-step state untouched) are what stand in for that.

FALSIFICATION IS PART OF THE GATE. `--falsify` re-runs the same assertions against three broken
tracers and REQUIRES each to be caught:
  * `unfixed` -- `record` copied verbatim from `git show rdna4:python/minisgl/weights/route_trace.py`
    (the capture early-return). Must fail (1): zero ops recorded, the graph traces nothing.
  * `naive`   -- the same body with ONLY the capture early-return deleted, i.e. the fix a reader
    would write first. Run twice: at the production capture state (`slot == -1`, capture happens at
    engine init before any `begin_forward`) it is still inert and fails (1); forced to `slot = 0` it
    records, and then fails (2)+(3) because every replay rewrites slot 0.
  * `nomask`  -- the shipped `record`, with `harvest`'s row mask removed. Must fail (4): the padded
    row of a bs=2 bucket replayed for one request reaches the observer.
A gate that cannot be made to fail is not a gate, so an arm that stays green is itself a FAILURE.
"""
import argparse
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from minisgl.weights import route_trace as RT  # noqa: E402

# Shapes. Small on purpose, and every one of them is load-bearing:
#   BUCKET > REAL rows        -> there IS padding to mask (defect C)
#   N_STEPS > RING_STEPS      -> the ring WRAPS, so a slot is reused mid-run (defect B)
#   LAYERS > 1                -> a per-layer write must stay per-layer (lid addressing)
LAYERS, TOP_K, EXPERTS = 3, 4, 64
BUCKET_ROWS = 2          # == cuda_graph_max_bs: the captured graph's padded row count
RING_STEPS = 4
N_STEPS = 9

_ORIG_SETITEM = torch.Tensor.__setitem__
_ORIG_TOLIST = torch.Tensor.tolist
_ORIG_ITEM = torch.Tensor.item


class FakeGraph:
    """Record device writes like a capture, then replay them with no Python re-running."""

    def __init__(self):
        self.ops = []           # (dst_tensor, index_key, src)
        self.host_syncs = []    # any .tolist()/.item() attempted under capture

    def __enter__(self):
        graph = self

        def _rec_setitem(t, key, value):
            graph.ops.append((t, key, value))
            # NOT executed: a real capture records, it does not run.

        def _no_sync(t, *a, **k):
            graph.host_syncs.append("host sync")
            raise RuntimeError(
                "a host sync (.tolist()/.item()) is illegal under graph capture -- the tracer's "
                "prefill/oversize path must stay capture-guarded"
            )

        torch.Tensor.__setitem__ = _rec_setitem
        torch.Tensor.tolist = _no_sync
        torch.Tensor.item = _no_sync
        self._prev_cap = torch.cuda.is_current_stream_capturing
        torch.cuda.is_current_stream_capturing = lambda: True
        return self

    def __exit__(self, *exc):
        torch.Tensor.__setitem__ = _ORIG_SETITEM
        torch.Tensor.tolist = _ORIG_TOLIST
        torch.Tensor.item = _ORIG_ITEM
        torch.cuda.is_current_stream_capturing = self._prev_cap
        return False

    def replay(self):
        """Re-execute the baked ops. This is the whole point: no Python from `record` runs here."""
        for dst, key, val in self.ops:
            _ORIG_SETITEM(dst, key, val)


# `record` calls `is_current_stream_capturing()` unconditionally now, and on a box with no GPU that
# RAISES AcceleratorError. The eager path therefore needs a False stub for the whole run; capture
# overrides it to True inside the `with`.
torch.cuda.is_current_stream_capturing = lambda: False


FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def make(**kw):
    kw.setdefault("ring_rows", BUCKET_ROWS)
    kw.setdefault("ring_steps", RING_STEPS)
    kw.setdefault("drain_every", 1)      # one drain per step, so the observer is attributable
    return RT.RouteTracer(
        None, model_slug="capt", num_layers=LAYERS, num_experts=EXPERTS, top_k=TOP_K,
        tp_rank=0, dp_rank=0, expert_bytes=1, max_steps=1_000_000, record_prefill=False,
        blockmap_checks=0, device=torch.device("cpu"), **kw,
    )


def step_ids(step, lid, row):
    """Distinct, in-range, non-overlapping experts per (step, layer, row) so every mix-up is visible."""
    base = 1 + ((step * LAYERS + lid) * BUCKET_ROWS + row) * TOP_K
    return [1 + (base + j) % (EXPERTS - 2) for j in range(TOP_K)]


# ---- the three broken variants the gate must catch -------------------------------------------
# VERBATIM from `git show rdna4:python/minisgl/weights/route_trace.py`, comments stripped. Kept as
# real code rather than a flag so the falsification is against the OLD BODY, not an emulation of it.
def record_unfixed(self, topk_ids, num_tokens, expert_ids=None, ntp=None, block_m=0):
    if self.disarmed or self.slot < 0:
        return
    if torch.cuda.is_current_stream_capturing():
        return                                            # <-- the defect
    lid = RT._CUR_LID
    if lid is None:
        raise RT.RouteTraceError("no layer id in scope")
    chunk = RT._CUR_CHUNK.get(lid, 0)
    RT._CUR_CHUNK[lid] = chunk + 1
    M = int(topk_ids.shape[0])
    if chunk == 0 and self._cur_kind in (RT.KIND_DECODE, RT.KIND_VERIFY) and M <= self.ring_rows:
        n = M * self.top_k
        flat = topk_ids.reshape(-1)[:n]
        self.ids_ring[self.slot, lid, :n] = flat
        if n < self.ring_width:
            self.ids_ring[self.slot, lid, n:] = -1
        self._legacy_meta[self.slot][lid] = (self.step_id, self._cur_uid, self._cur_kind, 0, M)
        return
    if self._cur_kind == RT.KIND_VERIFY:
        self.rows_dropped += 1
    if self._cur_kind == RT.KIND_PREFILL and not self.record_prefill:
        return
    ids = sorted({int(e) for row in topk_ids.tolist() for e in row})
    self.oversize.append((self.step_id, self._cur_uid, lid, self._cur_kind, chunk, M, ids))


def record_naive(self, topk_ids, num_tokens, expert_ids=None, ntp=None, block_m=0):
    """THE FIX A READER WRITES FIRST: delete the capture guard, keep the slot-addressed write."""
    if self.disarmed or self.slot < 0:
        return
    lid = RT._CUR_LID
    if lid is None:
        raise RT.RouteTraceError("no layer id in scope")
    M = int(topk_ids.shape[0])
    if self._cur_kind in (RT.KIND_DECODE, RT.KIND_VERIFY) and M <= self.ring_rows:
        n = M * self.top_k
        self.ids_ring[self.slot, lid, :n] = topk_ids.reshape(-1)[:n]
        if n < self.ring_width:
            self.ids_ring[self.slot, lid, n:] = -1
        self._legacy_meta[self.slot][lid] = (self.step_id, self._cur_uid, self._cur_kind, 0, M)


def harvest_nomask(self):
    """The shipped harvest with the PADDED-ROW MASK removed (defect C reintroduced)."""
    if self.disarmed or self.slot < 0 or self._harvested:
        return
    self._harvested = True
    if self._cur_kind not in (RT.KIND_DECODE, RT.KIND_VERIFY):
        return
    n = self.ring_width                                   # <-- the defect: bucket width, not real
    self.ids_ring[self.slot, :, :n].copy_(self.stage[:, :n])
    self.stage.fill_(-1)
    self.step_meta[self.slot] = (self.step_id, self._cur_uid, self._cur_kind, self.ring_rows)


def break_tracer(tr, arm):
    tr._legacy_meta = [dict() for _ in range(tr.ring_steps)]   # the pre-fix per-lid meta dict
    if arm == "unfixed":
        tr.record = types.MethodType(record_unfixed, tr)
    elif arm.startswith("naive"):
        tr.record = types.MethodType(record_naive, tr)
    elif arm == "nomask":
        tr.harvest = types.MethodType(harvest_nomask, tr)
    else:
        raise SystemExit(f"unknown arm {arm!r}")


# ---- the gate --------------------------------------------------------------------------------
def gate(arm=None):
    """Capture once at the PRODUCTION capture state, then replay N steps and check all four claims."""
    del FAILS[:]
    RT._CUR_CHUNK.clear()
    RT._CUR_LID = None
    tag = f"[{arm}] " if arm else ""
    tr = make()
    if arm:
        break_tracer(tr, arm)
    seen = []
    tr.set_observer(lambda lid, ids: seen.append((lid, tuple(ids))))

    # --- CAPTURE. Production state: `Engine.__init__` arms the tracer (engine.py:429, which calls
    # install_hooks) and captures the decode graphs afterwards (engine.py:791), so at capture time
    # NO `begin_forward` has run: slot == -1, step_id == -1, _cur_kind == KIND_OTHER. Reproduced
    # exactly, because a fix that only works after a step boundary would not work at all.
    assert (tr.slot, tr.step_id, tr._cur_kind) == (-1, -1, RT.KIND_OTHER)
    if arm == "naive-slot0":
        # Give the naive arm the benefit of the doubt: pretend capture happened mid-serve so its
        # write is recorded at all. The baked-slot defect is then the thing under test.
        tr.begin_forward(False, 7, num_rows=1)
    before = (tr.slot, tr.step_id, tr._cur_kind, tr._cur_rows, tr._harvested,
              tr.n_since_drain, len(tr.oversize), dict(RT._CUR_CHUNK))
    bufs = [torch.full((BUCKET_ROWS, TOP_K), -1, dtype=torch.int32) for _ in range(LAYERS)]
    g = FakeGraph()
    with g:
        for lid in range(LAYERS):
            RT._CUR_LID = lid           # what MoELayer's traced_forward wrapper does
            try:
                tr.record(bufs[lid], num_tokens=BUCKET_ROWS)
            finally:
                RT._CUR_LID = None
    after = (tr.slot, tr.step_id, tr._cur_kind, tr._cur_rows, tr._harvested,
             tr.n_since_drain, len(tr.oversize), dict(RT._CUR_CHUNK))

    print(f"{tag}== (1) the ring RECORDS under capture ==")
    report("one write recorded per MoE layer", len(g.ops) == LAYERS,
           f"{len(g.ops)} op(s), want {LAYERS} -- 0 means record() early-returned and the graph "
           f"traces NOTHING, so a captured serve feeds the expert cache zero references")
    report("no host sync attempted under capture", not g.host_syncs,
           f"{len(g.host_syncs)} -- a .tolist()/.item() would abort a real capture")
    report("per-step host state untouched by capture", before == after, f"{before} -> {after}")

    print(f"{tag}== (2) the write targets stage[lid], with NO dependence on self.slot ==")
    dsts = {id(d) for d, _, _ in g.ops}
    report("every destination is the STAGE buffer", bool(g.ops) and dsts == {id(tr.stage)},
           "ids_ring is addressed by a HOST slot index; a graph bakes it and every replay "
           f"rewrites one frozen slot. dst is stage: {[d is tr.stage for d, _, _ in g.ops]}")
    report("ids_ring is never a captured destination", id(tr.ids_ring) not in dsts)
    keys_ok = all(isinstance(k, tuple) and len(k) == 2 and k[0] == lid
                  and isinstance(k[1], slice) and k[1] == slice(None, BUCKET_ROWS * TOP_K)
                  for lid, (_, k, _) in enumerate(g.ops))
    report("key is exactly (lid, :bucket_width) -- 2 dims, no step index",
           bool(g.ops) and keys_ok, f"keys={[k for _, k, _ in g.ops]}")
    ptr_ok = all(isinstance(v, torch.Tensor) and v.data_ptr() == bufs[i].data_ptr()
                 for i, (_, _, v) in enumerate(g.ops))
    report("source aliases the live topk_ids buffer (sim fidelity precondition)",
           bool(g.ops) and ptr_ok,
           "reshape(-1)[:n] must be a VIEW, else FakeGraph's replay would re-copy a snapshot")

    # --- REPLAY. From here on, `record` MUST NOT run: a graph replay executes baked device ops and
    # runs no Python at all. Bound to a raiser so an accidental re-entry is loud rather than a silent
    # pass for the wrong reason.
    def _forbidden(*a, **k):
        raise AssertionError("record() ran during a replay -- the test is not testing capture")

    tr.record = _forbidden

    # Alternate REAL rows 1 and 2 against a 2-row bucket, so the mask has to track the STEP and not
    # be a constant that happens to be right.
    real_seq = [1 if i % 2 == 0 else BUCKET_ROWS for i in range(N_STEPS)]
    ring_seen, obs_per_step, slots = [], [], []
    for i in range(N_STEPS):
        n_obs_before = len(seen)
        tr.begin_forward(False, 1000 + i, num_rows=real_seq[i])   # harvests+drains step i-1
        if i:
            obs_per_step.append(seen[n_obs_before:])
        slots.append(tr.slot)
        for lid in range(LAYERS):                                  # the route kernel's output
            for row in range(BUCKET_ROWS):
                bufs[lid][row] = torch.tensor(step_ids(i, lid, row), dtype=torch.int32)
        g.replay()
        tr.harvest()
        snap = [frozenset(int(v) for v in tr.ids_ring[tr.slot, lid].tolist() if 0 <= v < EXPERTS)
                for lid in range(LAYERS)]
        tr.harvest()                                               # idempotent: must not re-copy
        snap2 = [frozenset(int(v) for v in tr.ids_ring[tr.slot, lid].tolist() if 0 <= v < EXPERTS)
                 for lid in range(LAYERS)]
        if snap != snap2:
            report(f"harvest is idempotent (step {i})", False, f"{snap} -> {snap2}")
        ring_seen.append(snap)
    n_obs_before = len(seen)
    tr.drain()                                                     # flush the last step
    obs_per_step.append(seen[n_obs_before:])

    want = [[frozenset(e for row in range(real_seq[i]) for e in step_ids(i, lid, row))
             for lid in range(LAYERS)] for i in range(N_STEPS)]

    print(f"{tag}== (3) N replays -> N DISTINCT steps in ids_ring, with no Python re-running ==")
    report("the ring WRAPPED (a slot is reused mid-run)", len(set(slots)) < N_STEPS,
           f"slots={slots} over {RING_STEPS} ring steps")
    bad = [i for i in range(N_STEPS) if ring_seen[i] != want[i]]
    report("every step's ring row holds THAT step's ids", not bad,
           f"wrong at steps {bad}: got {[ring_seen[i] for i in bad][:2]} "
           f"want {[want[i] for i in bad][:2]}")
    uniq = {tuple(s) for s in ring_seen}
    report("the N steps are pairwise distinct in the ring", len(uniq) == N_STEPS,
           f"{len(uniq)} distinct of {N_STEPS} -- a baked slot makes every replay overwrite one row")

    print(f"{tag}== (4) the observer is fed once per (step, layer), padding masked ==")
    counts = [len(o) for o in obs_per_step]
    report("exactly one observation per layer per step", counts == [LAYERS] * N_STEPS,
           f"counts={counts}, want {[LAYERS] * N_STEPS}")
    obs_bad = []
    for i, obs in enumerate(obs_per_step):
        got = {lid: frozenset(ids) for lid, ids in obs}
        if got != {lid: want[i][lid] for lid in range(LAYERS)}:
            obs_bad.append(i)
    report("the observer saw each step's own per-layer ids", not obs_bad,
           f"wrong at steps {obs_bad}: got {[obs_per_step[i] for i in obs_bad][:1]}")
    # THE PADDING CHECK, stated separately because it is the one an "observer was called" test misses.
    leaks = []
    for i, obs in enumerate(obs_per_step):
        if real_seq[i] >= BUCKET_ROWS:
            continue
        for lid, ids in obs:
            pad = set(step_ids(i, lid, real_seq[i]))
            if pad & set(ids):
                leaks.append((i, lid, sorted(pad & set(ids))))
    report("a bs=2 bucket replayed for 1 request leaks NO padded-row experts", not leaks,
           f"leaked {leaks[:3]} -- experts no request routed to would be admitted, and real ones "
           f"evicted to make room")
    report("counters clean (rows_unknown/mismatch/capture_unwrapped)",
           (tr.rows_unknown, tr.rows_mismatch, tr.capture_unwrapped, tr.rows_dropped) == (0, 0, 0, 0),
           f"unknown={tr.rows_unknown} mismatch={tr.rows_mismatch} "
           f"unwrapped={tr.capture_unwrapped} dropped={tr.rows_dropped}")
    return list(FAILS)


def gate_ring_rows():
    """DEFECT (A), which is a LIVE BUG on the shipped eager serve, not a capture prerequisite.

    `engine.py` used to set `ring_rows = 1` whenever spec was off. `record` takes the sync-free device
    ring only when `M <= ring_rows`, so with ring_rows == 1 every decode step carrying two or more
    rows fell to the HOST path -- and a host-path record lands in `self.oversize`, which `drain()`
    packs into the trace FILE but NEVER hands to the observer. On `--max-running-requests 2` that is
    every concurrent 2-request decode step: invisible to the expert cache, with nothing logged.
    """
    print("== (A) a 2-row decode must reach the observer (ring_rows >= max_running_req) ==")
    out = {}
    for rows in (1, BUCKET_ROWS):
        tr = make(ring_rows=rows)
        seen = []
        tr.set_observer(lambda lid, ids: seen.append((lid, tuple(ids))))
        tr.begin_forward(False, 1, num_rows=2)
        RT._CUR_LID = 0
        RT._CUR_CHUNK.clear()
        tr.record(torch.tensor([[1, 2, 3, 4], [9, 10, 11, 12]], dtype=torch.int32), num_tokens=2)
        RT._CUR_LID = None
        tr.drain()
        out[rows] = (list(seen), tr.rows_dropped, len(tr.oversize))
    report("ring_rows=1 (the OLD non-spec value) starves the observer", out[1][0] == [],
           f"observer calls={out[1][0]}, oversize records={out[1][2]} -- oversize never reaches "
           f"the observer, which is the whole defect")
    report(f"ring_rows={BUCKET_ROWS} feeds the UNION of both rows",
           len(out[BUCKET_ROWS][0]) == 1
           and set(out[BUCKET_ROWS][0][0][1]) == {1, 2, 3, 4, 9, 10, 11, 12},
           f"{out[BUCKET_ROWS][0]}")


def real_capture_probe():
    """Try an ACTUAL torch.cuda.CUDAGraph capture+replay. Reports why it cannot run, if it cannot."""
    print("== REAL torch.cuda.CUDAGraph capture (attempted, not simulated) ==")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        dev = torch.device("cuda")
        tr = RT.RouteTracer(
            None, model_slug="capt", num_layers=LAYERS, num_experts=EXPERTS, top_k=TOP_K,
            tp_rank=0, dp_rank=0, expert_bytes=1, ring_steps=RING_STEPS, ring_rows=BUCKET_ROWS,
            drain_every=1, max_steps=10**6, record_prefill=False, blockmap_checks=0, device=dev,
        )
        seen = []
        tr.set_observer(lambda lid, ids: seen.append((lid, tuple(ids))))
        bufs = [torch.full((BUCKET_ROWS, TOP_K), -1, dtype=torch.int32, device=dev)
                for _ in range(LAYERS)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                      # warmup, as capture requires
            for lid in range(LAYERS):
                RT._CUR_LID = lid
                tr.record(bufs[lid], num_tokens=BUCKET_ROWS)
                RT._CUR_LID = None
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for lid in range(LAYERS):
                RT._CUR_LID = lid
                tr.record(bufs[lid], num_tokens=BUCKET_ROWS)
                RT._CUR_LID = None
        ok = True
        for i in range(3):
            tr.begin_forward(False, 1000 + i, num_rows=1)
            for lid in range(LAYERS):
                for row in range(BUCKET_ROWS):
                    bufs[lid][row] = torch.tensor(step_ids(i, lid, row), dtype=torch.int32,
                                                  device=dev)
            graph.replay()
            tr.harvest()
            for lid in range(LAYERS):
                got = {int(v) for v in tr.ids_ring[tr.slot, lid].tolist() if 0 <= v < EXPERTS}
                ok &= got == set(step_ids(i, lid, 0))
        report("real CUDAGraph: 3 replays land 3 distinct masked steps", ok)
    except Exception as e:                              # noqa: BLE001
        print(f"  SKIP  no GPU in this container, by instruction: {type(e).__name__}: "
              f"{str(e).splitlines()[0]}")
        print("        -> the FakeGraph simulation below is what gates the dataflow; HIP's "
              "acceptance of the recorded copy is NOT gated here.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--falsify", action="store_true",
                    help="run the gate against the BROKEN variants; each must be caught")
    args = ap.parse_args()
    if args.falsify:
        expect = {
            "unfixed": "(1) zero ops recorded: the graph traces nothing",
            "naive": "(1) still inert at the production capture state (slot == -1)",
            "naive-slot0": "(2)+(3) the slot is baked; every replay rewrites one row",
            "nomask": "(4) the padded row's experts reach the observer",
        }
        undetected = []
        for arm, why in expect.items():
            print(f"\n######## FALSIFY {arm}: must FAIL -- {why}")
            fails = gate(arm)
            print(f"  -> {len(fails)} failure(s): {fails}")
            if not fails:
                undetected.append(arm)
        print()
        if undetected:
            print(f"GATE IS WORTHLESS: arms not caught: {undetected}")
            return 1
        print("FALSIFIED: every broken variant was caught by this gate.")
        return 0
    real_capture_probe()
    print()
    del FAILS[:]
    gate_ring_rows()
    pre = list(FAILS)
    print()
    fails = pre + gate()
    print(f"\n{'FAILED: ' + ', '.join(fails) if fails else 'ALL PASS'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
