#!/usr/bin/env python3
"""Expert-cache CPU REPLAY — drive the real `ExpertResidencyCache` on the real route trace, on CPU.

WHY THIS EXISTS. This cache has now frozen replacement FOUR times, and three of the four diagnoses
were wrong. Every one of them was made by reading the source and doing arithmetic on counters
printed by a serve that costs ~20 minutes and both GPUs to boot. This file closes that loop: it
runs the SHIPPING class — loaded verbatim from `python/minisgl/weights/expert_cache.py`, manager
thread and all — against the captured 22,001-step qwen4exp route trace, with `torch` replaced by a
shim that models the only part of torch this cache's correctness depends on: STREAM ORDERING and
EVENT COMPLETION, on a virtual clock that ticks once per scheduler step.

    python3 tools/offload/expert_cache_replay.py                     # stock knobs, 22k steps
    MINISGL_EXPERT_CACHE_MAX_INFLIGHT=256 python3 ... --steps 6000   # any knob, no GPU

WHAT IT MEASURES THAT A SERVE DOES NOT
  * served_h -- the hit rate the KERNEL would see, read off the device-visible `slot_of` tables,
    not the policy's own `observed_h`. The two differ, and the difference was the bug.
  * WRONG-BYTES reads -- references whose `slot_of` entry points at a slot another expert now
    owns. This is the failure the module docstring calls the one forbidden direction, it is
    silent in production, and on the pre-2026-09-22 code it was 13.05% of every expert read.
  * per-DRAIN-EPISODE installs -- promotions between route-trace drains. `min == max == _low_water`
    is the signature of the freeze this file was written to reproduce.
  * IDLE-MANAGER TICKS -- free slots parked on a tick where the manager has no queued work. 25.00
    on 11,000 of 11,000 ticks was the smoking gun: the pool was full and nobody was placing.

MEASURED WITH IT, 22,000 steps / 6,000 warm / 2,056 slots / LOW_WATER=25 / MAX_INFLIGHT=64:

                                 served_h   promotions/tick   installs per drain   WRONG-BYTES
    a8b6c95e (before)             0.3556         0.447        25.00 (min==max)      1,001,873
    after `_service` + retract    0.5019         8.379        537                           0
    offline SLRU ceiling          0.5425            --          --                         --

RUN IT BEFORE AND AFTER ANY CHANGE TO THIS CACHE. It takes 20 seconds and needs no lease.

SEE ALSO `expert_cache_freeze_repro.py`, the sibling harness that reached the same root cause from
a GENERATED reference stream (it can sweep locality and drain interval; it needs real torch, so it
runs in the image, and it scores the policy's `observed_h` rather than the device table).
"""

import argparse
import contextlib
import importlib.util
import os
import struct
import sys
import threading
import time
import types
from array import array
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SRC = os.path.join(REPO, "python", "minisgl", "weights", "expert_cache.py")
DEFAULT_TRACE = os.path.join(REPO, "tools", "offload", "route_traces", "route._model.rank0.bin")

# ============================================================================================
# THE torch SHIM.  Streams and events are modelled; arithmetic is not, because the freeze is
# pure host-side bookkeeping and the slabs' CONTENTS are what tests/expert_cache_test.py checks.
# ============================================================================================

import contextlib, threading

class Clock:
    now = 0
CLOCK = Clock()

COMPUTE_LAT = 1     # ticks before an event recorded on the compute stream completes
COPY_LAT    = 1     # ticks for a 1.36 MiB H2D once its wait_event is satisfied

class FakeStream:
    def __init__(self, device=None, compute=False):
        self.device = device; self.compute = compute; self.frontier = 0
    def wait_event(self, ev):
        self.frontier = max(self.frontier, ev.ready_at)
    def wait_stream(self, other):
        self.frontier = max(self.frontier, other.frontier)

class FakeEvent:
    __slots__ = ("ready_at",)
    def __init__(self):
        self.ready_at = float("inf")
    def record(self, stream=None):
        if stream is None or stream.compute:
            self.ready_at = CLOCK.now + COMPUTE_LAT
        else:
            self.ready_at = max(stream.frontier, CLOCK.now) + COPY_LAT
    def query(self):
        return CLOCK.now >= self.ready_at
    def synchronize(self):
        pass

class FakeRow:
    __slots__ = ()
    def copy_(self, src, non_blocking=False):
        return self

_ROW = FakeRow()

class FakeTensor:
    """Dense enough for shape checks; __setitem__ is recorded so the harness can read back the
    device-visible slot_of table (what the kernel would actually see)."""
    def __init__(self, shape, dtype=None, fill=None):
        self.shape = tuple(shape); self.dtype = dtype
        self.fill = fill
        self.vals = {} if fill is not None else None   # only `torch.full` tables track values
    def __getitem__(self, i):
        if isinstance(i, int) and self.vals is not None:
            return self.vals.get(i, self.fill)
        return _ROW
    def __setitem__(self, i, v):
        if self.vals is None:
            return                      # a slab row assignment: bytes, not tracked
        self.vals[i] = v
    def get(self, i):
        return self.vals.get(i, self.fill) if self.vals is not None else self.fill

class _Device:
    def __init__(self, type_): self.type = type_
    def __repr__(self): return f"device({self.type})"

class _CudaNS:
    Stream = FakeStream
    Event = FakeEvent
    _cur = threading.local()
    @staticmethod
    def stream(s):
        return contextlib.nullcontext()
    @staticmethod
    def current_stream(dev=None):
        s = getattr(_CudaNS._cur, "s", None)
        if s is None:
            s = _CudaNS._cur.s = FakeStream(dev, compute=True)
        return s
    @staticmethod
    def set_device(dev): pass
    @staticmethod
    def is_current_stream_capturing(): return False
    @staticmethod
    def current_device(): return 0

class FakeTorch:
    cuda = _CudaNS
    int32 = "int32"
    float16 = "float16"
    @staticmethod
    def device(t, i=None): return _Device(t)
    @staticmethod
    def full(shape, val, dtype=None, device=None): return FakeTensor(shape, dtype, fill=val)
    @staticmethod
    def empty(shape, dtype=None, device=None): return FakeTensor(shape, dtype)
    @staticmethod
    @contextlib.contextmanager
    def inference_mode():
        yield


# ============================================================================================
# TRACE IO -- the MSGLRT01 reader (same format tools/offload/expert_cache_oracle.py writes).
# ============================================================================================
MAGIC=b"MSGLRT01"
HDR_FMT="<8sIIIIIIQIIQ8x"; HDR=struct.calcsize(HDR_FMT)
REC_FMT="<IIHBBHH"; REC=struct.calcsize(REC_FMT)
KIND_PREFILL,KIND_DECODE,KIND_OTHER=0,1,2

def _read_trace(path, limit_records=None):
    blob=open(path,'rb').read()
    (magic,ver,nl,ne,tk,tp,dp,nrec,eb,flags,mh)=struct.unpack(HDR_FMT,blob[:HDR])
    assert magic==MAGIC,magic
    pos=HDR; out=[]; end=len(blob)
    while pos+REC<=end:
        step,uid,layer,kind,chunk,ntok,nids=struct.unpack_from(REC_FMT,blob,pos); pos+=REC
        if pos+2*nids>end: break
        ids=array("H"); ids.frombytes(blob[pos:pos+2*nids]); pos+=2*nids
        out.append((step,uid,layer,kind,chunk,ntok,list(ids)))
        if limit_records and len(out)>=limit_records: break
    return dict(num_layers=nl,num_experts=ne,top_k=tk,tp_rank=tp,expert_bytes=eb,nrec=nrec),out



# ============================================================================================
# THE REPLAY
# ============================================================================================

import argparse, importlib.util, os, sys, threading, time, types
from collections import defaultdict



def load_cache_module(src):
    sys.modules["torch"] = FakeTorch
    pkg = types.ModuleType("ecpkg"); pkg.__path__ = []
    sys.modules["ecpkg"] = pkg
    ct = types.ModuleType("ecpkg.cpu_tier")
    ct.CpuTierError = type("CpuTierError", (RuntimeError,), {})
    sys.modules["ecpkg.cpu_tier"] = ct
    spec = importlib.util.spec_from_file_location("ecpkg.expert_cache", src)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ecpkg.expert_cache"] = m
    spec.loader.exec_module(m)
    return m


def build(mod, slots, num_experts, registered):
    dev = FakeTorch.device("cuda", 0)
    c = mod.ExpertResidencyCache(num_experts=num_experts, expert_bytes=1,
                                 budget_bytes=slots, device=dev)
    w = FakeTensor((num_experts, 64), "u8")
    s = FakeTensor((num_experts, 8), "u8")
    for lid in registered:
        c.register_layer(lid, (w, s, None), (w, s, None))
    return c


def quiesce(c, dwell):
    """Give the manager thread the wall time a real 50 ms step would give it, then wait until it
    has visibly stopped making progress (no queue, and no placeable work left)."""
    t0 = time.monotonic()
    time.sleep(dwell)
    while time.monotonic() - t0 < 0.05:
        with c._lock:
            q = len(c._q); free = len(c._free); infl = len(c._inflight)
        pend = len(getattr(c, '_pending', ()))
        if q == 0 and (not pend or free == 0 or infl >= c._max_inflight):
            return
        time.sleep(0.0002)


def settle(c, budget_s=0.5):
    """Let the manager thread finish whatever the last observe() handed it."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < budget_s:
        with c._lock:
            n = len(c._q)
        if n == 0:
            break
        time.sleep(0.0002)
    time.sleep(0.0003)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--trace", default=DEFAULT_TRACE)
    ap.add_argument("--slots", type=int, default=2056)
    ap.add_argument("--steps", type=int, default=22000)
    ap.add_argument("--drain-every", type=int, default=64)
    ap.add_argument("--registered", type=int, default=48, help="how many of the 48 MoE layers are host-tier")
    ap.add_argument("--report", type=int, default=0)
    ap.add_argument("--progress", type=int, default=0)
    ap.add_argument("--admit-probe", action="store_true",
                    help="classify every miss against the second-reference filter")
    ap.add_argument("--warm", type=int, default=6000, help="steps to skip before scoring served-h")
    ap.add_argument("--dwell", type=float, default=0.0004,
                    help="seconds of wall time the manager gets per simulated tick. A real tick is "
                         "~50 ms of TPOT, so the manager always finishes; this emulates that.")
    args = ap.parse_args()

    for k, v in (("MINISGL_EXPERT_CACHE_LOW_WATER", "25"),
                 ("MINISGL_EXPERT_CACHE_MAX_INFLIGHT", "64")):
        os.environ.setdefault(k, v)
    os.environ["MINISGL_EXPERT_CACHE_REPORT"] = str(args.report)

    mod = load_cache_module(args.src)
    hdr, recs = _read_trace(args.trace)
    E = hdr["num_experts"]
    by_step = defaultdict(list)
    for (step, uid, layer, kind, chunk, ntok, ids) in recs:
        if kind == 1:
            by_step[step].append((layer, sorted(set(ids))))
    steps = sorted(by_step)[: args.steps]
    registered = set(range(args.registered))

    probe = {"admitted": 0, "rej_first": 0, "rej_lost": 0,
             "rej_lost_after_admit": 0, "rej_lost_window": 0}
    if args.admit_probe:
        seen_miss = {}          # key -> "miss" | "admitted"
        orig = mod.ExpertResidencyCache._admit_ok
        def probed(self, key):
            prior = seen_miss.get(key)
            out = orig(self, key)
            if out:
                probe["admitted"] += 1
                seen_miss[key] = "admitted"
            else:
                if prior is None:
                    probe["rej_first"] += 1
                else:
                    probe["rej_lost"] += 1
                    if prior == "admitted":
                        probe["rej_lost_after_admit"] += 1
                    else:
                        probe["rej_lost_window"] += 1
                seen_miss[key] = "miss"
            return out
        mod.ExpertResidencyCache._admit_ok = probed

    c = build(mod, args.slots, E, registered)
    c.start()
    assert c._thread is not None, "manager thread did not start"

    served_hits = served_refs = served_wrong = served_retracting = 0
    idle_free = []          # free slots sitting unused on a tick where the manager has no demand
    episodes = []           # (promotions, evictions) delta per drain burst
    last = (0, 0)
    pending = []
    t0 = time.monotonic()
    for i, s in enumerate(steps):
        CLOCK.now = i
        c.apply_pending()
        quiesce(c, args.dwell)
        # what the kernel would see THIS step, from the device-visible slot_of tables
        if i >= args.warm:
            for (lid, ids) in by_step[s]:
                if lid not in registered:
                    continue
                t = c._layers[lid]["slot_of"]
                for e in ids:
                    served_refs += 1
                    slot = t.get(e)
                    if slot is not None and slot >= 0:
                        served_hits += 1
                        # WHAT THE KERNEL WOULD ACTUALLY READ: the slab row for `slot` holds
                        # whatever key the cache last copied into it.
                        own = c._key_of_slot[slot]
                        if own != lid * E + e:
                            # own >= 0: ANOTHER expert's bytes are in that slot -> wrong numbers.
                            # own == -1: the slot is queued for retraction but not yet reused, so
                            # it still holds THIS expert's bytes -> stale table, correct read.
                            if own >= 0:
                                served_wrong += 1
                            else:
                                served_retracting += 1
        with c._lock:
            if len(c._q) == 0:
                idle_free.append(len(c._free))
        if args.progress and i % args.progress == 0:
            with c._lock:
                print(f"  [i={i}] free={len(c._free)} q={len(c._q)} infl={len(c._inflight)} "
                      f"pend={len(getattr(c,'_pending',()))} prom={c.stats['promotions']} "
                      f"evic={c.stats['evictions']} t={time.monotonic()-t0:.1f}s", flush=True)
        pending.extend(by_step[s])
        if (i + 1) % args.drain_every == 0:
            for (lid, ids) in pending:
                c.observe(lid, ids)
            pending.clear()
            settle(c)
            now = (c.stats["promotions"], c.stats["evictions"])
            episodes.append((now[0] - last[0], now[1] - last[1]))
            last = now
    c.stop()
    dt = time.monotonic() - t0

    print(c.summary())
    st = c.stats
    ticks = st["ticks"]
    print(f"[replay] wall={dt:.1f}s steps={len(steps)} ticks={ticks} "
          f"served_h={served_hits/max(1,served_refs):.4f} ({served_hits}/{served_refs}) "
          f"WRONG-BYTES reads={served_wrong} ({served_wrong/max(1,served_refs):.2%} of expert "
          f"reads, {served_wrong/max(1,served_hits):.2%} of cache hits); "
          f"stale-but-correct (retract queued)={served_retracting}")
    print(f"[replay] per-tick: refs={(st['hits']+st['misses'])/ticks:.1f} "
          f"promotions={st['promotions']/ticks:.3f} evictions={st['evictions']/ticks:.3f} "
          f"deferred={st.get('deferred',0)/ticks:.1f} "
          f"admit_deferred={st.get('admit_deferred',0)/ticks:.1f} "
          f"throttled={st.get('throttled',0)/ticks:.3f} "
          f"abandoned={st.get('abandoned',0)/ticks:.3f} "
          f"stale_pub={st.get('stale_publishes_dropped',0)/ticks:.3f}")
    # THE ONE CORRECTNESS INVARIANT, checked against the DEVICE-VISIBLE tables: no two experts may
    # be published into the same slot, and a published slot must be the one the cache claims.
    bad = []
    owner = {}
    published = 0
    dangling = 0
    # A slot sitting in _to_retract is NOT yet reusable (apply_pending un-publishes it before it
    # reaches _free), so a table entry pointing at one still points at that expert's OWN bytes.
    inflight_retract = {slot for _v, slot in c._to_retract}
    for lid in sorted(registered):
        t = c._layers[lid]["slot_of"]
        for e in range(E):
            slot = t.get(e)
            if slot is None or slot < 0:
                continue
            key = lid * E + e
            published += 1
            if c._slot_of_key.get(key) != slot and slot not in inflight_retract:
                dangling += 1
            if slot in owner:
                bad.append(("double-claimed slot", slot, owner[slot], key))
            owner[slot] = key
            if c._key_of_slot[slot] != key and slot not in inflight_retract:
                bad.append(("slot_of disagrees with _key_of_slot", slot, key, c._key_of_slot[slot]))
    print(f"[replay] INVARIANT: slot_of publishes {published} experts over {len(owner)} distinct "
          f"slots (cache claims {len(c._slot_of_key)} resident); DANGLING (table says resident, "
          f"cache does not)={dangling}; violations={len(bad)} {bad[:2]}")
    if idle_free:
        h2 = idle_free[len(idle_free)//2:]
        print(f"[replay] IDLE-MANAGER TICKS (queue empty at the step boundary): "
              f"{len(h2)} of {len(steps)//2} 2nd-half ticks, free slots parked on them: "
              f"mean={sum(h2)/len(h2):.2f} max={max(h2)}")
    if args.admit_probe:
        tot = sum(v for k, v in probe.items() if k in ("admitted", "rej_first", "rej_lost"))
        print(f"[admit-probe] misses={tot} admitted={probe['admitted']} "
              f"({probe['admitted']/max(1,tot):.1%}) rejected_first_sighting={probe['rej_first']} "
              f"({probe['rej_first']/max(1,tot):.1%}) rejected_credit_lost={probe['rej_lost']} "
              f"({probe['rej_lost']/max(1,tot):.1%}) "
              f"[of which window-aged={probe['rej_lost_window']} "
              f"consumed-by-a-deferred-promote={probe['rej_lost_after_admit']}]")
    tail = episodes[len(episodes)//2:]
    if tail:
        pr = [a for a, _ in tail]; ev = [b for _, b in tail]
        print(f"[replay] per-DRAIN-EPISODE (2nd half, n={len(tail)}): "
              f"promotions mean={sum(pr)/len(pr):.2f} min={min(pr)} max={max(pr)} | "
              f"evictions mean={sum(ev)/len(ev):.2f} min={min(ev)} max={max(ev)}")


if __name__ == "__main__":
    main()
