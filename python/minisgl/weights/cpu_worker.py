"""The CPU-tier EXECUTOR: the asynchronous handoff between the GPU forward and the AVX-512 cores.

`cpu_tier` owns the RULES (which layers, what they cost, what orderings are legal). This module
owns the MECHANISM: a persistent worker thread, a submit/join interface that obeys
`cpu_tier.HandoffLedger`, and the backends that actually compute an expert MLP.

WHY A PERSISTENT THREAD AND NOT `concurrent.futures.ThreadPoolExecutor`
    Three reasons, all measured or structural:
      * The native core keeps its OWN pinned pool of T worker threads (`tools/cpu_moe/`, measured:
        linear to ~6 threads, and it goes BACKWARDS above 6 under a live serve because there are no
        free cores to take). A futures pool on top would be a second, competing scheduler.
      * The GIL. The submitting thread is the engine's forward thread; it must not be the thread
        that spins. A single dispatcher thread that releases the GIL for the duration of the native
        call is the whole requirement, and one thread is enough because the parallelism lives
        INSIDE the call.
      * Determinism of the ordering rules. `HandoffLedger` is a state machine with checkable
        invariants; wrapping a `Future` would hide the SEALED edge (see `HandoffState`), which is
        the one that distinguishes "the worker read a staging buffer the GPU had already recycled"
        from correct behaviour, and which produces plausible numbers rather than a crash.

THE THREE THINGS THAT CROSS THE PCIe BUS PER CPU LAYER, AND NOTHING ELSE
    down   hidden state, (M, 2560) bf16          5 KB at M=1
    down   the route, (M, top_k) int32 + f32     80 B at M=1, top_k=10
    up     the expert partial, (M, 2560) f32     10 KB at M=1
    That is ~15 KB against the 30.7 MB of expert weights a streamed layer moves — four orders of
    magnitude, which is the entire thesis of the CPU tier.

WHAT IS NOT HERE
    The native backend's `.so`. `tools/cpu_moe/` built and validated the core as an EXECUTABLE
    (`cpu_moe_layer`), not a shared library, so `NativeBackend` declares the exact ABI it needs and
    refuses to load rather than guessing. `ReferenceBackend` is a numpy float64 implementation of
    the same math — it is what the correctness tests compare the fixture's stored float64 reference
    against, and it is 3 orders of magnitude too slow to serve. Neither one is a stub that silently
    returns zeros: `ReferenceBackend` is exact and slow, `NativeBackend` is fast and absent.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from .cpu_tier import CpuHandoff, CpuTierError, HandoffError, HandoffLedger

__all__ = [
    "ExpertBlock",
    "ReferenceBackend",
    "NativeBackend",
    "CpuMoEWorker",
    "e2m1_lut",
    "e4m3_to_float",
    "dequant_nvfp4",
    "expert_mlp_f64",
]

# OCP E2M1 codebook indexed by the raw 4-bit code (bit 3 = sign). Must match
# tools/cpu_moe/wload.hpp::kE2M1, python/minisgl/quant/mxfp4.py::FP4_E2M1_LUT and the HIP kernel's
# e2m1_to_e4m3 table. One definition per process; a second copy is how the three drift.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def e2m1_lut():
    import numpy as np

    return np.array(_E2M1, dtype=np.float64)


def e4m3_to_float():
    """256-entry float64 table for float8_e4m3fn (bias 7, no inf, 0x7F/0xFF = NaN).

    Exactly `tools/cpu_moe/wload.hpp::build_e4m3_lut` with `global_mul = 1`. The NaN slot maps to
    0.0, which is what the C core does and which is safe here for the same reason: a real
    `weight_scale` never contains it, and a table lookup that produced NaN would poison a whole
    expert's output silently.
    """
    import numpy as np

    out = np.zeros(256, dtype=np.float64)
    for b in range(256):
        sign = -1.0 if (b & 0x80) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = (m / 8.0) * 0.015625  # 2**-6
        elif e == 15 and m == 7:
            v = 0.0  # NaN slot
        else:
            v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        out[b] = sign * v
    return out


def dequant_nvfp4(codes, scales, global_scale: float, n: int, k: int, group: int = 16):
    """(N, K) float64 weights from NVFP4 bytes, in the checkpoint's OWN convention.

        W[n, k] = E2M1[code] * e4m3(scale[n, k//16]) * global_scale

    NOTE THE SIGN OF THE GLOBAL FOLD. `python/minisgl/quant/nvfp4.py` documents and implements
    llm-compressor's `weight_global_scale = 448*6/amax`, a DIVISOR. This checkpoint spells the
    tensor `weight_scale_2` and stores its RECIPROCAL, i.e. a MULTIPLIER. Using the documented sign
    gives |W| ~ 4.2e6 instead of rms 0.0135. Pinned by four independent checks — see
    `tools/cpu_moe/RESULTS_KERNEL_2026-09-04.txt` §0 and `tools/cpu_moe/make_fixture.py`. It is not
    a live bug (config.py already ignores the `.weight_scale_2` name, so these experts never reach
    the NVFP4 fold path), but it IS the reason this function multiplies.

    `codes` is uint8 (N, K/2) with the LOW nibble holding the LOWER k index.
    """
    import numpy as np

    codes = np.asarray(codes, dtype=np.uint8).reshape(n, k // 2)
    scales = np.asarray(scales, dtype=np.uint8).reshape(n, k // group)
    lut = e2m1_lut()
    lo = lut[codes & 0x0F]
    hi = lut[codes >> 4]
    w = np.empty((n, k), dtype=np.float64)
    w[:, 0::2] = lo
    w[:, 1::2] = hi
    s = e4m3_to_float()[scales] * float(global_scale)
    return w * np.repeat(s, group, axis=1)


@dataclass(frozen=True)
class ExpertBlock:
    """One expert's three matrices, already dequantized to float64. The REFERENCE representation.

    Deliberately not the resident representation: a real CPU tier holds packed E2M1 + e4m3 scale
    bytes (2,764,800 B/expert) and dequantizes inside the inner loop. This is the oracle, so it
    trades 16x the memory for being obviously correct.
    """

    gate: Any  # (I, H)
    up: Any  # (I, H)
    down: Any  # (H, I)

    @property
    def hidden(self) -> int:
        return int(self.gate.shape[1])

    @property
    def intermediate(self) -> int:
        return int(self.gate.shape[0])


def expert_mlp_f64(x, block: ExpertBlock):
    """SiLU-gated expert MLP in float64: `down @ (silu(gate @ x) * (up @ x))`.

    Same order of operations as `tools/cpu_moe/moe_core.hpp`'s driver and as
    `MoELayer`'s `silu_and_mul` convention (w13 = gate | up, w2 = down), because a reference that
    reassociates is a reference that disagrees at 1e-7 for reasons that are not bugs.
    """
    import numpy as np

    g = block.gate @ x
    u = block.up @ x
    act = g / (1.0 + np.exp(-g)) * u
    return block.down @ act


class ReferenceBackend:
    """Exact, slow, float64. The correctness ORACLE, and the only backend that needs no build.

    `compute(x, ids, weights) -> partial` computes `sum_j weights[j] * E_{ids[j]}(x)` over the
    routed slots, which is exactly the contribution the GPU would have produced for the same slots.
    Summation is in SLOT ORDER, not sorted, so it is bit-reproducible across runs and across thread
    counts (the native core's two-phase disjoint-row partition gives the same guarantee — measured
    bit-identical at T=1/4/8).
    """

    name = "reference-f64"

    def __init__(self, experts: Sequence[ExpertBlock]) -> None:
        if not experts:
            raise CpuTierError("ReferenceBackend needs at least one expert")
        self._experts = tuple(experts)

    @property
    def num_experts(self) -> int:
        return len(self._experts)

    def compute(self, x, ids: Sequence[int], weights: Sequence[float]):
        import numpy as np

        ids = [int(i) for i in ids]
        weights = [float(w) for w in weights]
        if len(ids) != len(weights):
            raise CpuTierError(
                f"route mismatch: {len(ids)} expert ids vs {len(weights)} routing weights. A "
                f"truncated pair would silently drop experts from the sum."
            )
        bad = [i for i in ids if not (0 <= i < len(self._experts))]
        if bad:
            raise CpuTierError(
                f"expert id(s) {bad} out of range for {len(self._experts)} LOCAL experts. Under EP "
                f"the ids must already be remapped to the local shard "
                f"(`ExpertStackTable.local_view`'s convention); a global id here would index "
                f"another rank's expert and produce plausible numbers."
            )
        xf = np.asarray(x, dtype=np.float64).reshape(-1)
        out = np.zeros(self._experts[0].hidden, dtype=np.float64)
        for e, w in zip(ids, weights):
            if w == 0.0:
                # A zero-weight slot is the MASK convention (`MoELayer`'s adaptive-K path, and the
                # GPU/CPU route split in SPLIT mode): it must contribute exactly nothing, and it
                # must not cost a GEMM either.
                continue
            out += w * expert_mlp_f64(xf, self._experts[e])
        return out


class NativeBackend:
    """ctypes binding to the AVX-512 core. Declares its ABI; REFUSES to load rather than guess.

    The ABI, which `tools/cpu_moe/cpu_moe_layer.cpp` must be built into a `.so` to expose:

        int cpu_moe_layer_f32(
            const void*  table,      // packed resident slab, E x 2,764,800 B (e4m3 policy)
            const float* lut256,     // e4m3 byte -> float, global scale already folded
            const float* x,          // (H,) fp32 activation
            const int32_t* ids,      // (top_k,) LOCAL expert ids
            const float* rweights,   // (top_k,) routing weights
            int32_t top_k, int32_t hidden, int32_t inter,
            float* out,              // (H,) fp32 partial, WRITTEN not accumulated
            int32_t threads);

    `out` is WRITTEN, never accumulated into: an accumulating ABI makes a double-join or a retried
    handoff silently double-count, and a partial sum in the residual stream is fluent and wrong.
    """

    name = "native-avx512"

    def __init__(self, so_path: str, table_ptr: int, lut_ptr: int, num_experts: int,
                 hidden: int, inter: int, threads: int) -> None:
        import ctypes

        try:
            lib = ctypes.CDLL(so_path)
        except OSError as exc:  # pragma: no cover - depends on a build that does not exist yet
            raise CpuTierError(
                f"cannot load the CPU MoE core from {so_path!r}: {exc}. tools/cpu_moe/ currently "
                f"builds an EXECUTABLE (cpu_moe_layer), not a shared library. Build "
                f"cpu_moe_layer.cpp with -shared -fPIC and export `cpu_moe_layer_f32` with the ABI "
                f"in NativeBackend's docstring. There is deliberately no fallback: a backend that "
                f"quietly degraded to the float64 reference would turn a 0.5 ms layer into a "
                f"multi-second one and read as a hang, not as a missing build."
            ) from exc
        self._fn = lib.cpu_moe_layer_f32
        self._fn.restype = ctypes.c_int
        self._fn.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ]
        self._table = table_ptr
        self._lut = lut_ptr
        self.num_experts = int(num_experts)
        self._hidden = int(hidden)
        self._inter = int(inter)
        self._threads = int(threads)

    def compute(self, x, ids, weights):  # pragma: no cover - needs the .so
        import ctypes

        import numpy as np

        xf = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1))
        idi = np.ascontiguousarray(np.asarray(ids, dtype=np.int32))
        rwf = np.ascontiguousarray(np.asarray(weights, dtype=np.float32))
        out = np.zeros(self._hidden, dtype=np.float32)
        fp = ctypes.POINTER(ctypes.c_float)
        ip = ctypes.POINTER(ctypes.c_int32)
        rc = self._fn(
            ctypes.c_void_p(self._table), ctypes.cast(self._lut, fp),
            xf.ctypes.data_as(fp), idi.ctypes.data_as(ip), rwf.ctypes.data_as(fp),
            len(idi), self._hidden, self._inter, out.ctypes.data_as(fp), self._threads,
        )
        if rc != 0:
            raise CpuTierError(f"cpu_moe_layer_f32 returned {rc}")
        return out


class CpuMoEWorker:
    """One persistent dispatcher thread executing CPU-tier layers, under `HandoffLedger`'s rules.

    Lifecycle: `start()` -> per step `begin_step()` -> per CPU layer `submit()` ... `join()` ->
    `barrier()` at every captured-graph segment boundary and at the end of the step -> `stop()`.

    `submit` takes the activation and the route ALREADY COPIED to the host — the caller is
    responsible for the D2H and for OBSERVING its completion event before calling, which is what
    `CpuHandoff.seal_input` records. The worker never touches a device tensor and never calls a HIP
    API; that is what lets the whole class be exercised, exactly as the engine uses it, on a box
    with no card.
    """

    def __init__(
        self,
        backend: Any,
        *,
        max_inflight: int = 1,
        name: str = "cpu-moe",
        join_timeout_s: float = 30.0,
    ) -> None:
        self._backend = backend
        self._ledger = HandoffLedger(max_inflight=max_inflight)
        self._q: "queue.Queue[Optional[CpuHandoff]]" = queue.Queue()
        self._done: dict[int, threading.Event] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._name = name
        self._join_timeout_s = float(join_timeout_s)
        self._stopped = False
        self.layers_computed = 0
        self.compute_seconds = 0.0

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self) -> "CpuMoEWorker":
        if self._thread is not None:
            raise CpuTierError("worker already started")
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        return self

    def stop(self, *, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        self._ledger.barrier("worker stop")
        self._stopped = True
        self._q.put(None)
        self._thread.join(timeout=timeout)
        self._thread = None

    def __enter__(self) -> "CpuMoEWorker":
        return self.start()

    def __exit__(self, *exc) -> None:
        # Drain rather than assert on the way out of an exception: `barrier()` in `stop()` would
        # otherwise mask the original traceback with an ordering complaint.
        with self._lock:
            pending = [h for h in self._ledger._inflight]  # noqa: SLF001 - teardown only
        for h in pending:
            self._done.get(h.seq, threading.Event()).wait(timeout=self._join_timeout_s)
        try:
            self._ledger._inflight.clear()  # noqa: SLF001
        finally:
            self._stopped = True
            if self._thread is not None:
                self._q.put(None)
                self._thread.join(timeout=5.0)
                self._thread = None

    # -- the step protocol ---------------------------------------------------------------------
    def begin_step(self, step: int | None = None) -> int:
        return self._ledger.begin_step(step)

    def barrier(self, where: str = "graph-segment boundary") -> None:
        self._ledger.barrier(where)

    @property
    def ledger(self) -> HandoffLedger:
        return self._ledger

    @property
    def backend(self):
        """The executor. Public because BOOT has to reach it: the placement path registers each
        CPU layer's tiled tensors with the backend as that layer is baked, and the counters the
        serve reports (`NativeVnniBackend.counters`) live on it, not on the ledger."""
        return self._backend

    def submit(self, layer_index: int, x, ids, weights) -> CpuHandoff:
        """Hand one CPU layer's work to the worker. NON-BLOCKING; the GPU proceeds meanwhile.

        `x`, `ids` and `weights` must be host copies the caller will not mutate before `join`. The
        contract is enforced by `seal_input` being a separate edge: a caller that submits a live
        staging buffer and recycles it has violated a checked state, not merely a comment.
        """
        if self._thread is None:
            raise CpuTierError("worker not started")
        h = self._ledger.submit(layer_index, payload=(x, ids, weights))
        self._done[h.seq] = threading.Event()
        h.seal_input()
        self._q.put(h)
        return h

    def join(self, handoff: CpuHandoff, *, timeout: float | None = None):
        """Block until this layer's partial is ready, then consume it exactly once (FIFO)."""
        ev = self._done.get(handoff.seq)
        if ev is None:
            raise HandoffError(f"unknown handoff seq={handoff.seq}")
        if not ev.wait(timeout=self._join_timeout_s if timeout is None else timeout):
            raise HandoffError(
                f"CPU MoE handoff seq={handoff.seq} (layer {handoff.layer_index}) did not complete "
                f"within {self._join_timeout_s}s. The GPU is now blocked on a CPU partial that is "
                f"not coming; this is a hang, and it is reported as one rather than being papered "
                f"over with a zero partial."
            )
        try:
            return self._ledger.join(handoff)
        finally:
            self._done.pop(handoff.seq, None)

    # -- the thread ------------------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            t0 = time.perf_counter()
            try:
                x, ids, weights = item.payload  # type: ignore[misc]
                item.start()
                # WHICH LAYER, for a backend that is a TABLE of them.
                #
                # The declared single-slab ABI encoded the layer in the expert ids (each CPU layer
                # occupied `num_experts` consecutive slots of one packed table). The engine's real
                # layout cannot: it holds each layer as its own pair of stacked tensors, so the
                # backend has to be told. `item.layer_index` is the seam's `backend_expert_offset`,
                # i.e. exactly the plan-order running expert count the single-slab design used as
                # its base index — the same key, passed explicitly instead of by arithmetic.
                #
                # Opt-in ON THE BACKEND rather than a signature change, because `ReferenceBackend`
                # is the float64 correctness oracle every seam/handoff test drives and it genuinely
                # holds one layer's experts.
                out = (self._backend.compute(x, ids, weights, layer=item.layer_index)
                       if getattr(self._backend, "per_layer", False)
                       else self._backend.compute(x, ids, weights))
                item.finish(out)
                self.layers_computed += 1
            except BaseException as exc:  # noqa: BLE001 - re-raised at join()
                try:
                    if item.state.name == "SEALED":
                        item.start()
                    item.fail(exc)
                except HandoffError:
                    pass
            finally:
                self.compute_seconds += time.perf_counter() - t0
                ev = self._done.get(item.seq)
                if ev is not None:
                    ev.set()


def split_route_for_cpu(
    ids: Sequence[int],
    weights: Sequence[float],
    is_cpu_expert: Callable[[int], bool],
):
    """SPLIT mode's route partition: (gpu_ids, gpu_weights, cpu_ids, cpu_weights).

    The GPU route keeps FULL LENGTH and the CPU-assigned slots are neutralised with the convention
    `MoELayer`'s adaptive-K path already uses and already documents: **weight 0, and the id
    replaced by a slot the row KEEPS**. Both halves matter.

      * weight 0 makes the slot contribute exactly zero to the combine.
      * aliasing the id to a kept expert (rather than leaving the CPU expert's id, or using a -1
        sentinel) is what stops the slot adding an expert to the layer's expert UNION, which is
        what the grouped kernel's cost is actually a function of. And -1 is not available:
        `moe_align` does `atomicAdd(&cnt[topk_ids[t]], 1)` with no bounds check, so a negative id
        is an out-of-bounds shared-memory write.
      * a row whose slots are ALL CPU-assigned has no kept expert to alias to, so slot 0's id is
        left as-is with weight 0. That is safe (weight 0 contributes nothing) and it is why the
        aliasing is a best-effort cost optimisation rather than a correctness mechanism.

    Requires NO kernel change on either side, which is the reason SPLIT mode is expressible at all.
    """
    ids = [int(i) for i in ids]
    weights = [float(w) for w in weights]
    if len(ids) != len(weights):
        raise CpuTierError(f"route mismatch: {len(ids)} ids vs {len(weights)} weights")
    cpu_slots = [j for j, e in enumerate(ids) if is_cpu_expert(e)]
    kept = [j for j in range(len(ids)) if j not in set(cpu_slots)]
    alias = ids[kept[0]] if kept else ids[0]
    gpu_ids = list(ids)
    gpu_w = list(weights)
    for j in cpu_slots:
        gpu_ids[j] = alias
        gpu_w[j] = 0.0
    return gpu_ids, gpu_w, [ids[j] for j in cpu_slots], [weights[j] for j in cpu_slots]
