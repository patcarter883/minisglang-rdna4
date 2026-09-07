"""The THIRD placement tier: experts that are COMPUTED on the CPU instead of streamed to the GPU.

WHY THIS EXISTS
    The shipped placement model has two tiers and one mechanism. A MoE layer is either DEVICE
    (weights in VRAM, read at 692 GB/s) or HOST (weights in a pinned `hipHostMalloc(...Mapped)`
    arena, read BY THE GPU KERNEL across PCIe at 28.93 GB/s on card 0 and 14.48 GB/s on card 1).
    Both tiers move the WEIGHTS to the compute. At 37 offloaded layers x 10 experts x 3.072 MB that
    is 1.137 GB per token over a link whose slow half is 14.48 GB/s, which is where the measured
    11.85 tok/s comes from.

    llama.cpp's `-ncmoe` does the structurally different thing: it moves the COMPUTE to the weights.
    This module is the placement-model half of that.

THIS FILE WAS REWRITTEN AGAINST THE *SECOND* KERNEL MEASUREMENT, NOT THE FIRST
    The first version of this module was designed against the fp32-activation AVX-512 core, whose
    honest summary is "it reaches the DDR wall only by spending 6-16 threads". Every design choice
    that followed from that — `default_threads=6`, "the CPU tier substitutes for the GPU", the claim
    that a contiguous block is capture-cheap — is either wrong or unnecessary now. The int8/VNNI
    core (`tools/cpu_moe/RESULTS_VNNI_2026-09-04.txt`) reaches **27.38 GB/s on ONE physical core and
    saturates the DDR bus at three**, so the tier finally fits inside the core budget a live serve
    leaves free. Three corrections are baked into this file and each has a measurement behind it:

      1. `default_threads` is 2 PER RANK, not 6 (and the budget is checked in PHYSICAL cores).
      2. The DDR bus is ZERO-SUM with PCIe and the projection now says so. `project_cpu_tier`
         used to add a PCIe term and a DDR term as if they were independent resources; they are
         not, and a SPLIT-mode plan costed that way over-promises by exactly the amount the bus is
         oversubscribed.
      3. `graph_segments` used to count boundaries BETWEEN layers. The cut is INSIDE a layer — the
         attention, the norms and the router of a CPU-MoE layer are still GPU work — so K CPU
         layers cost K+1 device segments whether or not they are contiguous. Contiguity buys
         nothing for capture. That was the stated reason `assign_cpu_block` takes a contiguous
         block, and it was wrong.

THE ACCURACY DECISION, STATED WHERE THE PLANNER CAN SEE IT
    The int8-activation core costs 8.3e-03 rel_rms on a full real-checkpoint layer, against the
    fp32 core's 2.4e-07. That is ~34,000x, and it is IRREDUCIBLE for int8 (16 roughly-Gaussian
    activations give amax ~ 2 sigma, so the step is 2 sigma/127; a layer quantizes twice).
    It is NOT, however, the decision-relevant comparison. The GPU path this offloads FROM already
    serves these experts with PER-TOKEN FP8 E4M3 activations (`quant/method.py:440`), measured on
    the same fixture at 4.09e-02. So a layer moved to the VNNI core is **4.4-4.9x more accurate
    than the layer that is served today**, and the outlier in the system is the fp32 CPU core,
    which is more accurate than anything else in the serving path. `CpuActPolicy` carries both
    numbers so a plan cannot quote the speed without carrying the error.

THE TWO MODES, AND WHY THE DEFAULT IS THE SERIAL ONE
    The residual stream is sequential: layer L+1's input is layer L's output, and layer L+1's
    ROUTE is a function of that input. So there is no cross-layer prefetch and no cross-layer
    overlap available at batch size 1 — which is the operating point this whole feature exists for.

    BLOCK  Whole layers are computed on the CPU, serially with the GPU: the GPU idles while the
           CPU MoE runs. It STILL wins, and by a lot, because a CPU layer is cheaper than a
           streamed layer even with zero overlap:
               streamed (card 1, quiet)   10 x 3.072 MB / 14.48 GB/s = 2.12 ms
               streamed (card 1, loaded)  10 x 3.072 MB / 12.36 GB/s = 2.49 ms
               CPU/VNNI, 4 threads node   10 x 2.7648 MB / 53.50 GB/s = 0.517 ms
           i.e. 4.1-4.8x per layer with no concurrency required and nothing unmeasured in it.

    SPLIT  Each layer's top-k expert set is PARTITIONED between GPU and CPU and the two partials
           summed — the only genuine bs=1 concurrency there is. And it is worth almost nothing
           here, which is a *measured* conclusion rather than a preference. See `split_speedup`:
           because DDR and PCIe are one bus, the best a split can do is
               min(pcie_gbps + cpu_gbps, ddr_wall_gbps) / cpu_gbps
           and at the operating point (cpu 53.5, pcie 12.4, wall 57.0) that is **1.065 — a 6.5%
           gain**, worth ~1.2 ms of a ~40 ms step. Against it: one extra host fence INSIDE every
           layer, the unmeasured handoff on the critical path rather than beside it, and the loss
           of even the possibility of segmented capture. BLOCK is the default and SPLIT is
           opt-in-and-gated.

    A third mechanism — pipelining ACROSS the decode batch, so the CPU runs micro-batch A's layer L
    while the GPU runs micro-batch B's layer L — is deliberately NOT implemented and should not be
    proposed without a new measurement. It yields exactly zero at bs=1 (the operating point), it
    requires the scheduler to split a decode batch into offset halves (every attention kernel then
    runs at half the batch width, and this engine's decode is already 71% busy in-kernel), and it
    converts a throughput win into an inter-token-latency loss on every sequence. `CpuTierMode` has
    no member for it on purpose.

CAPACITY — THE PART THAT IS PURE ARITHMETIC AND THEREFORE TESTABLE HERE
    A CPU-tier layer consumes:
      * ZERO device bytes — same as a HOST layer.
      * ZERO PINNED-ARENA bytes — unlike a HOST layer. Its weights are read by CPU cores with
        normal loads, so they need no `hipHostMalloc`, no device-visible mapping, and no page
        pinning. Ordinary pageable host memory is enough, and pageable memory is not subject to
        `OffloadPrior.host_arena_ceiling_bytes` (P3b: 62 GiB node-wide, reached only by swapping
        114,813 pages, then derated x0.90 to 55.8 GiB). That ceiling is the binding constraint on
        the shipped plan — 37 host layers at TP=2 is 54.2 GiB node-wide against 55.8 — so moving
        layers to this tier is the only lever that relieves it without surrendering KV.
      * 10% FEWER resident bytes than the same layer on the GPU tier, because the CPU core reads
        the checkpoint's own e4m3 group-scale byte (2,764,800 B/expert) where the GPU path holds
        an fp16-folded scale (3,072,000 B/expert). Measured, not assumed: the e4m3 policy scores
        rel_rms 2.0e-07 and the fp16 policy 3.6e-04 on the same tensors, so the smaller layout is
        also the ~1800x more accurate one — and, after the specialised positive-normal decode
        (RESULTS_VNNI section 4), the faster one wherever DDR binds.

    `cpu_layout_fraction` is 0.9 EXACTLY for this checkpoint (2764800/3072000) and defaults to 1.0
    for every other format, because it is a property of the scale encoding and nothing else.
    A plan may only claim the 0.9 when a repacker is actually wired — see
    `CpuTierPrior.layout_fraction_for`.

NO TORCH, NO HIP, NO /proc. This module is in the same torch-free planning layer as `chunk_plan`,
`host_capacity`, `prior`, `placement` and `stacks`: the byte arithmetic and the ordering rules are
the parts that can be silently wrong, so they are the parts that must be testable on a box with no
card and no working torch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Sequence, Tuple

GiB = 1 << 30
GB = 1_000_000_000

__all__ = [
    "CpuTierMode",
    "CpuTierError",
    "CpuActPolicy",
    "ACT_FP32",
    "ACT_VNNI_INT8",
    "ACT_POLICIES",
    "CoreBudget",
    "CORE_BUDGET",
    "CpuTierPrior",
    "CPU_TIER_PRIOR",
    "CpuTierAssignment",
    "CpuTierProjection",
    "assign_cpu_block",
    "cpu_layer_ms",
    "ddr_gbps",
    "ddr_share",
    "split_speedup",
    "graph_segments",
    "project_cpu_tier",
    "HandoffState",
    "HandoffError",
    "CpuHandoff",
    "HandoffLedger",
]


class CpuTierError(RuntimeError):
    """A CPU-tier placement or accounting request that cannot be satisfied."""


class CpuTierMode(Enum):
    """How CPU-computed experts are scheduled against the GPU.

    OFF     no CPU tier; the plan is exactly the shipped two-tier plan.
    BLOCK   whole layers on the CPU, serial with the GPU. The default.
    SPLIT   per-layer expert-set partition, concurrent with the GPU. Opt-in and gated.

    There is deliberately NO batch-pipelined member — see the module docstring.
    """

    OFF = "off"
    BLOCK = "block"
    SPLIT = "split"

    @property
    def is_concurrent(self) -> bool:
        return self is CpuTierMode.SPLIT

    @property
    def is_capturable(self) -> bool:
        """Can this mode's forward be captured as ONE `torch.cuda.CUDAGraph`? Only OFF.

        NOT a hedge and NOT "BLOCK is nearly capturable". `engine/graph.py` captures the WHOLE
        model forward per batch-size bucket into a single `CUDAGraph` and replays it with one
        `g.replay()`. A host call anywhere inside is not a capturable node, so ANY CPU-tier layer
        makes that single-graph capture impossible in both modes. What BLOCK additionally offers
        is that the forward is *expressible* as an alternating sequence of device segments and
        eager host calls — see `graph_segments`, which counts them, and which is a statement about
        a segmented replay the engine DOES NOT HAVE YET, not about today's capture path.
        """
        return self is CpuTierMode.OFF


# ────────────────────────────────────────────────────────────────────────────────────────────────
# Activation-format policies — the WLoad idea, applied to the ACTIVATION side
# ────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CpuActPolicy:
    """One measured CPU expert core: its thread curve AND the error it costs. Never one without the other.

    `KERNEL_CORE_POLICY.md` says a new WEIGHT format is a WLoad policy on the shared core. The
    activation format is the one axis that genuinely bought a second core (`VPDPBUSD` puts ROWS in
    lanes, so k cannot be the lane index and every line of the inner loop differs — RESULTS_VNNI
    section 3 argues the exemption). Both cores are still format-parameterised over the same
    WLoad policies, and both are represented here as a policy so a plan chooses between them by
    name instead of by which binary happened to be built.

    `gbps_by_threads` is a NODE-WIDE AGGREGATE keyed by TOTAL threads across all ranks, in weight
    bytes consumed per second. It is not per-core and it is not per-rank: two ranks running two
    threads each contend for one DDR bus, so the only sound lookup is by the total.
    """

    name: str
    wload: str                      # the tools/cpu_moe policy name this row was measured with
    gbps_by_threads: Mapping[int, float]
    gbps_by_threads_p5: Mapping[int, float]
    rel_rms: float                  # measured against the float64 make_fixture reference, L0
    rel_rms_range: Tuple[float, float]  # across L0-L3
    saturates_at_threads: int       # first measured thread count within 5% of this core's own max
    bytes_per_weight: float
    provenance: str

    def gbps(self, threads: int, *, p5: bool = False) -> float:
        """Measured aggregate at `threads`. Resolves DOWNWARD; interpolates nothing.

        Both curves are non-monotonic past their knee (fp32 measured slower at 12 than at 8 on a
        quiet box; VNNI slower at 16 than at 8), so resolving upward would invent throughput the
        box does not have.
        """
        if threads <= 0:
            raise CpuTierError(f"threads must be >= 1, got {threads}")
        table = self.gbps_by_threads_p5 if p5 else self.gbps_by_threads
        keys = sorted(k for k in table if k <= threads)
        if not keys:
            raise CpuTierError(
                f"{self.name}: no measurement at or below {threads} thread(s); measured points "
                f"are {sorted(table)}"
            )
        return float(table[keys[-1]])

    def threads_for(self, target_gbps: float, *, max_threads: int) -> int:
        """Fewest measured thread counts that reach `target_gbps`. Raises if none does.

        The core-budget question in its useful direction: not "how fast is T threads" but "how
        many cores must I take from the engine to hit the number the token budget needs".
        """
        for t in sorted(k for k in self.gbps_by_threads if k <= max_threads):
            if self.gbps_by_threads[t] >= target_gbps:
                return t
        raise CpuTierError(
            f"{self.name} cannot reach {target_gbps:.1f} GB/s within {max_threads} thread(s); its "
            f"best measured point at or below that is "
            f"{max((v for k, v in self.gbps_by_threads.items() if k <= max_threads), default=0.0):.1f}"
            f" GB/s"
        )


# The fp32-activation core. THE ORACLE: nothing else in the serving path is this accurate, and it
# is what every correctness comparison in tools/cpu_moe is made against. Curve is the quiet-box
# median from RESULTS_VNNI section 1 (the paired, interleaved run — NOT the earlier standalone
# table, which was taken in a different box condition).
ACT_FP32 = CpuActPolicy(
    name="fp32",
    wload="nvfp4_e4m3_g16",
    gbps_by_threads={1: 8.78, 2: 17.02, 3: 24.96, 4: 33.64, 6: 46.39, 8: 42.71, 16: 55.29},
    gbps_by_threads_p5={1: 9.09, 2: 18.02, 3: 26.77, 4: 35.51, 6: 51.94, 8: 58.94, 16: 60.09},
    rel_rms=2.416e-07,
    rel_rms_range=(2.188e-07, 2.557e-07),
    saturates_at_threads=16,
    bytes_per_weight=0.5625,
    provenance=(
        "tools/cpu_moe/RESULTS_VNNI_2026-09-04.txt section 1, quiet box (0.43-2.25 busy hw "
        "threads of 16), 4 paired interleaved reps, 5.27 GiB DDR-resident table (56x the 96 MB "
        "V-cache), top-10, real checkpoint bytes. rel_rms vs a float64 dequant-and-GEMM over the "
        "raw safetensors (make_fixture.py), layers L0-L3. NOTE the unexplained drift flagged in "
        "RESULTS_VNNI section 2: RESULTS_KERNEL recorded 1.988e-07 for L0 with the same fixture "
        "and the same on-disk binary and it reproduced today at 2.416e-07."
    ),
)

# The int8-activation VNNI core. Saturates DDR at THREE threads where fp32 needs 8-16 — a
# core-occupancy win, not a bandwidth win: both cores end at the same ~56 GB/s wall.
ACT_VNNI_INT8 = CpuActPolicy(
    name="vnni_int8",
    wload="vnni_nvfp4_e4m3_g16",
    gbps_by_threads={1: 27.38, 2: 44.64, 3: 54.00, 4: 53.50, 6: 56.16, 8: 56.72, 16: 53.32},
    gbps_by_threads_p5={1: 32.04, 2: 51.19, 3: 60.04, 4: 63.23, 6: 64.55, 8: 64.20, 16: 63.99},
    rel_rms=8.279e-03,
    rel_rms_range=(7.637e-03, 9.077e-03),
    saturates_at_threads=3,
    bytes_per_weight=0.5625,
    provenance=(
        "tools/cpu_moe/RESULTS_VNNI_2026-09-04.txt sections 1-2, same conditions and same fixture "
        "as ACT_FP32 (paired and interleaved with it within each rep). The 8.3e-03 is IRREDUCIBLE "
        "for int8 activations, not a bug: predicted 6.4e-03 from group-16 amax quantization run "
        "twice per layer, cross-validated to five digits by an independent numpy implementation "
        "with float64 weights (act_format_error.py, 8.2786e-03 vs 8.279e-03). The GPU path this "
        "offloads FROM serves the same experts at 4.094e-02 (per-token fp8 e4m3, "
        "quant/method.py:440), so this is 4.9x MORE accurate than what is served today."
    ),
)

ACT_POLICIES: Dict[str, CpuActPolicy] = {p.name: p for p in (ACT_FP32, ACT_VNNI_INT8)}


# ────────────────────────────────────────────────────────────────────────────────────────────────
# The core budget — a FIRST-CLASS constraint, denominated in PHYSICAL cores
# ────────────────────────────────────────────────────────────────────────────────────────────────
_ENV_CORE_BUDGET = "MINISGL_CPU_MOE_CORE_BUDGET"


@dataclass(frozen=True)
class CoreBudget:
    """How many PHYSICAL cores the CPU tier may take, and from whom.

    PHYSICAL, not hardware threads, and that is a measurement rather than a convention: pinning a
    MoE worker onto the SMT sibling of a busy core cost ~50% (fp32 T1 4.06 med / 6.2 p5 on cpu9,
    sibling of a loaded cpu1, against 8.4 / 8.6 on a free physical core — RESULTS_PERCORE section
    4). A budget stated in hardware threads therefore double-counts every core and produces a plan
    that measures at half its projection.

    ARBITRATION RULE, and it is a refusal rather than a degradation. `assert_fits` RAISES when the
    request does not fit. The reason is measured: this core's thread pool uses a sense-reversing
    spin barrier, and when a worker is descheduled it loses a whole timeslice — unpinned and
    oversubscribed under load, T=16 collapsed to a FIXED 6.0013 / 6.0018 ms per layer in two
    independent runs (RESULTS_PERCORE section 4). A CPU tier that cannot get its cores does not get
    slower in proportion; it falls off a cliff and takes the scheduler's cores with it. So an
    over-budget configuration is a boot error, not a warning.

    WHICH cores, not just how many. Core 0 is the only one that boosts (4.95-5.01 GHz against
    3.95-4.07 on cores 1-7, measured with an inline-asm dependent-add chain). The VNNI kernel is
    clock-INSENSITIVE at the operating point (34.72 GB/s on core 4 vs 33.90 on core 0 — at 1T
    DDR-resident it is memory-bound), while the engine's Python forward/dispatch thread is not. So
    the CPU tier takes cores from the TOP and leaves core 0 to the engine: `core_ids()` returns the
    high-numbered physical cores. This costs the CPU tier nothing measurable and is the one piece
    of free arbitration available.
    """

    physical_cores: int = 8
    threads_per_core: int = 2  # SMT width; recorded so nothing computes a budget in hw threads
    engine_cores: float = 1.88
    os_cores: float = 0.5
    engine_cores_measured: bool = True
    provenance: str = (
        "physical_cores/threads_per_core: Ryzen 7 7800X3D 8C/16T. engine_cores: the TP=2 minisgl "
        "serve measured 0.962 + 0.922 = 1.88 cores at steady state over a 20 s window from "
        "/proc/PID/stat deltas (RESULTS_PERCORE section 6). ONE OBSERVATION, not a distribution — "
        "the workflow exited before it could be re-measured under a known decode load. os_cores is "
        "an ESTIMATE covering kworkers, kswapd and the offload driver's own copy/dispatch threads; "
        "whole-system busy including those was 4.4-4.9 of 16 hw threads."
    )

    @property
    def usable_physical(self) -> float:
        """Physical cores left for the CPU MoE tier after the engine and the OS."""
        return max(0.0, self.physical_cores - self.engine_cores - self.os_cores)

    @property
    def max_threads(self) -> int:
        """The most PINNED PHYSICAL cores the CPU tier may take, node-wide, across all ranks.

        `MINISGL_CPU_MOE_CORE_BUDGET` OVERRIDES THE DERIVED CAP, and it exists because the derived
        one is built on `engine_cores`, which `provenance` above admits is ONE OBSERVATION taken
        without a known decode load. A cap resting on a single measurement must not be the thing
        that makes the opposing measurement unrunnable — otherwise "2 threads is the maximum" is
        true only because nothing was ever allowed to test 3.

        It is an override, not a new default: unset, the refusal is exactly what it was. Set, the
        caller is deliberately overcommitting and owns §1.5's cliff (an oversubscribed pool does not
        degrade in proportion; it goes to a FIXED ~6.0 ms/layer and takes the engine's cores with
        it), which is why the sweep that uses it must read the per-layer counters and not just the
        end-to-end tok/s.
        """
        override = os.environ.get(_ENV_CORE_BUDGET)
        if override:
            n = int(override)
            if n < 1:
                raise CpuTierError(
                    f"{_ENV_CORE_BUDGET}={override!r}: the node-wide core budget must be >= 1"
                )
            return n
        return int(self.usable_physical)  # floor: a fractional core is not a core

    def assert_fits(self, total_threads: int, *, what: str = "the CPU MoE tier") -> None:
        if total_threads < 1:
            raise CpuTierError(f"{what}: total_threads must be >= 1, got {total_threads}")
        if total_threads > self.max_threads:
            raise CpuTierError(
                f"{what} asks for {total_threads} physical core(s) but only {self.max_threads} are "
                f"free: {self.physical_cores} physical - {self.engine_cores:.2f} (engine) - "
                f"{self.os_cores:.2f} (OS) = {self.usable_physical:.2f}. This is a REFUSAL and not "
                f"a clamp on purpose: the native pool's spin barrier does not degrade gracefully "
                f"when starved — unpinned and oversubscribed it measured a FIXED 6.0 ms/layer in "
                f"two independent runs, ~12x the 0.49 ms it reaches with its cores. Reduce "
                f"threads-per-rank, or reduce ranks, or measure a smaller `engine_cores`."
            )

    def core_ids(self, total_threads: int) -> Tuple[int, ...]:
        """The physical core ids to pin to, highest first-free. See the class docstring on core 0."""
        self.assert_fits(total_threads)
        return tuple(range(self.physical_cores - total_threads, self.physical_cores))

    def smt_sibling_ids(self, core_ids: Sequence[int]) -> Tuple[int, ...]:
        """The hw-thread ids that MUST NOT also be handed out — Linux numbers siblings +N."""
        return tuple(int(c) + self.physical_cores for c in core_ids)


CORE_BUDGET = CoreBudget()


# ────────────────────────────────────────────────────────────────────────────────────────────────
# The measured prior
# ────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CpuTierPrior:
    """Measured CPU-side constants for one box, with provenance. Mirrors `prior.OffloadPrior`.

    Bandwidths are WEIGHT BYTES CONSUMED per second by the expert core — the same quantity
    `OffloadPrior.host_read_gbps` measures on the PCIe side, so the two are directly comparable and
    a projection can put them in the same expression without a unit conversion. They are NODE-WIDE
    aggregates keyed by TOTAL threads; see `CpuActPolicy`.
    """

    name: str
    provenance: str

    policy: CpuActPolicy = ACT_VNNI_INT8
    oracle: CpuActPolicy = ACT_FP32
    cores: CoreBudget = field(default_factory=lambda: CORE_BUDGET)

    # --- the ZERO-SUM resource ------------------------------------------------------------------
    # Everything the CPU cores read and everything the GPU DMAs out of the pinned arena crosses the
    # SAME DDR controller. Both cores independently top out here, which is what makes it a wall and
    # not a coincidence: fp32 reaches 55.29 at 16 threads and VNNI 54.00 at 3.
    ddr_wall_gbps: float = 57.0
    # Contention, measured once, at an extreme point: a 2-thread sequential DDR stream co-load
    # taking ~93% of the bus cut VNNI T2 from 41.98 to 20.60 GB/s (-51%). A pure proportional share
    # of `ddr_wall_gbps` predicts 25.3 for that point, so proportional sharing is OPTIMISTIC by
    # 0.815x there. `ddr_share` applies this as a derate on the CONTENDED portion only. ONE
    # calibration point, at a severity the real PCIe path (capped at 12.4-14.5 GB/s) never reaches.
    ddr_contention_derate: float = 0.815
    ddr_contention_points: int = 1

    # --- layout ---------------------------------------------------------------------------------
    # Resident bytes for ONE Qwen3.8-Flash-Next expert (gate+up 2x640x2560 + down 2560x640 =
    # 4,915,200 weights) in each layout. E2M1 codes are 2,457,600 B in both; only the group scale
    # differs (e4m3 byte vs folded fp16).
    bytes_per_expert_cpu_native: int = 2_764_800
    bytes_per_expert_gpu_fp16_scale: int = 3_072_000
    layout_fraction_by_policy: Mapping[str, float] = field(
        default_factory=lambda: {
            "nvfp4_e4m3_g16": 0.9,
            "vnni_nvfp4_e4m3_g16": 0.9,
            "nvfp4_fp16_g16": 1.0,
            "vnni_nvfp4_fp16_g16": 1.0,
            "mxfp4_e8m0_g32": 1.0,
        }
    )

    # --- scheduling -----------------------------------------------------------------------------
    # PER RANK. At TP=2 the node runs 2 x this many pinned physical cores and they share one DDR
    # bus, so 2 is also the largest value that fits `CoreBudget.max_threads` (5) at two ranks — and
    # 4 total threads already reach 53.50 GB/s, 94% of the wall. 3/rank would be 6 total and does
    # not fit. This is the whole reason the tier became viable: the fp32 core needed 6.
    default_threads_per_rank: int = 2

    # --- UNMEASURED -----------------------------------------------------------------------------
    # The activation round trip per CPU layer: D2H 2560xbf16 (5 KB) + a host-visible completion
    # signal + H2D 2560xf32 (10 KB) + the fence the GPU needs to consume it. NOT MEASURED — no card
    # was available (the graph-capture workflow held both). The bracket is a PCIe-latency argument;
    # every projection states which end of it it used.
    handoff_us_bracket: Tuple[float, float] = (10.0, 40.0)
    handoff_measured: bool = False

    # --- the honesty factor ----------------------------------------------------------------------
    # `prior.project_step`'s two-tier model projects 18.6 tok/s for the plan that MEASURES 11.85
    # tok/s on this box. Ratio 0.637. It is not a fudge and it must not be applied silently: it is
    # the one place a three-tier projection can be checked against reality at all, and a report that
    # quotes the raw projection is quoting a number the same model got wrong by 1.57x on the only
    # case anybody has run. `CpuTierProjection` carries BOTH.
    projection_fidelity: float = 0.637
    projection_fidelity_points: int = 1

    def gbps(self, threads: int, *, policy: CpuActPolicy | None = None, p5: bool = False) -> float:
        return (policy or self.policy).gbps(threads, p5=p5)

    def layout_fraction_for(self, policy: str | None, *, repacked: bool) -> float:
        """Resident-byte ratio CPU layout : device layout for `policy`.

        `repacked=False` returns 1.0 unconditionally. The shrink is a property of a REPACK that
        actually ran; a plan that copies the device-layout tensors verbatim into pageable memory
        holds device-layout bytes and must be charged for them. Claiming 0.9 on the strength of the
        format alone is exactly how a capacity number becomes fiction.

        NOTE the VNNI core needs a repack anyway — its resident bytes are tiled 16 rows x 16 k,
        and ours are row-major — so on the VNNI path `repacked=True` is the normal case rather
        than an optimisation. That does not make it automatic: the caller still has to have wired
        the repacker.
        """
        if not repacked or policy is None:
            return 1.0
        return float(self.layout_fraction_by_policy.get(policy, 1.0))

    def with_overrides(self, **kw) -> "CpuTierPrior":
        from dataclasses import replace

        return replace(self, **kw)


CPU_TIER_PRIOR = CpuTierPrior(
    name="ryzen7-7800x3d-ddr5-vnni-2026-09-04",
    provenance=(
        "tools/cpu_moe/RESULTS_VNNI_2026-09-04.txt (+ RESULTS_PERCORE for the core budget and the "
        "taskset effects, RESULTS_KERNEL for the layout arithmetic), worktree minisgl-rdna4-cpumoe "
        "@ b1f52659. AVX-512 E2M1 MoE cores (moe_core.hpp + wload.hpp), g++ 16.2.1 -O3 "
        "-march=znver4, Ryzen 7 7800X3D 8C/16T 96MB V-cache, DDR5 dual channel. Bandwidths are "
        "WEIGHT BYTES consumed at top-10 over a 5.27 GiB / 2048-expert resident table (56x the "
        "V-cache), route redrawn every iteration, threads pinned to physical cores 0..T-1. "
        "ddr_wall_gbps is where BOTH cores independently top out. ddr_contention_derate is fit "
        "from ONE point and that point is a deliberate worst case. handoff_us_bracket is NOT "
        "MEASURED. projection_fidelity is the single measured-vs-projected ratio available "
        "(11.85 / 18.6 on the shipped two-tier plan). RE-MEASURE after any DDR/BIOS change; the "
        "card-1 root port is Gen4 x8 and a BIOS retrain would move the PCIe side too."
    ),
)


# ────────────────────────────────────────────────────────────────────────────────────────────────
# The zero-sum bus
# ────────────────────────────────────────────────────────────────────────────────────────────────
def ddr_share(
    cpu_gbps: float, pcie_gbps: float, *, prior: CpuTierPrior = CPU_TIER_PRIOR
) -> Tuple[float, float]:
    """Split the ONE DDR bus between concurrent CPU-core reads and concurrent GPU DMA reads.

    Every byte the GPU streams over PCIe is first read from DDR, so the two demands are not
    independent resources and a cost model that adds them as if they were over-promises by exactly
    the amount the bus is oversubscribed. When the demands fit under `ddr_wall_gbps` both are
    granted in full — the co-load measurement is unambiguous that CORE isolation is free (half a
    TFLOP of AVX-512 on the neighbouring cores cost the MoE 2%). When they do not fit, both are
    scaled proportionally and the CONTENDED result is derated by `ddr_contention_derate`, because
    the one measured contention point came in 18.5% below what proportional sharing predicts.

    Returns `(cpu_effective_gbps, pcie_effective_gbps)`.
    """
    if cpu_gbps < 0 or pcie_gbps < 0:
        raise CpuTierError(f"bandwidth demands must be >= 0, got {cpu_gbps}, {pcie_gbps}")
    total = cpu_gbps + pcie_gbps
    if total <= prior.ddr_wall_gbps or total == 0.0:
        return (cpu_gbps, pcie_gbps)
    scale = prior.ddr_wall_gbps / total * prior.ddr_contention_derate
    return (cpu_gbps * scale, pcie_gbps * scale)


def split_speedup(
    cpu_gbps: float, pcie_gbps: float, *, prior: CpuTierPrior = CPU_TIER_PRIOR
) -> float:
    """How much faster SPLIT is than BLOCK for the same set of layers. >= 1.0.

    BLOCK runs the layer's whole expert set on the CPU: time = B / cpu_gbps.
    SPLIT gives a fraction to the GPU (which streams it over PCIe) and the rest to the CPU, and the
    two run concurrently. The optimum partition equalises the two halves, so the effective rate is
    the SUM of the two rates — capped by the one bus they share:

        split_rate = min(pcie_gbps + cpu_gbps, ddr_wall_gbps)
        speedup    = split_rate / cpu_gbps

    THE POINT OF THIS FUNCTION IS THAT THE ANSWER IS SMALL. At the measured operating point
    (4 node threads = 53.50 GB/s CPU, 12.36 GB/s loaded PCIe, 57.0 GB/s wall) it is **1.065**.
    The CPU tier alone already uses 94% of the bus, so the GPU has almost nothing left to add.
    SPLIT's entire case rests on this number and it does not carry the extra in-layer host fence,
    the unmeasured handoff moving onto the critical path, or the loss of segmented capture.
    """
    if cpu_gbps <= 0:
        raise CpuTierError(f"cpu_gbps must be > 0, got {cpu_gbps}")
    return min(pcie_gbps + cpu_gbps, prior.ddr_wall_gbps) / cpu_gbps


# ────────────────────────────────────────────────────────────────────────────────────────────────
# Assignment
# ────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CpuTierAssignment:
    """Which layer INDICES go to the CPU tier, and the structural facts about that choice.

    `indices` are indices into the caller's layer sequence, not layer numbers in the model — the
    caller (`placement.plan_three_tier`) owns the mapping, exactly as `plan_layer_granular` does.
    """

    mode: CpuTierMode
    indices: Tuple[int, ...]
    contiguous: bool
    eligible: Tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.indices)

    def __contains__(self, i: int) -> bool:
        return i in self.indices


def assign_cpu_block(
    eligible: Sequence[int],
    num_cpu_layers: int,
    *,
    mode: CpuTierMode = CpuTierMode.BLOCK,
    from_end: bool = True,
) -> CpuTierAssignment:
    """Pick `num_cpu_layers` of `eligible` for the CPU tier. Deterministic, integer-only.

    CONTIGUITY IS NOT A CAPTURE ARGUMENT AND THIS DOCSTRING USED TO SAY IT WAS. Every CPU-MoE layer
    cuts the captured region *inside itself* — its attention, its norms and its router are still
    GPU work — so K CPU layers cost K+1 device segments whether they are contiguous or scattered
    (`graph_segments`). What contiguity actually buys is smaller and real:

      * ONE `HandoffLedger` epoch and one arena-free region rather than K interleaved ones, so the
        "nothing crosses a segment boundary" invariant is checked at one place;
      * the CPU worker's thread pool is entered and left once per run rather than K times, which
        matters because the pool's spin barrier is what collapses when it is repeatedly descheduled;
      * a deterministic, budget-independent set. Running the greedy device fill first and taking
        "whatever is left" would make the CPU set a function of a per-rank byte budget, which is
        exactly the class of per-rank divergence `agreement_digest` exists to catch.

    The block is taken from the DEEP end by default so the shallow layers — the ones the device
    tier's `(-priority, index)` fill prefers anyway — stay on the GPU and stay capturable as a
    prefix.

    `eligible` must be given in declaration order and must be free of duplicates.
    """
    if mode is CpuTierMode.OFF or num_cpu_layers <= 0:
        return CpuTierAssignment(CpuTierMode.OFF, (), True, tuple(eligible))
    elig = tuple(int(i) for i in eligible)
    if len(set(elig)) != len(elig):
        raise CpuTierError(f"eligible layer indices contain duplicates: {elig}")
    if list(elig) != sorted(elig):
        raise CpuTierError(
            f"eligible layer indices must be in declaration order, got {elig}. The CPU block is a "
            f"SLICE of this sequence; an unordered input would produce a different block on a rank "
            f"whose discovery walk yielded a different order, and placement must be rank-identical."
        )
    if num_cpu_layers > len(elig):
        raise CpuTierError(
            f"asked for {num_cpu_layers} CPU layers but only {len(elig)} are eligible"
        )
    chosen = elig[-num_cpu_layers:] if from_end else elig[:num_cpu_layers]
    contiguous = all(b - a == 1 for a, b in zip(chosen, chosen[1:]))
    return CpuTierAssignment(mode, tuple(chosen), contiguous, elig)


def graph_segments(num_layers: int, cpu_indices: Sequence[int], mode: CpuTierMode) -> int:
    """How many separately-captured DEVICE segments this placement forces. 1 == unbroken capture.

    THE CUT IS INSIDE THE LAYER, NOT BETWEEN LAYERS, and the previous version of this function got
    that wrong in the optimistic direction. A CPU-tier layer is not "a layer that runs on the CPU";
    it is a layer whose ATTENTION, norms, router and residual adds all still run on the GPU and
    whose expert MLP is a host call in the middle. So each CPU layer splits the device work into
    a before and an after, and:

        segments = len(cpu_indices) + 1

    unconditionally, for any placement, contiguous or not. There is always device work before the
    first MoE (embedding + attention) and always device work after the last (final norm + lm_head),
    so neither end collapses. Under the old "count maximal runs of GPU LAYERS" rule a 21-layer
    contiguous block reported 2 segments; the truth is 22.

    This is a statement about a SEGMENTED replay the engine does not have. Today `engine/graph.py`
    captures the whole forward into one `CUDAGraph` per batch-size bucket and replays it with a
    single `g.replay()`, so any value > 1 here means *this plan cannot use graph capture at all
    until that path exists*. `CpuTierMode.is_capturable` says so directly.

    SPLIT returns 0 — "not expressible as segments". It needs the host INSIDE a layer's expert
    reduction, which is not a cut but a hole. Callers must treat 0 as "eager only", never as "free".
    """
    if mode is CpuTierMode.OFF or not cpu_indices:
        return 1
    if mode is CpuTierMode.SPLIT:
        return 0
    idx = set(int(i) for i in cpu_indices)
    if min(idx) < 0 or max(idx) >= num_layers:
        raise CpuTierError(f"cpu index out of range for {num_layers} layers: {sorted(idx)}")
    return len(idx) + 1


# ────────────────────────────────────────────────────────────────────────────────────────────────
# Projection
# ────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CpuTierProjection:
    """A projected decode step for a THREE-tier plan. A PROJECTION, never a measurement."""

    step_ms: float
    tok_s: float
    compute_ms: float
    device_ms: float
    host_ms: float
    cpu_ms: float
    handoff_ms: float
    overlapped_ms: float
    mode: CpuTierMode
    threads_per_rank: int
    total_threads: int
    cpu_gbps: float
    host_gbps: float
    ddr_contended: bool
    handoff_us_used: float
    handoff_measured: bool
    num_cpu_layers: int
    policy: str
    rel_rms: float
    graph_segments: int
    # tok_s x the one measured-vs-projected ratio this model has ever been checked against.
    calibrated_tok_s: float
    projection_fidelity: float

    def describe(self) -> str:
        h = "MEASURED" if self.handoff_measured else "UNMEASURED"
        return (
            f"cpu-tier projection [{self.mode.value}/{self.policy}] {self.tok_s:.2f} tok/s raw, "
            f"{self.calibrated_tok_s:.2f} tok/s at the measured x{self.projection_fidelity:.3f} "
            f"model fidelity ({self.step_ms:.2f} ms/step = compute {self.compute_ms:.2f} + device "
            f"{self.device_ms:.2f} + host/PCIe {self.host_ms:.2f} + cpu {self.cpu_ms:.2f} + "
            f"handoff {self.handoff_ms:.2f} - overlapped {self.overlapped_ms:.2f}); "
            f"{self.num_cpu_layers} CPU layers at {self.threads_per_rank}T/rank "
            f"({self.total_threads} physical cores) = {self.cpu_gbps:.1f} GB/s"
            f"{' [DDR CONTENDED]' if self.ddr_contended else ''}, handoff "
            f"{self.handoff_us_used:.1f} us/layer [{h}], act rel_rms {self.rel_rms:.2e}, "
            f"{self.graph_segments} capture segment(s)"
        )


def cpu_layer_ms(
    active_bytes: int,
    *,
    threads: int,
    prior: CpuTierPrior = CPU_TIER_PRIOR,
    policy: CpuActPolicy | None = None,
    p5: bool = False,
) -> float:
    """Milliseconds for the CPU cores to consume `active_bytes` of expert weights at `threads`.

    `threads` is the NODE-WIDE TOTAL (see `CpuActPolicy`), and `active_bytes` must be the
    NODE-WIDE total to match — two ranks each reading their half of a layer put the whole layer
    on one DDR bus.
    """
    if active_bytes < 0:
        raise CpuTierError(f"active_bytes must be >= 0, got {active_bytes}")
    if active_bytes == 0:
        return 0.0
    return active_bytes / (prior.gbps(threads, policy=policy, p5=p5) * GB) * 1000.0


def ddr_gbps(
    threads: int, *, prior: CpuTierPrior = CPU_TIER_PRIOR, policy: CpuActPolicy | None = None
) -> float:
    return prior.gbps(threads, policy=policy)


def project_cpu_tier(
    *,
    host_bytes_per_rank: int,
    device_bytes_per_rank: int,
    cpu_bytes_per_rank: int,
    num_cpu_layers: int,
    mode: CpuTierMode,
    handoff_us_per_layer: float,
    threads_per_rank: int = CPU_TIER_PRIOR.default_threads_per_rank,
    prior: "object" = None,
    cpu_prior: CpuTierPrior = CPU_TIER_PRIOR,
    policy: CpuActPolicy | None = None,
    num_ranks: int = 1,
    loaded: bool = True,
    num_layers: int = 0,
    check_cores: bool = True,
) -> CpuTierProjection:
    """Project one decode step for a three-tier plan.

    THE CORE BUDGET IS CHECKED FIRST. `threads_per_rank x num_ranks` physical cores must fit inside
    `CoreBudget.max_threads`, and not fitting is a refusal (see `CoreBudget.assert_fits`). A
    projection that quietly assumed 6 cores it cannot have is precisely the over-promise this
    rewrite exists to remove.

    THE TWO MODES DIFFER IN WHETHER THE DDR BUS IS SHARED, WHICH IS THE WHOLE MODEL:

      BLOCK  step = floor + device + host + cpu + handoff
             Serial, so the CPU phase and the PCIe phase do NOT overlap and therefore do NOT
             contend: each gets the whole bus in its own window. Nothing is subtracted because
             nothing overlaps — layer L+1's route depends on layer L's output, so at BLOCK
             granularity there is no independent GPU work to hide the CPU behind. This is the
             default and every number in it is measured.

      SPLIT  The CPU and PCIe halves of the SAME layers run concurrently and share one DDR bus, so
             both are put through `ddr_share` first and the effective layer rate is capped at
             `ddr_wall_gbps`. `host_bytes_per_rank` must already EXCLUDE the CPU-assigned experts'
             bytes; what SPLIT adds back is the GPU streaming a *fraction* of them, which is what
             `split_speedup` prices. The `max(0, ...)` form is kept so the CPU can never "save"
             more than it costs.

    `cpu_bytes_per_rank` is PER RANK and is scaled to a node total internally, because the DDR bus
    is a node resource: at TP=2 both ranks' CPU MoE work lands on the same eight cores and the same
    memory controller. Passing a per-rank figure and dividing by a per-rank bandwidth would
    under-count the node's traffic by `num_ranks`.

    `handoff_us_per_layer` is REQUIRED and has no default. `cpu_prior.handoff_measured` is False on
    this box, so every caller has to state which end of `handoff_us_bracket` it is quoting, and the
    returned projection carries the value and the flag so a report cannot launder it.
    """
    from .prior import PHASE0_PRIOR, OffloadPrior

    p: OffloadPrior = PHASE0_PRIOR if prior is None else prior  # type: ignore[assignment]
    pol = policy or cpu_prior.policy
    if mode is not CpuTierMode.OFF and num_cpu_layers <= 0 and cpu_bytes_per_rank > 0:
        raise CpuTierError("cpu_bytes_per_rank > 0 but num_cpu_layers == 0")
    if handoff_us_per_layer < 0:
        raise CpuTierError(f"handoff_us_per_layer must be >= 0, got {handoff_us_per_layer}")
    num_ranks = max(1, int(num_ranks))
    total_threads = int(threads_per_rank) * num_ranks
    if cpu_bytes_per_rank and check_cores:
        cpu_prior.cores.assert_fits(
            total_threads,
            what=f"the CPU MoE tier at {threads_per_rank} thread(s)/rank x {num_ranks} rank(s)",
        )

    host_gbps = p.slow_host_gbps(num_ranks, loaded=loaded)
    cpu_gbps = pol.gbps(total_threads) if cpu_bytes_per_rank else 0.0
    contended = False

    if mode is CpuTierMode.SPLIT and cpu_bytes_per_rank:
        # The only mode in which the two buses are the same bus AT THE SAME TIME.
        cpu_gbps, host_gbps = ddr_share(cpu_gbps, host_gbps, prior=cpu_prior)
        contended = (pol.gbps(total_threads) + p.slow_host_gbps(num_ranks, loaded=loaded)
                     > cpu_prior.ddr_wall_gbps)

    host_ms = host_bytes_per_rank / (host_gbps * GB) * 1000.0 if host_bytes_per_rank else 0.0
    device_ms = device_bytes_per_rank / (p.device_read_gbps * GB) * 1000.0
    # NODE-wide CPU traffic: every rank's share crosses the same controller.
    cpu_node_bytes = cpu_bytes_per_rank * num_ranks
    cpu_ms = cpu_node_bytes / (cpu_gbps * GB) * 1000.0 if cpu_bytes_per_rank else 0.0
    handoff_ms = num_cpu_layers * handoff_us_per_layer / 1000.0

    if mode is CpuTierMode.SPLIT:
        overlapped = min(cpu_ms + handoff_ms, host_ms)
    else:
        overlapped = 0.0

    step_ms = p.compute_floor_ms + device_ms + host_ms + cpu_ms + handoff_ms - overlapped
    tok_s = (1000.0 / step_ms) if step_ms > 0 else float("inf")
    segs = graph_segments(
        max(int(num_layers), num_cpu_layers), tuple(range(num_cpu_layers)), mode
    ) if num_cpu_layers else 1
    return CpuTierProjection(
        step_ms=step_ms,
        tok_s=tok_s,
        compute_ms=p.compute_floor_ms,
        device_ms=device_ms,
        host_ms=host_ms,
        cpu_ms=cpu_ms,
        handoff_ms=handoff_ms,
        overlapped_ms=overlapped,
        mode=mode,
        threads_per_rank=int(threads_per_rank),
        total_threads=total_threads,
        cpu_gbps=cpu_gbps,
        host_gbps=host_gbps,
        ddr_contended=contended,
        handoff_us_used=float(handoff_us_per_layer),
        handoff_measured=bool(cpu_prior.handoff_measured),
        num_cpu_layers=int(num_cpu_layers),
        policy=pol.name,
        rel_rms=pol.rel_rms,
        graph_segments=segs,
        calibrated_tok_s=tok_s * cpu_prior.projection_fidelity,
        projection_fidelity=cpu_prior.projection_fidelity,
    )


# ────────────────────────────────────────────────────────────────────────────────────────────────
# The async handoff — ordering, as a state machine, with no threads and no torch in it
# ────────────────────────────────────────────────────────────────────────────────────────────────
class HandoffError(RuntimeError):
    """An ordering or lifetime rule of the CPU handoff was violated."""


class HandoffState(Enum):
    """States of ONE layer's CPU handoff. Transitions are the only legal edges; see `CpuHandoff`.

        NEW ──seal_input──> SEALED ──start──> RUNNING ──finish──> DONE ──join──> JOINED
                                                     └──fail────> FAILED ──join──> raises

    The reason this is a state machine and not a `concurrent.futures.Future` is `SEALED`. The
    hidden state handed to the CPU lives in a device-to-host staging buffer that the next layer
    will reuse; the CPU worker must not read it after that reuse, and the GPU must not overwrite it
    before the worker has copied it. `seal_input` is the point the copy is complete, and it is a
    separate, checkable edge precisely so "the worker read a buffer the GPU had already recycled"
    is an exception instead of numerically-plausible garbage.
    """

    NEW = "new"
    SEALED = "sealed"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    JOINED = "joined"


_LEGAL: Dict[HandoffState, Tuple[HandoffState, ...]] = {
    HandoffState.NEW: (HandoffState.SEALED,),
    HandoffState.SEALED: (HandoffState.RUNNING,),
    HandoffState.RUNNING: (HandoffState.DONE, HandoffState.FAILED),
    HandoffState.DONE: (HandoffState.JOINED,),
    HandoffState.FAILED: (HandoffState.JOINED,),
    HandoffState.JOINED: (),
}


class CpuHandoff:
    """One layer's CPU work item. Pure ordering bookkeeping — carries payloads opaquely.

    `payload` and `result` are whatever the caller puts there (torch tensors in the engine, numpy
    arrays in the tests). This class never touches them, which is what keeps the ordering rules
    testable with no torch and no GPU.
    """

    __slots__ = ("seq", "layer_index", "step", "payload", "result", "_state", "_error", "_log")

    def __init__(self, seq: int, layer_index: int, step: int, payload: object = None) -> None:
        self.seq = int(seq)
        self.layer_index = int(layer_index)
        self.step = int(step)
        self.payload = payload
        self.result: object = None
        self._state = HandoffState.NEW
        self._error: BaseException | None = None
        self._log: list[HandoffState] = [HandoffState.NEW]

    # -- state ---------------------------------------------------------------------------------
    @property
    def state(self) -> HandoffState:
        return self._state

    @property
    def history(self) -> Tuple[HandoffState, ...]:
        return tuple(self._log)

    def _to(self, nxt: HandoffState) -> None:
        if nxt not in _LEGAL[self._state]:
            raise HandoffError(
                f"handoff seq={self.seq} layer={self.layer_index}: illegal transition "
                f"{self._state.value} -> {nxt.value}. Legal from {self._state.value}: "
                f"{[s.value for s in _LEGAL[self._state]] or ['(terminal)']}."
            )
        self._state = nxt
        self._log.append(nxt)

    # -- edges ---------------------------------------------------------------------------------
    def seal_input(self, payload: object = None) -> "CpuHandoff":
        """The device->host copy of this layer's activation is COMPLETE. Only now may a worker read it.

        Called after the D2H copy's event has been observed, never merely after it was enqueued.
        """
        if payload is not None:
            self.payload = payload
        self._to(HandoffState.SEALED)
        return self

    def start(self) -> "CpuHandoff":
        self._to(HandoffState.RUNNING)
        return self

    def finish(self, result: object) -> "CpuHandoff":
        self.result = result
        self._to(HandoffState.DONE)
        return self

    def fail(self, error: BaseException) -> "CpuHandoff":
        self._error = error
        self._to(HandoffState.FAILED)
        return self

    def join(self) -> object:
        """Consume the result exactly once. Raises if the work failed, or if it is not finished.

        NEVER returns a zero/None on a failure. A CPU MoE partial that silently became zero is a
        model that keeps generating fluent text with 21 of its 48 layers' experts missing -- the
        exact class of bug the whole offload feature's read-back verification exists to prevent.
        """
        if self._state is HandoffState.FAILED:
            self._to(HandoffState.JOINED)
            assert self._error is not None
            raise HandoffError(
                f"CPU MoE handoff seq={self.seq} layer={self.layer_index} step={self.step} "
                f"failed: {self._error!r}"
            ) from self._error
        if self._state is not HandoffState.DONE:
            raise HandoffError(
                f"handoff seq={self.seq} layer={self.layer_index} joined in state "
                f"{self._state.value}; only `done` (or `failed`, which raises) may be joined. A "
                f"join that returned early would feed the residual stream a partial sum."
            )
        self._to(HandoffState.JOINED)
        return self.result

    def __repr__(self) -> str:
        return (
            f"CpuHandoff(seq={self.seq}, layer={self.layer_index}, step={self.step}, "
            f"{self._state.value})"
        )


class HandoffLedger:
    """Enforces the CROSS-handoff ordering rules for one model-forward.

    Four invariants, each of which is a real failure mode rather than a tidiness rule:

    G1  FIFO JOIN. Handoffs are joined in submission order. The residual stream is sequential, so
        joining layer L+1 before layer L means layer L+1 consumed an input that layer L had not
        yet contributed to. There is no reordering that is safe here and therefore none is allowed.

    G2  BOUNDED FLIGHT. At most `max_inflight` handoffs are un-joined at once. BLOCK mode sets 1,
        which makes the mode's own claim ("serial, no overlap") a checked property rather than a
        comment. SPLIT sets 1 as well *per layer* -- its concurrency is CPU-vs-GPU within one
        layer, not multiple layers in flight -- and only a future batch-pipelined mode would raise
        it. A ledger that silently allowed more would be describing a schedule nothing implements.

    G3  DRAIN AT A BARRIER. `barrier()` (a captured-graph segment boundary, or the end of the
        step) refuses to pass while anything is in flight. A handoff that outlived a graph segment
        would have its result added into a buffer the next segment's captured nodes already read.

    G4  ONE STEP AT A TIME. A handoff from step N may not be joined during step N+1. Decode steps
        are pipelined by the scheduler elsewhere; a leaked handoff would add the previous token's
        expert partial to this token's residual, which is fluent, plausible and wrong.
    """

    __slots__ = ("_max_inflight", "_next_seq", "_inflight", "_join_cursor", "_step", "_completed")

    def __init__(self, *, max_inflight: int = 1) -> None:
        if max_inflight < 1:
            raise HandoffError(f"max_inflight must be >= 1, got {max_inflight}")
        self._max_inflight = int(max_inflight)
        self._next_seq = 0
        self._inflight: list[CpuHandoff] = []
        self._join_cursor = 0
        self._step = 0
        self._completed = 0

    @property
    def inflight(self) -> int:
        return len(self._inflight)

    @property
    def completed(self) -> int:
        return self._completed

    @property
    def step(self) -> int:
        return self._step

    def begin_step(self, step: int | None = None) -> int:
        """Start a new decode step. Refuses while anything is still in flight (G4)."""
        if self._inflight:
            raise HandoffError(
                f"begin_step while {len(self._inflight)} CPU handoff(s) are still in flight "
                f"{[h.seq for h in self._inflight]}. A handoff that crosses a step boundary adds "
                f"the PREVIOUS token's expert partial to this token's residual."
            )
        self._step = self._step + 1 if step is None else int(step)
        return self._step

    def submit(self, layer_index: int, payload: object = None) -> CpuHandoff:
        if len(self._inflight) >= self._max_inflight:
            raise HandoffError(
                f"CPU handoff in-flight bound exceeded: {len(self._inflight)} >= "
                f"{self._max_inflight}. Raising this bound is a SCHEDULE change (it claims layers "
                f"L and L+1 can run concurrently, which the sequential residual forbids), not a "
                f"tuning knob."
            )
        h = CpuHandoff(self._next_seq, layer_index, self._step, payload)
        self._next_seq += 1
        self._inflight.append(h)
        return h

    def join(self, handoff: CpuHandoff) -> object:
        if not self._inflight or self._inflight[0] is not handoff:
            want = self._inflight[0].seq if self._inflight else None
            raise HandoffError(
                f"out-of-order join: asked for seq={handoff.seq} (layer {handoff.layer_index}) "
                f"but the oldest in-flight handoff is seq={want}. Joins are FIFO because the "
                f"residual stream is sequential."
            )
        if handoff.step != self._step:
            raise HandoffError(
                f"handoff seq={handoff.seq} belongs to step {handoff.step}, ledger is on step "
                f"{self._step}"
            )
        try:
            return handoff.join()
        finally:
            self._inflight.pop(0)
            self._join_cursor += 1
            self._completed += 1

    def barrier(self, where: str = "graph-segment boundary") -> None:
        if self._inflight:
            raise HandoffError(
                f"{len(self._inflight)} CPU handoff(s) still in flight at {where}: "
                f"{[h.seq for h in self._inflight]}. Every CPU handoff must be joined inside the "
                f"eager region that opened it -- a captured segment replays device nodes with no "
                f"host involvement, so a result arriving after the cut lands in a buffer the "
                f"replay has already read."
            )
