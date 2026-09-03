"""The BAKED PRIOR for weight offload: every measured constant the placement planner needs.

Why a frozen table and not scattered literals: the planner must be a pure function of config, and
"config" here includes the hardware facts that were measured ONCE in Phase 0 and can never be
re-derived at boot without a timing run (which would make the plan timing-dependent, hence
rank-divergent). So the measurements are frozen into a dataclass with provenance attached, the
planner takes one as an argument, and a test can pin the shipped values against the report.

EVERY number below is measured on THIS box, not derived. The ones that were derived in the original
plan were wrong by up to 6.8x (llama.cpp 33.8 -> 4.990 tok/s), which is why the provenance string on
each field names the probe that produced it. Re-measure, do not re-derive.

Source of truth:
  docs/measurements/WEIGHT_OFFLOAD_2026-09-02/PHASE0_REPORT.md
  docs/measurements/WEIGHT_OFFLOAD_2026-09-02/p1.json  (host read bandwidth, per card)
  docs/measurements/WEIGHT_OFFLOAD_2026-09-02/p3b.json (pinned host capacity ceiling)
  docs/measurements/WEIGHT_OFFLOAD_2026-09-02/p4.json  (PCIe topology / concurrency)
  docs/measurements/WEIGHT_OFFLOAD_2026-09-02/p2prime.json (miss-cost curve: LINEAR)
  docs/measurements/WEIGHT_OFFLOAD_2026-09-02/p0.json  (llama.cpp baseline)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

GiB = 1 << 30
GB = 1_000_000_000  # bandwidths are quoted in decimal GB/s; capacities in binary GiB. Keep both.


@dataclass(frozen=True)
class OffloadPrior:
    """Measured hardware constants for one box, with provenance.

    `host_read_gbps` is indexed by PHYSICAL card index and is deliberately NOT symmetric: card 1's
    root port trained Gen4 x8 against card 0's Gen5 x8 (P4, 78/78 mid-DMA samples), so it reads host
    memory at exactly half card 0's rate. Every TP=2 ceiling is gated by the SLOWEST rank -- P4
    measured the two links as independent (efficiency 0.999), so the ranks stream concurrently and
    the slow one sets the step. A planner that averages the two, or assumes a symmetric pair, is
    wrong by ~2x on the card-1 rank. Use `slow_host_gbps()`.
    """

    name: str
    provenance: str

    # --- bandwidth ------------------------------------------------------------------------------
    # P1 tiled kernel-read from hipHostMalloc(Mapped) pages; reproduced to 3 digits by P2' (28.92 /
    # 14.47) through a real grouped MoE GEMV, and to ~92-96% by P5b at the torch level (26.6 / 13.87).
    host_read_gbps: Tuple[float, ...] = (28.93, 14.48)
    # Same read with the host DDR bus loaded. P4 copy-engine PROXY, not a shader measurement --
    # bracket, do not quote as fact (Phase 0 report 3.5).
    host_read_gbps_loaded: Tuple[float, ...] = (12.38, 12.36)
    # Device (HBM) streaming read, both cards. P1/P3 VMM location=Device arm.
    device_read_gbps: float = 692.0

    # --- capacity -------------------------------------------------------------------------------
    # P3b, IDLE box, nothing else loaded: pinned hipHostMalloc reached 34.0 GiB on one rank but only
    # 62.0 GiB (34.0 + 28.0) across two, stopped by the MemAvailable floor after swapping out 114,813
    # pages. This is a NODE-WIDE ceiling (host RAM is shared by every rank on the box), and it is a
    # near-miss rather than a fit for the 68.8 GB the target checkpoint wants.
    host_arena_ceiling_bytes: int = 62 * GiB
    # The 62 GiB was measured with NO engine resident. A live serve additionally holds the CPU-side
    # tokenizer/scheduler processes, the page cache the checkpoint load just filled, and pinned
    # staging buffers -- and the measurement itself was already swapping. Derate. This is the single
    # knob that decides "does it boot", so it is explicit rather than buried in an inequality.
    host_arena_headroom_fraction: float = 0.90

    # --- compute floor --------------------------------------------------------------------------
    # Non-expert per-step time (attention, norms, router, dense linears, launch overhead) at bs=1.
    # Phase 0 3.2 uses a 5-10 ms bracket and quotes the 7.5 ms midpoint in every published table;
    # matching it is what lets test_matches_phase0_table lock this arithmetic to the report.
    compute_floor_ms: float = 7.5

    # --- gates ----------------------------------------------------------------------------------
    baseline_tok_s: float = 4.990  # P0, llama.cpp bs=1 decode on the target checkpoint
    kill_tok_s: float = 3.178  # K4 = baseline / 1.57; below this, ship nothing
    accept_fraction: float = 0.75  # A1.7 = accept_fraction x the mechanism ceiling

    # --- sweep grid -----------------------------------------------------------------------------
    # The device-fraction grid the Phase 0 report publishes. Reported, never silently chosen: the
    # resolved plan uses the exact byte budget it was given, and the grid is the advisory Pareto
    # table (A2.4) that goes on the boot banner.
    f_grid: Tuple[float, ...] = (0.0, 0.10, 0.20, 0.25, 0.30)

    # --- curve shape ----------------------------------------------------------------------------
    # P2' verdict=LINEAR on both cards (cliff_index 0.090 / 0.094 vs pure-linear 0.100). Misses cost
    # ADDITIVELY at full PCIe rate: expected layer time = t_device + (1-f) * bytes / host_GB_s, with
    # no cliff term, no concurrency term and no residency-probability term. The h^10
    # all-resident-layer fear is REFUTED. Carried as a field so a future box that measures a CLIFF
    # makes the planner's linear projection loudly wrong instead of quietly optimistic.
    miss_cost_is_linear: bool = True
    # P2': per-expert placement beats layer-granular by 1.063x at its PEAK (h=0.75) and 1.013x at the
    # h~0.25 operating point capacity actually forces. That is why this planner is layer-granular.
    per_expert_gain_at_operating_point: float = 1.013

    def slow_host_gbps(self, num_ranks: int, *, loaded: bool = False) -> float:
        """Host-read bandwidth of the SLOWEST of the first `num_ranks` cards.

        The links are independent, so at TP=N every rank streams its own shard concurrently and the
        step is set by whichever rank finishes last. Ranks beyond the measured card list reuse the
        slowest measured card rather than extrapolating.
        """
        if num_ranks <= 0:
            raise ValueError(f"num_ranks must be >= 1, got {num_ranks}")
        table = self.host_read_gbps_loaded if loaded else self.host_read_gbps
        if not table:
            raise ValueError("prior carries no host_read_gbps measurements")
        return min(table[: max(1, min(num_ranks, len(table)))])

    def usable_host_arena_bytes(self) -> int:
        """Node-wide pinned-host budget the planner is allowed to commit, after the derate."""
        return int(self.host_arena_ceiling_bytes * self.host_arena_headroom_fraction)

    def with_overrides(self, **kw) -> "OffloadPrior":
        """A copy with fields replaced -- for what-if sweeps and for tests, never for boot.

        Exists so a caller who wants to ask "what if card 1 trains Gen5" does it by constructing a
        DIFFERENT prior with its own provenance string, instead of mutating the shipped one.
        """
        from dataclasses import replace

        return replace(self, **kw)


PHASE0_PRIOR = OffloadPrior(
    name="gfx1201-dual-2026-09-02",
    provenance=(
        "Phase 0 gate report 2026-09-02/03, worktree minisgl-rdna4-offload @ 8bcc7035. "
        "host_read_gbps P1 tiled kernel-read (card0 RX 9070 XT 0000:03:00.0 Gen5 x8, "
        "card1 RX 9070 0000:07:00.0 Gen4 x8), reproduced by P2' at 28.92/14.47 and by P5b at "
        "26.61/13.87 through torch. host_arena_ceiling_bytes P3b pinned 2-rank ceiling on an idle "
        "box (34.0+28.0 GiB, MemAvailable floor, 114813 pages swapped). device_read_gbps P1/P3 "
        "location=Device arm. compute_floor_ms Phase 0 3.2 midpoint of a 5-10 ms bracket. "
        "baseline_tok_s / kill_tok_s P0 llama.cpp 4.990 bs=1 (K4 = 4.990/1.57). "
        "miss_cost_is_linear P2' cliff_index 0.090/0.094 vs pure-linear 0.100. "
        "ROCm 7.2.4, libamdhip64.so.7.2.53211-3d9ef42, kernel 7.0.10-1-cachyos-custom. "
        "RE-MEASURE after any ROCm bump or BIOS PCIe change."
    ),
)

# What the box would look like if card 1's root port were retrained to Gen5 x8 -- the single
# highest-leverage open action in the Phase 0 report (18.7 -> 32.8 tok/s at TP=2). Kept here so the
# what-if is a named, provenance-carrying object rather than a number someone types into a table.
CARD1_GEN5_PRIOR = PHASE0_PRIOR.with_overrides(
    name="gfx1201-dual-if-card1-gen5",
    host_read_gbps=(28.93, 28.93),
    provenance=(
        "HYPOTHETICAL, NOT MEASURED. PHASE0_PRIOR with card 1 assumed retrained to Gen5 x8. "
        "Use only for what-if sweeps; A1.7 must be re-derived before any gate cites this."
    ),
)


@dataclass(frozen=True)
class Projection:
    """A projected decode step under the P2'-validated LINEAR miss-cost model.

    step_ms = compute_floor + host_bytes/host_BW + device_bytes/device_BW, per rank, with the
    slowest rank setting the step. No cliff term (P2' refuted it), no residency-probability term
    (layer-granular placement is deterministic), no concurrency term (P4: links are independent).
    """

    step_ms: float
    tok_s: float
    host_ms: float
    device_ms: float
    compute_ms: float
    host_gbps: float
    active_bytes_per_rank: int = 0


def project_step(
    *,
    host_bytes_per_rank: int,
    device_bytes_per_rank: int,
    prior: OffloadPrior,
    num_ranks: int,
    loaded: bool = False,
) -> Projection:
    """Project one decode step from per-token ACTIVE bytes (not resident bytes).

    `host_bytes_per_rank` / `device_bytes_per_rank` are the bytes this rank READS per token, i.e.
    top_k experts per MoE layer, already TP/EP-sharded -- not the size of the arena.
    """
    host_gbps = prior.slow_host_gbps(num_ranks, loaded=loaded)
    host_ms = host_bytes_per_rank / (host_gbps * GB) * 1000.0
    device_ms = device_bytes_per_rank / (prior.device_read_gbps * GB) * 1000.0
    step_ms = prior.compute_floor_ms + host_ms + device_ms
    return Projection(
        step_ms=step_ms,
        tok_s=(1000.0 / step_ms) if step_ms > 0 else float("inf"),
        host_ms=host_ms,
        device_ms=device_ms,
        compute_ms=prior.compute_floor_ms,
        host_gbps=host_gbps,
        active_bytes_per_rank=host_bytes_per_rank + device_bytes_per_rank,
    )
