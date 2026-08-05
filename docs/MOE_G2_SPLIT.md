# The MoE decode gemm2 was occupancy-starved, and the split-K that was supposed to fix it was pointed at the wrong kernel

**Date** 2026-08-06 · **Engine** `minisgl-rdna4` @ `4d719ad6` · **Kernels** `rdna4-hip-kernels` @
`8a8bca6` · **Card** RX 9070 XT (`multiProcessorCount` = 32 **WGP**; never assert 64) · **Model**
Qwen3.6-35B-A3B-AWQ-4bit TP=2 — `hidden 2048`, `moe_intermediate 512 → 256/rank`, `E 256`,
`top_k 8`, `group_size 32`, 40 MoE layers (from `config.json`, not from a probe default).

---

## The correction that had to come first: there are TWO gemm2 kernels, and they serve different M

`docs/COUNTER_SCORECARD.md` measured `moe_gemm2_gather_reduce_core` and labelled `K=256 M=1` **"the
served decode case"** — 8 workgroups, 64 waves, **2.2% occupancy**, 15.2% of roofline. The occupancy
diagnosis is right and it is the headline finding of that document. The *label* is not: at M=1 the
engine never calls that kernel.

`python/minisgl/quant/kernels.py:w4a8_moe` branches `if M <= 2:` into the atomic **scatter** first,
and only M ≥ 3 reaches the fused gather-reduce. Confirmed by calling the real dispatch entry point at
the real shape and recording `engaged(...)` per M (`tools/moe_g2_served_probe.py`):

| M | gemm2 arm actually launched |
|---|---|
| 1, 2 | `fp8_wmma.mmq_fp8_moe_gemm_scatter` |
| 3, **5**, **6**, 8, 16, 32 | `fp8_wmma.mmq_fp8_moe_gemm2_gather_reduce` |

So the served points split across both arms: **bs=1 decode is M=1 → scatter**, and **MTP verify
(M=5) and CONC=6 (M=6) → the fused gather-reduce**. Both were starved, and each had its own frozen
constant. The scorecard's M=1 row for the fused kernel is a real measurement of a configuration the
engine does not run; its M=5 / M=6 rows (12.3% / 14.8% occupancy) are the served ones.

---

## What was wrong

**Fused gather-reduce (M ≥ 3): no split axis at all.** `grid = (ceil(N/256), M)`. Parallelism comes
only from the output, and at decode the output is one row per token. The kernel's own comment
identified the fix and then talked itself out of it:

> *"The real M=1 headroom would be parallelising the top_k loop across blocks, which changes the
> ascending-k fp32 fold order this kernel's bit-exactness note relies on."*

That objection is **false**, and it kept the kernel at 2.2–14.8% occupancy for as long as it stood.
It is only true if a slice pre-folds its own experts.

**Scatter (M ≤ 2): a split, but a frozen and unswept one.** `_moe_split_k(M) = 4 if M == 1 else 1`,
whose docstring said "Workload-derived". It was neither derived nor swept — and measured, `4` is the
**worst** useful value at the only M it was written for.

---

## The fix

### 1. A top_k split on `grid.z` of the fused kernel, at max|Δ| = 0

`k_slices` blocks share each (token, N-tile); block `z` takes `k = z, z+k_slices, …`.

The numerics are **unchanged — max|Δ| = 0 at every slice count and every M** — and that is a
property of the partial layout, not luck. A slice does *not* pre-fold its experts. It writes each
`k`'s term `tw_k · (float)cvt_out<OT>(…)` — computed by the identical code — into its own slot of a
`(top_k, M, N)` fp32 buffer, and the last-arriving block folds all `top_k` slots in **ascending k**,
which is the same left fold the unsplit kernel does in registers. Missing slots (`r < 0`) write an
explicit `0.0f`.

Two consequences the neighbouring scatter arm does *not* have:

* **No atomic accumulation into `out`**, so this stays deterministic against itself. (`minv.py`
  records the scatter at 9.5e-7 … 2.4e-4 between two identical consecutive calls.)
* **`k_slices` may depend on M without breaking M-invariance**, because the fold order is fixed by
  `top_k`, not by the slice count. That is what lets the launcher derive the split from occupancy —
  which genuinely is a function of M — while a token's value is identical at every batch size.
  M-invariance is what prefix-caching / chunked-prefill / spec-verify depend on.

The cross-block handshake costs **no extra dispatch and no extra allocation**: the arrival counters
ride in the tail of `slot_to_row` and are zeroed by `moe_build_inverse_map_kernel`, a prelude that
already launches — which also re-zeroes them on every cudagraph replay.

### 2. One occupancy rule, replacing both frozen constants

`moe_split_slices(...)` serves the fused kernel's top_k carve and the scatter's K carve, because
they are the same question asked of two reductions.

    want_blocks = multiProcessorCount × blocks-resident-per-WGP   [÷ 2 if the split costs a pass]

The first two terms come from the device and from the *compiled kernel*, never from literals:
`multiProcessorCount` (WGPs — the 64-CU 9070 XT answers 32, the 9070 answers 28), and
`hipOccupancyMaxActiveBlocksPerMultiprocessor` for the exact instantiation about to launch. The third
term is fitted and is discussed below.

The engine now passes `split_k = 0` ("derive it") instead of a number it cannot compute: BN and
WARPS_N — which set the grid — are chosen inside the kernel launcher and are invisible from Python.

---

## The measurement

`tools/moe_g2_bench.py`, median of 3 × 200 iterations, **64 independent routes cycled** so the
expert slabs rotate past the 64 MB MALL by byte count rather than replaying out of L2. µs; best in
each row in **bold**.

### Fused gather-reduce — the `top_k` carve

| M | base blocks | sk=1 | sk=2 | sk=4 | sk=8 | rule picks | rule vs best |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 24 | 39.45 | 34.48 | **31.16** | 31.70 | 4 | 1.00× |
| **5** (MTP verify) | 40 | 45.24 | 39.12 | **36.88** | 43.66 | 2 | 1.06× |
| **6** (CONC=6) | 48 | 56.90 | **50.70** | 50.73 | 56.72 | 2 | 1.00× |
| 8 | 64 | 57.67 | **53.56** | 61.60 | 57.81 | 2 | 1.00× |
| 16 | 128 | **65.19** | 82.38 | 73.67 | 69.41 | 1 | 1.00× |
| 32 | 256 | 137.13 | 127.29 | 119.08 | **117.43** | 1 | 1.17× |

The rule **never regresses against not splitting**, at any M.

**At the served points the split is worth 1.12× (M=6, 56.90 → 50.60 µs) and 1.16× (M=5, 45.24 →
39.12 µs), bit-exactly.**

### Scatter — the `K` carve, the M ≤ 2 arm

| M | base blocks | sk=1 | sk=2 | sk=4 | sk=8 | rule picks |
|---|---:|---:|---:|---:|---:|---:|
| **1** (bs=1 decode) | 256 | 20.72 | **19.11** | 26.64 ← *shipped* | 28.76 | 2 |
| 2 | 512 | **38.34** ← *shipped* | 42.94 | 42.21 | 59.04 | 1 |

**The shipped `4` was 39% off the optimum at the one M it was written for, and 29% worse than not
splitting at all** — a straight regression on the bs=1 decode path, in a constant whose docstring
called itself workload-derived.

### Where the rule is knowingly wrong, and the term that is fitted

The two arms do **not** get the same fill target, and that is not a fudge — it is the one real
difference between them. The scatter carve is **free**: the epilogue's `atomicAdd` already *is* the
reduction, so k-slices need no second pass and no buffer, and it wants the machine full. The fused
carve **materialises a `(top_k, M, N)` fp32 partial buffer and reads it back**, a round-trip that
grows with M, so it saturates earlier. Measured, "earlier" is *half* the residency limit — and that
halving is a **fitted constant**, named `split_costs_a_pass` rather than buried, because targeting
full residency there picks sk=2 at M=16 where sk=1 is best: a **27% regression against not splitting
at all**.

I first tried to remove that factor by asking the runtime for *achievable* rather than architectural
residency, on the theory that the kernel could not reach 16 waves/SIMD. **It can** — the query
returns the cap. So the gap is not occupancy, it is the cost of the extra pass, and the honest
resolution is a named term plus the surface it was fitted to, not a cleaner-sounding derivation that
does not predict the data.

**M=32 is where the rule is simply wrong: it declines to split and leaves 14%.** That is the only cell where more parallelism
still pays *past* residency, and no occupancy argument predicts it. `max-running-requests` is 6 on
this model at 16 GB, so M ≤ 6 is the band that can actually be served; the trade — never regress the
served band to chase an unreachable shape — is deliberate, and it is recorded here rather than
smoothed over.

---

## Method notes worth carrying forward

**A cross-process parity harness reported a kernel bug that did not exist.** The first version ran
each slice count in its own subprocess and rebuilt inputs from a seed. It reported max|Δ| ≈ 1.3e-1 at
M ≥ 16 — and, decisively, **two legs with the *same* slice count disagreed**. `moe_align`'s row
assignment within an expert block is not stable across processes, so the legs were not reading the
same `buf2` rows. The control that catches this (`sk1b`: a second recording of the *same* unsplit
configuration) was not in the harness, and adding it is what ended a build-and-measure cycle spent
chasing a phantom race. **Run the control first.** The gate now runs every leg in one process against
byte-identical tensors, and passes at every M and slice count.

**Two `__restrict__` pointers into one allocation is UB even when it looks harmless.** The arrival
counters ride in `slot_to_row`'s tail; passing both as separate `__restrict__` parameters is a lie to
the compiler that pays out as dropped stores under a vectoriser, not as a fault. Both are now derived
from one pointer. (This was found while chasing the phantom above; it is kept because it is correct,
**not** because it was measured to fix anything.)

---

## Served end-to-end

Same binary in both legs; the base leg forces the unsplit grid with an env override, so a compiler or
layout difference cannot be read as the effect. `--num-pages 2048` pinned so the legs admit identical
work. Aggregate decode tok/s (TTFT excluded), **median of 5**.

### Run 1 — serve.sh defaults, which means **MTP is on**

| bs | base (unsplit) | new (derived) | ratio | spread |
|---|---:|---:|---:|---|
| 1 | 40.59 | **40.99** | **1.010×** | base 40.04–40.70, new 40.93–41.29 (disjoint) |
| 5 | 98.82 | **100.11** | **1.013×** | base 98.13–99.32, new 98.24–101.29 |
| 6 | 113.53 | 113.47 | 0.999× | base 112.66–114.44, new 112.18–115.67 |

**+1.0% and +1.3% at bs=1 and bs=5; nothing at bs=6.** The bs=1 legs do not overlap at all
(base max 40.70 < new min 40.93), so that one is outside the run-to-run spread. bs=5 is at the edge
of it. bs=6 is a wash.

**But these bs labels do not mean what they look like.** With a draft width the decode batch the MoE
sees is a spec-VERIFY width, not `bs`. The `[g2-split]` ledger for this run shows the fused kernel
being called at M = 3, 4, 5, 6, 8, 10, 12, 16, 18, 20, 24, 30, 32 — and **M=1 and M=2 never appear**,
so the scatter arm is not on the decode path at all under MTP. Run 2 pins `SPEC=none` so that
`bs = n` gives `M = n` and the served points line up with the swept surface.

### Honest sizing, before anyone extrapolates

The isolated win is 1.12–1.39× **on one kernel**, and that kernel is roughly 40 × 50 µs ≈ 2 ms of a
step measured in tens of ms. A ~1% end-to-end result is the *expected* size of this change, not a
disappointment and not evidence that the kernel measurement was wrong. It is worth having because it
is free at runtime and bit-exact — but the decode step has many other kernels, and this one is no
longer among the starved ones.
