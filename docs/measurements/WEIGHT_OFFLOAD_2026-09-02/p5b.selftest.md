# P5b — torch over a `hipHostGetDevicePointer` address (in-image, under capture)

**Run:** 2026-09-03T08:33:03+1000 · host `blue` · in-container `False` · image `None` (`None`) · schema `p5b/1`
**THIS IS A SELFTEST ARTIFACT — no GPU was touched and NOTHING below was measured.**

> Mechanism under test: **`hipHostMalloc(Mapped|Portable)` + `hipHostGetDevicePointer`** —
> the T1 tier Phase 0 selected after `hipMemCreate(location=Host)` was found to return
> device VRAM silently. P5 validated torch plumbing over a VMM reservation, i.e. over
> **device** memory; this is the first time the *selected* mechanism meets torch.

## Answers

| Question | Answer |
|---|---|
| are the arena's pages **provably host-resident** (VRAM delta, MemAvailable/GTT delta, CPU accessibility, read rate vs the PCIe ceiling)? | not measured |
| `t.data_ptr()` == the host-derived device pointer, under `use_mem_pool` | not measured |
| zero `hipMalloc` fallbacks inside the pool | not measured |
| coexists with `expandable_segments:True` (compose default) | not measured |
| arena survives `torch.cuda.empty_cache()` (`engine/graph.py:314`) | not measured |
| free callback fires for a **live** pool block | not measured |
| free callback fires for a **cached (dropped)** pool block | not measured |
| graph capture + replay over the arena is correct | not measured |
| ≥ 20 replays **bit-identical** to a CPU-computed ground truth | not measured |
| **both physical cards** exercised | not measured (cards: []) |

**Exit classification:** n/a

**If this probe is RED:** selftest placeholder

**Reasons**

- selftest: nothing was measured

## Placement proof — per leg

| leg | physical card | PCI | classification | VRAM Δ / size | MemAvail Δ / size | GTT Δ / size | CPU readable | arena read GB/s | copy engine GB/s | HBM ref GB/s | arena / HBM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expandable_card0 | None | `None` | - | - | - | - | None | - | - | - | - |
| expandable_card1 | None | `None` | - | - | - | - | None | - | - | - | - |
| plain_card0 | None | `None` | - | - | - | - | None | - | - | - | - |

> A read that outruns the copy engine by >1.5× or reaches ≥0.25× the HBM reference did **not** cross PCIe — those pages are device memory whatever the counters said. That check is the one the first P2 lacked (device 117.9 vs "host" 118.2 GB/s, ratio 1.00).
> Expected PCIe ceilings on this box: **card 0 = 28.70 GB/s (Gen5 x8)**, **card 1 = 14.34 GB/s (Gen4 x8)** — sampled mid-DMA, never at idle (ASPM downtrains).

## Arms — per leg

| leg | alloc conf | status | arena | placement | pool_alloc | bandwidth | correctness | empty_cache | capture | alloc µs (via our allocator) | replay µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expandable_card0 | expandable_segments:True | selftest | - | - | - | - | - | - | - | - | - |
| expandable_card1 | expandable_segments:True | selftest | - | - | - | - | - | - | - | - | - |
| plain_card0 | &lt;unset&gt; | selftest | - | - | - | - | - | - | - | - | - |

> `alloc µs` counts only reps that actually reached the custom allocator; reps torch served from its own cache are reported separately in the JSON.
> Replay times are a functional latency over an L2/MALL-resident working set — **not** a bandwidth measurement. The bandwidth column above is the ≥256 MiB one.

## Box state at run start

- MemTotal 91.8 GB, MemAvailable 70.8 GB, MemFree 67.4 GB, Cached 3.1 GB
- swap: SwapTotal 53.9 GB, SwapFree 34.9 GB, pswpout 407142868
- loadavg `0.88 1.02 1.12 2/3762 3674633`
- GPU lease **waived by explicit user instruction**; legs run strictly serially, never two GPU jobs at once.

Raw: `p5b.json` (this directory), per-leg `p5b_leg_<name>.json` + `.log`.
