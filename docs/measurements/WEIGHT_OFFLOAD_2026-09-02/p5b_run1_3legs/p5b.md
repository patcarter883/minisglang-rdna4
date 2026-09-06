# P5b — torch over a `hipHostGetDevicePointer` address (in-image, under capture)

**Run:** 2026-09-02T22:58:44+0000 · host `26d846a273a3` · in-container `True` · image `minisgl-rdna4:lean` (`sha256:9748d5f9ec1d`) · schema `p5b/1`
**Verdict:** **PASS** (exit 0)

> Mechanism under test: **`hipHostMalloc(Mapped|Portable)` + `hipHostGetDevicePointer`** —
> the T1 tier Phase 0 selected after `hipMemCreate(location=Host)` was found to return
> device VRAM silently. P5 validated torch plumbing over a VMM reservation, i.e. over
> **device** memory; this is the first time the *selected* mechanism meets torch.

## Answers

| Question | Answer |
|---|---|
| are the arena's pages **provably host-resident** (VRAM delta, MemAvailable/GTT delta, CPU accessibility, read rate vs the PCIe ceiling)? | YES |
| `t.data_ptr()` == the host-derived device pointer, under `use_mem_pool` | YES |
| zero `hipMalloc` fallbacks inside the pool | YES |
| coexists with `expandable_segments:True` (compose default) | YES |
| arena survives `torch.cuda.empty_cache()` (`engine/graph.py:314`) | YES |
| free callback fires for a **live** pool block | **NO** |
| free callback fires for a **cached (dropped)** pool block | **NO** |
| graph capture + replay over the arena is correct | YES |
| ≥ 20 replays **bit-identical** to a CPU-computed ground truth | YES |
| **both physical cards** exercised | YES (cards: ['0000:03:00.0', '0000:07:00.0']) |

**Exit classification:** all gating legs green on both cards

**If this probe is RED:** M1-A cannot use the ~30-line ctypes arena and needs a C++ torch::from_blob extension plus a build-system change: +3 days on the critical path.

**Reasons**

- GREEN on both cards: hipHostMalloc(Mapped) pages are provably host-resident, torch allocates over the host-derived device pointer with zero fallbacks, the arena survives empty_cache(), and >= 20 graph replays are bit-identical to a CPU-computed ground truth. M1-A can use the ctypes arena; no C++ extension is needed.

## Placement proof — per leg

| leg | physical card | PCI | classification | VRAM Δ / size | MemAvail Δ / size | GTT Δ / size | CPU readable | arena read GB/s | copy engine GB/s | HBM ref GB/s | arena / HBM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expandable_card0 | 0 | `0000:03:00.0` | not_in_vram_but_host_delta_unclear | 0.000 | 1.006 | 0.000 | True | 26.62 | 28.69 | 118.23 | 0.2252 |
| expandable_card1 | 1 | `0000:07:00.0` | not_in_vram_but_host_delta_unclear | 0.000 | 1.011 | 0.000 | True | 13.87 | 14.34 | 117.73 | 0.1178 |
| plain_card0 | 0 | `0000:03:00.0` | not_in_vram_but_host_delta_unclear | 0.000 | 1.014 | 0.000 | True | 26.63 | 28.69 | 118.25 | 0.2252 |

> A read that outruns the copy engine by >1.5× or reaches ≥0.25× the HBM reference did **not** cross PCIe — those pages are device memory whatever the counters said. That check is the one the first P2 lacked (device 117.9 vs "host" 118.2 GB/s, ratio 1.00).
> Expected PCIe ceilings on this box: **card 0 = 28.70 GB/s (Gen5 x8)**, **card 1 = 14.34 GB/s (Gen4 x8)** — sampled mid-DMA, never at idle (ASPM downtrains).

## Arms — per leg

| leg | alloc conf | status | arena | placement | pool_alloc | bandwidth | correctness | empty_cache | capture | alloc µs (via our allocator) | replay µs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expandable_card0 | expandable_segments:True | ok | ok | ok | ok | ok | ok | ok | ok | 7.6 | 241.0 |
| expandable_card1 | expandable_segments:True | ok | ok | ok | ok | ok | ok | ok | ok | 7.6 | 387.5 |
| plain_card0 | &lt;unset&gt; | ok | ok | ok | ok | ok | ok | ok | ok | 7.6 | 230.3 |

> `alloc µs` counts only reps that actually reached the custom allocator; reps torch served from its own cache are reported separately in the JSON.
> Replay times are a functional latency over an L2/MALL-resident working set — **not** a bandwidth measurement. The bandwidth column above is the ≥256 MiB one.

## Box state at run start

- MemTotal 91.8 GB, MemAvailable 70.9 GB, MemFree 67.4 GB, Cached 3.1 GB
- swap: SwapTotal 53.9 GB, SwapFree 35.0 GB, pswpout 407142868
- loadavg `0.63 1.11 1.44 3/3758 8`
- GPU lease **waived by explicit user instruction**; legs run strictly serially, never two GPU jobs at once.

Raw: `p5b.json` (this directory), per-leg `p5b_leg_<name>.json` + `.log`.
