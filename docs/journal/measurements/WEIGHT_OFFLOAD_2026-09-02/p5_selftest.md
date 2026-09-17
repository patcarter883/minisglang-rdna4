# P5 — torch over a foreign device pointer (in the serve image, under capture)

**Run:** 2026-09-02T17:05:15+0000 · host `98455b815417` · in-container `True` · schema `p5/1`
**THIS IS A SELFTEST ARTIFACT — no GPU was touched and NOTHING below was measured.**

## Answers

| Question | Answer |
|---|---|
| `t.data_ptr() == reserved_base` under `use_mem_pool` | not measured |
| coexists with `expandable_segments:True` (compose default) | not measured |
| graph capture + replay over the foreign VA is correct | not measured |
| arena tensor survives `torch.cuda.empty_cache()` (graph.py:314) | not measured |
| `empty_cache()` invokes the custom **free** cb for a **live** pool block | not measured |
| `empty_cache()` invokes it for a **cached (dropped)** pool block | not measured |
| host-located `hipMemCreate` pages behind a device VA (**bytes verified**, not just `hipSuccess`) | not measured |
| any `hipMalloc` fallback inside the user MemPool | not measured |
| **COLLATERAL** — torch's *own* allocator returns stale memory after `empty_cache()` under `expandable_segments:True` | not measured (reps/leg: None) |

**Exit classification:** n/a

**Reasons**

- selftest: nothing was measured

## Legs

| leg | alloc conf | backing | status | reservation (bytes verified) | pool_alloc | correctness | empty_cache | capture | granularity B | alloc µs (median, via our allocator) | replay µs (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expandable_host | expandable_segments:True | host | selftest | - | - | - | - | - | - | - | - |
| expandable_device | expandable_segments:True | device | selftest | - | - | - | - | - | - | - | - |
| plain_host | <unset> | host | selftest | - | - | - | - | - | - | - | - |

> Replay times are a functional latency over an L2/MALL-resident working set. They are **not** a bandwidth measurement — that is P1's job, on a ≥256 MB working set.
> `alloc µs` counts only reps that actually reached the custom allocator; reps torch served from its own cache are reported separately in the JSON.

## Box state at run start

- MemTotal 91.8 GB, MemAvailable 71.9 GB, MemFree 68.9 GB, Cached 2.9 GB
- swap: SwapTotal 53.9 GB, SwapFree 34.2 GB, pswpout 407142868
- loadavg `2.84 2.15 2.84 1/3791 8`


Raw: `p5.json` (this directory), per-leg `p5_leg_<name>.json` + `.log`.
