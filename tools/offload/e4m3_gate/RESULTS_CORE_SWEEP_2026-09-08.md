# CPU expert tier: it serves, and it is NOT core-bound — 2026-09-08

Four legs, same box state, same harness, back to back on a freshly rebooted box.
`tools/offload/run_cpu_tier_serve.sh`, qwen4exp TP=2 CONC=2 MEM_RATIO=0.75 GRAPH_BS=0, all MoE
off-device (`WOFF_DEVICE_GB` unset → serve.sh's all-host default), 3 reps × 128 decode tokens.

## 0. The headline: the tier runs, and the deadlock is gone

`§3B` of `RESULTS_e4m3_2026-09-07.md` recorded a TP-overlap deadlock at 2, 12 and 48 CPU layers and
never at 0, which is why **no throughput number had ever existed for any CPU-tier configuration**.
Four legs now completed with `bench rc=0`. The fix is in `models/qwen3_5_moe.py`: pass
`num_chunks=1` to `rowchunked_ar_span` when the expert seam reports `computes_on_cpu`, so the CPU
layer takes the plain `produce + one all_reduce` arm instead of running its blocking `.to("cpu")`
inside a chunked async-collective region.

It is SCOPED, and the logs prove the scope held: `TP comms/compute overlap ENGAGED (side stream;
row chunks=2)` appears in the same run as `[cpu-moe] ... layers_registered=12`. The 36 GPU-streamed
layers keep their overlap; only the 12 CPU layers give it up.

## 1. Served numbers

| arm | CPU layers | cores/rank | M1 TPOT ms | M1 tok/s | M2 tok/s | prefill tok/s |
|---|---:|---|---:|---:|---:|---:|
| `cpu0base` | 0 | — | **68.22** | **9.54** | 11.21 | 59.7 |
| `cpuT2` | 12 | [1,2] / [3,4] | 70.36 | 9.42 | 11.21 | 56.0 |
| `cpuT3` | 12 | [1,2,3] / [4,5,6] | 70.59 | 9.46 | 11.40 | 56.6 |
| `cpuT4` | 12 | [0,1,2,3] / [4,5,6,7] | 72.04 | 9.29 | 10.96 | 57.6 |

Medians of 3 reps. Against the 0-layer baseline: T=2 **−3.04%**, T=3 −3.35%, T=4 −5.29%.

**M2 is not a usable channel at 3 reps.** Its spread within a single arm (T=2 saw 107.0, 124.0,
127.0 ms) exceeds every between-arm difference. M1 is stable to ~1% and is the only channel any
claim below rests on.

## 2. MORE CORES DOES NOT HELP — measured, not projected

This was run because the arithmetic said it would not help and that arithmetic rested on a
microbench nobody had confirmed described the live tier. It does describe it, and the answer stands.

Native compute, **decode bs=1 windows only** (counter deltas where `Δnative_tokens == Δlayer_calls`,
n=54 per arm):

| arm | p10 | median | p90 | × 12 layers |
|---|---:|---:|---:|---:|
| T=2 | 0.575 | **0.620** | 0.675 | 7.44 ms/step |
| T=3 | 0.565 | **0.600** | 0.680 | 7.20 ms/step |
| T=4 | 0.590 | **0.665** | 0.755 | 7.98 ms/step |

T=2's 0.620 reproduces `docs/CPU_MOE_OFFLOAD.md` §1.1's 0.621 ms/layer exactly. T=3 buys 3%, not
the 18% §1.1 projects, because §1.1 measured ONE pool on a quiet box: at TP=2 there are TWO pools on
one memory controller, so 3 threads/rank is 6 node-wide against a bus §1.1 itself shows saturating
at 3. **T=4 is worse in the kernel** (+7%) as well as end to end — 8 threads node-wide is past
saturation, and it also takes core 0, the only core that boosts and the one the engine's dispatch
thread runs on.

**Why the ceiling is low regardless:** native compute is 7.44 ms of a 70.4 ms step = **10.6%**. A
free kernel would buy ~10%; a 3× core increase moved the served number by ≤2.3% in either direction.

`CoreBudget`'s refusal is therefore protecting something real and stays a refusal. The override
added here (`MINISGL_CPU_MOE_CORE_BUDGET`) exists so the opposing measurement is *runnable* — the
derived cap rests on `engine_cores`, which `provenance` admits is one observation, and a cap resting
on a single measurement must not be the thing that makes its own falsification impossible.

## 3. Where the step actually goes (`MINISGL_HOSTPROF`, steady-state decode M1)

| | total | fwd_launch | gpu_wait |
|---|---:|---:|---:|
| 0 CPU layers | 68.31 ms | 25.31 (37%) | **41.86 (61%)** |
| 12 CPU layers, T=2 | 70.86 ms | **68.05 (96%)** | 1.05 (1%) |

The tier does not add 42 ms of overhead — it **destroys host run-ahead**. `cpu_forward` is BLOCK
mode (submit and join in one call), so the host can no longer be ahead of the device: what was
41.9 ms of `gpu_wait` becomes GPU time billed inside `fwd_launch`. The net is +2.1 ms/step.

> An earlier reading of this session called the difference "~34 ms of handoff". That was wrong: it
> compared against the 0-layer hostprof recorded in commit `cd40a9ab`, a different configuration and
> box state. Against a same-session baseline the net is +2.1 ms, and the mechanism is serialization,
> not per-call overhead.

## 4. So what is the tier FOR

Not throughput: at 12 layers it costs **3%**. It is a CAPACITY feature, and that is the number to
quote — 12 CPU layers moved 7.98 GiB/rank out of the pinned host arena (`36 host (23.950 GiB INSIDE
the pinned arena) ... 12 cpu-compute (7.983 GiB PAGEABLE, proven off-device)`). Pinned arena is the
binding constraint on this box (see `pinned-host-ceiling-measured-12gib-not-28`), so trading 3% of
decode for 8 GiB/rank of pinning relief is the trade to evaluate — against KV pool and context, not
against tok/s.

**Still unmeasured: 48 CPU layers**, the configuration where the pinned tier is empty entirely. At
0.620 ms/layer that is ~29.8 ms/step of added CPU compute against the ~7 ms/step of GPU MoE it
removes, so expect a materially worse decode than 12 layers buys. That leg is the one that prices
the capacity ceiling.

## 5. Defects fixed to make any of this measurable

1. **`_serve_probe.py` could never pass on a thinking model.** It asked for `max_tokens=8` and read
   only `message.content`. Measured: "Say READY and nothing else." costs 24 reasoning tokens before
   a 1-token answer, so at 8 tokens the think span is still open and `reasoning.py` files the whole
   reply under `reasoning_content` with `content=""` — read by the probe as "generated nothing". A
   serve answering correctly in 6.4 s was reported NOT READY for the full timeout. Now 64 tokens,
   and reasoning counts as generation.
2. **Three knobs never reached the container** (`MINISGL_CPU_MOE_THREADS` / `_CORES` / `_STATS`) —
   the five-hops trap. `MINISGL_CPU_MOE_STATS=200` had been exported by this harness for four prior
   legs and silently did nothing, which is why per-layer cost had never been collected and why §3B
   could only guess at the split. Every number in §2 exists because this was plumbed.
3. **`default_core_list` hands out SMT siblings at ≥4 threads/rank.** `base = 1 + rank*threads` is
   unbounded, so at ranks=2/threads=4 rank 1 gets `[5,6,7,8]` — and cpu8 is cpu0's sibling
   (verified: `thread_siblings_list` = "0,8"), i.e. §1.4's ~50% penalty, on the engine's own core.
   Unreachable while the cap was 5; reachable the instant it is raised. Now a refusal that names the
   fix.
