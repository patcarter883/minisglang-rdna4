# Per-expert VRAM cache — BUILD. The decision, the numbers, and the architecture it forces

Measured 2026-09-08 on a REAL qwen4_exp routing trace. Instruments: `weights/route_trace.py`
(capture), `tools/offload/expert_cache_oracle.py` (simulate). Trace: 38 MB/rank, 13 distinct
prompts, 20,004 warm decode steps, 9.6M references, sampled at temperature 1.0, captured on the
38-host/10-device arm. Rank 0 and rank 1 records are BYTE-IDENTICAL (the only differing byte in the
file is offset 24, the `tp_rank` header field) — both ranks routed the same experts, which TP
requires and whose failure would have invalidated everything.

## 1. The decision — all gates pass at the budget we ALREADY have

| policy | h_decode | ms/step | tok/s | vs shipped | derated |
|---|---:|---:|---:|---:|---:|
| **static-layer (SHIPPED)** | 0.2083 | 60.06 | **16.65** | 1.000 | 16.65 |
| lfu | 0.3597 | 54.13 | 18.47 | 1.109 | 17.56 |
| static-prior | 0.5559 | 46.44 | 21.53 | 1.293 | 19.09 |
| **lru** | 0.8558 | 34.70 | **28.82** | 1.731 | 22.73 |
| **slru** | **0.8644** | **34.36** | **29.10** | **1.748** | **22.88** |
| belady (bound) | 0.9369 | 31.52 | 31.72 | 1.905 | 24.19 |

    G1 PASS   h_Belady = 0.9352 >= 0.40      (exploitable structure exists)
    G2 PASS   h_LRU    = 0.8529 >= 0.7014    (LRU captures >=75% of the offline optimum)
    G3 BUILD  h*       = 0.8605 (slru)       -> 34.51 ms = 28.97 tok/s = 1.740x, 1.370x derated
    G4 OK     prefill pollution moves LRU by -0.0008 (LFU by +0.028); insert policy is not a lever
    G5 OK     13 uids / 20,004 steps / temp>0 / rank0 == rank1 records

## 2. Capacity curve — the cache substitutes for VRAM we cannot buy

| budget/rank | coverage | static-layer | lru | slru | belady |
|---|---:|---:|---:|---:|---:|
| 3.35 GiB | 10.6% | 15.59 | 24.58 | 24.78 | 28.64 |
| **6.7 GiB (today)** | 21.2% | **16.65** | 28.82 | **29.10** | 31.72 |
| 10 GiB | 31.6% | 17.86 | 31.32 | 31.48 | 33.10 |
| 13.4 GiB | 42.4% | 19.27 | 32.79 | 32.87 | 33.78 |

Full residency — all 48 layers in VRAM, needing ~32 GiB/rank we do not have — is **34.4 tok/s**.
**SLRU at 13.4 GiB reaches 32.87 = 95% of that on 42% of the bytes**, and at HALF today's budget it
still returns 24.8 against static's 15.6.

## 3. WHY every earlier reading was wrong — the locality is TEMPORAL, not distributional

    per-layer Gini of expert access counts   0.6135  (0.4306 .. 0.7196)   <- only moderate skew
    top-C mass at 6.7 GiB                    0.6476                        <- only moderate
    stack distance, frac(d <= slots)         0.8404                        <- THIS is the LRU hit
    static-prior half-split @108/layer       0.5540 vs uniform 0.2109
    adjacent-decode-step route overlap       0.3708  (llama.cpp ~0.442)

The checkpoint carries `router_aux_loss_coef: 0.001`, and it does what it says: it flattens the
MARGINAL expert load, which is why the distribution-based policies do badly — **static-prior reaches
only 0.556 and LFU only 0.360**. But an aux loss places NO constraint on the AUTOCORRELATION of the
route sequence, and that is what a recency policy eats: `frac(stack distance <= cache size)` IS the
LRU hit condition, and it is 0.84.

This is the exact step the plan's reasoning skipped. "Linear miss curve -> hit rate == resident
fraction -> routing unskewed -> LRU cannot beat static" — measured, the first link does not even
mean what it was read as (see `RESULTS_PLACEMENT_SWEEP_2026-09-08.md` §5 and the P2-prime audit),
and the third link is false: routing IS near-uniform in its marginals and is still hugely
cacheable. **Skew and locality are different properties. The oracle separates them; the prior
analysis conflated them, and it cost 12 tok/s.**

## 4. THE ARCHITECTURE THIS FORCES — the manager can be fully ASYNCHRONOUS

The one design question that decides the build: must the cache manager know `topk_ids`
synchronously? A synchronous route costs a D2H per MoE layer per step (~0.13 ms on card 1 x 48
layers = ~6 ms/step) and would eat a third of the win.

MEASURED with `--lag-steps` (the manager observes references N steps late; hits are still evaluated
against what is resident NOW, which is the physical situation — the kernel dereferences whatever
`slot_of` currently says and the manager's update lands later):

    lag =  0 steps   h_lru = 0.8558
    lag = 64 steps   h_lru = 0.8563      <- IDENTICAL

**A 64-step-stale manager costs nothing**, because the per-layer stack distance is P50 29.9 / P90
138.4: the hot set is stable over hundreds of steps. So the manager drains the route ring
`weights/route_trace.py` already builds, updates residency in the background, and **the hot path
takes no host sync at all**.

## 5. The implementation, and why it is not a new kernel

`KERNEL_CORE_POLICY.md` governs: a new residency scheme is a **WLoad policy on the existing shared
core**, never a fork. The seam is already a policy call —
`fp8_wmma/fp8_wmma_rocm/moe_gemm_flag.h:64`:

```cpp
const long e = expert_ids[block_idx];
const typename WLoad::WT* wq_e = WLoad::wq_expert(w_data, e, N, K);   // w + e*N*K
```

so the change is `slot_of[e]` indirection inside `wq_expert`, plus a second base pointer:

    slot = slot_of[e]                      // device-resident int32 table, per layer
    base = (slot >= 0) ? w_dev  : w_host   // VRAM slab, else the pinned host arena
    idx  = (slot >= 0) ? slot   : e

One `if constexpr` and two kernel args — the same shape as the e4m3 policy the CPU tier added.
**Stable VAs by construction**, so it is graph-capture-safe with plain `hipMemcpyAsync` and no VMM
(which is dead here anyway — [[hip-vmm-remap-silently-broken-gfx1201]]).

Correctness rests on ONE invariant: **a slot marked resident must hold that expert's bytes.** A
stale table is otherwise safe in both directions — it only ever means "read from host", which is
correct and merely slower. That is what makes the asynchronous manager sound.

### Phases
1. `WLoad` slot indirection + the device table, with a parity gate against the non-cached arm.
2. Host cache manager: VRAM slab, SLRU over (layer, expert), copy stream, ring drain. Off the
   critical path by §4.
3. Wire, A/B against static-layer at 6.7 GiB, and check the realized-vs-nominal gap.

### The build risk, stated up front
The 1.370x "derated" column applies the llama.cpp ROCm haircut (that project measured +3.2%/+15.3%
REALIZED against 81-93% nominal hit rates on dual 7900 XTX, because its cache hot path cost 2.99x
stream syncs, 9.80x H2D ops and 12.8x H2D engine time, and misses still ran dummy zero-expert
kernels). Our nominal is 1.748x. **Phase 3 must measure realized, not assume it** — and if the gap
looks like theirs, the cause will be the manager's hot-path cost, which §4 says we are free to move
entirely off the critical path.

### Known limits of this trace
One workload shape: single-turn, long generations, 13 prompts, CONC=1 (sequential), decode-heavy.
Multi-turn chat with prefix reuse, or CONC=2 interleaving two sequences, would both reduce locality
and are not measured here. The capture's ring is gated on `M == 1` (its buffer holds one row), so a
CONC=2 trace needs the ring widened to `M_max * top_k` first — 20 int32/layer instead of 10.
