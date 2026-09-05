# qwen4_exp endgame — measured 2026-09-04/05

Written by the session owner after the workflow's report agent hit a hard usage limit. Everything
below is taken from the four agent results that DID complete (merge, attribute, serve, plus the
capture A/B from the prior run). **MEASURED** and **PROJECTED** are marked; nothing here is
smoothed.

---

## 1. Headline

| | measured |
|---|---|
| 48-layer TP=2 offloaded serve, graphs ON | **12.54–12.59 tok/s** |
| same, CPU expert tier ON (36 CPU layers) | **5.49–5.50 tok/s** |
| target (no MTP, no spec decode) | 23.3 tok/s |

**The CPU expert tier is 2.28x SLOWER and is switched OFF.** The best split is
`device 11 / host 37 / cpu 0` — the sweep is monotone against the tier at **+0.976 ms/step per
layer moved**, so there is no interior optimum.

Quality was never the problem: both legs are coherent at 200 sampled tokens (checkpoint sampler,
temp 1.0 / top_k 20 / top_p 0.95), correct heat-pump physics, and clean against all four
degeneration signatures. This is purely a throughput verdict.

---

## 2. The premise this whole detour rested on is REFUTED

> "decode is PCIe-bound — 41.72 ms of host-expert reads = 49% of the step"

**Wrong, and the disproof is arithmetic on the measurement**: the `cpu=0` step is **26.5–27.0 ms
TOTAL**, which is *less than the 39.25 ms attributed to host-expert PCIe reads alone*. And if the
premise held, moving 36 of 37 layers OFF PCIe could not have made decode slower — it made it 2.28x
slower.

A host-PCIe layer costs **~0.18 ms**, not the 2.12–2.49 ms prior the CPU tier was justified against.
The 0.517 ms/layer kernel figure was never wrong; **the seam around it is the cost, and no standalone
benchmark could see it.**

Root cause of the mis-attribution: the transfer is not a separate H2D copy. The grouped NVFP4 kernel
dereferences the pinned-arena host pointer directly, so **the transfer IS the kernel** and is fully
overlapped within a layer. Substituting CPU compute for an already-free transfer adds pure serial
latency. Every comparison of "CPU compute time vs PCIe transfer time" was measuring two things that
do not trade off.

**Lesson, recorded because it recurred three times today:** a microbenchmark of a component cannot
price a seam. The kernel hit its ceiling (55.46 GB/s, then 27.38 GB/s/core int8-VNNI — both real);
the integration lost 2.3x.

---

## 3. Where the "32.9 ms gap" went — it was mostly measurement error

| source | ms |
|---|---|
| `prior.compute_floor_ms = 7.5`, a Phase-0 GUESS (`weights/prior.py:68-72`, documented as the midpoint of an unmeasured 5–10 ms bracket) vs a MEASURED 26.7 ms of non-expert device work | **~19** |
| the 79.86 ms baseline itself — a WALL figure dividing a run containing 8 prefill forwards by 119 decode steps, in a harness whose own doc records 41/59/80 ms across three boots | remainder |

The same operating point measures **62.36 ms/step**, reproduced to 0.5% across two independent boots
(62.36 / 62.57). The serve round's forward-only steady decode at `cpu=0` is **26.5 ms/step**.

**UNRESOLVED:** 62.36 (attribution boot) vs 26.5 forward-only + 79.7 end-to-end (serve boot) do not
reconcile against the 2.69 ms of measured sampler/scheduler/detokenize. Different boots, possibly
different `--memory-ratio` or batch. **The operating point's true ms/step is therefore not settled**
and should be pinned before any further perf claim.

---

## 4. The two real levers, both untouched

### 4.1 Rank-0 idle on link asymmetry — 20.59 ms/step, 33% of the step

Rank 0 sits idle inside the 48 MoE all-reduces waiting for rank 1, because the expert tier is
sharded **50/50 across two links that differ 1.92x** (card 0 Gen5 x8 28.93 GB/s; card 1 Gen4 x8
14.48 GB/s — see `card1-root-port-trained-gen4-x8`, riser parked as unfixable hardware).

An **asymmetric expert shard matched to link speed** is a planner change, not a kernel change, and it
is the single largest identified win. NOT ATTEMPTED.

### 4.2 Hyper-connections are launch-bound, not bandwidth-bound — 7.9 of 10.15 ms

Of the 97 HC blocks' 10.15 ms, only **2.64 ms is bandwidth**; the other **7.9 ms is ~1000 launch-bound
kernels on 20 KB tensors**. A fusion target. NOT ATTEMPTED.

Measured step decomposition (attribution boot, eager, instrument overhead removed):

```
host_expert_PCIe_read_37_layers   41.72   <- see §2; this number is now SUSPECT
hyper_connections_97_blocks       10.15
gdn_mixers_36_layers               5.54
moe_arith + inter-kernel RESIDUAL  3.56
sampler/scheduler/detokenize       2.69
full_attn_mixers_12_layers         2.27
moe_shared_expert + gate x48       1.82
device_resident_experts_11_layers  1.70
moe_all_reduce x48                 1.26
lm_head (248,320 vocab)            1.00
moe_router_gate x48                0.65
ple_ngram_gather                   0.29
embed                              0.12
```

---

## 5. What the CPU tier DID buy, and what it cost

**Bought:** **52.73 GiB of pinned arena freed** (measured). The planner's claim that a CPU layer costs
zero device AND zero pinned bytes is verified: swept `--weight-offload-device-gb` in {0,2,5,8,40} x
`num_cpu_layers` in {0,21}, the device tier is BYTE-IDENTICAL in all 10 cells, and 21 CPU layers
relieve exactly 15.381 GiB of arena while holding 13.843 GiB pageable. That makes the morning's
capacity fight moot regardless of throughput.

**Cost:** 2.28x decode. Roughly half of the per-layer cost is **handoff, not compute** — `cpu_submit`
does THREE separate synchronising `.cpu()` copies (hidden state, weights, ids).

**Also blocking:** graph capture with any CPU layer is **architecturally incompatible**, not merely
unwired — `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture unless
the CPU tensor is pinned`. Pinning is necessary but insufficient: CPU compute must happen BETWEEN
device segments and a captured graph replays as one unit, so **segmented capture (K+1 segments) is
required**. `cpu_tier.graph_segments()` models the cost; nothing implements it. All CPU-tier numbers
are therefore EAGER, with an eager baseline for a like-for-like A/B.

---

## 6. Two silent bugs the integration exposed

1. **`plan.cpu_tier_gate` was keyed on the wrong namespace.** `diagnostics['schemes']` carries scheme
   KINDS from `build_planned_layers` (the config fallback) but CONTAINER CLASS NAMES from
   `observed_planned_layers` (the path every real serve takes). The gate refused the CPU tier on
   **100% of real serves** while passing every unit test, because the tests drive the config path.
   FIXED.
2. **`bind_seam` billed every CPU layer against `device_resident_bytes`** — the one independent
   measurement `assert_device_accounting` compares against the plan, and the term the KV pool is
   sized from (~0.6 GiB of phantom VRAM at 21 layers). FIXED, with a test.

---

## 7. Still undischarged

- **vision**, **MTP** (the checkpoint ships a 4B head; ruled out as an unfair comparison), **QSA
  indexer above 2048 context**, **dense-linear offload**
- **e4m3 scale policy** — a MEASURED 4.37e-04 max weight rel-err on the NVFP4 path, ~4 days, legal as
  a WLoad policy, blocked on templating two hand-written non-WLoad kernels at `moe_kernel.hip:299`
  and `:450`. Independently corroborated by the CPU work: the checkpoint-native e4m3 scale is **10%
  smaller AND ~1800x more accurate** (1.99e-07 vs 3.58e-04) than the GPU's fp16 fold.
- **`ct_sign_cross_rank`** — reported "0 CT containers in this checkpoint" = **UNTESTED, not cleared**
- **capture above bs=2**, **prefill capture**, **`--memory-ratio 0.96` WITH capture** (both capture
  boots ran 0.90)
- **`test_kv_budget_keeps_exactly_five_subtrahends`** — pre-existing failure, carried, not fixed
- **The §3 ms/step reconciliation.**

## 8. Housekeeping

`/home/pat/.cache/hf-awq` (**176 GiB**) is reclaimable — AWQ was never needed once TP=2 landed. The
box is at **97% disk, 69 GB free**. User's call.

Format finding worth carrying: this checkpoint's `weight_scale_2` is a **MULTIPLIER**, the reciprocal
of what `quant/nvfp4.py` documents — pinned four independent ways. These experts are **not served
through the NVFP4 fold path at all**.
