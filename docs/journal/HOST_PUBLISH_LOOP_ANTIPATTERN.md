# Antipattern: host-side per-position state publish (spec-decode verify)

Written 2026-07-28 out of the vhip (vLLM 0.24 HIP) spec-decode work, where this pattern was found,
measured, fixed, and the fix validated bit-exact. minisgl has the **same** pattern in three places,
all on the spec-decode verify hot path. This note records the evidence so the fix here is a measured
decision rather than a rediscovery.

## The shape of it

A verify kernel computes the recurrent state after *every* query position into a **scratch** tensor,
then host Python gathers the one position that was actually accepted and scatters it into the paged
state cache:

```python
seq_ar = torch.arange(n, device=slots.device)
for lid in range(self.num_gdn_layers):            # <- per LAYER, on the hot path
    cs, ss = conv_scratch[lid], ssm_scratch[lid]  # [Q, N, ...]
    self.conv_state[lid, slots] = cs[t_index, seq_ar].to(self.conv_state.dtype)
    self.ssm_state[lid, slots]  = ss[t_index, seq_ar].to(self.ssm_state.dtype)
```

The fused alternative: hand the kernel the **2-D slot table** (and the acceptance selector) and let it
store each position straight into `cache[slot[n, t]]` inside its own token loop. No scratch, no
gather, no scatter.

## Why it matters — MEASURED, not asserted

From the vhip port (Qwen3.6-35B-A3B, TP=2, MTP K=2, 30 GDN layers, bs=1, qlen=3, standalone
per-layer timing on gfx1201):

| phase | us/layer | ms/step x30 layers |
|---|---|---|
| the recurrence kernel itself | 104.4 | 3.13 |
| **the host publish loop** | **141.4** | **4.24** |

**The publish loop cost more than the kernel it existed to serve.** It moves ~3 MiB in 141 us
(~21 GB/s), i.e. it is op-overhead and `.to()`-copy bound, not bandwidth bound — so it does not get
better with a faster card, only with fewer ops.

Fixing it in vhip: the verify kernels gained a templated `INPLACE` publish policy (one core, two
store policies — per `KERNEL_CORE_POLICY.md`, NOT a forked kernel), taking `slot_table [N, max_qlen]`
+ `num_accepted [N]`. Validated **bit-exact** against the scratch+scatter path across fp32/bf16 state,
int32/int64 slot tables, several acceptance counts, and ragged query lengths.

### The end-to-end result (this is the number to plan against)

| gdn_hip spec-decode, Qwen3.6-35B-A3B TP=2 | bs=1 | concurrency-4 | accept_len |
|---|---|---|---|
| scratch + Python publish loop | 53.8 tok/s | 143.4 | 2.51/3 |
| **fused in-kernel publish** | **74.1 tok/s** | **194.0** | **2.511/3** |
|  | **+37.7%** | **+35.3%** | unchanged |

**WARNING for whoever sizes this work: the standalone microbench UNDER-predicted the win by ~3x.**
It measured the publish loop at 141 us/layer = 4.24 ms/step, so ~4 ms of gain was expected. In-serve
the fusion was worth ~12.8 ms/step. A warm, N=1, tight-loop microbench is a LOWER BOUND on host-op
cost — the same ops cost far more once they compete with a full model forward for the launch queue.
Do not deprioritize this on microbench numbers.

Note the accept_len is unchanged to three decimals, which is the correctness signal to watch: a
publish that installs the wrong state shows up as an acceptance collapse, not as a crash.

**vLLM 0.24 does not have this antipattern anywhere** — it is the reference for the good pattern.
`v1/attention/backends/gdn_attn.py:58` declares `spec_state_indices_tensor` as `[batch, num_spec]`,
and `qwen_gdn_linear_attn.py` / `olmo_gdn_linear_attn.py` forward it plus `num_accepted_tokens` into
`fused_sigmoid_gating_delta_rule_update_kernel`, which does the per-position store in-kernel
(`INPLACE_FINAL_STATE`). `mamba_utils.py:306-309` goes further and *asserts* that the
`num_accepted_tokens > 1` case "must be handled by the fused postprocess kernel" — they explicitly
refuse to do the offset publish on the host.

## The three sites in minisgl (all verified by reading the code, 2026-07-28)

### 1. `python/minisgl/kvcache/gdn_state.py:139-148` — `GDNStateCache.install_verify_state`
Loops over **layers** (the `t` axis is folded into the scratch's leading `Q` instead).
* ~`num_gdn_layers` x 6 ops (2 advanced-index gathers, 2 dtype casts, 2 `index_put_`) + 1 `arange`.
  Qwen3.6-35B-A3B has **30** linear-attention layers (of 40) -> **~181 launches per verify step**.
* Callers at `python/minisgl/scheduler/scheduler.py:3683-3690` add 2 host->device `torch.tensor(...)`
  constructions (`sel`, `t_index`) built from Python lists — H2D per step.
* `python/minisgl/models/qwen3_5.py:220-223` adds 2 more full-scratch `copy_` per layer on the
  cudagraph path (+60 ops/step) to keep the captured pointer valid.
* The verify kernels already receive `state_idx` and the live cache (`gdn/layer.py:411,427`), so the
  slot side needs no new plumbing.

### 2. `python/minisgl/cca/metadata.py:146-158` — `capture_cca_verify_state`
The **producer** side, done entirely in Python: a per-sequence loop building the whole scratch with
`cat` / `transpose` / `unfold` / slice-assign, and `int(seg[i])` forces a host read per sequence.
~5 ops x `num_seqs`, **per CCA layer**. Likely the most expensive of the three (per-layer AND
per-sequence). There is no kernel here to hand a slot table to — fixing #3 requires fixing this first.

### 3. `python/minisgl/kvcache/cca_state.py:119-124` — `CCAStateCache.install_verify_state`
Structurally identical to #1 (its own docstring says so). ~`num_cca_layers` x 6 ops. Two hot call
sites: `scheduler/scheduler.py:2981` (TiDAR fused path) and `:3699`.

## The one real blocker — why we cannot just copy vLLM

**minisgl computes `t_index` AFTER the verify forward.** `scheduler/scheduler.py:3683-3690` runs
post-sampling, once acceptance is known, and only then installs. vLLM instead consumes the
**previous** step's `num_accepted_tokens` at the **start** of the next forward, which is what lets its
kernel do the selection internally.

So there are two routes, and they are not equivalent in effort:

* **(a) One publish kernel** — collapse all ~181 launches into 1 by passing
  `(conv_scratch_stack, ssm_scratch_stack, slots, t_index)` to a single kernel that does the
  gather+scatter for every layer at once. Keeps the current scheduler structure and the scratch.
  Mechanical, low risk, captures most of the launch win.
* **(b) vLLM-style deferral** — carry `num_accepted` into the *next* forward and publish in-kernel,
  which removes the scratch entirely (its allocation, its write, and `qwen3_5.py`'s `copy_`). Strictly
  better steady-state, but it is a scheduler restructure and changes the state-install contract.

Start with (a) and measure before considering (b).

## Rule of thumb

If a kernel is writing per-position state to a scratch so that Python can pick one position and
scatter it, the kernel should have been given the slot table. On this box a decode step is
inter-kernel-gap bound, so ~180 extra launches per step is not a rounding error — it was 57% of the
GDN layer cost in the case we measured.
