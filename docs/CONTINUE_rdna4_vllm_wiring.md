# CONTINUATION PROMPT — wire every RDNA4 HIP kernel into vLLM 0.24

Paste this whole file as the opening prompt of a new session.

---

## Task

Land **every** RDNA4 HIP kernel that has a real vLLM 0.24 call site, in `vllm24-hip:combined`, under
these binding constraints from the user:

* **No Triton fallbacks** on any wired path — unsupported shape/dtype must RAISE, never silently defer.
* **No monkeypatching** — register through supported vLLM hooks. Where 0.24 genuinely has no hook
  (sampler, `rocm_unquantized_gemm`), use a **baked source patch in a derived image**, which is
  existing precedent (`rocm_hip_attn.py` already ships as an in-tree fork).
* **Highest-performing variant**, **no hacks, no casts, no copies.**
* Maximum effort — do not skip a path because it looks unused.

Read `docs/RDNA4_VLLM_WIRING_SPEC.md` first — op inventory, per-op override points (file:line), and
the clean "cannot be wired, no call site" list.

## Where to work

Worktree `/home/pat/code/minisgl-rdna4-vhipspec` (branch `task/vhip-glm-spec`), **not** the shared
tree. Nothing is committed. Launch: `tools/vhip_launch.sh qwen` (no-spec) / `qwen-mtp` (MTP).
Stop + release lease: `docker compose -p lease-vhip-qwen[-mtp] down`.

Boot is ~131 s. Do NOT re-derive the all-stock number by booting `VLLM_PLUGINS=""` — that hangs
>20 min in the cold GDN Triton compile. The recorded all-stock figure is **92 tok/s decode**.

`VHIP_CMD=<shell>` (added this session) runs anything in the vhip service with the serve's exact
mounts + env instead of the server — that is how you profile or run a kernel test against the same
overlay set. The `run` compose profile carries none of the vhip mounts; do not use it for this.

## MEASURED decode profile — this is now the map, use it

Cudagraph-on trace of the CURRENT stack (`tools/profiling/prof_out`, no-spec, 32768 ctx, MNS=8, fp8
KV, TP=2, `--dtype float16`). GPU-active 76.7% busy / 23.3% idle-gap, 1322 dispatches/token.
Decode kernels, prefill one-shots excluded:

| kernel | % of decode GPU | calls/token | avg |
|---|---|---|---|
| `fused_moe_kernel_gptq_awq` (STOCK TRITON) | ~26% | 80 | ~21 µs |
| `ncclDevKernel_Generic_4` | ~25% | 82 | 42 µs |
| `wvSplitK_hf_sml_<__half>` (UNQUANTIZED dense) | ~16% | 193 | 11.6 µs |
| `gdn_decode_conv` | ~5.5% | 29 | 26.3 µs |
| `gemv_decode Bf16GemvLoader` (our LM head) | ~5.4% | 1 | 750 µs |
| `Cijk_..MT16x16x32` (rocBLAS) | ~5% | 40 | 17.5 µs |
| `flash_decode_paged_fp8` | ~3% | 10 | 44.3 µs |

**Two long-standing beliefs are now dead:**

1. The `elementwise_kernel_manual_unroll<128,4>` that memory
   `qwen-decode-88pct-busy-elementwise-dominates` records at **64.4% / 231 µs avg** is now **1.0% /
   2.6 µs avg**. It was the fp32 full-cache shadow copy; the slot-fix + stride-aware state killed it.
   That memory's kernel table is STALE — this one supersedes it.
2. **The LM head is worth zero because it is already at bandwidth**, not because it is inert. It
   ENGAGES (`[rdna4_vllm] lmhead ENGAGED: x(8,2048) fp16 @ w(124160,2048) -> dense_bf16_gemv`) and
   moves 508 MB in 750 µs = **680 GB/s**. Do not tune it. Question closed.

## Measured — ALL at one config (32768 ctx, MNS=8, fp8 KV, TP=2, `--dtype float16`)

decode = server-side `(tokens-1)/delta(vllm:request_decode_time_seconds_sum)`; e2e = `tokens/wall`.
Use `tools/vhip_patches/bench_decode.py` — it prints BOTH from one request, so a row can never again
be ambiguous about which metric it used.

| config | decode | e2e |
|---|---|---|
| stock vLLM (recorded) | **92** | — |
| no-spec, prior baseline (stock Triton MoE at decode) | 83.0–83.3 | 81.5–81.8 |
| **CURRENT DEFAULT: W4A16 MoE + BYLANE + fused epilogues** | **90.5–90.7** | **89.2–89.4** |
| W4A16 unfused, K-on-lanes (why it looked dead) | 71.2 | 69.8 |
| W4A16 + BYLANE, unfused epilogues | 82.4 | 81.2 |
| fla-Triton spec (`VLLM_GDN_HIP_SPEC=0`) | 82.5 | 79.2 |
| gdn_hip spec, fused publish | 77.1 | 74.3 |

## W4A16 MoE — LANDED, VALIDATED, and DEFAULT OFF (measured regression)

Written this session (the previous doc said "never applied"; it is applied now):
`tools/vhip_patches/moe_experts.py` `_run_grouped_moe_w4a16` + a `_moe_should_engage` override,
mounted over `w4a8_vllm/moe_experts.py`. Env `VLLM_ROCM_W4A16_MOE=off|auto|on`.

* **The engage gate was the real bug.** `_moe_should_engage` consults a crossover cache that only
  ever described the fp8-ACTIVATION kernel, whose decode window is empty — so every bs=1 batch fell
  through to **stock Triton `fused_moe`**. "Our fp8 MoE is worth nothing at decode" was really "our
  MoE never ran at decode".
* **Accuracy: a large, clean win.** vs a pure-torch fp16 dequant reference on the real expert shape,
  rel error **0.0006 (W4A16) vs 0.045 (fp8-act)** — ~70×. 4/4 cases incl. symmetric and has_zp, plus
  the no-fallback contract (bf16 in → raises). `tools/vhip_patches/test_w4a16_moe.py`.
* **Speed: −14% (83.0 → 71.2), and it is ALL gemm2.** Per-layer decode:

  | gemm | shape | bytes | best of 32-point NWARPS×COLS sweep | BW |
  |---|---|---|---|---|
  | gemm1 (w13) | K=2048 N=512 | 4.19 MB | 13.5 µs | **310 GB/s** |
  | gemm2 (w2) | K=256 N=2048 | 2.10 MB | **53.2 µs** | **39 GB/s** |

  (launcher's own heuristic picks 125.8 µs for gemm2; stock Triton is ~21 µs per gemm.)

* **Root cause = LANE STARVATION on the K axis** (verified by reading both kernels + a K sweep;
  an earlier framing in this doc blamed the `__shfl` tree — that is a secondary term, not the
  driver). The K loop is `for (base = lane*4; base < K/8; base += 32*4)`: a lane's slot is a
  `v4i_t` = 4 int32 = **32 k-values**, so filling a 32-lane wave needs **K >= 1024**. gemm1's
  K=2048 fills it; gemm2's K=256 gives `base = lane*4 < 32`, i.e. **only lanes 0-7 do any work
  while 24/32 idle through the full 5-step reduction anyway.**

  K sweep at fixed N=2048 (`tools/vhip_patches/bench_w4a16_k_cliff.py`) — 8x the bytes for 1.55x
  the time, i.e. the extra work is free until the lanes fill:

  | K | active lanes | MB | us | **us/MB** |
  |---|---|---|---|---|
  | 256 (gemm2) | 8 | 2.10 | 155.5 | **74.1** |
  | 512 | 16 | 4.19 | 179.8 | 42.9 |
  | 1024 | 32 | 8.39 | 230.4 | 27.5 |
  | 2048 (gemm1) | 32 | 16.78 | 241.6 | **14.4** |

  (K=4096 jumps to 20.8 us/MB: BK=3072 forces a second LDS chunk + `__syncthreads` + re-stage.)

* **What stock Triton does differently.** `fused_moe_kernel_gptq_awq` keeps a
  `[BLOCK_M, BLOCK_N]` fp32 accumulator in registers and reduces over K **sequentially** —
  `accumulator = tl.dot(a, b, acc=accumulator)` inside a `for k in range(cdiv(K, BLOCK_K))` loop.
  K never touches the lane axis, so there is **no cross-lane reduction and no minimum K to keep a
  wave busy**; lane occupancy comes from BLOCK_M x BLOCK_N (the output tile), which is large at
  gemm2's N=2048 regardless of how short K is. Our kernel put K on the lane axis, which is the
  right call at K=2048 and falls off a cliff at K=256.

### GEMV-by-lanes + epilogue fusion — LANDED, +9.3% (2026-07-28)

**W4A16 MoE decode is now the DEFAULT (`VLLM_ROCM_W4A16_MOE=auto`): 90.5 tok/s vs the 83.0 stock
Triton baseline (+9.3%), and ~70x more accurate.** Two independent fixes were required; either one
alone measures as a regression or a wash, which is why this looked like a dead end twice:

| step | decode | e2e |
|---|---|---|
| stock Triton fused_moe (what ran before) | 83.0 | 81.5 |
| W4A16, K-on-lanes gemv, unfused epilogues | 71.2 | 69.8 |
| + BYLANE tiling for gemm2 | 82.4 | 81.2 |
| **+ fused SiLU gemm1 and scatter gemm2** | **90.5** | **89.2** |

Coherence green; parity 4/4 (rel err 0.00057-0.00063 vs 0.045 for fp8-act); concurrency-4 199 tok/s.

**1. BYLANE — a tiling axis on the SHARED core**, not a new kernel (KERNEL_CORE_POLICY.md: a
different *tiling* justifies new code, but it must be a variant of the core so every WLoad
inherits it). `gemv_decode_core<..., NWARPS, MMAX, COLS, BYLANE=false>` — trailing param, all
existing call sites untouched. Each LANE owns a whole column and sweeps K serially: no `__shfl`
reduce, no minimum K, occupancy from N. Loaders opt in via `accum_bylane()`; the branch is
discarded otherwise. Also added `Int4Fp16GemvLoader` (int4 weights, fp16 activations direct — what
the forked `mmq_regdirect_w4a16_moe_gemv_kernel` should always have been) and a `uses_act_scale<>`
trait so no-act-quant loaders skip the act_scales read instead of passing an all-ones vector.

MEASURED us/MB at N=2048 (`bench_w4a16_k_cliff.py`) — BYLANE is flat until its 32 resident weight
rows stop fitting L1, which is exactly why it is a SHORT-K tiling:

| K | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| K-on-lanes | 75.0 | 43.6 | 27.3 | **14.4** | **21.0** |
| BYLANE | **10.3** | **9.4** | **9.2** | 16.4 | 23.8 |

Threshold `K<=1024`. COLS=1 is the clear win (21.7 vs 30.2/50.1 for COLS=2/4 — COLS multiplies the
wave's resident row footprint, the very L1 reuse BYLANE depends on). NWARPS flat; 8 for CU coverage.
In-serve gemm2 **64.4 -> 19.2 us (3.35x)**.

**2. EPILOGUE FUSION — this is what released the win.** With BYLANE alone the gemms were 19%
cheaper than Triton (1.68 -> 1.36 ms/token) yet e2e was flat, because the unfused form paid a
(P, 2*inter) intermediate round-trip plus a separate `gather_reduce` — almost exactly the gemm
saving. Both now fuse, using axes the core already had:
* `mmq_regdirect_w4a16_moe_gemv_silu` (new op) — gemm1 writes `silu(gate)*up` straight from the
  epilogue; the (P, 2*inter) intermediate is never materialised. BYLANE+SILU is supported too, so a
  short-K gemm1 cannot silently starve the way gemm2 did.
* `mmq_regdirect_w4a16_moe_gemv_scatter` — gemm2 folds topk-weight + scatter, dropping
  `gather_reduce`. The atomicAdd **is** HIP-graph-capturable (output re-zeroed inside the graph,
  exactly as the fp8 `use_gemv` path has always done); the earlier claim otherwise in this doc was
  wrong.

**3. BYLANE wired into the fp8-act path too, DEFAULT OFF.** `Int4Fp8GemvLoader::accum_bylane` +
`Fp8MoeGemvLoader::accum_bylane` exist; `VLLM_W4A8_MOE_GEMV_BYLANE=1` routes the int4/fp8-act MoE
decode GEMV through it. Isolated measurement on the whole `_run_grouped_moe` call: **74.2 -> 66.7 us
(1.11x)**, rel diff 1.4e-5 (reassociation only). Far smaller than W4A16's 3.35x because only its
gemm2 benefits and it already used a scatter epilogue. **Left OFF because that has not been
validated e2e** — flip the env var to A/B it.

GENERAL RULE this exposed, worth applying anywhere: **a K-on-the-lane-axis GEMV has a MINIMUM K.**
Check `K >= 32 * k_per_lane_slot` before reusing one on a new shape, or it runs at a fraction of
throughput and nothing complains. Per-loader minimums today: int4 loaders and `Fp8MoeGemvLoader`
1024, `Fp8DenseGemvLoader` 512, `Bf16GemvLoader` 256.

### Still-open fold (policy debt, tracked)

`mmq_regdirect_w4a16_moe_gemv_kernel` is NOT deleted — the launcher routes to the core for BYLANE
(K<=1024), so gemm1's K=2048 non-fused entry still runs the fork. `Int4Fp16GemvLoader::accum()` is
written, so finishing is a launcher change plus a parity re-run; it is not done because the fork's
K-on-lanes path LDS-stages activations while the core reads them from global, which needs its own
measurement rather than an assumption.

## Other levers the profile exposes, ranked

1. **NCCL 25%** — 82 all-reduces/token at 42 µs for ~4 KB messages, with `NCCL_P2P_DISABLE=1`
   (host-staged). 42 µs for 4 KB is ~10× worse than the latency floor. NOTE `custom_ar` was SHELVED
   as "0% win" — that verdict predates this profile and deserves a re-read, but the shelving also
   cited **corruption under cudagraph** and **deadlock under `--enforce-eager`**, which are
   correctness blockers, not perf ones. See `/home/pat/code/vllm-gfx1201-custom-ar/.../FINDINGS.md`.
2. **`wvSplitK` 16%** — 193 calls/token, 11.6 µs each. These are the **UNQUANTIZED** dense linears:
   this checkpoint's `quantization_config.ignore` list covers every `linear_attn.*`, every
   `self_attn.*`, every `shared_expert.*`, and `lm_head` — only the MoE experts are int4. That is
   also why the awq-dense A/B was legitimately flat: there is nothing dense to quantize. The lever
   here is `rocm_unquantized_gemm` (baked source patch) → `dense_bf16_gemv`, which the LM head shows
   hits 680 GB/s. Estimate its headroom from the measured per-call bytes BEFORE building it —
   wvSplitK may already be near bandwidth at these shapes.
3. `gdn_decode_conv` 5.5% and `Cijk` rocBLAS 5% are the next tier.

## Landed and validated in earlier sessions (unchanged)

1. **GDN SSM state stride-aware** — bit-exact paged-vs-contiguous, fp32+bf16.
2. **Fused in-kernel per-position publish** — +37.7% bs=1; bit-exact vs the old path.
3. **Boot 257 s → 131 s** — persistent attn-autotune cache + `/root/.cache/vllm` AOT mount.
4. **`awq_dense_hip` in the compose default `VLLM_PLUGINS`** (inert for THIS checkpoint — nothing
   dense is quantized — but correct for AWQ models that do quantize dense).
5. **`${VLLM_PLUGINS-...}` (was `:-`)** so an explicit empty value really means no plugins.
6. **`dense_gemm` built** and mounted at `/opt/kernels/dense_gemm`.
7. **`rdna4_vllm` plugin package** — real `vllm.general_plugins` entry point; LM head via
   `PluggableLayer.register_oot(name="LogitsProcessor")`. **Confirmed engaged this session.**

## Next steps, in order

1. **GEMV-by-lanes kernel for short-K/wide-N** (above). Biggest measured, well-understood lever.
   Rebuild `fp8_wmma` via `local/build_local.sh` in the image and mount the rebuilt `.so` the way
   the gdn `.so` already is.
2. **`tail_hip.store_kv`** — the only per-layer decode op still genuinely Triton
   (`triton_reshape_and_cache_flash`, `triton_attn.py:1075`). Override `do_kv_cache_update` on
   `RocmHipAttentionImpl`, a class we own. Risks: `kv_cache.unbind(1)` gives non-contiguous k/v
   views; `k_scale`/`v_scale` are device tensors but the op takes host doubles (a `.item()` would
   break graph capture). It did NOT show up in the decode top-20, so measure its share first.
3. **`SiluAndMul`/`GeluAndMul`** via `CustomOp.register_oot` — inert without
   `--compilation-config {"custom_ops":["+silu_and_mul",...]}`.
4. **Sampler** (`sampler_hip.top_k_top_p_sampling_from_logits`) — baked patch of
   `v1/sample/ops/topk_topp_sampler.py`. Temperature is ALREADY applied before the sampler
   (`sampler.py:228`) so pass `ones`; warmup passes `top_k` as fp32 → normalize to int32; greedy
   never reaches it, so benchmark with sampling on. `rejection_sampler.py:20` is a SEPARATE site.
5. **Remove the casts/copies in `tools/profiling/vllm_oot_slotfix.py`** — the spec path does
   `mqkv_spec.float().contiguous()`, `a_spec…`, `b_spec…` per layer per step; the gdn kernels are
   `AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, …)` and take activations natively. Also 3×
   `.reshape().contiguous()` after the conv split (19.1 µs/layer).

## Do NOT re-derive — settled by measurement

* Attention backend is irrelevant at bs=1 (3 backends, identical 83.3); `attn_decode` /
  `attn_prefill_paged` are already wired via the in-tree `rocm_hip_attn.py` fork.
* LM head engages and is at 680 GB/s — worth zero, correctly.
* Nothing dense is quantized in `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`; the awq-dense flat A/B is real.
* `cca` (8 ops), `gdn` backward ops, non-paged `flash_decode` — no vLLM call site.
* The gather/scatter was NOT the GDN spec perf cause (+8%); the publish loop was (+37.7%).
* `_SPEC_NO_SCATTER` cannot answer perf questions — dropping the publish collapses accept_len.
* A warm N=1 microbench UNDER-predicts host-op cost ~3× (predicted 4 ms/step, actual 12.8).
* `max_num_seqs=4` does not boot (less KV headroom than 8); 60k ctx does not boot under MTP.
* The MoE crossover cache describes the fp8-act kernel ONLY. Never gate a different kernel on it.

## Verification standard

Every wiring needs: (a) proof it ENGAGED (a print on first call, not an assumption), (b) a bit-exact
or parity check against the path it replaces, (c) a decode + e2e number at the config above, and
(d) a coherence check. Note that enabling our `rms_norm` CHANGED greedy output vs stock — numerics
parity of the tail kernels is an open question worth settling.
