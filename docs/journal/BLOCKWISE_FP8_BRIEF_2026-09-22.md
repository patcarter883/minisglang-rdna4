# Native blockwise-fp8 GEMM — implementation brief, 2026-09-22

Produced by an 8-area read-only mapping pass (33 agents, 152 findings, 24 steering findings
adversarially tested: 4 CONFIRMED, 19 PARTLY, 1 REFUTED — plus 9 claims refuted outright in §7).
No GPU was used, nothing was built or benchmarked, and the live serve was never touched.

Replaces `Fp8BlockDequantLinearMethod`'s dequantize-to-bf16-at-load, which costs +2.49 GiB
model-wide / ~1.25 GiB per card at TP=2 on 16 GiB cards where the expert cache is the residual
claimant on VRAM. Companion to `ENGINE_COMPARISON_2026-09-22.md` item 7, which proposed it.

# Native blockwise-fp8 GEMM — implementation brief

**Scope:** 156 checkpoint tensors / ~120 engine modules of `tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ` declared `num_bits:8, type:float, strategy:block, block_structure:[128,128]`, today dequantized to bf16 at load by `Fp8BlockDequantLinearMethod` (`python/minisgl/quant/method.py:835-931`) at a cost of **+2.49 GiB model-wide / ~1.25 GiB per card at TP=2** (`docs/journal/ENGINE_COMPARISON_2026-09-22.md` item 7, sized from the served tree's safetensors headers; I did not re-derive it — no shards are on disk in either HF cache).

Everything below was read this pass unless marked otherwise. Nothing was built, benchmarked, or run on a GPU.

---

## 1. Verdict

**Buildable as a loader policy on the existing shared cores — with one bounded axis addition on the prefill body, of exactly the same shape as the `GATHER` axis that already landed on that same body.** It is not a new kernel and not a new package, and KERNEL_CORE_POLICY is satisfiable as written. The decode half (M≤16) needs **zero** core edits; the prefill half (M>16) needs one new template axis on one body plus a scale-index seam widening.

Justification, by half:

**Decode (M≤16) is a pure policy add.** `gemv_decode_core` (`gemv_decode.h:2153`) delegates the entire K sweep to `WLoad::accum(acc, M_real, x_fp8, s_src, wq_e, ws_e, wz_e, nc, N, K, group_size, lane, e2m1)` — the contract is spelled out at `gemv_decode.h:556-580` and it **already carries `ws_e`, `N`, `K` and `group_size` into the loader**. `Fp8DenseGemvLoader::accum` (`gemv_decode.h:1152-1194`) takes both and ignores them; `Int4Fp8GemvLoaderT::consume_chunk` is a working reference for folding a per-K-group scale inside the chunk loop. A blockwise loader is a new struct and a launcher. Nothing in the core body moves.

**Prefill (M>16) needs one axis, and the candidate that needs the least is already dense and already called.** `moe_bf16act_tiled_kernel` (`moe_gemm_tiled.h:1306`) is the only dense-capable WLoad-templated tiled body: it carries `GATHER` (`:1305`, `GATHER=false` instantiated by `bf16_launch_bn_dense`, `moe_bf16_ops.hip:152`), it carries an fp8-byte weight loader (`W8A16Loader<Elem>`, `:982`, `WT = unsigned char`, exact `narrow_exact` widen), and it ships as `dense_w8a16_gemm` (`moe_bf16_ops.hip:692`; binding `torch_binding.cpp:1465/2249/2395`) with a live caller (`models/draft_linear.py:64-68`). What it lacks is the **K-group fold**: one `Compute::CFrag acc[NFRAG]` held across the whole `k0` loop (`:1348`, loop at `:1360`) with the weight scale applied once in the epilogue (`WLoad::wscale(wse, an)`, `:1441`/`:1446`). A `[128,128]` scale varies along K, so the fold is mandatory — but the fold *mechanism* is a shipped WLoad policy two screens up in the same file (`float running[NFRAG_W][8]` at `:539`, per-group flush at `:606-613`), and the flag bodies key the identical thing on `WLoad::kGroupFold` (`moe_gemm_flag.h:88/144/157/258-278`), a trait **already declared false on every loader this body compiles with** (`:894`, `:1065`, `:1208`). The body simply does not read it yet.

**What the core genuinely does not have, stated plainly so nobody re-litigates it as "so we need a new kernel":**

1. **No blockwise-in-N scale policy.** All three shipped WScale policies decode at N-granularity 1 — `WSP::decode(p, off)` takes a precomputed flat offset (`tile_config.h:352`), and the callers open-code `g*N + abs_n` (`gemm_tiled.h:222-223`, `moe_gemm_flag.h:273`). `group_wscale(ws_e, block_n, nfrag_abs, frag_col, N, num_groups, g)` (`moe_gemm_tiled.h:142-166`) is the one hook that owns the offset itself and already receives enough arguments to compute `(abs_n>>7, g)`. On the a16 body the seam is `WLoad::wscale(wse, an)` — per-channel, and it must widen.
2. **No bf16 `ScaleT`.** `Fp16GroupScale` is `__half` (`tile_config.h:364`), `E8m0GroupScale`/`E4m3GroupScaleGlobal` are `unsigned char` (`:399`, `:427`); the byte-weight loaders use bare `float`. The checkpoint ships `weight_scale_inv` BF16 `[N/128, K/128]` (`method.py:880`). A load-time cast to f32 is exact and costs N·K/16384 elements.
3. **No per-(token, K-group) activation scale anywhere.** Every activation scale in the package is 1-D over tokens and applied once (`gemm_tiled.h:236`, `moe_gemm_flag.h:316`, `w8a8_dense_kernel.hip:278`); the op ABI pins it (`torch_binding.cpp:160`, `quant/kernels.py:687`). This is the real fork — see §5(a).
4. **Zero prior art.** `grep` over all of `rdna4-hip-kernels` for `weight_scale_inv` / `block_structure` / `blockwise` returns nothing. There is no partial implementation to extend.

None of these is a new algorithm. Do **not** copy `dense_gemm_tiled_kernel` (`w8a8_dense_kernel.hip:116`) and edit its epilogue: it is the fastest dense fp8 path (152.2 TF/s at M=2048, `PERFORMANCE.md`) and precisely because of that it is the tempting fork KERNEL_CORE_POLICY:13-14 forbids by name — it has *no* WLoad, *no* WSP, one accumulator at BK=64, and a hardcoded `acc * a_scale[m] * w_scale[n]` epilogue (`:263-281`).

---

## 2. The constraint set

### C1 — M-invariance. The most precise statement, because this is why the dequant path exists at all.

`Fp8BlockDequantLinearMethod.apply` is one line: `return minv_linear(x, layer.weight, bias)` (`method.py:931`), and its docstring gives the reason (`:842-845`): dequantizing "lands the module on `minv_linear`, the same M-invariant chokepoint every other unquantized Linear uses, so a chunked / prefix-cached / spec-verify forward still matches a fresh one bit-for-bit."

M-invariance here is **three layered properties, and only the first is a kernel property**:

1. **Reduction order within an arm.** Each kernel arm does a full-K reduction in a fixed 16-wide order with no split-K, so a row's value does not depend on M. Measured, not assumed: `minv.py:29-42` records max|Δ| = 0 for rows[0:m] alone vs inside a batch of 2048, for all three dense arms.
2. **Tile/schedule selection may depend on M** — but only because every tile is measured bit-identical (`minv.py:60-62`, `:391-395`: "verified 0.000e+00 over 36-54 configs × 4 shapes × 4 M"). The freedom is earned by measurement.
3. **Dispatch across kernel families is the lossy seam.** `minv_linear` already has one: the decode GEMV at `_DECODE_GEMV_MAXM = 16` (`minv.py:178`) is "NOT bit-identical to the dense_gemm family, so the threshold is a crossing; MAXM=16 puts decode AND spec-verify on this side of it" (`minv.py:256-258`). The same constant is written in three places with three separate justifications — `minv.py:178`, `gdn/layer.py:77`, `quant/kernels.py:1608` — so grep all three if you ever move it.

**What rests on it:** recurrent-radix prefix caching is enabled for this model *because* minv made chunked prefill bit-identical (`scheduler/scheduler.py:276-281`; the spec-decode lossless gate was REMOVED on that strength); `rowchunked_ar_span`'s bit-exactness is enforced by `_MIN_CHUNK_ROWS = 33` (`layers/tp_overlap.py:165`, `:399`), a constant chosen to sit above every crossover the engine currently has; and spec-verify losslessness (MEMORY: approximate verify costs +3–7% per-draft acceptance).

**What a native arm must do:**
- Each of its arms individually M-invariant (fixed full-K order; no split-K derived from M or occupancy; no tile that changes reduction order).
- The GEMV/GEMM crossing at **M=16, not 8**. The existing fp8 dense op gets this wrong: `const bool use_gemv = (kernel == 2) || (kernel < 0 && M <= 8)` (`w8a8_dense_kernel.hip:667`) — the exact defect `_W4A8_GEMV_MAX_INT4` was raised 8→16 to fix. `dense_w8a16_gemv` already enforces `TORCH_CHECK(M <= 16, ...)` (`torch_binding.cpp:1447`) and `draft_linear.py` splits at 16, so taking that pair inherits the right seam for free.
- The GEMV launcher must pin `mreal_cap` so `select_gemv_tiling` cannot return two reduction orders for one (N,K). The pattern is written twice already: `kBf16GemvMrealCap = 16` (`w8a8_dense_kernel.hip:332`) and `kW8A16GemvMrealCap = 16` (`:434`), with the reasoning at `:385-391` — "BYLANE and K-on-lanes reduce K in DIFFERENT orders... So we pass a CONSTANT mreal_cap rather than the real M." Group 128 clears the BYLANE guard (default `bylane_min_group` is 32, `gemv_decode.h:156`), so BYLANE is *available* for this format and an unpinned cap could flip the order mid-serve.
- **Honest limit, state it:** the seam cannot be placed above the whole decode+verify band, because verify M = `padded_bs*(width+1)` and at bs>1 it is already past 16. `quant/kernels.py:1608-1625` says it in as many words — "this MOVES the seam to M=16, it does not remove it." The arm inherits that limit; it does not create it.
- **If the arm ever introduces a seam BELOW 16**, register it in `spec/width.py::live_m_thresholds()` and `_M_THRESHOLD_DOC`, or `max_verify_rows()` caps the verify ladder at a stale 16 (`scheduler.py:722-729` is the drift warning and can only see constants width.py imports).

### C2 — Split-K is committed by SHAPE, never by M

`dense_gemm_split_k_slices(int64_t IN)` takes no M argument by signature (`dense_gemm_kernels.hip:548-551`); `minv.py:117-141` states the rule and records the gate (fixture `tools/_fixtures/splitk_router_ladder_card0.txt`; 0/22400 top-k index flips). Any K partition in the new arm must be a pure function of (N,K). Note for scope: at TP=2 only `k_proj`/`v_proj` (local OUT=256, IN=2560) fall in the band `OUT ≤ 256 and IN ≥ 1024` (`minv.py:358-360`) — 24 of 156 modules, none at TP=1. `o_proj`/`out_proj` at OUT=2560 can never reach it.

### C3 — `BK` is locked to the scale group by the scale index, not by style

`const int BK = group_size` at `gemm_tiled.h:118`, `moe_gemm_tiled.h:533` and `:705`. This is a **correctness lock**: the scale is indexed by the BK-group ordinal (`WSP::decode(w_scales, (long)g*N + abs_n)`, `gemm_tiled.h:222-223`), so BK ≠ group_size reads the wrong tile and returns plausible wrong numbers silently. For the blockwise arm the fold cadence is 128 K. `MAX_GROUP_SIZE = 128` (`tile_config.h:34`) already sizes the LDS worst case and the launchers already validate `group_size % 16 == 0 && group_size <= MAX_GROUP_SIZE`. Every shipped dimension divides 128 (2560/128=20, 512/128=4, 6144/128=48, 10240/128=80), including after the TP=2 split — so **no K tail and no N tail on this checkpoint** — but do not bake that in; `method.py:871-876` asserts it engine-side and raises.

### C4 — The a16 body's staging block is NOT the scale group

`BF16_BK = 32` (`moe_gemm_tiled.h:842`), loop at `:1360`. A 128-K block spans exactly four staging iterations, so the fold point is **not** a `k0` loop boundary. `moe_gemm_flag.h:250-256` states the rule and why it is not obvious: "A group can span staging chunks, so the cadence is counted in 16-K WMMA STEPS, not in chunks" — and `:259-263` records that deriving the step from the loop indices rather than a mutable counter "is the difference between the shared body being 2.7% slower and being perf-neutral." Get this wrong and a partial is scaled by the *neighbouring* block's scale: correctly shaped, plausible, off by whatever the adjacent-block scale ratio is.

### C5 — No debug knob may become a kernel argument; no guarded load in a hot loop

`group_size` is already a runtime arg on these cores and that is fine. What is not: a `getenv`-fed kernel arg (measured −13.4% instructions when made constexpr) and a runtime null test around a load. `moe_gemm_flag.h:311-313` records `act_scales != nullptr` as exactly that defect; `:119-125` records losing `__restrict__` provenance on the weight-scale pointer costing +131 branch instructions / ~+3% wall. The activation-scale-presence axis is compile-time in three places for this reason (`kActScales` `moe_gemm_tiled.h:252`, `quantizes_act` `tile_config.h:257`, `uses_act_scale_v` `gemv_decode.h:277-280`) — keep any new axis compile-time.

### C6 — `weight_scale_inv` is a MULTIPLIER despite its name

`layer.weight = (w.to(torch.float32) * sc).to(torch.bfloat16)` (`method.py:922`). The identical trap is documented for NVFP4's global at `tile_config.h:420-424` ("The global is a MULTIPLIER... direction normalisation is the loader's job, host-side"). A kernel that divides produces finite plausible output wrong by the square of the scale, and is invisible if the parity reference is a transcription of the loader.

### C7 — The weight dtype is protected; the scale dtype is not

`_CASTABLE_FLOAT_DTYPES` excludes `float8_e4m3fn` deliberately (`layers/base.py:24-30`), so a wrong weight declaration fails loudly. But bf16↔f32↔f16 are all mutually castable, and **`cast_checkpoint_tensor` has no `.weight_scale_inv` rule** — it exempts `.scales`, fp8 bytes, `.weight_scale`, `.weight_global`, `.A_log`/`.dt_bias`, then falls through to `return v.to(model_dtype)` (`models/weight.py:~45-100`). The 48 attention projections get the declared dtype restored by `_coerce_dtype`; the 72 GDN projections do **not** — `GDNLinearAttn` loads with `assign=True` (`layers/base.py:87`), which takes the incoming dtype with no declared-dtype check. That is the trap `.weight_global` exists for, one tensor kind later (`weight.py:74-79` says so verbatim). Declare the scale f32 and it must get its own rule, or one layer hands f32 to q/k/v/o and bf16 to `in_proj_qkvz`/`out_proj`.

### C8 — Every op tensor must be `.contiguous()`

`derive_granule_spec` raises `GranuleError` on any non-contiguous tensor it reaches, and it reaches underscore attributes on both `BaseOP` and `nn.Module` nodes (`granule.py:572-590`, `:838-848`). A transposed scale kept as a *view* breaks the residency walk at boot, not at runtime.

### C9 — Two call sites, not one

`create_weights` is called from `layers/linear.py:62` (48 attention modules) **and** `gdn/layer.py:108` (72 GDN modules). On the GDN path everything `create_weights` sets is lifted to an `nn` buffer (`gdn/layer.py:104-110`), post_load runs via `_MethodLinear.process_quant` (`:112-115`), and forward is `self._method.apply(self, x, None)` (`:117-118`) — **bias always None, and no `supports_producer_actquant` plumbing at all**. Whatever activation policy the arm picks, the GDN half cannot receive an engine-supplied activation pair the way `_LinearTPImpl.forward` can (`linear.py:80-82`).

### C10 — Pinned CU count

Any occupancy-aware choice must reason about a pinned 64, not `multiProcessorCount`: `constexpr int PINNED_CU = 64` (`tile_select.h:~236`) mirroring `minv.py:143` `_CUS = 64` ("do NOT read this from device properties"). This box is a mismatched 64/56-CU pair and the arbiter leases whichever card is free, so a per-device rule makes the same token dispatch differently run to run — and at TP=2 it would latch different plans per rank.

---

## 3. What already exists and is reusable (CONFIRMED only)

| Asset | Where | How the arm reaches it |
|---|---|---|
| **Dense fp8-weight pair, both M bands, one weight layout** | `dense_w8a16_gemv` (`torch_binding.cpp:1427`, `M<=16` checked at `:1447`) + `dense_w8a16_gemm` (`:1465` → `moe_bf16_ops.hip:692`) | This IS the arm's skeleton. Both take `(N,K)` e4m3 bytes; today both take `(N,) f32` per-output-channel scales (`torch_binding.cpp:1445`, `moe_bf16_ops.hip:698`). Swap the scale contract for `(N/128, K/128)` + fold. **Has a live caller**: `models/draft_linear.py:64-68` maps `torch.float8_e4m3fn → ("dense_w8a16_gemv","dense_w8a16_gemm")` and splits at M≤16. |
| **Dense axis on the tiled body** | `GATHER` at `moe_gemm_tiled.h:1305`, dense launcher `bf16_launch_bn_dense` at `moe_bf16_ops.hip:152` ("Templated on WLoad so this is an AXIS, not a fork") | Already landed (FORMAT_MATRIX G2, table at `:20`). No fabricated `sorted_token_ids` needed. |
| **K-group fold mechanism, as a WLoad policy** | `moe_gemm_tiled.h:539/606-613` (fp32 `running`, per-group flush); `moe_gemm_flag.h:88/144/157/258-278` under `kGroupFold`; `gemm_tiled.h:150/219-226` on the dense int4 body | The pattern to port into the a16 body. `kGroupFold` is already declared on every loader in `moe_gemm_tiled.h` (`:174` true, `:315`/`:894`/`:1065`/`:1208` false) — the a16 body just does not read it. |
| **Scale-offset hook with enough arguments** | `WLoad::group_wscale(ws_e, block_n, nfrag_abs, frag_col, N, num_groups, g)`, `moe_gemm_tiled.h:142-166`, called at `:610`/`:771` | A blockwise `(abs_n>>7, g)` index is expressible here with no body edit. Note `moe_gemm_flag.h:271-274` deliberately **bypasses** it for a measured reason — do not assume it is the seam on every body. |
| **GEMV WLoad contract carries the group already** | `gemv_decode.h:556-580` contract; `Fp8DenseGemvLoader` (`:1152`) takes `ws_e`/`group_size` and ignores them; `Int4Fp8GemvLoaderT::consume_chunk` folds per group inside the chunk loop | A blockwise GEMV loader needs **no core edit**. A 16-fp8 chunk divides 128 exactly, so a chunk never straddles a block boundary — the straddle-split NVFP4's group-16 forced is not needed. |
| **Exact e4m3 → bf16/fp16 widen** | `narrow_exact` (`tile_config.h:97-106`), used by `W8A16Loader::stage_w` (`moe_gemm_tiled.h:~1008`), host-verified over all 254 finite codes | Free for the A16 route. |
| **TP shard rules for the tile scale** | `models/weight.py:789-826` — `in_proj_qkv` via `_shard_blocks_dim0` over `[key,key,value]//bn` with a raise if a head block is not a whole block count divisible by tp; col-parallel chunk dim 0 for `in_proj_z`/q/k/v; row-parallel chunk dim 1 for `out_proj`/`o_proj`; **raise** for anything unruled | Done, and format-independent — they key on leaf name and module suffix only. Runs at READ on the checkpoint name (`weight.py:1954`), before the concat. |
| **qkv+z scale merge** | `_Q4_QKVZ_SCALE` → `in_proj_qkvz.weight_scale_inv` concat on dim 0 (`weight.py:1213-1218`, `:1243-1244`); native-key allow-list `:1273-1274` | Done. At TP=2 the merged scale is exactly (40,20)+(24,20) = (64,20). |
| **Checkpoint-native weight declaration** | `create_weights` (`method.py:862-880`): `weight` fp8_e4m3fn `(N,K)` + `weight_scale_inv` bf16 `(N/bn, K/bk)` at LOCAL per-TP sizes, with a raise on non-division | Reusable **verbatim**. Keep the declared tensor names — the loader's shard/merge/allow-list machinery is keyed on them. |
| **Dispatch slot and its ordering guard** | `create_linear_method` returns `Fp8BlockDequantLinearMethod(quant)` at `method.py:702`, gated on `is_fp8_block` (`quant/config.py:357-368`) and placed **before** `is_fp8_w8a8` with the reason in-line (`:700-701`) | Arm selection is one `return`. Covered by `tests/ct_block_fp8_test.py:112-133`. |
| **Structural classification** | `.weight_scale_inv` in `_QSUFFIX` (`models/config.py:601`), consulted by `for_module` before the ignore list | A module is blockwise iff the checkpoint ships it a scale. No scope creep possible. |
| **GDN bridge for a non-Unquantized method** | `_MethodLinear` (`gdn/layer.py:98-120`), selected by `_make_proj` (`:121-133`) | Already carries this method today. No new bridge work; see C9 for what it *lacks*. |
| **Measurement harnesses** | `tools/quant_m_invariance.py` (5 legs incl. ARGMAX agreement), `tools/quant_m_invariance_serve.py --check-arm`, `tools/minv_tile_ab.py` (chunk equivalence), `fp8_wmma/tests/test_dense_w8a16_gemv.py` (BLAS-free fp64 reference with a BLAS tripwire) | Extend `ARMS` and shape lists; do not write new ones. |

**Exists but has no caller / is stale — flagged:**
- `moe_bf16act_regdirect_kernel` (`moe_gemm_tiled.h:1460`) is recorded **engine-dead** in `FORMAT_MATRIX.md` Matrix B.
- `w8a8_dense_kernel.hip:565-570` and `FORMAT_MATRIX.md:256` both still say the a16 tiled body "is grouped-only and has no GATHER axis / G2 still open". **Both are stale** — `GATHER` is at `moe_gemm_tiled.h:1305`, the launcher ships at `moe_bf16_ops.hip:692`, and the file's own update table (`FORMAT_MATRIX.md:20`) records G2 closed with its caller. FORMAT_MATRIX states its own rule: where a doc and the source disagree, the source wins.
- `method.py:695-696` / `:937-938` say the dense fp8 W8A8 path reuses the MoE core as a single-expert grouped GEMM. **Stale** — it runs `mmq_w8a8_gemm`, a genuine dense kernel (`w8a8_dense_kernel.hip:6-7`).

---

## 4. Work list, ordered

**Kernel side (`rdna4-hip-kernels/fp8_wmma`)**

1. **`Bf16BlockScale` / f32 blockwise WScale policy** — `tile_config.h`, beside the three at `:363/:398/:426`. Either a bf16 `ScaleT` (high-half widen; the exact-narrow machinery at `:97-106` is the inverse and already proven) or `ScaleT = float` fed by a load-time cast. **Recommend f32**: scale bytes are N·K/16384, and it dodges both the fp16 saturation cliff `tile_config.h:385-388` records and the dispatch collision in item 8.
2. **Blockwise GEMV loader** — new struct in `gemv_decode.h` modelled on `Fp8DenseGemvLoader` (`:1152`) for byte staging and on `Int4Fp8GemvLoaderT::consume_chunk` for the fold. Per KERNEL_CORE_POLICY, share `Fp8DenseGemvLoader`'s chunk text rather than copying it. Set `bylane_min_group_v` to the real group (or drop the override; the default 32 already admits 128). **No core edit.**
3. **GEMV launcher** in `w8a8_dense_kernel.hip` beside `run_w8a16_gemv`, with `mreal_cap` pinned to 16 and `group_size` actually passed (the existing W8A16 launcher passes `/*group_size=*/0`).
4. **K-group fold axis on `moe_bf16act_tiled_kernel`** (`moe_gemm_tiled.h:1306`): add `float running[kGroupFold ? NFRAG : 1][8]` and a flush gated on `if constexpr (WLoad::kGroupFold)`, cadence derived from the loop indices (C4), accumulator re-zeroed per block. The second-accumulator move is precedented *in this body* by SILU's `accu[SILU ? NFRAG : 1]` (`:1349`). Must be compiled out with proven-identical codegen for `Bf16DirectLoader`/`W8A16Loader`/`Int8A16Loader`.
5. **Widen the a16 scale seam.** `WLoad::wscale(wse, an)` (`:1441`, `:1446`) is per-channel; the blockwise policy needs `(an, k-block)`. Widen the hook, keep the existing loaders' answers textually identical.
6. **Blockwise tiled loader** (`W8A16BlockLoader<Elem>`), sharing `W8A16Loader`'s `stage_w`/`stage_b` text, declaring `kGroupFold = true` and owning the `(abs_n>>7, g)` index.
7. **Launcher + `TORCH_CHECK`s** in `moe_bf16_ops.hip` beside `launch_dense_w8a16_gemm_gfx1201` (`:692`). **Check the scale's 2-D shape against `(N/bn, K/bk)` and its contiguity explicitly** — the existing op checks `numel() == w_fp8.size(0)`, which a blockwise tensor can satisfy by accident.
8. **Dispatch by PRESENCE/SHAPE, not dtype.** The existing scale-format dispatches say so in their own comments (`w4a8_fp8_wmma_kernel.hip:1012-1028` "PRESENCE, not dtype"). A 2-D scale must select the block policy and nothing else may.
9. **Op plumbing — six sites plus a build.toml entry.** decl in `torch-ext/torch_binding.h`; launcher in the `.hip`; `*_forward` + `ops.def` + `ops.impl` in `torch_binding.cpp`; Python wrapper in `torch-ext/fp8_wmma/__init__.py`; **its name in `__all__`** (miss this and the op is invisible to `hasattr` and silently unused — the `_tail_hip` re-export gotcha); `_register_fake` meta kernel. A new header also needs a line in `fp8_wmma/build.toml`'s `src` list or the Hub build compiles against a missing header. Choose the (tiling × MMAX) instantiation set deliberately — the macro ladders are hand-written and every point is binary size.
10. **`OperandFormat` entry in `tile_select.h`** if the tiled arm uses the chooser: `b_bits 8`, 1 B/elem staging on the weight, and a scale term ~1/128 of `FMT_W4A8`'s. `tile_select.h:279-291` warns that the scale-line term "is the term that separates BN, which no ISSUE term can" — reusing `FMT_W4A8` prices ~128× the real scale traffic and reusing `FMT_BF16` zeros it. Treat `vgpr_*` as UNFITTED, per the `FMT_BF16` precedent at `:307-311`.

**Engine side (`minisgl-rdna4/python/minisgl`)**

11. **`.weight_scale_inv` rule in `cast_checkpoint_tensor`** (`models/weight.py:~45-100`), exactly as `.weight_global` got one. Closes C7 **and** the `--dtype float16` exponent-range narrowing in one edit.
12. **New method class** (or rewrite of `Fp8BlockDequantLinearMethod`). Keep `create_weights` verbatim. `process_weights_after_load` now *retains* the scale: build op-layout tensors (`layer._w_op = layer.weight.contiguous().view(torch.uint8)` — the `Fp8W8A8LinearMethod` pattern at `method.py:954-960`, documented as a byte-identical alias at `granule.py:596`), promote the scale to the op dtype, `.contiguous()` everything, `del` the checkpoint names. Any per-shape plan is latched **here**, as a plain Python selector, never as a pointer or handle passed into the kernel.
13. **`apply`** — own M ladder (GEMV at M≤16, tiled above), `engaged("<arm name>")` on every leg the way `w8a8_dense_linear` does (`quant/kernels.py:1776`), flatten N-D x (`minv_linear` accepts N-D and flattens, `minv.py:223-226`; the closest native precedent `TORCH_CHECK`s `dim()==2`).
14. **Do not widen `minv_supported`.** It is a public predicate with a second consumer that reads it as a *row*-invariance oracle (`layers/hyperconnection.py:250-285`, with a measured 9.54e-07 logit shift behind it). Give the fp8 arm its own predicate. Consequence: the "single chokepoint" claim at `method.py:843-846` becomes false and must be rewritten — the invariance argument moves into the method, which is the established W4A8 pattern (`quant/kernels.py:1494-1516`).
15. **Fix the stale comments** named in §3, and the `_added_bytes` tally/log (`method.py:882-914`), which becomes dead and whose text literally advertises "a real VRAM cost a native blockwise-fp8 GEMM would not pay."
16. **VRAM: no second sizing site.** KV sizing bills `model_memory` from a *measured* device window around post_load (`engine/engine.py:297-315`, `:356-368`), so the freed ~1.245 GiB/rank flows automatically — **into the KV pool** (`engine.py:1893`, the residual). The expert cache (`--expert-cache-gb`) and the device offload tier do **not** grow on their own. Land the kernel *with* a `serve.sh` operating-point move and put the new point in serve.sh's table, or the A/B reads as "+108k KV tokens nobody needed, TPOT flat."
17. **MoE dispatch asymmetry.** `create_moe_quant_method` (`layers/moe.py:958-988`) tests `quant.is_fp8_w8a8` FIRST at `:971`, and `is_fp8_w8a8` (`config.py:389`) is strictly weaker than `is_fp8_block` — a blockwise MoE group routes to `_FP8MoEMethod` and declares a per-channel `(E,N)` scale against a file shipping `(N/128,K/128)`. Inert today (MTP unloaded), but the dense side got its guard and the MoE side did not. Add the same guard in this change.
18. **`quant.group_size` is 32 for these modules.** `config.py:~567` is `group_size=int(w["group_size"]) if w.get("group_size") else 32`, and a block group ships no weight `group_size`. Read `block_structure`, never `group_size`, anywhere in the new path — the 4×-wrong scale field would be plausible and silent.

**Explicitly out of scope, state it in the handover:** the 1543 `mtp.*` blockwise scale leaves (config `group_2`, same `[128,128]`) contribute nothing today — `spec_default="none"` zeroes `mtp_num_hidden_layers` and the loader skips every `mtp.*` key (`ENGINE_COMPARISON` item 7). They have no TP rule and would hit the raise at `weight.py:822`. Note also that `_qwen4_exp_mtp_remap`'s docstring asserting "no `*_scale` under `mtp.`" is false for this repack.

---

## 5. Decision points

### (a) Activation scheme — A16 vs A8-per-128 vs A8-per-token. **The only fork that changes what the arm is scored against.**

The checkpoint's `group_1` declares `input_activations: {num_bits:8, type:float, strategy:group, group_size:128, dynamic:true}`, and `quant/config.py:~572` collapses it to `act_type="fp8"` — strategy, group_size and dynamic are dropped, and `act_type` has **no reader anywhere in the tracked tree**. `config.py:315-318` says an env "must NEVER substitute a different activation scheme than the checkpoint declares"; against that, this engine drops declared activation schemes *by design* at the method (`config.py:166-168` for e2m1; `method.py:853-855` for exactly these modules).

| | A16 (bf16 acts, weight-only) | A8-per-128 (checkpoint-faithful) | A8-per-token (existing quantizer) |
|---|---|---|---|
| VRAM prize | full | full | full |
| Activation numerics | **exact** — strictly better than today | new accuracy surface | new accuracy surface, *coarser* than declared |
| Core work | fold + scale seam on the a16 body | fold exists; needs `GATHER` on the fp8-act grouped cores (`moe_gemm_tiled.h:485/657`, SCATTER-only) | same as A8-per-128 minus the act axis |
| Activation-side work | **none** | a (M, K/128) producer, a new op-schema slot, a row-slicing rule in `tp_overlap`, and an act-scale fold in the K loop — **exists nowhere** | none |
| Dense arms today | **both exist and are called** (`draft_linear.py:64-68`) | dense axis missing on the fp8-act spine | same |
| Compute ceiling | bf16 matrix (`PERFORMANCE.md`: dense bf16 115.3 TF/s at M=2048) | fp8 matrix (dense W8A8 152.2 TF/s at M=2048) | fp8 matrix |
| Extra launch | none | +1 act-quant kernel + an (M,K) HBM round-trip per linear, ×3 over the same rows in attention | same |

**RECOMMEND A16.** It delivers the entire stated prize — the VRAM saving comes from not widening the weight, not from the activation — with zero activation-side machinery, on a dense pair that already exists end-to-end with the correct M=16 seam, and it *improves* numerics on 156 modules served exactly today (see §6). A8's only benefit is prefill speed, and prefill on this arm is not compute-bound: `tools/serve.sh:~1086` records "prefill measured 4.0 tok/s (`minisgl_prefill_seconds_total` 558.6 s for 2243 tokens) because each chunk re-streams the host expert set over PCIe at ~52 MB/s effective." Say out loud in the docstring that the declared activation scheme is deliberately dropped, name the same reason the W4A16 arms use, and encode the choice in the `engaged()` arm name so it is visible in the ledger. Adding `act_strategy`/`act_group_size` to `QuantConfig` is worthwhile provenance hygiene (it is the only way to *assert* declared-vs-served) but is not a prerequisite. Severity is low: both blockwise groups are `dynamic:true` and the checkpoint ships zero `input_scale` tensors, so no calibration data is lost.

**Do not split by M** (A16 at decode, A8 at prefill): an activation-format seam inside one request is what `method.py:557-576` and `kernels.py:1729-1745` forbid for the W4A16/W4A8 pair.

### (b) M-invariance vs the reference engine's per-M-bucket plan latch

Their `Fp8HipPlan` latches arm/tile/grid per `(N,K,device)` for a fixed set of M buckets (cudagraph capture sizes + `max_num_batched_tokens`), and the `skinny` arm splits K across waves and reduces through LDS — a different summation order per bucket. **RECOMMEND: latch by shape only.** Copy the *mechanism* (a load-time, per-shape decision table; `process_weights_after_load` is the right home, and the decode bucket ladder `[1,2,4,8,12,16,24,32]` at `engine/graph.py:209-212` makes the reachable M set finite) and reject the *per-bucket re-plan*, which would forfeit chunked-prefill bit-identity, recurrent-radix losslessness on the 36 GDN layers, `rowchunked_ar_span`'s enforced bit-exactness, and spec-verify losslessness. Their registry is also keyed on device, which C10 forbids here outright.

### (c) Scale memory order — keep disk order, or transpose to group-major at load

NVFP4 transposes to group-major for coalescing (`method.py:756-759`); the reference does not transpose. **RECOMMEND: transpose at load into `(K/128, N/128)` group-major**, matching the `(G,N)` convention every core already uses (`moe_gemm_tiled.h:46-53`, `gemv_decode.h:572-580`, both citing the coalescing reason), and making the kernel index `g*(N/128) + (abs_n>>7)`. Constraints: the transpose must happen in `process_weights_after_load`, **after** the TP shard and **after** the qkvz concat, and must end in `.contiguous()` (C8). This ordering is self-enforcing — none of the 156 scale tensors is square, so an early transpose raises in `torch.split`, `torch.cat`, or as a `load_state_dict` shape mismatch.

### (d) Scale element type — bf16-native policy vs f32 cast vs fp16

**RECOMMEND f32 at load.** Exact, negligible bytes, no new policy struct, and it keeps the failure loud: a bf16 scale passed raw currently hits a `TORCH_CHECK`. fp16 is refused — `tile_config.h:385-388` records the saturation cliff ("an out-of-window checkpoint is served with saturated scales — wrong numbers, no failure"), and bf16 carries fp32's exponent range.

### (e) Pre-shuffle the weight into WMMA fragment order?

The reference does (`shuffle_weight_gfx1201`, keeping the `(N,K)` view). **RECOMMEND: no, not in v1.** It de-aliases `_w_op` from `weight` (breaking the assumption `granule.py:592-600` states), needs a `_granule_policy` declaration so `fingerprint()` moves if two TP ranks disagree (`layers/linear.py:37-49`), costs a transient full-size device allocation at the moment VRAM is tightest, and is invisible in shape/strides so any later consumer reading it row-major produces silent garbage. If it is ever adopted: delete the unshuffled weight, declare it in `_granule_policy`, budget the transient.

### (f) Pin `bn == bk == 128`, or parameterise?

No real checkpoint can exercise a non-square block, and `[128,128]` makes a bn/bk swap a permanent no-op in every test that could exist. **RECOMMEND: pin it and refuse anything else loudly** in the launcher, and say so — the test suite can never catch a silent swap.

---

## 6. Test plan

### Bit-exactness against the dequant path is NOT achievable, and the dequant path is not the exact reference its docstring claims

`method.py:842` calls the dequant "EXACT (an fp8 value times its block scale, widened — no second approximation)". That is exact *relative to a bf16 target only*: `:922` is `(w.to(float32) * sc).to(bfloat16)`, and e4m3 (4 significand bits) × bf16 scale (8) needs up to 12 bits where bf16 holds 8. **The shipped baseline carries up to a half-ULP of bf16 weight rounding that a native arm does not.** A native arm that widens fp8 exactly and applies an f32 block scale into an fp32 accumulator is therefore *closer to fp64 truth in weight space* — and it also accumulates in a different order, so output equality is impossible in both directions.

**The gate is relative, not absolute**, and the template is already in-repo: `fp8_wmma/tests/test_dense_w8a16_gemv.py` — a BLAS-free fp64 elementwise reference with `torch.mm`/`matmul`/`F.linear` monkeypatched to **raise** while it is built, both arms scored against it, `WORSE_RATIO_MAX = 1.05` on rel_rms, arm-vs-arm distance printed as a tolerance and never asserted to zero; docstring line 12 says it "must never be turned into" a bit-exactness test. The engine-side twin is `tests/draft_linear_w8a16_parity_test.py`'s three-way gate. Under the A16 recommendation the direction is "native ≤ 1.05 × dequant" and it should clear it comfortably; **if A8 is chosen instead, this gate direction is invalid and must be re-derived**, because a correct fp8-activation kernel is *expected* to be worse than a bf16-activation reference built from the same weights.

**Bit-exactness IS achievable and required on the weight decode.** `tests/ct_block_fp8_test.py:164-170` already asserts `torch.equal` against an independent per-element tile-lookup reference, built with an explicit double loop and not the implementation's `repeat_interleave` spelling, with the reason in-line: "a shared `repeat_interleave` spelling would agree with itself even transposed." Keep that leg, pointed at the native path's decoded weight.

### Fixture shapes that make a transposed or mis-axed tile scale VISIBLE

This is the dominant silent failure mode: **both linearisations are always in bounds**, because a `[A,B]` tensor and its transpose have identical numel and the swapped max index `(B-1)·A + (A-1) = AB-1 < AB`. No bounds check, no ASAN run, no NaN, no shape assertion can catch it in a HIP kernel taking `data_ptr()`.

Block grids at TP=1: `q_proj` (96,20) · `k_proj`/`v_proj` (4,20) · `o_proj` (20,48) · `in_proj_qkv` (80,20) · `in_proj_z` (48,20) · merged `in_proj_qkvz` (128,20) · `out_proj` (20,48). At TP=2 the merged is (64,20), `k/v_proj` (2,20), `o_proj`/`out_proj` (20,24). **None is square, at either TP degree — use real shapes.** The obvious synthetic fixture (N=K=256 → a 2×2 grid) is square and passes both spellings.

Fixture requirements, all load-bearing:
- **Non-square tile grid**, and `M ≠ G` on any activation-scale leg (K=2560 gives G=20, and M=20 is a reachable batch).
- **A distinct scale per tile, varying along BOTH axes, spanning decades** — not `rand()+0.5` near 1.0 (`ct_block_fp8_test.py:159`), where a reciprocal-direction error (C6) is small and survives a tolerance test. Never `ldexp(1, exp)`: powers of two are lossless through the bf16 narrowing and make both arms tie exactly.
- **K > 128 on every leg.** K=128 is one block and makes the whole K axis constant, blinding the fold entirely.
- **Emit three references from one fixture** — correct index, transposed tile index, and "left-neighbour's K-block scale applied across the chunk" — and **assert the fixture separates them before scoring the kernel.** The exemplar is `fp8_wmma/tests/test_gemv_group16_split.py`, whose docstring says it outright: "a fixture that cannot see the bug is the failure mode here, not a kernel that cannot pass."
- No e4m3 NaN codes (0x7F/0xFF) anywhere — `tests/nvfp4_mxfp4_dense_loadpath_test.py:9-13` records a fixture that could encode them reporting a correct path as broken.
- **TP=2 leg at sharded widths**, including `k/v_proj` at local N=256 (only 2 N-blocks — the narrowest case and the most likely to expose an N-block index bug).

### M-invariance legs (mandatory, and the harness exists)

Extend `tools/quant_m_invariance.py` — its `ARMS` tuple is hardcoded to the three W4A8 arms — and run all five legs: arm agreement, per-arm self-invariance, engine-dispatch invariance, row split, and **ARGMAX agreement**, which the file itself calls "the acceptance-relevant metric." Score argmax flips, not max|Δ|: top-k is a step function and one ULP reroutes a token (the split-K gate used 0/22400 index flips for exactly this reason). Add a chunk-equivalence leg through the real `method.apply` (not through `minv_linear`), M=512 vs chunks of 32/64/128/192/256, per `tools/minv_tile_ab.py:286-311`.

### Timing gate (mandatory — the parity suite cannot see this class of regression)

The blockwise policy lands on a body that eight live ops compile from, so it can regress every *other* format on that core with every parity test green. MEMORY records precisely that: templating the fused gemm1+silu body rewrote its A staging into byte-granular LDS writes, cost −16.6% on the shipping fp8 arm, and **shipped**, with the parity tests added by the same commit green throughout. Required:
- A before/after leg on an **unrelated arm of the same core** (bf16 / W8A16 dense) as the regression guard.
- Paired, interleaved, **in-process** A/B with a **control arm** (the same untouched kernel timed twice per round), ~300-iteration windows, on a cool card, recording card id and temperature. A hot card once faked a 2.7% regression that was really +0.36% against a ±4.16% control floor; `moe_gemm_flag.h:29-38` is the house standard for how to report it.
- Price the M-pad **inside** the timed region if any tile sweep is run: `minv.py:309-312` records a fitting surface that hoisted `_padded()` out and shipped a +19.1% regression.
- Watch specifically for a vector copy going element-wise and for `(long)` appearing in an inner-loop index.

### Provenance

Two images / two worktrees, not an env switch (`no-env-gating-on-merge`; two builds of one kernel package cannot coexist in one process — TORCH_LIBRARY double registration). Each leg prints `module.__file__` and hard-asserts where it resolved; refuse to run if the two sources hash the same. **Diff `engaged()` counts per leg before reading any timing number** — the arm is done when the ledger shows the new name on all 120 dispatches and `fp8_wmma.dense_bf16_gemv(minv_decode)` has vanished from those layers. Do **not** use "differs from the dequant output" as proof of dispatch (`draft_linear_w8a16_parity_test.py:104` does this; it is the anti-pattern).

### End-to-end output equality is invalid by default on this model

MEMORY records, for qwen4_exp specifically with spec OFF, 4/5 greedy prompts diverging run-to-run at 160 tokens and 1/5 at 16. Any "the tokens are the same" claim needs a published zero-floor high-confidence prompt set verified identical across three runs of the **control** first. Without that, an e2e diff measures scheduler/MoE-atomic noise.

### How the tests get RUN

`pyproject.toml:106-125` sets `testpaths=["tests"]` with `-m "not gpu"`, but `ct_block_fp8_test.py`, `qwen4exp_mxfp4_remap_test.py` and `draft_linear_w8a16_parity_test.py` define **zero `test_*` functions** — `pytest` collects nothing from them and reports success. Decide deliberately: either `test_*` + `@pytest.mark.gpu`, or a named runner script called out in the handover. Otherwise the coverage exists on disk and never executes.

---

## 7. What this pass could NOT establish

**Binary-only.** `/app/fp8hip/libfp8hip_gemm.so` (93,920 bytes) — I did not open it this pass. Everything about their device algorithm (tile internals, LDS staging, which wave stores the skinny arm's reduction, the exact 4th template bool) comes from a prior pass's disassembly, not from me. Their readable Python *does* establish: both scale operands are `float const*`, the only output store is bf16, `can_implement` requires group shape (128,128) / K%128==0 / N%16==0, activations are dynamically quantized per-token-per-128-K by a **separate** vLLM launch, weights are pre-shuffled at load while keeping the `(N,K)` view, and the plan is latched per `(N,K,device)` — with `world_size` stored but never compared, so do not design a plan law that takes it.

**Their accuracy claim is unusable.** The module docstring's "error bit-identical on the fp32 oracle" cites `fp8hip/README.md`, which is **not in the image**, is ambiguously worded, and accompanies a 280-cell table measured on **Qwen3.8-27B TP2/TP4 (N=5120 family) shapes**, not this checkpoint. Do not cite "they proved it accurate" as a reason to skip our own gate, and do not reuse their 0.59–0.95× / 0.79–0.98× numbers as a prediction.

**No quality scalar exists here.** `ENGINE_COMPARISON:149` states it: they ship a WikiText-2 gate against the live server; we have no format-agnostic quality scalar, "which is why every 'never measured for accuracy' item stays open." Their harness is not portable — it scores every token via `echo=true`, which `server/api_server.py:1526-1527` rejects with a 400. If a lossy arm (A8) is chosen, the loss cannot be priced today. A buildable route exists that bypasses the `echo` gap: sampler-hook logit capture (`tools/qwen3_5_decode_oracle_ours.py`) teacher-forced over a fixed corpus.

**No measurement exists for the fold on an fp8 body.** The only A/B of the shared-vs-forked fold is on int4 (+0.36% median against a ±4.16% control floor, `moe_gemm_flag.h:29-38`); `moe_gemm_tiled.h:594-607` records that for the fp8 arm the fold-vs-no-fold split is DCE-equivalent with unchanged VGPR/occupancy — i.e. the measured neutrality is about turning the fold **OFF**, never about turning it **ON** for an fp8 weight.

**No prefill baseline for these shapes.** `tools/_fixtures/dense_gemm_surface_card0.csv` covers six shapes, all at IN=2816 (the Gemma4/DiffusionGemma family). None of q4e's (IN 2560/3072, OUT 8192/6144/2560/256) appear, so today's bf16 prefill baseline for these modules is unmeasured and minv's tile rule was fitted on shapes that are not these.

**Roofline denominators conflict in-tree.** `PERFORMANCE.md` records HBM **674 GB/s** measured and "dense bf16 gemv 410 GB/s (61%)"; MEMORY derives **706.6 GB/s** from the live 1380 MHz OC and asserts the bf16 decode GEMV at 672 = 95.1%. `PERFORMANCE.md`'s own header disclaims §1 as a 2026-07-29 snapshot. Unresolved without a GPU; it moves any decode prize estimate by ~1.6×. For compute, MEMORY's matrix table is fp8/int8 389.3 / bf16 194.6 TFLOPS; `PERFORMANCE.md:30` gives fp8 WMMA hardware peak ≈389 and hipBLASLt ~230.

**Needs a GPU window, explicitly:** whether the fold is perf-neutral on the a16 body; whether the a16 body's M-dependent `block_m` ladder (`moe_bf16_ops.hip:710`, 16→128) is bit-neutral for the *dense* W8A16 instantiation — the analogous grouped-MoE ladder was measured neutral (`minv.py:36`), this one has no record; within-arm M-invariance for the new fold; A16-vs-A8 accuracy on real hidden states; and the prefill cost of the tiled arm on these shapes, which decides whether a transient per-chunk dequant scratch beats a tiled arm.

### REFUTED — do not re-derive these

1. **"The fold and an fp8 byte weight have never been composed."** `moe_gemm_tiled_kernel`'s shipped W8A8 arm already runs `BK = group_size` at 128 with raw e4m3 bytes and an fp32 `running` sum (`:533`, `:606-608`; `w8a8_moe_kernel.hip` picks group_size=128). What is absent is the scale *decode and multiply* at that boundary, not the boundary break.
2. **"`if constexpr (WLoad::is_fp8)` blocks a blockwise fp8 loader."** It is read at exactly two places (`moe_gemm_tiled.h:606`, `:767`) and gates **only** the fold. Staging is dispatched through `WLoad::stage_b` unconditionally (`:572`, `:733`). A blockwise loader declaring `is_fp8 = false` gets byte staging from its own `stage_b` *and* the fold, touching zero shared code — which is exactly what `W8A16GemvLoader` already ships (`gemv_decode.h:1701`, `is_fp8 = false` with an e4m3 `WT`, its own comment: "this flag describes the ACTIVATION path, not the weight"). Re-keying to `kGroupFold` is an optional cleanup, provably a compile-time no-op per loader, and would additionally require adding `kGroupFold` to `NlInt8Loader`, which lacks it.
3. **"The dense tiled core has no WLoad axis, so hosting a blockwise arm requires adding one."** True of `gemm_tiled_kernel` (`gemm_tiled.h:89-98`, weight hardwired `const int* w_packed` with `PACK_FACTOR=8`) — but that is the int4×fp8 cell of a six-cell dense matrix, and nothing forces it to host this arm. The fp8-weight dense cell already has the axis, the dense instantiation, and a live caller.
4. **"`moe_bf16act_tiled_kernel` is grouped-only / G2 is open."** Stale in two source comments and one doc row; `GATHER` is at `moe_gemm_tiled.h:1305` and the dense launcher ships.
5. **"The fold has nowhere to go without restructuring the a16 body."** Overstated: the mechanism is a shipped WLoad policy in the same header, `kGroupFold` is already declared on all five loaders, the compile-out neutrality is recorded twice (`moe_gemm_tiled.h:601-607` VGPR/occupancy unchanged; `FORMAT_MATRIX.md:89-94` max|Δ|=0 at group 32/64/128, both bodies in one binary), and the second-accumulator move is precedented in situ by SILU.
6. **"An early scale transpose is a silent corruption risk."** Self-enforcing: no scale tensor among the 156 is square at TP=1 or TP=2, so an early transpose raises in `split`, `cat`, or `load_state_dict`.
7. **"Asserting 'more accurate' at a bf16 output will read as a tie."** The committed W8A16 precedent measured the streaming arm closer to fp64 in **36 of 36 cases** at a bf16 output (ratio 0.545–0.616). The effect here is same-order-as-the-quantum, which is the regime where it stays visible; an *arm-vs-arm* equality assertion would tie, which is why the precedent prints that distance and never gates on it.
8. **"A weight-format change has a second sizing site."** Not for a dense arm — `model_memory` is a measured free-memory delta (`engine.py:297-315`), and the analytic byte models cover MoE expert containers only. The second site here is the *operating point*, not the accounting.
9. **`--spec-algorithm mtp` on this checkpoint** fails to **boot** loudly at TP=2 (`create_moe_quant_method` has no `is_fp8_block` arm) — it is not a reachable operating point, so decode M on the shipped launch is ≤ 2 (`tools/serve.sh:842`, `CONC` capped at 2 and fed to `--max-running-requests`). That governs *priority*, never correctness bounds: `EXTRA_ARGS` wins and the engine default is 256.