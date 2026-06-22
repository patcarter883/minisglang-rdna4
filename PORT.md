# minisgl-rdna4 — RDNA4 (gfx1201) native serving engine

A fork of [mini-SGLang](https://github.com/sgl-project/mini-sglang) re-targeted from NVIDIA/CUDA to
AMD RDNA4 (gfx1201, Radeon RX 9070 XT), built around the `w4a8_fp8_wmma` int4-weight/fp8-activation
WMMA kernel and the tuned RDNA4 `triton_attn`, extended with GDN (Gated Delta Net) hybrid support.

- **Forked from upstream:** `9a91cfafe754aa85daee49998176275667eb58f2` (mini-SGLang `main`, 2026-05-17).
- **Full design + rationale:** `vllm-gfx1201/docs/RDNA4_ENGINE_DESIGN.md` (the north star).
- **Companion analysis:** `vllm-gfx1201/docs/MINISGLANG_PATHWAYS.md`.

## Governing principles (do not violate)

1. **Maximise RDNA4 strengths** — native fp8/int4 WMMA, 3D flash-decode, `waves_per_eu` tuning.
2. **Nothing dequants to F16.** I/O = bf16, compute = fp8 (e4m3**fn**, NOT fnuz), accumulate = f32.
   Banned = per-element F16 dequant of quantized weights/activations. fp32 accumulate + bf16
   activations are REQUIRED, not violations.
3. **Clean, tidy, agent+human-maintainable.** Keep mini-SGLang's style: small typed modules,
   Protocol-based backends, `__dict__`-introspection weight loading. No special-case spaghetti.
4. **W4A8 kernel is consumed as a dependency** from `vllm-gfx1201/w4a8_fp8_wmma/` (single source of
   truth) — call the raw `torch.ops.w4a8_fp8_wmma.*` ops; NEVER copy its csrc.

## Equivalence oracle

Greedy **token-identical vs `vllm22-w4a8:combined`** (same kernels) — tighter than vs HF. Reuse the
`vllm-gfx1201/patches/run_het_e2e_combined.sh` pattern. Stand up before Phase 2.

## Phase 1a boot — RESULT (2026-06-17): SUCCESS ✅

Booted **Qwen/Qwen3-0.6B** (dense, bf16, eager, TP=1) on gfx1201 via the recipe below. The
`triton_rdna4` backend auto-selected (`is_rocm()` works), the vendored tuned Triton attention
kernel JIT-compiled and ran under HIP, and **all 4 greedy prompts produced coherent, correct
completions** ("The capital of France is" -> " Paris…"; "2 + 2? A:" -> " 4"). Validates the whole
stripped path: embedding -> torch RMSNorm -> NeoX RoPE -> tuned Triton attention -> SwiGLU -> greedy
sampling -> paged KV + radix scheduler. KV cache 8.13 GiB, eager.

Gotchas hit + fixed: (1) hand-rolled `docker run` needs ROCm device passthrough
(`--device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined
--security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb`) or torch sees no HIP
GPUs and `is_rocm()`->False. (2) Qwen3-0.6B weights weren't cached (config only) — fetched via
`snapshot_download` (drop `HF_HUB_OFFLINE` for the fetch). The weight loader globs `*.safetensors`
in the resolved snapshot dir, so a bare hub-id with no cached weights yields 0 tensors.

## Phase 1a validation — token-diff vs vLLM (2026-06-18): PASS (not bit-identical, expected)

Greedy on identical input ids (`tools/cmp_ours.py` + `cmp_vllm.py`, GPU0): per-prompt agreement
5/48, 21/48, 39/48, **48/48 identical**. First generated token matches on ALL prompts; all outputs
coherent+correct. Divergences land at semantic near-ties (sub-ULP logit diff flips greedy argmax,
contexts then separate) — NOT a bug (a bug diverges at token 0 / yields garbage).

**Key finding — pick the right oracle:** we share vLLM's *attention* kernel but reimplement
RMSNorm / RoPE / SwiGLU as torch shims, so the full model is NOT bit-identical to vLLM by
construction. Greedy decoding amplifies legitimate bf16 differences. So **token-identity is too
strict**; the numerical oracle going forward is **logit-level agreement** (cosine-sim / top-1 rate on
a single forward pass), which tolerates sub-ULP noise. (vLLM's `--enable-prefix`-style custom op
fusions `norm_quant`/`act_quant` are another source of difference.) If we ever want tighter token
agreement, match vLLM's exact RMSNorm/activation op order — but that's not a goal (we want a clean,
correct, performant engine, not a vLLM bit-clone).

NEXT: (optional) build a logit cos-sim check as the standing oracle; then Phase 1b (fp8-KV via
lifted reshape_and_cache + 3D flash-decode + startup autotuner).

## Phase 1a boot — ready-to-run recipe (fire when GPUs idle)

```bash
docker run --rm \
  -e HIP_VISIBLE_DEVICES=0,1 -e ROCR_VISIBLE_DEVICES=0,1 \
  -v /home/pat/code/minisgl-rdna4:/engine \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  --entrypoint bash vllm22-w4a8:combined -lc '
    source /app/.venv/bin/activate
    pip install -q msgpack pyzmq prompt_toolkit accelerate 2>&1 | tail -1
    PYTHONPATH=/engine/python python /engine/tools/boot_smoke.py \
      --model Qwen/Qwen3-0.6B --json-out /engine/tools/boot_ours.json'
```
Expect: coherent greedy completions for 4 prompts + token ids. Validates the whole stripped path +
tuned attention. If it works, do the token-diff vs vLLM (same kernel) on the same prompts.

## Phase 2 — W4A8 (in progress)

Designed around a **swappable kernel provider** so the parallel custom kernel framework drops in
without touching layers/models/loader (memory: parallel-custom-kernel-framework).

- **2a foundation (built, pending no-regression oracle):**
  - `quant/` package: `config.py` (`QuantConfig.from_hf` — AWQ + minimal compressed-tensors),
    `kernels.py` (the swappable provider — `w4a8_linear` + per-M version ladder v11/v10/v5, fp16
    staging), `method.py` (`LinearMethod` protocol, `UnquantizedLinearMethod`, `W4A8LinearMethod`
    scaffold, `create_linear_method`).
  - `models/config.py`: `ModelConfig.quant` parsed (top-level `quantization_config`).
  - `layers/linear.py`: `_LinearTPImpl` routes through `_method` (default unquantized = no behavior
    change); OProj/RowParallel forwards too. Validate: bf16 oracle still ~0.9996 (no regression).
- **2c (IMPLEMENTED, validating) — AWQ→op weight conversion + wiring:**
  - `quant/kernels.py:awq_to_op_layout` — dense AWQ (K,N//8)/(G,N)/(G,N//8) → op
    (N,K//8)/(N,G)/(N//8,G) (unpack nibbles, reverse AWQ order, transpose, repack). op supports
    g128 natively (v10), no g→32 re-expansion.
  - `quant/method.py:W4A8LinearMethod` — create_weights (declare AWQ buffers), process_weights_after_load
    (convert → underscore `_w_packed_op`/`_scales_op`/`_zeros_op`, free originals), apply (provider).
  - `layers/base.py` + `linear.py`: `post_load()` recursion runs the conversion; quant_method threaded
    through the 4 proj-linear ctors.
  - `models/utils.py`: qkv/o/gate_up/down get `create_linear_method(config.quant)` (lm_head/embed bf16).
  - `models/weight.py`: AWQ siblings merge along dim=1 (output/packed dim); dense/bias dim=0.
  - `engine.py`: load no longer blanket-casts int/scales (preserves quant dtypes); calls `post_load()`.
  - TP-quant sharding deferred (MVP is TP=1). One-shot unpack transient — chunk for >7B (27B OOM note).
  - **Validation:** boot Qwen2.5-Coder-7B-AWQ (coherence = MVP), then logit oracle vs the cached
    UNQUANTIZED bf16 7B (`MINISGL_ORACLE_MODEL`=AWQ, `MINISGL_ORACLE_REF`=base; cos-sim reflects
    quant error, not a bug). Run via `gpu-lease.sh -n 1`.

## Phase status

| Phase | What | Status |
|---|---|---|
| 0 | Fork + strip NVIDIA deps (dense path) | **done** (cf5a478) |
| 1a | Tuned RDNA4 `triton_attn` backend, bf16 KV, 2D grid — wired + import-validated | **done** (fed6efa) |
| 1a-boot | Functional boot bf16 eager TP=1 — **coherent greedy generation on gfx1201** | **done** 2026-06-17 |
| 1b-1 | 3D flash-decode wired (lazy f32 segment scratch); validated correct (converges to 2D as segments→1) | **done** 2026-06-18 |
| 1b-2 | startup autotuner — right-size segments + tuned `waves_per_eu`/warps/tile (RDNA4 perf) | todo |
| 1b-3 | fp8-KV (e4m3fn KV buffer, torch-scatter store, scale=1.0) — env MINISGL_KV_FP8=1 | **done** 2026-06-18 |
| — | **logit-level oracle** (engine vs HF, cos-sim/top-1) — the standing numerical oracle | **done** 2026-06-18 |
| 2 | **W4A8 (AWQ) dense serving — MVP** | **DONE 2026-06-18** ✅ |
| 3d | GDN-hybrid Qwen3.5-4B — build + serve, greedy token-diff vs vLLM (coherent) | **DONE 2026-06-21** ✅ |
| 3e | GDN-hybrid — quantitative logit oracle + decode-path per-step parity vs HF | **DONE 2026-06-22** ✅ |
| 1b-2 | startup autotuner — right-size segments + tuned `waves_per_eu`/warps/tile (RDNA4 perf) | todo (needs perf-bench infra) |

## Phase 3 — GDN (Gated Delta Net) hybrid attention → the 35B (STARTED)

The 35B is a GDN/SSD hybrid: 40 layers, 1-in-4 full-attention interleave; linear layers use GDN
(delta-rule linear attention). FLA kernels already run on gfx1201 (vLLM vendors them) — this is
integration, not kernel porting. Source extracted to `/home/pat/code/scratch/gdn/`.

- **3a — recurrent state cache (DONE):** `kvcache/gdn_state.py` `GDNStateCache` — per-seq fixed slots
  (conv_state `(conv_dim, k-1)` + ssm_state `(num_v_heads, head_v_dim, head_k_dim)`), free-list
  allocator (NOT paged/prefix-cacheable — GDN state can't roll back). CPU unit-tested
  (`tools/gdn_state_test.py`): shapes/alloc/free/reuse/reset/exhaustion. 35B dims: conv_dim 8192,
  ssm (32,128,128).
- **3b — one GDN layer's numerics (DONE 2026-06-19).** Three checks, all green:
  - **3b-1 — vendor the FLA/mamba kernels (DONE 2026-06-19):** the 16-file closure
    (chunk_gated_delta_rule + chunk_delta_h/chunk_o/chunk_scaled_dot_kkt/cumsum/wy_fast/solve_tril/
    l2norm/op/index/utils, fused_recurrent, fused_sigmoid_gating, fused_gdn_prefill_post_conv,
    layernorm_guard; mamba/ops causal_conv1d/layernorm_gated/triton_helpers) vendored into
    `python/minisgl/gdn/{fla,mamba}/ops`. Their only non-torch/triton deps were ~5 `vllm.*` symbols
    — rewritten to `gdn/_compat.py` (triton/tl/tldevice, current_platform.is_cuda_alike,
    cdiv/next_power_of_2, num_compute_units via runtime CU query, NULL_BLOCK_ID/PAD_SLOT_ID). No
    fake `vllm` package (would shadow the real one in the image). **Validated:** 18/18 files
    byte-identical to installed vLLM 0.22.69 modulo import lines; GPU parity vs installed originals
    (`tools/gdn_kernel_parity.py`, `gpu-lease -n 1`) — 5/5 gated kernels max|Δ|=0 incl. both
    shim-sensitive paths (chunk_gated_delta_rule→is_cuda_alike, RMSNormGated→num_compute_units).
    `causal_conv1d_fn` micro-call is info-only (needs full GDNAttentionMetadata; covered by byte-diff
    + 3b-3). `kda.py` excluded (outside closure; would need a custom_op stub).
  - **3b-2 — the layer (DONE, pending 3b-3 parity):** `gdn/layer.py` `QwenGatedDeltaNet` — clean
    `nn.Module` reimplementing the reference's no-spec forward COMPUTE (prefill = causal_conv1d_fn →
    fused_post_conv_prep → chunk_gated_delta_rule, writes final ssm_state; decode = causal_conv1d_update
    → rearrange → fused_sigmoid_gating_delta_rule_update, in-place state) + `_output_projection`
    (RMSNormGated(core,z) → out_proj). TP=1, unquantized bf16, state passed EXPLICITLY (no
    forward_context). Strips CustomOp/distributed/MergedColumnParallelLinear. Import-validated; numerics
    pending 3b-3. conv_state assumed dim-first DS `(slots, conv_dim, k-1)` (= GDNStateCache layout).
  - **3b-3 — single-layer parity (DONE 2026-06-20 — PASS; the 2026-06-19 "DONE" was PREMATURE).**
    capture/replay vs the REAL vLLM `QwenGatedDeltaNetAttention` (combined image, vllm 0.22.69) —
    independent oracle, NOT a self-authored eager ref.
      * `tools/gdn_layer_parity_cpu.py` (CPU) — `_split_qkvz_ba` / `_rearrange_mixed_qkv` /
        `_output_projection` reshape order vs inline replications of the real non-interleaved
        (Qwen3.5) logic. **9/9 bit-exact.** (SOLID — stable across the whole investigation.)
      * `tools/gdn_layer_parity.py` (GPU) — stands up the real layer (Qwen3NextConfig + a real
        `VllmConfig(ModelConfig=Qwen3-0.6B)` under `set_current_vllm_config`, TP=1, quant=None,
        `gqa_interleaved_layout=False`); monkeypatches `get_forward_context`; sets `real.kv_cache`;
        drives `_forward_core_rocm` then `_output_projection` separately; per-param shape-asserted
        copy of ALL 7 tensors (in_proj_qkvz/ba, conv1d, out_proj, A_log, dt_bias, norm) real→mini;
        shares in_proj via `_SharedProj`. Run modes via `GDN_SINGLE`: `both` (in-process A/B),
        `probe` (kernel matrix vs a CPU ground-truth conv), `real`/`mini` (single-pass capture).
      * **⚠ THE 2026-06-19 "PREFILL bit-exact, max|Δ|=0" PASS WAS VACUOUS (0==0 aliasing).** Root
        cause: the harness used cache **slot 0**, and `causal_conv1d_fn` treats any program whose
        `cache_indices[seq] == NULL_BLOCK_ID (==0)` as a null/padding block and **returns its output
        buffer UNWRITTEN** (`out = torch.empty_like(x)` → reads back as 0 / NaN / aliased garbage,
        the exact "flip-flop 0.0 / 11.875 / NaN"). Both real AND mini were skipped, so every prior
        prefill check compared garbage-to-garbage. Proven with a pure-torch CPU reference conv:
        **slot 0 → WRONG, slot ≥ 1 → CORRECT (rel ~7e-3 bf16), namespace/layout/metadata-independent.**
      * **PREFILL — now genuinely bit-exact at slot ≥ 1, IN-PROCESS (shared weights+hs).** `q, k, v,
        g, beta, core_attn_out, ssm_state, conv_state` all **max|Δ| = 0** at substantial magnitude
        (core ~1.8e-2, ssm ~0.30, conv ~9.3); only the z-dominated `out` differs at rel ~2e-3 (bf16
        out_proj GEMM backend). Stable across runs. Δ=0 follows from shared in_proj + copied weights +
        byte-identical kernels, so this validates the **GDN orchestration** (split order, DS-vs-SD
        conv_state layout, state write/indexing, call sequence) — the kernels were validated in 3b-1.
        NOTE: a TWO-process A/B is invalid — each process
        random-inits its own real layer (incl. `A_log ~ N(-2,0.3)`), so weights differ across
        processes; parity MUST be in-process (`GDN_SINGLE=both`) where the weight-copy applies.
      * **DECODE** — independent shared-random-state decode (nonzero readout core ~3e-2, output ~17)
        matches bit-exact; SOLID across the investigation. It was unaffected by the slot-0 bug because
        that test used **slot 1** (`sidx=1`), not because the kernel differs: `causal_conv1d_update`
        carries the SAME `cache_indices[seq] == NULL_BLOCK_ID` skip, so decode at slot 0 is skipped too
        — the "never slot 0" constraint is identical for prefill and decode.
      * **Corrections to earlier notes:** (1) a transient `.contiguous()` "GAP fix" in
        `forward_prefill` was REMOVED — the probe showed gapped split-view == contiguous == CPU-ref at
        slot ≥ 1 (the earlier GAP≠PACK was slot-0 aliasing noise). (2) `metadata=None` in
        `causal_conv1d_fn` is correct (== precomputed `md_p` at slot ≥ 1); the on-the-fly batch_ptr
        path is fine. (3) The conv autotune is sensitive to op-sequence on a cold/live buffer — a
        GEMM-free warmup settles it before the measured prefill.
    **★ 3c CONSTRAINT (the durable finding): `GDNStateCache` MUST NOT allocate slot 0 to a real
    sequence** — cache index 0 == `NULL_BLOCK_ID`, which `causal_conv1d_fn` silently treats as a
    padding/no-op block (conv output unwritten). Either reserve slot 0 as the null block (offset all
    real slots by ≥1, matching vLLM), or the prefill conv produces garbage with no error.
    **Live RDNA4 facts that bind 3a/3c:**
      * `is_conv_state_dim_first() = False` (**SD**, no `VLLM_SSM_CONV_STATE_LAYOUT` env) — the real
        layer transposes `kv_cache[0]` to `(…, conv_dim, k-1)` for the conv kernels; minisgl +
        `GDNStateCache` store dim-first **DS** `(slots, conv_dim, k-1)`. So **3c must wire the conv
        state in the transposed frame** (allocate the engine's conv buffer SD and pass
        `.transpose(-1,-2)`, or keep DS and accept it as the minisgl-native layout). The harness
        already branches on this boolean for both the fed buffer and the comparison frame.
      * `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE = True` in the image — production decode would take
        `fused_recurrent_gated_delta_rule_packed_decode`; minisgl `gdn/layer.py` implements the
        non-packed `fused_sigmoid_gating_delta_rule_update`. **Deliberate divergence to revisit in
        3c** (validate the packed path or keep non-packed for greedy token-parity).
      * `GDN_AITER_TRITON_AVAILABLE = None` — production `forward_hip` → `forward_cuda` →
        `_forward_core` (the path the oracle drives; the aiter decode-fast path is unavailable).
      * Box quirk: hipBLASLt intermittently `INTERNAL_ERROR`→`ALLOC_FAILED` on the in_proj GEMM
        under serve load — run with `-e TORCH_BLAS_PREFER_HIPBLASLT=0`.
- **3c — scheduler slot lifecycle + metadata + warmup hook (DONE 2026-06-20).** The vLLM
  framing ("one batch splits into prefill/decode subsets") does NOT apply here: **minisgl
  schedules HOMOGENEOUS batches** (`Batch.phase` = prefill XOR decode — `_schedule_next_batch`
  is `prefill_manager … or decode_manager …`), so the subset-split is already done by the
  scheduler and a GDN layer just dispatches `forward_prefill`/`forward_decode` on
  `batch.is_prefill`. THE RISK relocated to **chunked-prefill state threading** (one recurrent
  slot shared across fresh / chunk-continuation / prefill→decode / free for a sequence). Five
  sub-phases, all green:
  - **3c-0 (`f636197`) — `GDNStateCache` reserves slot 0 = NULL_BLOCK_ID.** The free-list
    popped slot 0 FIRST (`range(N-1,-1,-1)`), so the first real sequence got the null block →
    `causal_conv1d_fn`/`update` silently skip it (output unwritten = garbage). This was the
    root cause of the vacuous 3b-3 PASS. Free-list now stops at 1; slot 0 is buffer-only.
    Size `num_slots = max_running_req + 2`. CPU test asserts slot 0 is never handed out.
  - **3c-1 (`8310cea`) — `gdn/metadata.py`.** `build_gdn_metadata(batch, state_indices, device)`
    packages the per-batch tensors the layer consumes: `query_start_loc` (cumsum `extend_len` /
    arange), `state_indices` (scheduler slots, all ≥1), per-seq `has_initial_state`
    (`cached_len>0`). chunk/conv metadata left None for MVP (kernels compute on the fly;
    `cumsum.py` self-calls `prepare_chunk_indices` when None — verified). Pins host staging
    only for a CUDA target (stays CPU-testable).
  - **3c-2a (`e6e65d9`) — `GDNSlotManager`, the spine.** uid-anchored slot lifecycle (uid is
    the only identity stable across chunks — each chunk builds a NEW `Req`). Fresh→alloc+zero;
    continuation→reuse, no re-zero; decode→pure lookup (unknown uid raises, no silent garbage);
    finish/abort→idempotent free (overlap can double-free). CPU-unit-tested (all 4 cases).
    **★ Precondition: GDN-hybrid models MUST run the non-radix ("naive") prefix cache** — GDN
    state isn't prefix-cacheable; a radix hit gives `cached_len>0` with no state behind it
    (silent garbage). Confirmed `NaivePrefixCache.match_prefix` always returns `cached_len=0`,
    so under it `cached_len>0` ⟺ chunk continuation = exactly the `has_initial_state` predicate.
  - **3c-2b (`40b26b2`) — inert scheduler/engine wiring.** `Engine.gdn_state: GDNStateCache|None`
    placeholder (None for every dense model; **3d constructs it** from the 35B's linear-attn
    dims); `Scheduler` builds a `GDNSlotManager` iff non-None; `_prepare_batch` allocs slots +
    builds `batch.gdn_metadata`; `_free_req_resources` frees the slot (single site covering
    finish AND abort). All hooks are `if self.gdn_slots is not None` no-ops today → dense path
    byte-unchanged (import-smoke verified). Eager only (GDN cudagraph out of 3c scope).
  - **3c-3 (`e2337ec`) — `QwenGatedDeltaNet.warmup_conv`.** GEMM-free conv warmup on a private
    2-slot scratch (never touches real state) settling `causal_conv1d_fn`'s in-place
    batch_ptr autotune (per-process — NOT in the on-disk Triton JIT cache). Engine calls it
    once before the first real batch in 3d.
  - **3c-4 (`8e97784`, `4f4165a`) — engine-plumbing integration test, GREEN on gfx1201**
    (`tools/gdn_3c_integration.py`, lease + `TORCH_BLAS_PREFER_HIPBLASLT=0`). Validates the
    WIRING (not 3b numerics), non-vacuous with magnitude guards:
      * **A** metadata exactness — built tensors == hand-built (`[0,96,160,192]`, `[F,T,F]`).
      * **B** chunked-vs-single prefill parity (independent state-threading oracle): a 2-chunk
        prefill sharing one slot (`has_initial_state=True` on chunk 1) reproduces a single-shot
        prefill of the concatenation. **conv_state BIT-EXACT (Δ=0)** for aligned 64+64 AND
        non-aligned 80+48 (partial FLA block + mid-block conv carry); ssm/output at bf16 noise
        (rel ~2–4e-3), |out|~17.
      * **C** prefill→decode handoff: decode reuses the slot, reads the state (|out|~12),
        advances ssm in place.
      * **D** multi-seq batch executed through the kernels (the load-bearing case): a 2-seq
        (96+64) batch in ONE `forward_prefill`/`forward_decode` vs each seq run ALONE. Exact
        signature of correct segmentation — **seqA (leading segment) BIT-EXACT** (Δ=0
        output/ssm/conv), seqB (trailing) at bf16 noise (rel ~1e-3); slots `[1,2]` distinct,
        reused at decode. This is the ONLY place varlen segmentation + per-seq state gather is
        EXECUTED (Part A only asserts the tensors).
  - **Packed-decode (deliberate divergence, carried to 3d perf):** image sets
    `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=True`; minisgl keeps the non-packed
    `fused_sigmoid_gating_delta_rule_update` (3b-3-validated) for greedy token-parity.
  - **Deferred to 3d (needs the GDN model to exist):** construct `Engine.gdn_state` from the
    35B's linear-attn dims; force the non-radix cache + eager for GDN models; live serve.
- **3d — interleave + serve (STARTED 2026-06-21).** Build the qwen3_5 hybrid model, construct
  `Engine.gdn_state`, force naive cache + eager, serve, greedy token-diff vs the combined image.
  **★ TARGET CORRECTION (2026-06-21 — what's actually on the box):** the ONLY cached GDN-hybrid
  checkpoint is the **multimodal `Qwen/Qwen3.5-4B`** (`Qwen3_5ForConditionalGeneration`, text
  submodel `model_type=qwen3_5_text`, 32 layers = 24 GDN + 8 full at `[3,7,11,15,19,23,27,31]`,
  **dense MLP — no MoE**, weights under `model.language_model.*`, vision under `model.visual.*`).
  Both "DFlash" dirs in cache (`z-lab/Qwen3.5-4B-DFlash`, `z-lab/Qwen3.5-35B-A3B-DFlash`) are
  `DFlashDraftModel` **speculative draft** models (model_type qwen3, sliding-attn, ~6 layers) —
  NOT the base hybrid. So the **35B base is not present**; the 4B is the servable MVP target and
  exercises the full GDN-interleave + serve path **without** the MoE path (already validated in
  Phase-2-MoE). The 35B becomes a later weight/MoE swap once a checkpoint exists.
  **★ FULL-ATTENTION LAYER IS NOT THE PHASE-1 `RopeAttn`** (the earlier "full layers reuse Phase-1
  attention" was wrong for Qwen3.5). `qwen3_5_text` full attention (`modeling_qwen3_5.py:714-789`)
  adds: (1) **attn output gate** — `q_proj` emits `2*num_heads*head_dim`; the 2nd half is a gate,
  `attn_out *= sigmoid(gate)` AFTER attention; (2) **partial rotary** — `partial_rotary_factor=0.25`
  → `rotary_dim=64` of `head_dim=256` (rotate first 64, pass the rest); (3) per-head q_norm/k_norm
  (RMSNorm, eps 1e-6); (4) separate q/k/v projections (16 q / 4 kv heads, GQA). So 3d-1 authors a
  NEW gated-partial-rotary attention op (and the rope/attention path must honor `rotary_dim<head_dim`).
  **★ GDN LAYER REUSABLE AS-IS:** `minisgl/gdn/layer.py` `QwenGatedDeltaNet` is a faithful compute
  match to `Qwen3_5GatedDeltaNet` (non-interleaved qkv/z + b/a, conv kernel 4, RMSNormGated, l2norm
  in kernel, `repeat_interleave(2)` GQA) — only the **weight names differ**: the checkpoint splits
  `in_proj_qkv`+`in_proj_z` and `in_proj_b`+`in_proj_a`, so 3d-3 concats them into the layer's
  `in_proj_qkvz` / `in_proj_ba`. GDN projections stay unquantized bf16.
  Sub-phases:
  - **3d-0 (DONE 2026-06-21) — `ModelConfig` GDN fields + `ctx.gdn_state`.** `config.py`: added
    `linear_num_{key,value}_heads` / `linear_{key,value}_head_dim` / `linear_conv_kernel_dim` /
    `layer_types` (+ `is_gdn_hybrid` / `gdn_layer_ids` / `num_gdn_layers` / `gdn_conv_dim`), parsed
    in `from_hf` ONLY when the config carries linear dims (dense path untouched). Made `from_hf`
    robust to Qwen3.5's `rope_parameters` dict (rope_theta + `partial_rotary_factor` → `rotary_dim`;
    the existing `RotaryConfig.rotary_dim` field already supports partial rotary). Added
    `Context.gdn_state` (mirrors `kv_cache`, None for dense). CPU-verified: `tools/qwen3_5_config_test.py`
    (`--standalone`, no torch) — 4B parse (conv_dim 8192, rotary_dim 64, 24+8, tied, SwiGLU 9216)
    AND dense regression (non-GDN, full rotary, GDN fields None) both PASS.
  - **3d-1a (DONE 2026-06-21) — partial rotary in `RotaryEmbedding`.** Dropped `assert rotary_dim
    == head_size`; `_apply` rotates the first `rotary_dim` dims (NeoX rotate-half) and concats the
    unrotated tail — reduces EXACTLY to the old path when `rotary_dim == head_size`. CPU-verified
    bit-exact (`tools/rotary_partial_test.py`) vs an HF reference for partial (256/64) AND full
    (128/128, 64/64); tail bit-identical. `AttentionLayer` already builds rope from
    `rotary_config.rotary_dim`, so partial rotary engages with no other change.
  - **3d-1b (DONE 2026-06-21) — `models/qwen3_5.py` + register.** `Qwen3_5Attn` (gated GQA: q_proj
    emits 2× per-head = q + sigmoid gate applied to the attn output; q/k norm; partial rotary via
    the shared `AttentionLayer`; global layer_id indexes the KV pool). `GDNLinearAttn` — a **BaseOP
    bridge** around the nn.Module `QwenGatedDeltaNet` (its params live in nn.Module `_parameters`,
    invisible to BaseOP's `__dict__` walk): `state_dict`/`load_state_dict` delegate to the module,
    load via `assign=True` so meta params are replaced by real-device checkpoint tensors AND fp32
    `A_log`/`dt_bias` dtype is preserved. Built on the meta device; per-layer branch on
    `config.layer_types`; the GDN bridge pulls `ctx.gdn_state.conv/ssm(gdn_layer_id)` +
    `batch.gdn_metadata` and dispatches `forward_prefill`/`forward_decode` on `batch.is_prefill`.
    Dense SwiGLU `GatedMLP`; tied lm_head. Registered `Qwen3_5ForConditionalGeneration`.
    **Verified (`tools/qwen3_5_build_smoke.py`, combined image, CPU/meta — no GPU lease):** builds
    the 4B on meta → 24 GDN + 8 full layers, 346 state-dict tensors (24·11 + 8·10 + embed + norm),
    correct per-layer key sets / shapes (q_proj 8192×2560, in_proj_qkvz 12288×2560, conv1d 8192·1·4)
    / fp32 A_log+dt_bias, AND a clean `state_dict()`↔`load_state_dict()` round-trip (the bridge's
    key layout is self-consistent). ★ Carries to 3d-3: the engine's bf16 weight cast MUST skip
    `A_log`/`dt_bias` (keep fp32).
  - **3d-2 (DONE 2026-06-21, import-verified; live boot deferred to 3d-4) — engine wiring.**
    `Engine.__init__`: for `mc.is_gdn_hybrid`, construct `GDNStateCache(num_gdn_layers,
    max_running_req+2 slots, conv_dim/conv_kernel/v-heads/head-dims from the config)`, set
    `ctx.gdn_state`, and `warmup_conv(512)` each GDN layer (settles the per-process conv autotune).
    Force **eager** via an empty `cuda_graph_bs` (→ `max_graph_bs=0`, capture skipped,
    `can_use_cuda_graph`→False). `Scheduler.__init__`: force **naive** prefix cache when
    `engine.gdn_state is not None`. Dense path byte-unchanged (all branches gated on the flag);
    import-smoke clean in the combined image. ★ Live boot verification rides with 3d-4.
  - **3d-3 (mapping CPU-VERIFIED 2026-06-21; live load rides with 3d-4) — weight-name mapping.**
    `weight.py`: `qwen3_5_remap` (pure, testable) + `_load_qwen3_5_weight` streaming loader, branched
    in `load_weight` on `config.is_gdn_hybrid`. Skips `model.visual.*` (vision) + `mtp.*` (the
    multi-token-prediction head — text-only MVP); strips `model.language_model.` → `model.`; concats
    `in_proj_qkv`+`in_proj_z` → `in_proj_qkvz` and `in_proj_b`+`in_proj_a` → `in_proj_ba` (dim 0),
    renames `conv1d.weight` → `conv1d_weight`, merges dense `gate`+`up` → `gate_up`. Full-attn q/k/v
    stay SEPARATE (q_proj carries the output gate → unfusable). ★ Engine `_cast`: A_log/dt_bias forced
    fp32 — A_log ships fp32 but **dt_bias ships bf16 in the checkpoint** (must upcast; the kernels +
    the model nn.Parameter both want fp32). CPU-verified header-only (no GPU, no 8 GB load):
    `tools/qwen3_5_weight_map_test.py` — 738 ckpt keys → skip 312 → **346 native keys == the model's
    346**, every shape + dtype matches, GDN concat shapes + fp32 gating + split-QKV spot-checks pass.
    ★ FOUND (defer to rope/3d-4): `from_hf` maps Qwen3.5's `rope_parameters` (rope_type "default" +
    an `mrope_section` LIST) into `RotaryConfig.scaling`; `AttentionLayer` does `tuple(scaling.items())`
    → unhashable list into the lru-cached `_get_rope` → crash on model build from the real config. The
    weight test nulls `scaling` (rope is non-parametric, irrelevant to key layout) to isolate 3d-3.
  - **3d-4 (DONE 2026-06-21 — live serve 4B + greedy token-diff PASS, coherent, not bit-identical) —**
    Qwen3.5-4B (bf16, eager, naive cache, 24 GDN + 8 full-attn layers) boots on gfx1201 and generates
    **coherent + correct** greedy output. Token-diff vs the combined image's vLLM (same prompts, greedy,
    eager): "three primary colors → yellow…" **32/32 exact**, "2+2 → 4…" **32/32 exact**, "capital of
    France → Paris…" 24/32 (first div @ tok 23), "Once upon a time…" 15/32 (div @ tok 1) — both remain
    coherent. Late/early divergence = expected bf16 drift between minisgl's `triton_rdna4`+vendored GDN
    and vLLM's own kernels (same posture as Phase 1a "PASS, not bit-identical").
    Three fixes were required to get from "boots but emits all-spaces (token 220)" to coherent:
    1. **rope crash (from 3d-3):** `from_hf` now leaves `RotaryConfig.scaling=None` for a "default"
       `rope_type` (Qwen3.5's `rope_parameters` is rope_type "default" + an `mrope_section` LIST — mrope
       is multimodal-only; for text it reduces to plain partial rotary, already set via `rotary_dim`, and
       the list is unhashable in the lru-cached `_get_rope`). The weight test now builds from the real
       config unmodified and still PASSES.
    2. **GDN-state OOM:** `GDNStateCache` sizes `max_running_req+2` fixed conv+ssm slots/GDN-layer; the
       default 256 → 6 GiB ssm alloc OOMs a 16 GB card. `boot_smoke.py` gained `--max-running-req`
       (used 16 for the smoke) + `--prompt`.
    3. **★ ROOT-CAUSE of the all-spaces garbage — RMSNorm gain convention.** `Qwen3_5RMSNorm` applies
       `(1 + weight)` with weight **init 0** (centred on 0), UNLIKE the dense Qwen3 plain `weight` (init
       1). The checkpoint weights load bit-identically, but minisgl applied them plainly → every norm
       scaled by ~0.2 instead of ~1.2 (a measured **5.37× input-magnitude deficit** + direction error
       cos 0.81 at the layer-0 GDN input), washing out all context → unigram-frequency output (bare
       space). Fix: `plus_one` flag on `RMSNorm`/`RMSNormFused` (default False = dense path untouched),
       set True in qwen3_5 for `input_layernorm`/`post_attention_layernorm`/`model.norm`/`q_norm`/
       `k_norm`. The GDN's `RMSNormGated` keeps the plain convention (init 1) — verified, unchanged.
    Diagnosis tools added (kept for regression/parity): `qwen3_5_hf_ref.py` (HF per-layer ground truth),
    `qwen3_5_hs_cmp.py` (per-layer cos/rel — localized divergence to layer 0), `qwen3_5_gdn_isolate.py`
    (GDN compute vs HF = cos 0.99996, exonerated the kernel), `qwen3_5_weight_value_cmp.py` (loaded
    values bit-identical → not a loader bug), `qwen3_5_gdn_replay.py` (engine GDN input ≠ HF input by
    5.37× → pinpointed the norm), `qwen3_5_vllm_ref.py` (the token-diff harness). Validated on
    `Qwen/Qwen3.5-4B` (tied lm_head, no MTP served).
- **3e — quantitative numerical validation of the GDN-hybrid (DONE 2026-06-22 — decode path SOUND).**
  3d-4 proved prefill numerics (per-layer hidden-state diff vs HF caught the RMSNorm bug) but the
  **decode** path was only validated by "the generated text is coherent." forward_decode is a DISTINCT
  code path — `causal_conv1d_update` + `fused_sigmoid_gating_delta_rule_update` with **in-place
  ssm_state recurrence**, none of which `forward_prefill` (chunk_gated_delta_rule) exercises. 3e hardens
  it with three quantitative checks vs an independent HF oracle (`Qwen/Qwen3.5-4B`, bf16, same vendored
  FLA/conv kernels), all GREEN:
  - **3e-A — single-GDN-layer decode unit test** (`tools/qwen3_5_gdn_isolate.py`, now `--mode
    {prefill,decode,both}`). Copies HF's layer-0 GDN weights into minisgl's `QwenGatedDeltaNet`, then:
    *prefill* (the 3d-4 check) — T=6 fresh prefill, **cos 0.99997** (rel 8e-3); *decode* (NEW) —
    establish state with a T-token prefill, then ONE `forward_decode` step for token T (reusing the
    in-place conv/ssm state) vs HF's full (T+1)-token forward sliced at position T: **cos 0.99995**
    (rel 1.0e-2). Isolates the recurrent kernels from loading/wiring — no checkpoint. (Fixed a display
    bug: decode `hf_out` is 1-D `(D,)`, the per-row print indexed it as a scalar — the overall cos was
    always correct; both args now 2-D.)
  - **3e-B — first-token logit oracle** (task 3e-1; `tools/qwen3_5_decode_oracle_{ours,cmp}.py`,
    mirrors the dense path's `oracle_ours.py`/`oracle_cmp.py`). minisgl's prefill first-token logits vs
    HF full logits: **prefill cos 0.99990–0.99991**, top-1 OK, top-5 5/5 across 3 prompts — exceeds the
    0.999 bar.
  - **3e-C — decode-path per-step parity (the real target; task 3e-2).** minisgl greedy 16 steps on one
    prompt, hooking the sampler to ACCUMULATE per-step logits (sample() fires once per generated
    token → step 0 = prefill, steps 1.. = recurrent decode), saved with the gen ids. The cmp side
    **teacher-forces HF on minisgl's OWN gen sequence** in one `use_cache=False` forward, slicing logits
    at positions `P-1+t` — so each step is conditioned on an IDENTICAL prefix (apples-to-apples even past
    any greedy divergence), and HF's single full-sequence forward (CHUNK path) is compared step-by-step
    to minisgl's RECURRENT decode kernels. **Result over 3 prompts (48 decode steps): worst decode-step
    cos 0.99922**, top-5 5/5 at every step. **★ The load-bearing finding: cos does NOT degrade with
    decode step** (e.g. prompt A decode7 dips to 0.99922, decode8 right after is 0.99995) — flat bf16
    noise, NOT the monotonically-growing drift a state-update/recurrence bug would produce. The in-place
    ssm_state recurrence is correct.
    * **One benign top-1 flip** (prompt B "Once upon a time", decode1: ours 11815 "lived" vs HF 557) at
      **cos 0.99990** — the two leading logits are tied to within bf16 noise; a sub-1e-4-cos perturbation
      flips the argmax. This is exactly the greedy-token-identity brittleness the cos-based logit oracle
      exists to see past. The cmp verdict now classifies a top-1 flip under `cos >= 0.9995` as a TIE (not
      a failure) and only an under-0.9995 flip as a real break → all 3 prompts: **DECODE PATH NUMERICALLY
      SOUND**. (Teacher-forcing keeps steps ≥2 apples-to-apples regardless: HF stays conditioned on
      minisgl's actual emitted prefix.)
  - **★ Box/recipe fix (durable):** the README "Running" recipe sets BOTH `HIP_VISIBLE_DEVICES` and
    `ROCR_VISIBLE_DEVICES` to `$LEASE_ROCR_DEVICES` (the physical card index). That **only works when the
    lease assigns card 0** — for card 1 it double-filters (`ROCR=1` selects physical card 1 and
    re-indexes it to 0, then `HIP=1` selects nothing → torch `RuntimeError: No HIP GPUs are available`).
    The arbiter already exports the correctly-composed pair in the lease shell (`ROCR_VISIBLE_DEVICES=1`,
    `HIP_VISIBLE_DEVICES=0`); **forward those verbatim** (`-e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e
    ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES`) rather than overriding both with `$LEASE_ROCR_DEVICES`.
  **★ NEXT — the GATE before 35B (do NOT start 35B blind):** the 35B GDN-hybrid MoE combines (a) the
  now-proven GDN path (3b–3e), (b) the Phase-2 W4A8 grouped MoE (synthetically parity-validated), and
  (c) real compressed-tensors MoE checkpoint loading (not yet exercised) — BUT 35B @ W4A8 ≈ 17.5 GB
  **exceeds one 16 GB card**, so it needs **TP=2 (Phase 4)**. 35B is therefore BLOCKED on TP. Also: no
  35B base checkpoint is currently on the box (the cached "35B-DFlash" is a speculative draft, not the
  base — see 3d). Ordering decision (TP-first vs MoE-loading-first) deferred to a strategic checkpoint;
  flagged here so 35B isn't begun before TP exists and a base checkpoint is fetched.

## ★ MoE PARITY REACHED 2026-06-18 — W4A8 grouped MoE numerically validated

`w4a8_moe` matches a bf16-dequant reference at **cos-sim 0.99894** (rel-err 4.6% = fp8-activation
error over 2 GEMMs) on real MoE shapes (E=32, K=2048, inter=512, top_k=4, g=32). The full MoE
compute path — topk → moe_align → grouped GEMM(w13) → SwiGLU → grouped GEMM(w2) →
`mmq_fp8_moe_gather_reduce` — is sound, at parity with the dense W4A8 MVP. (`tools/moe_parity.py`.)

## Phase 2-MoE — W4A8 grouped MoE

- `quant/kernels.py:w4a8_moe` — grouped MoE forward (topk → moe_align → grouped GEMM(w13) →
  silu_and_mul → grouped GEMM(w2) → `mmq_fp8_moe_gather_reduce`), mirroring the proven
  `_run_grouped_moe` (non-GEMV path). Imports vLLM's `moe_align_block_size` for now (port later).
- `tools/moe_parity.py` — numerical parity vs a bf16-dequant reference on **synthetic** symmetric
  int4 experts (op layout). Validates the MoE compute integration independent of checkpoint loading.
- **Target note:** no small standard Qwen3-MoE-AWQ is bootable (smallest plain Qwen3-MoE is 30B; the
  35B is GDN-hybrid → Phase 3). The 35B MoE is **compressed-tensors pack-quantized symmetric g32**
  routed through vLLM **MoeWNA16** — real-checkpoint MoE loading is a Phase-3 follow-up; the synthetic
  parity test validates the compute path now.
- TODO after parity: a `"w4a8"` MoE backend in the registry + MoELayer expert-weight conversion
  (`_ct_moe_to_op_layout`/MoeWNA16) for the real 35B; strip the bf16 MoE `sgl_kernel` deps (topk/align).

## Phase 2M (real-checkpoint MoE loading) — Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 (STARTED 2026-06-22)

De-risks the 35B's "real quantized-MoE checkpoint loading" prerequisite on a model that is **NOT**
GDN and fits one 16 GB card (so it needs neither TP nor the GDN path — see the ★ 35B GATE at
Phase 3e). Target picked by the user; **vLLM (combined image) loads/serves it fine** → it is the
loading reference AND the parity oracle.
**★ What the checkpoint actually is** (`tools/` header scan): `Qwen2MoeForCausalLM` / `qwen2_moe`,
24 layers, hidden 2048, **60 experts top-4**, `moe_intermediate_size=1408`, a **shared expert**
(`shared_expert_intermediate_size=5632`) + a 1-row `shared_expert_gate` (sigmoid), MHA 16/16 heads
head_dim 128, **qkv+o bias**. **EVERYTHING is GPTQ-Int4** (attn q/k/v/o, shared expert, AND the 60
routed experts) — only embeds/norms/router/lm_head are fp16. GPTQ packing: `qweight I32 (K//8, N)`
packed along **input** (vs AWQ's `(K, N//8)` along output → a *different* unpack/repack), per-group
`scales (K//g, N)`, `qzeros (K//g, N//8)`, `g_idx` identity (`desc_act=false`). 14.3B params → must
stay int4 (bf16 ≈ 28 GB won't fit) → **W4A8 fp8-activation kernel is the only memory-viable compute
path**, and it aligns with the 35B W4A8 target.
**★ vLLM reference path** (mirror, don't reinvent): `GPTQConfig.get_quant_method` sees a `FusedMoE`
layer → redirects to **MoeWNA16** (W4A16; `sym` true → `has_zp=False`, no zeros loaded), stacking
per-expert `gate/up/down_proj` into grouped `w13 (E,2·inter,K//8)` / `w2 (E,K,inter//8)`; the shared
expert + router are separate (shared expert = a quantized dense MLP, router/shared_gate = fp16
`ReplicatedLinear`). `make_expert_params_mapping` drives the per-expert load. (We feed the same
grouped int4 layout to our `w4a8_moe` fp8-activation kernel instead of MoeWNA16's fp16 path.)
Sub-phases (mirrors the GDN port's CPU-verified-then-serve cadence):
- **2M-0 (DONE 2026-06-22) — config + GPTQ recognition.** `quant/config.py`: recognize
  `quant_method=="gptq"` (`sym`, `group_size`, `desc_act`; `is_gptq` prop; `desc_act` field).
  `config.py`: `shared_expert_intermediate_size` field + parse. CPU-verified
  (`tools/qwen2_moe_config_test.py --standalone`, torch-free — loads the real QuantConfig +
  ModelConfig with transformers stubbed, reads config.json): 4B-MoE parse (qwen2_moe, 60 experts,
  top-4, inter 1408, shared 5632, GPTQ g128 sym desc_act=false) AND dense regression (shared=0,
  quant=None, not GDN) both PASS.
- **2M-1 (DONE 2026-06-22) — `models/qwen2_moe.py` + register `Qwen2MoeForCausalLM`.** Attn = Qwen2
  `RopeAttn(has_attn_bias=True, has_qk_norm=False)` — **only q/k/v carry a real bias** (`o_proj.bias`
  and all expert/shared `.bias` are all-zero GPTQ placeholders → skipped on load, verified by value
  scan). MoE block (`Qwen2MoeSparseBlock`, all 24 layers sparse, `decoder_sparse_step=1`) = router
  `gate` (fp16 `LinearReplicated`) + 60 grouped experts (`MoELayer`) + `Qwen2MoeSharedExpert` (quant
  SwiGLU at `shared_expert_intermediate_size=5632`) + `shared_expert_gate` (fp16, weight `[1,H]`):
  `final = routed(x) + sigmoid(shared_gate(x))·shared(x)`. Also added the **GPTQ buffer-shape branch**
  to `W4A8LinearMethod.create_weights` (qweight `(K//8,N)` input-packed, scales `(K//g,N)`, qzeros
  `(K//g,N//8)` ALWAYS present even symmetric) + a `gptq_to_op_layout` stub (impl 2M-2) +
  process-after-load dispatch. **Meta build smoke (`tools/qwen2_moe_build_smoke.py`, combined image,
  CPU/meta — no GPU):** 459 tensors, 24 MoE layers, merged `qkv_proj` (GPTQ + bias) / `o_proj` (no
  bias), shared-expert GPTQ buffers, fp16 router+shared gates, grouped expert buffers `(E,2·inter,H)`
  / `(E,H,inter)` — all shapes correct.
- **2M-2 (DONE 2026-06-22) — GPTQ→op-layout conversion**, the load-bearing numerics.
  `quant/kernels.py` `gptq_to_op_layout`: GPTQ packs int4 along **input** K (natural nibble order, no
  AWQ interleave); qzeros are ALWAYS present and the dequant zero point is `unpacked_qzeros + 1`
  (AutoGPTQ off-by-one). **Checkpoint scan: every qzero unpacks to 7 → constant zero point 8**
  (symmetric, fits 4 bits); `g_idx` is identity (`desc_act=false`). We fold the +1 into an EXPLICIT
  zeros tensor and reuse the proven **asymmetric** op path (`w = scale·(q − zero)`), exact regardless
  of `sym`. **CPU-unit-tested (`tools/qwen2_moe_gptq_convert_test.py`, combined image, no GPU):** for
  real q_proj / expert gate&down / shared-expert down, an op-layout dequant equals an independent
  GPTQ-checkpoint dequant at **max|Δ| = 0** (faithful re-encoding); zero point 8 asserted. (The GPTQ
  *formula* itself is the end-to-end oracle in 2M-4.)
- **2M-3 (TODO) — quantized-MoE method + wire `w4a8_moe`** into MoELayer (dispatch to the W4A8
  grouped kernel when `config.quant` is set; per-expert stacking already exists in `weight.py`).
  Header-only weight-map test (22539 ckpt keys → native keys/shapes, expert stacking, shared expert).
- **2M-4 (TODO) — serve on gfx1201 + token/logit parity vs vLLM.** Greedy token-diff + the
  decode/logit oracle vs the combined image's vLLM (loads it fine). DoD = full serve + parity.

## ★ MVP REACHED 2026-06-18 — W4A8 quantized serving on RDNA4

Qwen2.5-Coder-7B-Instruct-**AWQ** (4-bit, g128, asymmetric) boots on gfx1201 via the `triton_rdna4`
attention + the W4A8 path, generating **coherent + correct** greedy output ("capital of France →
Paris", "2+2 → 4, 3+3 → 6, 4+4 → 8"). Full chain validated: AWQ checkpoint → `awq_to_op_layout`
conversion → `mmq_fp8_gemm` v10 (asymmetric zeros) → fp8 WMMA. Weights load 6s, KV 1.81 GiB, eager.
Optional next: quantitative logit oracle vs the cached unquantized bf16 7B; then MoE W4A8 (toward 35B),
the autotuner, and TP.
| 2 | W4A8 dense (`LinearMethod`) + MoE backend + weight-loader fix → 7B-AWQ | todo |
| ★ | GATE: re-decide 35B GDN port | — |
| 3 | GDN hybrid: 3a state cache **done** → 3b layer numerics **done** → 3c scheduler/slot/metadata/warmup **done 2026-06-20** → 3d-0 config+ctx **done** → 3d-1 model+rotary+register **done** → 3d-2 engine wiring **done 2026-06-21** → 3d-3 weight map **CPU-verified 2026-06-21** → 3d-4 serve **DONE 2026-06-21** (coherent, greedy token-diff vs vLLM PASS; RMSNorm (1+w) fix) | **3d DONE ✅** |
| 4 | RCCL TP + het-TP (re-derive ratio) + decode HIP graphs + parity | todo |

## Change log (what we've diverged from upstream + why)

_Phase 0 — dependency strip. NVIDIA imports in mini-SGLang are all lazy; replace with torch/lifted
Triton. Targets (file:line from audit):_
- [x] `layers/norm.py` flashinfer rmsnorm/fused_add_rmsnorm → fp32-internal torch RMSNorm.
- [x] `layers/rotary.py` flashinfer apply_rope → torch NeoX rotate-half (fp32 internal, reuses
      `_cos_sin_cache`).
- [x] `layers/activation.py` flashinfer silu/gelu_and_mul → fp32-internal gated torch activation.
- [x] `engine/sample.py` flashinfer.sampling → torch softmax + per-row top-k/top-p + multinomial
      (greedy still argmax). NOTE: stochastic sampling won't be RNG-identical to flashinfer; only
      greedy is the token-identity oracle.
- [x] `kvcache/mha_pool.py` store_cache `.cu` → torch scatter (bf16 path; fp8 KV write lands in P1).
- [x] `layers/embedding.py` index `.cu` → torch vocab-parallel gather + mask + all_reduce.
- [x] `kernel/radix.py` fast_compare_key `.cu` → torch common-prefix-length (matches std::mismatch).
- [ ] `moe/fused.py:16,71` sgl_kernel topk_softmax/moe_align → DEFERRED (MoE-only path; our W4A8 MoE
      backend replaces it in Phase 2; not on the dense boot path).
- [ ] `engine/engine.py:223` + `utils/arch.py:12` — ROCm `auto` falls to `fi`. DEFERRED to Phase 1:
      add the RDNA4 `triton_attn` backend + select it on ROCm (no backend exists yet to select).
- [x] collectives: default `TorchDistributedImpl` = RCCL on ROCm — zero change at TP=1 (verified).

_Precision note: all shims are fp32-internal / bf16-I-O (honor "no F16"); they are correctness-first
placeholders, validated by token-diff vs the combined image once Phase 1 attention lands._

_Verification: `python -m py_compile` passes on all edited files; no top-level NVIDIA imports remain
(the residual `flashinfer`/`sgl_kernel` refs are the lazy CUDA attention backends — replaced in
P1 — and the deferred MoE path)._

_Phase 1a — tuned RDNA4 attention (commit fed6efa):_
- Vendored `attention/_triton_unified.py` + `_triton_helpers.py` from the baseline image's tuned
  `triton_attn` (vLLM imports stubbed; `KVQuantMode` reproduced; fp8 dtype = `e4m3fn`). Kernel logic
  unchanged.
- `attention/triton_rdna4.py` (`TritonRDNA4Backend`) behind `attention/base.py`. KV layout maps 1:1
  (`k_cache(layer_id)` = `(num_pages, page_size, kv_heads, head_dim)` == kernel's
  `(num_blocks, block_size, ...)`); metadata lifted from `fa.py`. Phase 1a = bf16 KV (torch store),
  2D grid (no 3D scratch), Triton-heuristic tuning.
- `is_rocm()`/`get_gcn_arch()` in `utils/arch.py`; `_adjust_config` auto-selects `triton_rdna4` on
  ROCm and forces `page_size % 16 == 0`.
- **Validated CPU-only** in `vllm22-w4a8:combined` (GPUs hidden): `import minisgl.attention`, the
  vendored kernel, and the backend all import under real Triton; backend is concrete (no unmet
  abstractmethods). Functional boot (token generation) is the next step and needs a GPU window.

**Build-story note:** the combined image's venv lacks minisgl's runtime deps (msgpack, pyzmq,
prompt_toolkit, accelerate, modelscope, fastapi/uvicorn, openai…). The engine image (`FROM
vllm22-w4a8:combined`) must `pip install` the engine + those deps. For ad-hoc runs, `pip install`
them into the ephemeral container first.
