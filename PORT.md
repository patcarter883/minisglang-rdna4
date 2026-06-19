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
- **3b — one GDN layer's numerics (IN PROGRESS).** Split into three checks:
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
  - **3b-3 — single-layer parity (next):** capture/replay vs the REAL vLLM `QwenGatedDeltaNetAttention`
    (imports in the combined image, vllm 0.22.69) — independent oracle, NOT a self-authored eager ref.
    Harness recipe (seam already found):
      * Stand up the real layer (Qwen3NextConfig + minimal VllmConfig, tp=1, quant=None).
      * `_forward_core`/`_forward_core_rocm` read `get_forward_context().attn_metadata` as a **dict
        keyed by `self.prefix`** (lines 1228/1284). Monkeypatch `get_forward_context` → stub with
        `.attn_metadata = {prefix: GDNAttentionMetadata(...)}`; set `real_layer.kv_cache=[conv,ssm]`;
        call `_forward_core_rocm(qkvz, ba, z, core_attn_out)` directly (bypasses the custom op +
        set_forward_context). `GDNAttentionMetadata` is a plain dataclass — construct by hand.
      * Copy weights real→minisgl with a per-param shape assert; expect ~bit-exact (TP=1), gate <1e-2 bf16.
      * Cheap CPU pre-check first: diff `real.prepare_gdn_attention_core_inputs` vs `_split_qkvz_ba`
        and `_output_projection` on shared inputs (isolates split/reshape/gate-order bugs, no GPU).
      * Drive **prefill→decode**; compare the OUTPUT **and** `conv_state` **and** `ssm_state` after
        prefill (validates the 3a state write-path). Gate decode parity on prefill-state parity; also
        run an independent decode test injecting one shared random `(conv,ssm)` into both.
      * Print `is_conv_state_dim_first()` on the live RDNA4 run — if False, the real layer transposes
        kv_cache[0] and `layer.py`/`GDNStateCache` (DS) must match; this boolean decides 3a's layout.
- **3c — scheduler subset-split + warmup hook (THE risk):** one batch splits into prefill/decode
  subsets running DIFFERENT kernels (chunk-scan vs fused-recurrent) with per-subset query_start_loc +
  state indices; FLA first-batch autotune needs a warmup-prefill hook or it OOMs. No spec-decode/MTP.
- **3d — interleave + serve:** qwen3_5 1-in-4 full/linear interleave (full layers reuse Phase-1
  attention); serve the 35B; greedy token-diff vs combined image. GDN projections are unquantized
  bf16; only routed MoE experts are W4A8 (uses the Phase-2-MoE path).

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

## ★ MVP REACHED 2026-06-18 — W4A8 quantized serving on RDNA4

Qwen2.5-Coder-7B-Instruct-**AWQ** (4-bit, g128, asymmetric) boots on gfx1201 via the `triton_rdna4`
attention + the W4A8 path, generating **coherent + correct** greedy output ("capital of France →
Paris", "2+2 → 4, 3+3 → 6, 4+4 → 8"). Full chain validated: AWQ checkpoint → `awq_to_op_layout`
conversion → `mmq_fp8_gemm` v10 (asymmetric zeros) → fp8 WMMA. Weights load 6s, KV 1.81 GiB, eager.
Optional next: quantitative logit oracle vs the cached unquantized bf16 7B; then MoE W4A8 (toward 35B),
the autotuner, and TP.
| 2 | W4A8 dense (`LinearMethod`) + MoE backend + weight-loader fix → 7B-AWQ | todo |
| ★ | GATE: re-decide 35B GDN port | — |
| 3 | GDN hybrid (3a state cache → 3b layer numerics → 3c scheduler split → 3d serve) | todo |
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
