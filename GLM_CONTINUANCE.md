# Continuance — GLM-4.7-Flash (glm4_moe_lite) GPU validation + production-readiness

## PROGRESS LOG (2026-06-27, session 2)
- **Task 1 DONE** — `mla_hip` rebuilt for this checkout; all serve-path HIP pkgs import OK
  (mla_hip/moe_hip/tail_hip/swiglu_hip/attn_hip/attn_decode/attn_prefill_paged/gdn_hip).
- **Task 2 DONE** — added the **GLM qk256/v256 prefill** template instantiation
  (`mla_prefill_kernels.hip`: BR/BC are now template params; v256 doubles the float O/PV LDS so
  BR=16,BC=16 fits 64 KB) + extended `mla_hip_parity.py`. **Parity ALL PASS** (decode 512/64, fp8,
  prefill 192/128, and the new prefill 256/256 — cos≥0.99999, 0 ULP violations).
- **Task 3 DONE (loader contract; CPU)** — checkpoint = **`QuantTrio/GLM-4.7-Flash-AWQ`** (true
  autoawq gemm, g128, zero_point — matches the kernel; cyankiwi is compressed-tensors and does NOT).
  Downloaded (~19 GB) to the HF cache. QuantTrio quantizes the **shared expert too** (its
  modules_to_not_convert keeps only attn+gate+layers.0 bf16), so `glm4_moe_lite.py` was changed to
  **quantize the shared expert when expert_quant is present** (relaxes commit 7b113b9; bf16 ckpt
  still keeps it bf16). New `tools/glm4_moe_lite_build_smoke.py`: meta-build + simulate the real
  loader over all 28033 ckpt keys → **ALL PASS** (1072 runtime keys exactly cover the model
  state_dict; 602 MTP tensors skipped; shared+routed experts AWQ-stacked, attn/gate/dense0/lm_head bf16).
- **31.2B params → AWQ ≈19 GB served > a single 16 GB card.** TP=1 single-card serve is INFEASIBLE
  (brief's "~9 GB" was wrong). So TP=2 was promoted from follow-up to prerequisite (user-confirmed).

- **TP=2 MLA sharding DONE + GPU-VALIDATED (serves, coherent).** Changes (working tree, uncommitted):
  - `glm4_moe_lite.py` GLMMLAAttention: dropped the `tp_size==1` assert; q_b_proj/kv_b_proj are now
    **head-parallel** (`LinearColParallelMerged`, num_qo_heads 20→10/rank), o_proj **row-parallel**
    (`LinearOProj`, all-reduce). q_a/kv_a bottlenecks + the MQA latent + the latent KV cache stay
    **replicated** (shared across heads). `self.num_heads = div_even(num_qo_heads, tp)`.
  - `weight.py`: AWQ-aware `_shard_tensor` — AWQ/GPTQ pack output on axis 1, so a col-parallel AWQ
    tensor shards axis 1 and a row-parallel one axis 0 (flipped vs bf16); added `.q_b_proj`/
    `.kv_b_proj` to the col-parallel set; **`.shared_experts.` tensors are replicated** (not sharded).
  - `glm4_moe_lite.py` GLMSharedExpert + `linear.py`: the shared expert is **REPLICATED** (full
    output added to the all-reduced routed output, no double-count). Why: a row-parallel shared
    down_proj has K=moe_intermediate/tp=768, but the W4A8 dense kernel needs **K%512==0** (1536 only
    un-sharded → `RuntimeError: v11 needs K % 512 == 0; got 768`). `LinearReplicated` gained a
    `quant_method` arg for this.
  - `tools/glm4_moe_lite_build_smoke.py` (new): meta-build + loader-contract (TP=1) AND TP=2 shape +
    real-tensor AWQ-axis-flip checks — `MINISGL_SMOKE_TP={1,2}`, both **ALL PASS** on CPU.
  - GPU: `QuantTrio/GLM-4.7-Flash-AWQ` boots **TP=2 eager (mla, page16)**, fits (≈13.5 GB weights +
    3.9 GB KV / card), and `tools/glm_coherence_smoke.sh` (new) → **"The capital of France is Paris."**
    + on-topic responses. Correct facts ⇒ W_UK/W_UV absorption + NeoX RoPE + MoE routing are sound.
    (Pure-greedy reasoning-model looping/`<think>` scaffold is expected, not a bug.)

- **TP=2 MLA sharding committed** at `1baf6e5`.

- **MLA cudagraph capture DONE + GPU-VALIDATED** (`attention/mla.py`): implemented
  `init_capture_graph`/`prepare_for_capture`/`prepare_for_replay` + `_decode_metadata_static`/
  `_fill_decode_static`, mirroring `HIPAttnBackend` (decode-only; static int32 `cache_seqlens` +
  `page_table`; latent store + decode run inside the graph; cu_seqlens are a placeholder arange).
  MLA is simpler than GDN — no recurrent state to thread. Validated: `GRAPH=8 MOE_SCATTER=0`
  captures bs [1,2,4,8] cleanly and the replayed graphs produce identical coherent output
  ("...Paris."). MoE-decode MUST use the graph-safe gather_reduce path (`MINISGL_MOE_SCATTER=0`).
  `tools/glm_coherence_smoke.sh` now takes `GRAPH`/`MOE_SCATTER` envs.

- **MLA cudagraph capture committed** at `c5035bb`.

- **BENCHMARK DONE — DOD COMPLETE.** Bench harness gained an `ATTN` env (`_bench_inner.sh` /
  `run_bench_window.sh`): GLM runs with `ATTN=auto MEMRATIO=0.85 TP=2 MOE_SCATTER=0` (engine forces
  `mla`). Ran eager (GRAPH=0) vs production graph (GRAPH=16) on a 2-card lease, **0 failures** all
  cells. **Decode TPOT (ms) / tok/s, M=1→16:**
    eager:  37.0/26.8 · 38.2/51.7 · 38.2/103 · 44.9/176 · 47.2/332
    graph:  22.9/43.2 · 24.5/80.4 · 37.5/106 · 42.1/187 · 44.4/352
  Graph capture = **1.61× at M=1** (37.0→22.9 ms; launch-overhead-bound), narrowing to ~1.05× at
  M≥4 (compute-bound). GLM graph M=1 = 22.9 ms / 43.2 tok/s ≈ the 35B-A3B baseline (20.4 ms /
  48.6 tok/s). Mixed M=1 TPOT 36.5→24.3. Prefill ~80–110 ms TTFT (not graphed).

  **All DOD items met:** mla parity qk256/v256 ✓ · AWQ ckpt loads ✓ · serve coheres ✓ · MLA graph
  capture ✓ · TP=2 ✓ · benchmarked (prefill/decode/mixed × M, graph) ✓.
  Optional leftover: logit oracle vs bf16 HF (needs CPU/offload — 59 GB ref doesn't fit a card).



## Mission
GLM-4.7-Flash (`Glm4MoeLiteForCausalLM`, `model_type=glm4_moe_lite`) serving was implemented and
committed **GPU-UNVALIDATED** at `56d8a29` on branch `rdna4`. CPU structure/naming/config checks
passed; nothing has run on a GPU. **Validate it on hardware, then lift the committed limits toward a
real production config (graph-capturable, TP=2).** AMD gfx1201/RDNA4, Triton-free native HIP.

## READ FIRST (mandatory)
- `CLAUDE.md` (this repo) + `/home/pat/code/vllm-gfx1201/CLAUDE.md` — **GPU LEASE protocol** (all GPU
  work via the absolute-path arbiter `/home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh`; `-n 1` single
  card, `-n 2` both) + the container-run conventions (mount source into `vllm22-w4a8:combined`,
  forward the lease's `HIP/ROCR_VISIBLE_DEVICES`).
- Auto-memory: `glm47-flash-bringup`, `mla-hip-kernel-status`, `production-serve-config-and-bench`,
  `triton-free-serve-hip-packages`, `gpu-lease-visible-devices-recipe`, `gdn-hip-serve-pythonpath`.
- The code: `python/minisgl/models/glm4_moe_lite.py`, `attention/mla.py` (MLABackend),
  `kvcache/mla_pool.py` (MLAKVCache, 576/token), `mla_hip/` (decode kernels), and the
  precomputed-`topk_weights`/`topk_ids` seam in `quant/kernels.py:w4a8_moe` (noaux_tc routing).

## Architecture (real checkpoint)
47 layers; **MLA attention** (kv_lora 512 / q_lora 768 / qk_nope 192 + qk_rope 64 = **qk 256**,
**v_head_dim 256**); 64 routed experts top-4 + 1 always-on shared expert; layer-0 dense;
`tie_word_embeddings=false`; MTP layer (num_nextn_predict_layers) skipped. Only the MoE/MLP linears
are AWQ-quantized — **MLA attention, router gate, and shared expert stay bf16** (standard
`modules_to_not_convert`). Engine auto-forces the `mla` attention backend + `page_size 16` when
`is_mla` (do NOT pass `--attn hip` for GLM — that overrides the MLA backend).

## Limits AS COMMITTED (the starting point — these are what to validate, then lift)
- **TP=1 only** (`GLMMLAAttention.__init__` asserts `tp_size==1`; MLA sharding is a follow-up).
- **AWQ-only** — the bf16 (non-AWQ) MoE path is NOT wired (fused-MoE has no precomputed-topk path),
  so the cached bf16 `zai-org/GLM-4.7-Flash` will NOT serve as-is.
- **`--cuda-graph-max-bs 0` (eager)** — `MLABackend.init_capture_graph` is a `NotImplementedError`
  stub (`attention/mla.py`), so MLA cannot be graph-captured yet.

## TASKS (validate first, in order)
1. **AOT-build the vendored `.so` for this checkout.** They are gitignored → rebuilt per checkout.
   At minimum `mla_hip` (GLM decode) + the serve-path pkgs (`moe_hip`, `attn_*`, `tail_hip`,
   `gdn_hip` — `import` check each). CPU-only, NO lease:
   `docker run --rm -v "$PWD":/engine --entrypoint bash vllm22-w4a8:combined -lc 'source
   /app/.venv/bin/activate && cd /engine/mla_hip && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace'`
2. **Extend `mla_hip/mla_hip_parity.py` to GLM's qk256 / v256 / latent512** — only qk192/v128 is
   tested today, and GLM runs the untested 256/256 shapes. Get parity green BEFORE trusting serve
   output. (1-card lease, in image.)
3. **Source/produce an AWQ-INT4 GLM-4.7-Flash checkpoint with attention left unquantized.**
   `cyankiwi/GLM-4.7-Flash-AWQ-4bit` is NOT in the HF cache (only bf16 `zai-org/GLM-4.7-Flash` is).
   Either download a ready AWQ-INT4 (drop `HF_HUB_OFFLINE` for a deliberate `snapshot_download`) or
   quantize the bf16 → AWQ g128 keeping `modules_to_not_convert` = the MLA attn + gate + shared
   expert. Verify the weight-name contract matches `glm4_moe_lite.py`'s loader.
4. **Smoke serve + logit-compare vs HF transformers** (a few tokens, greedy). Tools:
   `tools/oracle_ours.py` + `tools/oracle_cmp.py` (`MINISGL_ORACLE_MODEL`/`MINISGL_ORACLE_REF`), or
   `tools/boot_smoke.py` for a coherence pass first. **Watch two known risk spots:** the RoPE
   convention (NeoX `rotate_half`) and the index order of the **W_UK/W_UV absorption einsums**
   (`einsum("thn,hnl->thl", q_nope, w_uk)`) — these are the most likely sources of silent wrongness.

   Serve command (TP=1, eager, AWQ): under a 1-card lease, in the image —
   `python -m minisgl --model <AWQ-glm-path> --tensor-parallel-size 1 --cuda-graph-max-bs 0`
   (let the engine auto-pick the `mla` backend; do NOT pass `--attn hip`). Probe OpenAI
   `/v1/chat/completions` on the chosen port.

## THEN lift the limits (production-readiness — the actual goal)
- **MLA graph capture:** implement `MLABackend.init_capture_graph` (+ a decode-graph replay path
  that threads the MLA latent-cache slots through static buffers, mirroring `GDNGraphCapture`). This
  is the big one — **graph capture was ~20% faster TPOT than eager** on the 35B this session
  (`production-serve-config-and-bench`), so eager GLM is leaving ~20% on the floor. Once captured,
  production GLM = `--graph N` + `MINISGL_MOE_SCATTER=0` (the MoE-decode scatter atomicAdd is not
  graph-capturable — use the graph-safe gather_reduce path under capture).
- **TP=2 MLA sharding:** remove the `tp_size==1` assert; shard the MLA projections (head-parallel
  q/o; the latent kv cache is shared/replicated) + the latent pool. The 35B serves TP=2 today; GLM
  should too rather than being single-card-only.
- (Optional) wire the bf16 (non-AWQ) MoE path so `zai-org/GLM-4.7-Flash` serves without quantizing.

## Benchmark once validated
Reuse `tools/run_bench_window.sh` + `tools/_bench_inner.sh` + `tools/serve_matrix_bench.py`
(prefill/decode/mixed × M∈{1,2,4,8,16}, TTFT/TPOT/throughput). **The harness currently hardcodes
`--attention-backend hip` and `--graph 16` — add a GLM mode** that drops `--attn hip` (let the
engine force `mla`) and sets `GRAPH=0` until MLA capture lands. Run in the BACKGROUND
(`run_in_background`) + tail the log — boot is silent for a while (pip deps + load + capture).
Compare against the production baselines already captured: 35B-A3B decode M=1 20.4 ms / 48.6 tok/s,
4B GDN 20.7 ms / 48 tok/s.

## Gotchas (cost real leases if ignored)
- Servers spawn a scheduler/worker process tree — kill the whole **process group** (`setsid` launch
  + `kill -- -$PGID`), or the next boot hangs on a held GPU/port (already handled in `_bench_inner.sh`).
- Measure production (`--graph`), never eager — but GLM is eager-only until task "MLA graph capture".
- `mla_hip` (and the other HIP pkgs) import from the repo ROOT → `PYTHONPATH=/engine/python:/engine`.
- Separately: an RXF (W4-NL/A8) quant of GLM-4.7-Flash is deferred — `/tmp/quant_glm47_rxf.sh`
  (lease `-n 2`); distinct from the AWQ path this brief targets.

## DEFINITION OF DONE
mla_hip parity green at qk256/v256; an AWQ-INT4 GLM checkpoint loads; smoke serve coheres and
logit-matches HF (cos-sim / top-1) within bf16-rounding; then MLA graph capture works and GLM serves
TP=2 — benchmarked (prefill/decode/mixed × M) in the production config (graph capture, not eager).
