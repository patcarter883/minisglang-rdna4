# PERF_NOTES — optimization backlog (revisit once the ecosystem works)

These are performance opportunities deliberately deferred while building for **proof-of-concept +
correctness**. Each shim/path below is correct-but-not-optimal on purpose. Do NOT optimize these
until a working end-to-end ecosystem (dense + W4A8 + GDN serving) exists and we have a logit-level
oracle to keep correctness while tuning. Ordered roughly by expected payoff.

## Validation tooling (do FIRST — unblocks safe optimization)
- **[V1] Build a logit-level oracle.** Greedy token-diff amplifies sub-ULP rounding into full
  divergence at near-ties, so it CANNOT distinguish "correct rounding" from a bug (proven: 3D
  flash-decode converges toward 2D as segments→1). Add a single-forward logit dump from the engine
  + vLLM/HF and compare cosine-sim / top-1 rate / max-abs-diff. Every perf change below should be
  gated on "logit cos-sim unchanged," not token-identity.

## Attention backend (`attention/triton_rdna4.py`)
- **[A1] 3D flash-decode segment count is fixed at 64.** Over-segmentation on short sequences wastes
  work AND adds recombination rounding (observed: more divergence from 2D at seg=64 vs seg=1). The
  startup autotuner (Phase 1b-2) must right-size `num_par_softmax_segments` by batch × KV length,
  and set the RDNA4-tuned launch knobs (`waves_per_eu`, `num_warps`, `num_stages`,
  `tile_size_decode`) — currently all default to None (Triton heuristics). **This is where the
  tuned kernel's actual RDNA4 advantage lives** (the offline gfx1201 configs show 3D + tuned
  `waves_per_eu` is the win). Highest-payoff item here.
- **[A2] Output buffer `torch.empty_like(q)` every forward.** Reuse a persistent output buffer
  (also needed for cudagraph pointer stability in Phase 4).
- **[A3] `prepare_metadata` rebuilds the page table via a Python list-comprehension + `torch.stack`
  every step** (per-req CPU slicing → GPU). Vectorize the page-table gather; avoid per-step
  host-side Python loops over reqs.
- **[A4] fp8-KV store is a torch scatter** (`kvcache/mha_pool.py:store_kv`, `.to(e4m3fn)`); replace
  with the fused `reshape_and_cache_flash` Triton kernel for less store overhead + bandwidth.
- **[A5] fp8-KV uses a static per-tensor scale of 1.0** (direct e4m3 cast). This cost ~0.006 cos-sim
  vs HF (0.9996→0.9935, top-1 still all-match). A calibrated/dynamic per-tensor (or per-token-head,
  kv_quant_mode 2/3) scale would recover most of that. Add a calibration pass when accuracy matters.

## Phase-0 torch shims (all fp32-internal, correctness-first)
All of these upcast to fp32 and materialize intermediates for numerical safety. Once the logit
oracle exists, replace with fused Triton kernels (or lift vLLM's) and verify cos-sim unchanged.
- **[S1] `layers/norm.py` RMSNorm** — `.float()` upcast + full materialization every call (incl.
  per-head q/k norm). A fused Triton RMSNorm avoids the fp32 round-trip memory traffic. Hot path
  (2× per layer).
- **[S2] `layers/rotary.py` RoPE** — fp32 upcast, allocates new tensors (not in-place), re-gathers
  cos/sin and `torch.cat`-repeats each call. Fuse into a Triton RoPE (or lift vLLM's
  `apply_rope`), do it in-place.
- **[S3] `layers/activation.py` silu/gelu_and_mul** — fp32 upcast + intermediate materialization.
  Fuse SwiGLU into one Triton kernel (gate·up·act in a single pass); avoids the extra MLP-width
  memory traffic. Hot path.
- **[S4] `engine/sample.py`** — full-vocab `torch.sort` per row for top-k/top-p. Fine at low batch;
  a fused sampling kernel (flashinfer-equivalent) helps at high batch. Greedy path (argmax) is
  already fine.
- **[S5] `layers/embedding.py`** — `torch.where` materializes a full masked embedding tensor for
  the TP>1 vocab-parallel gather. Minor; a masked gather kernel avoids the temporary.

## Build / infra
- **[B1] Engine image should bake minisgl's deps.** Every ad-hoc run `pip install`s
  msgpack/pyzmq/prompt_toolkit/accelerate into the ephemeral container. Build a proper engine image
  `FROM vllm22-w4a8:combined` that pip-installs the engine + deps once (also bakes the W4A8 pkg).
- **[B2] Revisit `memory_ratio` / `page_size` defaults** for RDNA4 / 16 GB once serving real models.
