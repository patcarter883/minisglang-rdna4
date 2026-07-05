# minisgl-rdna4

A minimal LLM serving engine for **AMD RDNA4 (gfx1201 / Radeon RX 9070 XT)**, forked from
[mini-SGLang](https://github.com/sgl-project/mini-sglang) (upstream `9a91cfa`, tracked via the
`upstream` remote) and re-targeted from NVIDIA/CUDA to ROCm. Built around the in-repo
`w4a8_fp8_wmma` int4-weight / fp8-activation WMMA kernel and the tuned RDNA4 `triton_attn`, with the
kernel-call layer kept swappable for an incoming custom kernel framework.

> This README documents the RDNA4 fork. The original CUDA-targeted mini-SGLang README is in git
> history and on the `upstream` remote.

**Status:** dense engine working + numerically validated (Phase 1); W4A8 quantized serving in
progress (Phase 2). Live tracker: `PORT.md`. Optimization backlog: `PERF_NOTES.md`. Full design:
`vllm-gfx1201/docs/RDNA4_ENGINE_DESIGN.md`.

## What works today

- **Dense bf16** (Qwen2/Qwen3/Llama/Mistral), eager, TP=1 — boots and generates coherent output;
  logits match HF transformers to **cos-sim 0.9996** (the standing oracle).
- **Tuned RDNA4 attention** — the vLLM `triton_attn` unified prefill+decode kernel, lifted and
  running under HIP, with **3D flash-decode** and an **fp8 (e4m3) KV cache** (`MINISGL_KV_FP8=1`;
  e4m3 → bf16 → f32 accumulate, nothing dequants to F16).
- **W4A8 (AWQ) dense** — `quant/` package: swappable kernel provider + `LinearMethod` protocol +
  AWQ→op weight conversion (validation in progress).

## Design principles

1. Maximise RDNA4 strengths — native fp8/int4 WMMA, 3D flash-decode, `waves_per_eu` tuning.
2. **Nothing dequants to F16** — I/O bf16, compute fp8 (e4m3fn), accumulate f32.
3. Clean, tidy, agent+human-maintainable — small typed modules, Protocol-based backends.
4. The custom HIP kernels are a **dependency** built from the canonical `rdna4-hip-kernels/` repo
   (never copied into this repo); all quantized GEMMs route through `quant/kernels.py` so a
   different kernel backend can drop in.

## Running (via the GPU lease)

The purpose-built minisglang image (`Dockerfile` + `docker-compose.yml`) is the only serving
configuration — see `docs/LEAN_IMAGE.md`. GPU work goes through the shared-box `flock` lease (never
hand-set devices/ports): the bare `gpu-lease` command on `$PATH` (canonical repo
`/home/pat/code/gpu-lease`, installed via `lease install`).

```bash
# build (CPU only — kernel compiles need no GPU/lease):
docker compose build

# 35B AWQ MoE serve (TP=2):
gpu-lease -n 2 --detach --name leanmoe -- docker compose --profile serve up -d
docker compose -p lease-leanmoe logs -f serve   # follow boot
docker compose -p lease-leanmoe down            # stop -> frees the lease
```

## Tools

- `tools/boot_smoke.py` — load a model + greedy-generate (coherence smoke test).
- `tools/oracle_ours.py` + `oracle_cmp.py` — the **logit oracle**: capture first-token logits and
  compare to HF (cos-sim / top-1). `MINISGL_ORACLE_MODEL` / `MINISGL_ORACLE_REF` parameterize it.

## Layout (changes from upstream)

- `python/minisgl/attention/` — `triton_rdna4.py` backend + the vendored tuned kernel
  (`_triton_unified.py`, `_triton_helpers.py`).
- `python/minisgl/quant/` — W4A8: `config.py`, `kernels.py` (swappable provider), `method.py`.
- Phase-0 torch shims replace flashinfer/sgl_kernel/tvm ops in `layers/`, `engine/sample.py`,
  `kvcache/`, `kernel/radix.py`.
