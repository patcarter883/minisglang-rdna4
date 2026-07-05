# Lean serving image + canonical kernel sourcing

`Dockerfile` + `docker-compose.yml` build the purpose-built minisglang image — the **only** serving
configuration for this repo. It is **NOT** based on the shared vllm image (`vllm22-w4a8:combined`).
It contains only what the engine needs to run on gfx1201 (RDNA4), and it sources every custom HIP
kernel from the **canonical**
[`rdna4-hip-kernels`](/home/pat/code/rdna4-hip-kernels) repo — not from vendored copies in this repo
or in `vllm-gfx1201`.

## What's in the image

- Base `rocm/dev-ubuntu-24.04:7.2.1-complete` — ROCm 7.2.1 runtime + toolchain (hipcc 7.2.53211,
  rocWMMA, hipBLASLt), the **same** toolchain the combined image used, minus vllm.
- `torch` for ROCm 7.2 with the gfx1201 fat binary (nightly `download.pytorch.org/whl/nightly/rocm7.2`).
- The engine's pure-python deps only: transformers / tokenizers / safetensors / accelerate /
  modelscope / msgpack / pyzmq / fastapi / uvicorn / pydantic / starlette / prompt_toolkit / openai /
  numpy / psutil / sentencepiece / einops.
- **No** vllm, sglang, flashinfer, triton, sgl_kernel, apache-tvm-ffi. The native-HIP serve path
  needs none of them (routing/align/silu/attention/GDN/RMSNorm/RoPE all come from the canonical
  kernels; sampling is pure torch; MoE `topk_softmax` has a pure-torch fallback).
- The 11 canonical kernels, built at image-build time via each package's `local/build_local.sh`
  (hipcc, gfx1201) and collected on `PYTHONPATH=/opt/kernels`: `gdn_hip`, `zaya_cca`, `mla_hip`,
  `attn_hip`, `attn_decode`, `attn_prefill_paged`, `w4a8_fp8_wmma`, `moe_hip`, `moe_splitk_hip`,
  `swiglu_hip`, `tail_hip`.

## Kernel cutover (canonical callable API)

The vendored packages registered plain `torch.ops.<name>.*` ops. The canonical kernel-builder
packages register under a build-unique `torch.ops.<name>_C` namespace and expose the ops as
**module-level callables** (`tail_hip.silu_and_mul`, `gdn_hip.gdn_prefill_wmma`,
`mla_hip.mla_decode`, …). The engine was repointed accordingly:

- `torch.ops.<mod>.<op>(...)` → `<mod>.<op>(...)`
- `from gdn_hip import op as gdn` → `import gdn_hip as gdn`
- `import cca_hip.cca_op` + `torch.ops.zaya_cca.*` → `import zaya_cca` + `zaya_cca.*`
- `w4a8_fp8_wmma` now resolves to the canonical package (was imported from `vllm-gfx1201`).
- `vllm._custom_ops.topk_softmax` and `vllm … moe_align_block_size` removed: MoE routing falls back
  to pure torch (softmax → topk → renorm), align uses the native `moe_hip.moe_align` drop-in.

The vendored kernel dirs were deleted from this repo. The canonical packages are the single source
of truth (Hub repo-ids `pat883/*`, or the local build used here).

## Not yet de-vendored (blockers)

These three are imported by the engine but are **not in `rdna4-hip-kernels`** yet, so they remain
vendored here on non-default paths. To finish the cutover, port each into the canonical repo (see its
`PORTING.md`) and delete the local copy:

| dir | path | why still vendored |
|-----|------|--------------------|
| `w8a8_fp8_wmma/` | W8A8 fp8-WMMA dense/MoE GEMM | not ported to canonical (non-default quant) |
| `moe_bf16_wmma/` | bf16/fp16 grouped-MoE WMMA GEMM | canonical "upstream candidate", not yet vendored there |
| `rxf_hip/` | RXF W4-NL/A8 rotate+quant MoE | deliberately excluded from canonical ("not our work") |

## Build & run

```bash
# build (CPU only — kernel compiles need no GPU/lease):
docker compose build

# 35B AWQ MoE serve (TP=2). max-running-requests bounds the GDN state on 16 GB cards:
MINISGL_MEM_RATIO=0.8 MINISGL_EXTRA_ARGS="--max-running-requests 16" \
  gpu-lease -n 2 --detach --name leanmoe -- \
  docker compose --profile serve up -d
docker compose -p lease-leanmoe logs -f serve       # follow boot
docker compose -p lease-leanmoe down                # stop -> frees the lease
```

## Validation (2026-07-04, lean image)

- Dense smoke (Qwen3-0.6B, 1 card): coherent — "The capital of France is" → " Paris."; "2 + 2? A:" → " 4".
- 35B AWQ MoE + GDN (cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, TP=2): booted, coherent chat completions
  (correctly reasons the capital of Japan is Tokyo). Exercises the W4A8 MoE route (torch
  `topk_softmax` fallback), `moe_hip`/`moe_splitk_hip`, GDN kernels, and the HIP attention path — all
  from the canonical repo, with the vendored dirs removed.

> The legacy `Dockerfile` (CUDA/NVIDIA upstream) and its mount-into-`vllm22-w4a8:combined`
> compose were removed — this purpose-built image is now the only serving configuration.
