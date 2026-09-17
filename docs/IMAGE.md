# The minisglang-rdna4 image

Everything needed to serve a model on gfx1201 is baked into one image: the ROCm runtime, a torch
built for RDNA4, the engine, and the 14 custom HIP kernel packages compiled for gfx1201. There is
nothing to install on the host beyond the AMDGPU kernel driver and Docker, and nothing to compile
at first run.

Most people never need this document — `docker compose -f docker-compose.example.yml up` pulls the
published image and serves (see [`SERVING.md`](SERVING.md)). Read on if you want to know what you
are running, or you need to build it yourself.

---

## 1. What's in it

The `Dockerfile` is a **single stage** on top of `rocm/dev-ubuntu-24.04:7.2.1-complete`. There is no
builder/runtime split: the HIP kernels are compiled in the same image that runs them, on purpose. A
`.so` built against one image's torch will not load in another — the failure is
`libc10.so: cannot open shared object file`, and in a serve it shows up as a container that never
reaches the GPU rather than as a build error.

| Component | Detail |
|---|---|
| **Base** | `rocm/dev-ubuntu-24.04:7.2.1-complete` — ROCm 7.2.1 runtime **and** toolchain (hipcc, rocWMMA, hipBLASLt). The `-complete` tag is required: the kernels are compiled inside the image. |
| **Python** | 3.12, in an isolated venv at `/opt/venv` (Ubuntu's system python is PEP-668 externally-managed). |
| **torch** | Built for ROCm 7.2 with gfx1201 in the fat binary, from the PyTorch nightly ROCm 7.2 index. **Unpinned** — `ARG TORCH_SPEC=torch`, so the exact version is whatever was current at build time (`2.14.0.dev20260803+rocm7.2` in the image checked while writing this). |
| **Engine deps** | transformers, tokenizers, safetensors, huggingface-hub, accelerate, modelscope, sentencepiece, einops, numpy, msgpack, pyzmq, psutil, xgrammar, fastapi, uvicorn, pydantic, starlette, prompt_toolkit, openai. |
| **HIP kernels** | 14 packages built from [`rdna4-hip-kernels`](https://github.com/patcarter883/rdna4-hip-kernels) at image-build time (`GPU_ARCHS=gfx1201 bash local/build_local.sh` per package), collected under `/opt/kernels`. |
| **Engine source** | Copied to `/opt/minisgl/python`, so the image runs with no source mount and no repo clone. |
| **`libcpumoe.so`** | The AVX-512 CPU expert tier, built into `/opt/kernels`. See the caveat below. |
| **Profiling tools** | rpd_tracer (rocmProfileData), IntelliKit (accordo, kerncap, linex, metrix, nexus), py-spy, and optionally AMD uProf. Not needed to serve. |

The finished image is about **33 GB**, of which the ROCm base alone is **16.6 GB**. That is the cost
of shipping a compiler toolchain, and it is what buys you a host with no ROCm userland install.

### The 14 kernel packages

Each is an independent kernel-builder package in the canonical repo. The left column is the source
directory; the right is the python module name the engine imports. They differ for seven of them.

| Source dir | Import name | Covers |
|---|---|---|
| `gdn` | `gdn_hip` | Gated delta-net (Qwen3.5 / 3.6 hybrids) |
| `cca` | `zaya_cca` | ZAYA1 convolutional attention (conv-state decode, decode-qk, varlen prefill-qk) |
| `mla` | `mla_hip` | Multi-head latent attention (GLM-4.7-Flash) |
| `attn_hip` | `attn_hip` | Flash-attention prefill (contiguous, non-paged) |
| `attn_decode` | `attn_decode` | Paged decode attention |
| `attn_prefill_paged` | `attn_prefill_paged` | Paged prefill attention |
| `dense_gemm` | `dense_gemm` | Large-M dense GEMM |
| `fp8_wmma` | `fp8_wmma` | The shared WMMA GEMM/GEMV core, dense **and** MoE: fp8, W4A8 (int4/MXFP4/NVFP4) and bf16 all run through it as weight-loader policies |
| `moe` | `moe_hip` | MoE routing: fused softmax/sigmoid + top-k + renormalize + token align |
| `custom_ar` | `custom_ar` | One-shot P2P all-reduce |
| `swiglu` | `swiglu_hip` | Fused SwiGLU shared-expert |
| `sampler` | `sampler_hip` | Fused top-k / top-p sampling |
| `tail` | `tail_hip` | RMSNorm, RoPE, and the other elementwise tail ops |
| `qsa_index` | `qsa_index` | Query-sparse-attention indexer (Qwen4-Exp): score / top-k / expand |

The build fails loudly if any of them does not import: the final step of that layer imports all 14
(which registers their `torch.ops.<mod>_C.*` ops) with no GPU involved.

### Layout and environment

| Path / variable | Value | Why |
|---|---|---|
| `PATH` | `/opt/venv/bin:/opt/rocm/bin:$PATH` | The venv is already first — there is **no activate step**. Just run `python`. |
| `PYTHONPATH` | `/opt/kernels:/opt/minisgl/python` | Kernels first so nothing can shadow them, then the baked engine. |
| `HF_HUB_OFFLINE` | `1` | Baked on, for boxes with a pre-populated HF cache. The example compose sets it back to `0` so a fresh install can download. |
| `TORCH_BLAS_PREFER_HIPBLASLT` | `0` | A stability pin, not a perf choice: on ROCm 7.2.x hipBLASLt intermittently returned `INTERNAL_ERROR` -> `ALLOC_FAILED` on a projection GEMM under sustained serve load. rocBLAS reaches hipBLASLt on gfx1201 either way; the flag only picks whose heuristic selects the solution. |
| `WORKDIR` | `/opt/minisgl` | |
| `EXPOSE` | `1919` | The serve port. |
| `CMD` | `python -m minisgl --help` | The image has no default model, so the bare `docker run` prints usage rather than failing. |

### What is deliberately absent

No vLLM, no SGLang, no flashinfer, no `sgl_kernel`. The serve path does not reach the GPU through
any of them — routing, align, SwiGLU, attention, GDN, MLA, CCA, RMSNorm and RoPE all come from the
kernel packages above, and sampling runs through `sampler_hip` (one fused
temperature+softmax+top-k+top-p+multinomial kernel) with the pure-torch sampler as the fallback.

One honest caveat about Triton: `triton-rocm` is a hard dependency of the torch ROCm wheel and
`triton` is one of xgrammar's, so both are physically installed. Nothing on the default serve path
imports them — on ROCm the attention backend `auto` resolves to `hip`, and the MoE path is native —
so there is no Triton JIT autotune at boot and no Triton kernel in a decode step. The engine does
ship Triton code — a fused-MoE fallback, and a Triton path inside the non-default `rdna4` attention
backend — but neither is reached: both exist for hosts where the HIP kernels are missing, and every
one of those kernels is present here. `rdna4` and its deprecated alias `triton_rdna4` are not
supported on this image; use `auto`, which resolves to `hip`.

### The `libcpumoe.so` caveat — read this before using the CPU expert tier

`--weight-offload-cpu-layers` computes whole MoE layers on host cores, through `libcpumoe.so`. That
library is compiled **`-march=znver4`**, because the machine this engine is developed and measured
on is a Ryzen 7 7800X3D. The core needs AVX-512 with VNNI and F16C and has **no scalar fallback by
design** — a float64 path would run roughly 1000x too slowly and read as a hang, so
`weights/cpu_native.find_library` refuses rather than degrading.

On a host that lacks those instructions (Zen 3 and earlier, and most Intel consumer parts), the
library loads and then **SIGILLs** the moment that tier is used. Every other path in the image is unaffected. If you
need the CPU tier on a different microarchitecture, rebuild with `--build-arg CPU_MOE_ARCH=<arch>`
and change it deliberately — widening the ISA to something generic loses `VPDPBUSD`, which is the
instruction the whole tier is built around.

---

## 2. Building it yourself

The kernels live in a separate repo, outside this repo's build context, so they are injected as a
**named additional build context**. A second context supplies the AMD uProf `.deb`. **Both are
required**, even though the uProf one is usually empty:

```bash
mkdir -p /tmp/empty-uprof
git clone https://github.com/patcarter883/rdna4-hip-kernels.git ../rdna4-hip-kernels

docker build -t minisgl-rdna4:$(date +%Y%m%d) \
  --build-context kernels=../rdna4-hip-kernels \
  --build-context uprof=/tmp/empty-uprof \
  --build-arg KERNELS_REF=$(git -C ../rdna4-hip-kernels rev-parse --short HEAD) \
  .
```

**Why `uprof` is required even when you do not want uProf.** AMD serves the uProf `.deb` only behind
a EULA click-through — the direct URL answers 302 back to the landing page for any request that has
not been through the form — so it cannot be fetched at build time and is supplied as a build context
instead. The Dockerfile's `COPY --from=uprof .` is **unconditional**. Leave the context out and
BuildKit tries to resolve `uprof` as a *registry image*, and the build dies with
`pull access denied ... insufficient_scope`, which reads like a credentials problem rather than a
missing argument. An empty directory is enough: the install step self-skips when it finds no `.deb`.
To actually get uProf, point the context at a directory holding `amduprof_*.deb`.

**Why `KERNELS_REF` must be bumped.** It is a cache-buster label and nothing else — the real kernel
source is the copied `kernels` context, which BuildKit does not hash into the layer key here. Leave
`KERNELS_REF` unchanged after updating the kernels repo and the whole kernel layer is served from
cache: the build succeeds in seconds and you ship the *old* kernels believing otherwise. Deriving it
from `git rev-parse` as above makes that impossible to get wrong.

**Build from a clean checkout.** `COPY --from=kernels .` and `COPY python` copy what is *on disk*,
not what is committed, so a tree with uncommitted edits bakes those edits into the image with no
record of it.

### Build arguments

| Arg | Default | Effect |
|---|---|---|
| `KERNELS_REF` | a short sha | Cache-buster for the kernel layer. Bump it on every kernel change. |
| `TORCH_SPEC` | `torch` | Pin it (e.g. `torch==2.14.0.dev20260803+rocm7.2`) to make the image reproducible. |
| `TORCH_INDEX` | PyTorch nightly ROCm 7.2 index | Switch to a stable channel once ROCm 7.2 stable wheels carry gfx1201. |
| `MAX_JOBS` | `6` | Kernel compile parallelism. Raise it for a solo build; it is deliberately modest to avoid saturating all cores. |
| `CPU_MOE_ARCH` | `znver4` | `-march` for `libcpumoe.so`. See the caveat above. |
| `RPD_REF` | a pinned sha | rocmProfileData commit. Pinned because master does not build — its rlog v3 series includes a header that does not exist in the public rlog repo at any commit. |
| `INTELLIKIT_REF` | `main` | IntelliKit ref. |

### Building through compose

The repo's internal `docker-compose.yml` wires both contexts for you, from
`RDNA4_HIP_KERNELS` and `UPROF_PKG_DIR`. Set both — their defaults in that file are absolute paths
from the development box and will not exist on yours:

```bash
RDNA4_HIP_KERNELS=../rdna4-hip-kernels UPROF_PKG_DIR=/tmp/empty-uprof \
  docker compose --profile serve build
```

**The profile is not optional.** All five services in that file are profile-gated, so a bare
`docker compose build` matches nothing and exits with `No services to build` — a success message for
a build that never happened. Use `--profile serve`: two of the other profiles (`vhip`, `vllm`)
override `image:` to point at unrelated images and would tag the result wrongly. The result is tagged
`minisgl-rdna4:lean` unless you set `MINISGL_IMAGE` to the tag you want.

`docker-compose.example.yml`, the public one, has no build section at all — it only pulls.

Building needs no GPU. The kernel compiles are hipcc invocations and the import check loads the
modules without touching a device.

---

## 3. How it is published

Images go to **`ghcr.io/patcarter883/minisglang-rdna4`**.

| Tag | Meaning |
|---|---|
| `:latest` | The current build. This is what `docker-compose.example.yml` pulls by default. |
| `:<short-sha>` | The engine commit the image was built from. Use this to pin. |
| `:<custom>` | An optional extra tag, supplied per-run. |

Override the image with `MINISGL_IMAGE` if you want a different tag or a local build:

```bash
MINISGL_IMAGE=ghcr.io/patcarter883/minisglang-rdna4:<short-sha> \
  docker compose -f docker-compose.example.yml up
```

### CI: the `publish-image` workflow

[`.github/workflows/publish-image.yml`](../.github/workflows/publish-image.yml) builds and pushes.
It is **`workflow_dispatch` only** — it never runs on push, so it cannot fail CI on a fork. It takes
one optional input, `tag`, added alongside `:latest` and `:<sha>`.

It checks out this repo plus the public kernels repo into the `kernels` context, creates an *empty*
directory for the `uprof` context (for the reason in §2 — omitting it is why `:latest` sat frozen
from 2026-07-14), logs in to GHCR with the built-in `GITHUB_TOKEN` under `packages: write`, and
pushes. It needs no secrets. It does not pass `KERNELS_REF`, which is safe there only because the
job configures no build cache — a fresh runner has nothing stale to hit.

**It will not currently succeed on a GitHub-hosted runner.** The finished image is ~33 GB and a
`ubuntu-latest` runner has roughly 14 GB of free disk, so the job runs out of space. Fixing that
means either aggressively freeing disk on the runner or pointing `runs-on` at a self-hosted machine.
Until that is settled, **the local path below is the reliable one.**

### Local build and push

```bash
# Log in once. Pushing needs the write:packages scope, which `gh auth login` does not grant by
# default; add it first if `docker push` comes back with "denied":
gh auth refresh -h github.com -s write:packages
gh auth token | docker login ghcr.io -u <your-github-username> --password-stdin

# Build (see §2), then tag and push both a moving and a pinned tag:
IMG=ghcr.io/patcarter883/minisglang-rdna4
docker tag minisgl-rdna4:<local-tag> "$IMG:latest"
docker tag minisgl-rdna4:<local-tag> "$IMG:$(git rev-parse --short HEAD)"
docker push "$IMG:latest"
docker push "$IMG:$(git rev-parse --short HEAD)"
```

Push the pinned tag as well as `:latest`, always. `:latest` is the only thing the example compose
resolves, so without a second tag there is no way to roll a user back to the previous image, and no
way for a bug report to say which build it was against.

A first push creates the package as **private**. Make it public in the package settings on GitHub,
or `docker pull` fails with an authentication error for everyone else.
