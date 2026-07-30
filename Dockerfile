# syntax=docker/dockerfile:1
#
# minisgl-rdna4 — LEAN serving image. NOT based on the vllm image (vllm22-w4a8:combined); contains
# ONLY what minisglang needs to run on gfx1201 (RDNA4):
#
#   * ROCm 7.2.1 runtime + toolchain (hipcc / rocWMMA / hipBLASLt) — the -complete base
#   * torch built for ROCm 7.2 with the gfx1201 (RDNA4) fat binary
#   * the engine's pure-python deps (server + message + tokenizer + model-load)
#   * the custom HIP kernels, built HERE from the CANONICAL repo `rdna4-hip-kernels` — never the
#     vendored copies that used to live in this repo or in vllm-gfx1201.
#
# No vllm, no sglang, no flashinfer, no triton, no sgl_kernel — the native-HIP serve path needs none
# of them (fused MoE routing/align + SiLU + attention + GDN + RMSNorm/RoPE all come from the
# canonical kernels; sampling is pure torch).
#
# The canonical kernels are outside this repo's build context, so `docker compose` injects them as a
# named additional build context (`kernels` -> /home/pat/code/rdna4-hip-kernels). Building standalone:
#   docker build --build-context kernels=<CLEAN kernels worktree> \
#       --build-arg KERNELS_REF=<kernels sha> -t minisgl-rdna4:<tag> .
#
# Build from CLEAN worktrees for BOTH contexts: `COPY --from=kernels .` and `COPY python` copy what
# is ON DISK, so a shared tree bakes another agent's mid-edit files. KERNELS_REF is only a
# cache-buster label — bump it or the kernel layer is served from cache and you ship stale kernels.
# (Dockerfile.lean is RETIRED; stale copies survive in non-git dirs like minisgl-rdna4-leanimg/.)

FROM rocm/dev-ubuntu-24.04:7.2.1-complete

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-pip git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv (Ubuntu's system python is PEP-668 externally-managed).
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/rocm/bin:$PATH \
    PIP_NO_CACHE_DIR=1

# --- torch for ROCm 7.2, with the gfx1201 (RDNA4) arch in the fat binary --------------------------
# Nightly channel: torch 2.10 ROCm stable wheels are not yet published, and this is the same
# ROCm-7.2 / gfx1201 combination the kernels are compiled against (hipcc 7.2.53211 in this base).
# Pin TORCH_SPEC to a known-good build after the first green run to make the image reproducible.
ARG TORCH_INDEX=https://download.pytorch.org/whl/nightly/rocm7.2
ARG TORCH_SPEC=torch
RUN pip install --pre ${TORCH_SPEC} --index-url ${TORCH_INDEX} \
 && python -c "import torch; assert torch.version.hip, 'not a ROCm torch'; \
print('torch', torch.__version__, 'hip', torch.version.hip)"

# --- engine runtime deps (only what minisglang imports on the serve path) -------------------------
# server: fastapi/uvicorn/pydantic/starlette + openai (OpenAI-compatible API) ; message: msgpack/pyzmq ;
# model load: transformers/tokenizers/safetensors/huggingface-hub/accelerate/modelscope/sentencepiece/
# einops ; cli: prompt_toolkit ; util: numpy/psutil ; guided decoding: xgrammar (grammar compiler;
# minisgl does the bitmask masking device-agnostically in torch, so no xgrammar CUDA kernel needed).
# NB: NO apache-tvm-ffi (the tvm_ffi paths are the disabled pynccl/JIT build path — the combined image
# never had it and serve works without it).
RUN pip install \
        "transformers>=4.56" tokenizers safetensors "huggingface-hub" accelerate modelscope \
        sentencepiece einops \
        numpy msgpack pyzmq psutil xgrammar \
        fastapi uvicorn pydantic starlette prompt_toolkit openai

# --- build the custom HIP kernels from the CANONICAL repo (source of truth) ------------------------
# Each subdir of rdna4-hip-kernels is an independent kernel-builder package with a no-Nix local
# build (local/build_local.sh -> hipcc, gfx1201). We build every serve-path package and collect its
# importable python module (torch-ext/<pyname>) under /opt/kernels, which goes on PYTHONPATH. The
# import name of each module already matches what the engine imports (gdn_hip, mla_hip, tail_hip, …);
# only cca is exposed as `zaya_cca` (the engine is repointed to that name in the same change).
ARG KERNELS_REF=2fa1c38
# Bound the compile parallelism. torch's cpp_extension honours MAX_JOBS; unbounded it saturates all
# 16 cores, and on this SHARED box that perturbs whatever a concurrent opt_loop is timing (host-side
# contention shows up in kernel launch latency). Raise it for a solo build; leave it modest when the
# loops are running.
ARG MAX_JOBS=6
COPY --from=kernels . /opt/rdna4-hip-kernels
RUN set -eux; export MAX_JOBS="${MAX_JOBS}"; mkdir -p /opt/kernels; \
    for pkg in \
        gdn:gdn_hip \
        cca:zaya_cca \
        mla:mla_hip \
        attn_hip:attn_hip \
        attn_decode:attn_decode \
        attn_prefill_paged:attn_prefill_paged \
        dense_gemm:dense_gemm \
        fp8_wmma:fp8_wmma \
        moe:moe_hip \
        custom_ar:custom_ar \
        swiglu:swiglu_hip \
        sampler:sampler_hip \
        tail:tail_hip ; do \
      dir="${pkg%%:*}"; mod="${pkg##*:}"; \
      echo "=== building canonical kernel: ${dir} -> import ${mod} ==="; \
      ( cd "/opt/rdna4-hip-kernels/${dir}" && GPU_ARCHS=gfx1201 bash local/build_local.sh ); \
      ln -sfn "/opt/rdna4-hip-kernels/${dir}/torch-ext/${mod}" "/opt/kernels/${mod}"; \
    done; \
    python - <<'PY'
# Import-check every collected kernel module (registers torch.ops.<mod>_C.*). No GPU needed to load.
import sys; sys.path.insert(0, "/opt/kernels")
for m in ["gdn_hip","zaya_cca","mla_hip","attn_hip","attn_decode","attn_prefill_paged",
          "dense_gemm","fp8_wmma","moe_hip",
          "custom_ar","swiglu_hip","sampler_hip","tail_hip"]:
    __import__(m); print("ok import", m)
PY

# --- bake the engine source so the image is SELF-CONTAINED (turnkey `docker compose up`) ----------
# The internal dev compose hot-mounts the repo at /engine and prepends /engine/python to PYTHONPATH
# (edit-and-restart). But the PUBLIC prebuilt image must run with NO source mount and NO repo clone,
# so we also COPY the engine into the image at /opt/minisgl/python and put it on the DEFAULT
# PYTHONPATH below. A dev mount that overrides PYTHONPATH still wins; the baked copy is the fallback.
# .dockerignore (see the repo) already strips __pycache__/*.so/logs from this COPY.
COPY python /opt/minisgl/python

# /opt/kernels first so the canonical builds are authoritative, then the baked engine. The dev
# compose OVERRIDES PYTHONPATH to /opt/kernels:/engine/python:/engine (mounted source + the 3
# not-yet-canonical vendored kernels); the default below is what the turnkey image runs with.
# ROCm serve needs the RCCL-via-torch path.
ENV PYTHONPATH=/opt/kernels:/opt/minisgl/python \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TORCH_BLAS_PREFER_HIPBLASLT=0


EXPOSE 1919
WORKDIR /opt/minisgl
CMD ["python", "-m", "minisgl", "--help"]
