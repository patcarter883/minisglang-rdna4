# Serving a model with minisglang-rdna4

minisglang-rdna4 is a minimal, OpenAI-compatible LLM serving engine built specifically for
**AMD RDNA4** GPUs (gfx1201 — Radeon RX 9070 / RX 9070 XT). It ships as a **prebuilt Docker image**
with the engine and all custom HIP kernels baked in, so serving a model is a single command — no
building, no cloning of a kernel repo, no CUDA.

```bash
MODEL=Qwen/Qwen3-4B docker compose -f docker-compose.example.yml up
```

That pulls the image, downloads the model from Hugging Face on first run, and serves an
OpenAI-compatible API at **http://localhost:1919/v1**.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **AMD RDNA4 GPU** | Radeon RX 9070 or RX 9070 XT (gfx1201). The kernels are compiled for gfx1201 only. |
| **ROCm kernel driver** | The host needs the AMDGPU/ROCm kernel driver so `/dev/kfd` and `/dev/dri` exist. A full ROCm userland install is **not** required — the runtime lives inside the image. |
| **Docker** with GPU device access | Standard Docker; the compose file passes `/dev/kfd` + `/dev/dri` through. No `nvidia-container-toolkit` and no special runtime needed. |
| **~16 GB VRAM per card** | A 4B dense model fits comfortably on one card. Large quantized MoE models (35B) need **two** cards (TP=2). |
| **Disk + network** | Model weights download from Hugging Face on first run and are cached in a Docker volume. |

Quick host check — you should see your RDNA4 card and the render nodes:

```bash
ls -l /dev/kfd /dev/dri        # both must exist
rocminfo | grep -i gfx1201     # (if rocminfo is installed) confirms the arch
```

---

## 2. Quickstart

Grab the repo (or just the two files: `docker-compose.example.yml` + this guide) and run:

```bash
# Serve a small dense model on a single card (good first test):
MODEL=Qwen/Qwen3-4B docker compose -f docker-compose.example.yml up
```

Wait for the log line indicating the server is listening on `0.0.0.0:1919`, then in another shell:

```bash
# List the loaded model:
curl -s http://localhost:1919/v1/models | python3 -m json.tool

# Chat completion:
curl -s http://localhost:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
    "max_tokens": 64
  }' | python3 -m json.tool
```

Or with the OpenAI Python SDK (the API is drop-in compatible):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:1919/v1", api_key="dummy")  # no auth by default
resp = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Write a haiku about GPUs."}],
)
print(resp.choices[0].message.content)
```

To run detached: `... docker compose -f docker-compose.example.yml up -d`, follow logs with
`docker compose -f docker-compose.example.yml logs -f`, and stop with `... down`.

---

## 3. Choosing a model

Set `MODEL` to any Hugging Face repo id (or a local path — see §6) whose architecture is supported:

| Family | Example `MODEL` | Fits | Notes |
|---|---|---|---|
| **Qwen3 dense** | `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B` | 1 card | Default; bf16. |
| **Qwen2.5 / Qwen2 dense** | `Qwen/Qwen2.5-7B-Instruct` | 1 card | |
| **Llama / Mistral dense** | `mistralai/Mistral-7B-Instruct-v0.3` | 1 card | |
| **Qwen3 MoE** | `Qwen/Qwen3-30B-A3B` (quantized) | 1–2 cards | Use an AWQ/quantized variant on 16 GB. |
| **Qwen3.6-35B (GDN-hybrid MoE)** | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | 2 cards | W4A8 AWQ; needs `TP=2`. |
| **GLM-4.7-Flash (MoE + MLA)** | `QuantTrio/GLM-4.7-Flash-AWQ` | 2 cards | Needs `TP=2`. |

Supported architectures (from `config.json` → `architectures[0]`): `LlamaForCausalLM`,
`MistralForCausalLM`, `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM`, `Qwen3ForCausalLM`,
`Qwen3MoeForCausalLM`, `Qwen3_5*`, `Glm4MoeLiteForCausalLM`, `ZayaForCausalLM`, `LagunaForCausalLM`.
An unsupported architecture fails fast at load with `Model architecture ... not supported`.

**Gated models** (e.g. some Llama/Mistral): set `HF_TOKEN=hf_...` in your shell before `up`.

Quantization is read from the checkpoint's own `quantization_config` — AWQ (W4A8), compressed-tensors
(W4A16), and MXFP4 checkpoints load automatically; there is nothing to configure.

---

## 4. Two-card (TP=2) and large models

The big quantized MoE models don't fit on one 16 GB card. Run them tensor-parallel across two cards:

```bash
MODEL=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
TP=2 \
MEM_RATIO=0.82 \
MINISGL_EXTRA_ARGS="--max-running-requests 32" \
  docker compose -f docker-compose.example.yml up
```

If your host has more than two GPUs (or an integrated GPU), pin the discrete cards you want with
`HIP_VISIBLE_DEVICES` (0-based over the discrete GPUs), e.g. `HIP_VISIBLE_DEVICES=0,1`.

---

## 5. Common knobs

All are environment variables consumed by `docker-compose.example.yml`:

| Env | Default | Meaning |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-4B` | HF repo id or local path to serve. |
| `TP` | `1` | Tensor-parallel size (number of cards). |
| `PORT` | `1919` | Host port mapped to the container's API port. |
| `MEM_RATIO` | `0.8` | Fraction of VRAM for the KV cache. Lower it if you hit OOM at capture. |
| `GRAPH_BS` | `8` | Max batch size for CUDA-graph capture. `0` disables graph capture (slower, uses less memory). |
| `ATTN_BACKEND` | `auto` | Attention backend (`auto` / `hip`). |
| `REASONING_PARSER` | `auto` | Splits `<think>…</think>` into `reasoning_content`; safe no-op on non-thinking models. |
| `HF_TOKEN` | *(unset)* | Hugging Face token for gated/private models. |
| `HIP_VISIBLE_DEVICES` | *(all)* | Pin specific GPUs (0-based over discrete cards). |
| `MINISGL_IMAGE` | `ghcr.io/patcarter883/minisglang-rdna4:latest` | Override the image tag. |
| `MINISGL_EXTRA_ARGS` | *(empty)* | Any extra `python -m minisgl` flags (e.g. `--max-running-requests 32`, spec-decode flags). |

Run `docker run --rm ghcr.io/patcarter883/minisglang-rdna4:latest python -m minisgl --help` to see
every engine flag.

---

## 6. Serving a model from local disk

Mount your model directory and point `MODEL` at the in-container path. Edit
`docker-compose.example.yml` to uncomment the models bind mount, then:

```bash
# in the compose file, under volumes:
#   - /path/to/your/models:/models:ro
MODEL=/models/my-finetune docker compose -f docker-compose.example.yml up
```

Local paths skip Hugging Face entirely (no network, no `HF_TOKEN` needed).

---

## 7. Troubleshooting

- **`RuntimeError: No HIP GPUs are available`** — the container can't see the GPU. Confirm
  `/dev/kfd` and `/dev/dri` exist on the host and that the compose `devices:`/`group_add: [video]`
  block is intact. On a multi-GPU/APU box, set `HIP_VISIBLE_DEVICES` to the discrete card index.
- **Out-of-memory during startup / graph capture** — lower `MEM_RATIO` (e.g. `0.7`), lower
  `GRAPH_BS` (or set `GRAPH_BS=0` to disable capture), or add
  `MINISGL_EXTRA_ARGS="--max-running-requests 16"`. Large MoE models need `TP=2`.
- **`Model architecture ... not supported`** — the checkpoint's architecture isn't in the registry
  (§3). Pick a supported family.
- **Model won't download** — a fresh install needs online Hugging Face access; the example compose
  sets `HF_HUB_OFFLINE=0`. For gated repos, set `HF_TOKEN`.
- **Slow first request** — first run downloads weights and captures CUDA graphs; subsequent starts
  reuse the cached weights (`hf-cache` volume) and are much faster.

---

## 8. What's inside the image

- ROCm 7.2 runtime + torch built for gfx1201 (RDNA4).
- The minisglang engine (OpenAI-compatible server, tokenizer, scheduler, radix prefix cache).
- Custom HIP kernels compiled for gfx1201: attention (prefill/decode, fp8 KV cache), W4A8/W8A8/MXFP4
  GEMMs, grouped-MoE, GDN (gated delta-net), MLA, CCA, RMSNorm/RoPE, sampling.

No vLLM, no Triton, no flashinfer — the native-HIP serve path needs none of them. See
[`docs/LEAN_IMAGE.md`](LEAN_IMAGE.md) for the image internals and
[`README.md`](../README.md) for the project overview.
