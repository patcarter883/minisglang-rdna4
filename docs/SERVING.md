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
| **~16 GB VRAM per card** | A 4–8B dense model fits on one card. The large quantized MoE and hybrid checkpoints in §3.1 need **two** cards (`TP=2`). |
| **Disk + network** | The image is large (~33 GB), and model weights download from Hugging Face on first run into a Docker volume. |

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
# Liveness — 200 with the served model name once the backend is wired up:
curl -s http://localhost:1919/health

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
client = OpenAI(base_url="http://localhost:1919/v1", api_key="dummy")  # no auth — any string works
resp = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Write a haiku about GPUs."}],
)
print(resp.choices[0].message.content)
```

The `model` string is whatever you passed as `MODEL` — that is what `/v1/models` advertises, unless
you override it with `--served-model-name` (§5.2).

Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/generate` (raw prompt),
`/health`, and `/metrics` (Prometheus text exposition).

To run detached: `... docker compose -f docker-compose.example.yml up -d`, follow logs with
`docker compose -f docker-compose.example.yml logs -f`, and stop with `... down`.

---

## 3. Choosing a model

`MODEL` takes any Hugging Face repo id, or a local path (§8). Architecture and quantization are both
read from the checkpoint — there is nothing to configure.

### 3.1 Checkpoints validated on this hardware

Every row below has been booted and served on an RX 9070 XT + RX 9070 pair. `MEM_RATIO` is the value
measured to leave a usable KV cache for that checkpoint — §5.1 explains what the knob actually does.
The two-card rows do not fit one 16 GB card at any memory ratio.

| `MODEL` | Cards | `MEM_RATIO` | Notes |
|---|---|---|---|
| `Qwen/Qwen3-4B` | 1 | 0.8 (default) | bf16 dense. The first-run model. |
| `Zyphra/ZAYA1-8B-MXFP4-Experts` | 1 | 0.8 | MoE with a CCA backbone. The backbone does **not** tensor-parallelize — keep `TP=1` and use data parallelism for a second card (§4). |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | 2 | 0.8 | Gated-delta-net hybrid MoE, AWQ int4. Ships an MTP head (§6). |
| `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4` | 2 | 0.8 | Same model, MXFP4. |
| `QuantTrio/GLM-4.7-Flash-AWQ` | 2 | 0.8 | MoE + MLA. The attention backend is force-overridden to `mla`, which requires a page size that is a multiple of 16 (the compose file already passes 16). |
| `cyankiwi/Qwen3.6-27B-AWQ-INT4` | 2 | 0.8 | Dense GDN hybrid, int4. |
| `cyankiwi/Qwen3.6-27B-AWQ-BF16-NVFP4` | 2 | 0.8 | Same model, NVFP4. |
| `sakamakismile/Qwen3.8-27B-MTP-NVFP4` | 2 | **0.97** | All-NVFP4, plus a separate bf16 MTP head. ~10 GiB/card of resident weights leave little room, so the high ratio is what buys a usable KV cache. |
| `cyankiwi/Qwen3.8-27B-AWQ-INT4` | 2 | **0.92** | Same backbone, int4 — the lightest and fastest of the three Qwen3.8-27B builds, but **not the recommended one**: long agent-shaped sessions on it produced token-level corruption (a non-Latin script fused mid-word, duplicated fragments) in 6 of 19 archived sessions, against zero on every other checkpoint here. Root cause open; it did not reproduce from plain requests. The `0.92` is a *lower* ratio on purpose — because the model is so light, at `0.97` the KV cache grows to fill the card and graph capture then OOMs. |
| `unsloth/Qwen3.8-27B-NVFP4` | 2 | **0.97** | Mixed precision: NVFP4 for the bulk MLP, fp8 for attention, GDN projections and lm_head. The heaviest of the three (~11.6 GiB/card); at 0.80 its KV cache sizes negative and boot fails. |
| `poolside/Laguna-XS-2.1-NVFP4` | 2 | **0.85** | Sliding-window attention hybrid, NVFP4. |
| `RedHatAI/Muse-Glimmer-30B-NVFP4` | 2 | **0.85** | Dense sliding-window hybrid, NVFP4. A vision checkpoint — only the text decoder is built. |

Anything else whose architecture is listed in §3.2 should load; it just has not been measured here.

**Gated models** (some Llama/Mistral repos): `export HF_TOKEN=hf_...` in your shell before `up`.

### 3.2 Supported architectures

Read from the checkpoint's `config.json` → `architectures[0]`. **18 architecture strings map to 16
model implementations.** Anything else fails fast at load with `Model architecture ... not
supported`.

| Family | `architectures[0]` | Notes |
|---|---|---|
| Llama | `LlamaForCausalLM` | |
| Mistral | `MistralForCausalLM`, `Mistral3ForConditionalGeneration` | Mistral3 reuses the Mistral decoder. |
| Qwen2 / 2.5 | `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM` | |
| Qwen3 | `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM` | |
| Qwen3.5 / 3.6 / 3.8 | `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration` | Gated-delta-net hybrids, dense and MoE. |
| Qwen3.8-Flash-Next | `Qwen4ExpForConditionalGeneration` | GDN hybrid with hyper-connections and a PLE n-gram block. Larger than two cards on its own — needs weight offload (§7). |
| GLM-4.x MoE | `Glm4MoeLiteForCausalLM` | MLA attention; the backend is forced to `mla`. |
| ZAYA1 | `ZayaForCausalLM` | CCA backbone; does not tensor-parallelize. |
| Nemotron-H | `NemotronHForCausalLM` | Mamba-2 + MoE. **Builds and runs, but the Mamba-2 recurrence is a pure-torch reference path — not GPU-validated.** Do not expect serving performance from it. |
| Laguna | `LagunaForCausalLM` | Sliding-window hybrid. |
| Gemma-4 | `Gemma4ForConditionalGeneration`, `Gemma4ForCausalLM` | Both map to the same implementation. |
| DiffusionGemma | `DiffusionGemmaForBlockDiffusion` | Block-diffusion decode head on the Gemma-4 backbone. |
| Muse-Glimmer | `MuseGlimmerForConditionalGeneration` | |

For multimodal checkpoints (Mistral3, Qwen3.5/3.8, Muse-Glimmer, Gemma-4) only the **text decoder**
is built — the loader skips the vision tower.

### 3.3 Quantization — what is read, and what actually executes

The checkpoint's own `quantization_config` decides everything. Headers understood:

| `quant_method` | Handling |
|---|---|
| `awq`, `gptq` | int4 packed weights with group scales (and zero points). |
| `compressed-tensors` | int4 pack-quantized, MXFP4 (`e2m1` codes + per-group `e8m0`), NVFP4 (`e2m1` + per-16 `e4m3` block scale + fp32 global), and fp8 `e4m3`. |
| `modelopt` (NVIDIA TensorRT-ModelOpt) | Rewritten into the compressed-tensors shape at parse time — the same on-disk layouts under a different name. |
| `quark` (AMD) | Same treatment: rewritten at parse time. |

**What executes is not what the checkpoint's format name suggests.** The point of this engine is that
nothing dequantizes back to 16-bit:

- **4-bit weights in any format — AWQ, GPTQ, MXFP4, NVFP4 — execute as W4A8**: weights stay 4-bit in
  memory, activations are quantized **per token to fp8**, the WMMA units multiply them natively, and
  accumulation is f32.
- **fp8 checkpoints execute as W8A8.**
- Unquantized checkpoints run bf16/fp16 (`--dtype auto`).

16-bit-activation (weight-only) execution is **opt-in**, not the default: set the engine environment
variable `MINISGL_MOE_W4A16=1` (all shapes) or `MINISGL_MOE_W4A16=decode` (decode-sized batches
only, M≤2) for the routed MoE experts. It is silu-only and needs `group_size >= 64`. A checkpoint
card that says "W4A16" is describing the **file format**, not what this engine runs for it.

---

## 4. Two cards (TP=2)

The large quantized checkpoints in §3.1 do not fit on one 16 GB card. Run them tensor-parallel:

```bash
MODEL=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit \
TP=2 \
MEM_RATIO=0.80 \
GRAPH_BS=16 \
MINISGL_EXTRA_ARGS="--max-running-requests 16" \
  docker compose -f docker-compose.example.yml up
```

`GRAPH_BS` is raised alongside `--max-running-requests` on purpose: a graph cap below the running
cap silently lowers the running cap to match (§5.1), so asking for 16 concurrent requests at the
default `GRAPH_BS=8` would have served 8.

If the host has more than two GPUs, or a discrete card alongside an integrated one, choose the cards
with `HIP_VISIBLE_DEVICES` — comma-separated ROCm device indices, as `rocm-smi` enumerates them:

```bash
HIP_VISIBLE_DEVICES=0,1 TP=2 MODEL=... docker compose -f docker-compose.example.yml up
```

Export it only when you actually want to restrict the set: an **empty** value selects **zero** GPUs
(§9).

Two cards does not always mean tensor parallelism. A `ZayaForCausalLM` backbone cannot be sharded,
so a second card runs a second full replica behind the one endpoint — `--data-parallel-size 2`,
optionally with `--enable-ep` to shard the MoE experts across the replicas — rather than `TP=2`.

---

## 5. Knobs

### 5.1 Compose variables

Read from your shell (or a `.env` file) by `docker-compose.example.yml`:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-4B` | HF repo id, or an in-container path. |
| `TP` | `1` | Tensor-parallel size (number of cards). |
| `PORT` | `1919` | Host port mapped to the container's API port. |
| `MEM_RATIO` | `0.8` | Fraction of the card's **free** memory the engine may use in total. Weights, CUDA graphs and the other reserves come out of it, and **the KV cache is whatever is left**. So a heavy checkpoint needs a *higher* value to have any KV cache at all (§3.1); lower it if capture OOMs on a model that fits easily. |
| `GRAPH_BS` | `8` | Maximum batch size for CUDA-graph capture. **It also caps concurrency:** a graph cap of 1 or more that sits below `--max-running-requests` lowers that flag to match, so no decode batch runs uncaptured (fully eager, the worst launch-overhead case). With the default `8`, the server admits 8 concurrent requests however high you set `--max-running-requests`. `0` disables capture entirely — slower, but less memory, and then the admission cap is left alone. |
| `ATTN_BACKEND` | `auto` | On ROCm, `auto` resolves to the native-HIP backend `hip`; those are the only two values worth passing. An MLA checkpoint is force-overridden to `mla` either way. |
| `REASONING_PARSER` | `auto` | Splits reasoning into `reasoning_content`. `auto` derives the delimiter pair from the checkpoint's own chat template (so non-`<think>` markup works); `none` disables extraction. Named families: `qwen3`/`qwen`, `deepseek_r1`, `glm`, `poolside_v1`. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Torch allocator configuration. The default is a fragmentation guard that matters during graph capture on 16 GB cards, and it is what the tight checkpoints in §3.1 were measured with. |
| `HF_TOKEN` | *(unset)* | Hugging Face token for gated/private models. Forwarded only when exported. |
| `HIP_VISIBLE_DEVICES` | *(unset — all GPUs)* | Restrict which cards ROCm sees. Forwarded only when exported; unset is the normal case. |
| `MINISGL_IMAGE` | `ghcr.io/patcarter883/minisglang-rdna4:latest` | Override the image tag. |
| `MINISGL_EXTRA_ARGS` | *(empty)* | Extra `python -m minisgl` flags, appended last. |

### 5.2 Engine flags via `MINISGL_EXTRA_ARGS`

`MINISGL_EXTRA_ARGS` is appended to the end of the launch line, so repeating a flag there
**overrides** the value the compose file already set. Useful ones:

```bash
MINISGL_EXTRA_ARGS="--max-running-requests 16 --served-model-name my-model"   # raise GRAPH_BS too
MINISGL_EXTRA_ARGS="--cache-type recurrent_radix"       # overrides the compose default of radix
MINISGL_EXTRA_ARGS="--max-seq-len-override 32768 --max-prefill-length 2048"
MINISGL_EXTRA_ARGS="--chat-template-kwargs '{\"enable_thinking\": false}'"
```

Note the inner single quotes in that last one. `MINISGL_EXTRA_ARGS` is pasted into a shell command
inside the container, so a bare `{"enable_thinking": false}` loses its double quotes on the way and
the server raises on the first chat request, not at boot. Anything containing JSON or spaces needs
quoting that survives that second parse.

- `--cache-type {naive,radix,recurrent_radix}` — prefix-cache strategy. `recurrent_radix` also
  reuses linear-attention state across prefix hits on GDN/CCA hybrids.
- `--gdn-radix` / `--no-gdn-radix` — recurrent-radix prefix reuse for GDN/CCA hybrids. **On by
  default.** It forces the synchronous scheduler loop, so pass `--no-gdn-radix` to keep overlap
  scheduling.
- `--data-parallel-size N` / `--enable-ep` — replicas behind one endpoint, and expert sharding
  across them (§4).
- `--max-prefill-length N` — chunked-prefill chunk size, default `8192`. Lower it to `2048` if the
  scheduler worker dies part-way through a long prompt: a whole chunk's activations have to fit
  beside the KV pool, and on a wide-activation model an 8192-token chunk does not.
- `--chat-template <file.jinja|string>` — replace a checkpoint's broken or thinking-less template.

The full list comes straight from the image:

```bash
docker run --rm ghcr.io/patcarter883/minisglang-rdna4:latest python -m minisgl --help
```

### 5.3 Engine environment variables

A few behaviours are environment-only — they have no CLI flag. The compose file passes through only
the variables named in its `environment:` list (`HF_TOKEN`, `HIP_VISIBLE_DEVICES`, and the two it
sets itself), so **exporting one of these in your shell is not enough**: add it to that list in
`docker-compose.example.yml`.

| Variable | Meaning |
|---|---|
| `MINISGL_KV_FP8=1` | fp8 (e4m3) KV cache — roughly doubles the KV pool. Scales resolve at boot, first hit wins: `MINISGL_KV_FP8_SCALES=<file\|dir>`, a `kv_scales.safetensors` sidecar next to the weights, or the checkpoint's own `k_scale`/`v_scale` tensors. With none of those it **warns and falls back to identity scales** — uncalibrated, and it costs output quality. |
| `MINISGL_MOE_W4A16=1` \| `decode` | Weight-only 4-bit MoE with 16-bit activations (§3.3). |
| `MINISGL_DFLASH_QUANT=fp8\|int8\|nvfp4\|none` | Weight-only quantization of a DFlash **draft** model, so it fits beside the target (§6). |
| `MINISGL_SPEC_SAMPLED=1` | Rejection-sampling speculative verify, so `temperature > 0` requests speculate at all. Without it speculation runs only for all-greedy batches (§6) — set it on any serve that turns `--spec-algorithm` on. |
| `MINISGL_SPEC_MAX_BS=N` | Batch size above which speculative decoding falls back to plain decode (§6). |

---

## 6. Speculative decoding

Off by default. The target verifies every draft token, so speculation never changes what the model
is allowed to produce — it only removes steps.

**Set `MINISGL_SPEC_SAMPLED=1` as well, or it will do nothing for normal traffic.** With that
variable unset — the engine default — the only accept path is the greedy, bit-exact one, so
speculation engages only for a decode batch in which *every* request is greedy: any
`temperature > 0` request silently falls back to plain decode, and the draft model is loaded but
never used. With it set, non-greedy requests speculate through rejection sampling: the emitted
tokens are drawn from exactly the target's temperature/top-k/top-p distribution, though an
individual run is not byte-identical to an unspeculated one. It needs a line in the compose
`environment:` list (§5.3).

```bash
# with - MINISGL_SPEC_SAMPLED=1 added to the compose environment: list
MODEL=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit TP=2 \
MINISGL_EXTRA_ARGS="--spec-algorithm mtp --spec-num-draft 4" \
  docker compose -f docker-compose.example.yml up
```

| `--spec-algorithm` | Extra weights needed |
|---|---|
| `none` | — (default) |
| `ngram` | None. Prompt lookup: drafts come from repeated n-grams in the context. Tune with `--spec-ngram-min` / `--spec-ngram-max`. |
| `mtp` | None beyond the checkpoint — it uses the model's **own** appended next-token-prediction head, so it only works on a checkpoint that ships one. |
| `eagle3` | A separate EAGLE3 draft checkpoint, via `--spec-draft-model-path`. |
| `dflash` | A separate DFlash block-diffusion draft checkpoint, via `--spec-draft-model-path`. |
| `tidar` | None. Self-draft block diffusion on the target; reads `tidar_config.json` from the served model directory. |

`--spec-num-draft K` is the draft length — verify runs K+1 query rows per sequence. It is a
**ceiling**, not the width that runs: the engine captures a ladder of verify widths and an adaptive
controller picks a rung per step from measured acceptance.

### Validated pairings

These are the settings each pair was measured **booting and serving** at — the draft, the draft
length and the memory it takes to fit both models on the cards. They are not a claim that
speculation is a win on every one of them: it is off by default for most of these targets because it
measured neutral or negative at batch 1 (the DSpark row below, and MTP on GLM and on Qwen3.8-27B).
Measure your own workload — acceptance is content-dependent, and code/math accept far better than
prose.

| Target | `--spec-algorithm` | `--spec-draft-model-path` | `--spec-num-draft` | Also needs |
|---|---|---|---|---|
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | `mtp` | — | 4 | — |
| `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4` | `mtp` | — | 2 | — |
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | `dflash` | `z-lab/Qwen3.6-35B-A3B-DFlash` | 15 | `MEM_RATIO=0.86`, `MINISGL_DFLASH_QUANT=fp8`, `GRAPH_BS=4` |
| `QuantTrio/GLM-4.7-Flash-AWQ` | `eagle3` | `thoughtworks/GLM-4.7-Flash-Eagle3` | 6 | — |
| `poolside/Laguna-XS-2.1-NVFP4` | `dflash` | `poolside/Laguna-XS-2.1-DFlash-NVFP4` | 15 | `MEM_RATIO=0.93` |
| `cyankiwi/Qwen3.6-27B-AWQ-INT4` | `dflash` | `z-lab/Qwen3.6-27B-DFlash` | 15 | `MEM_RATIO=0.90`, `MINISGL_DFLASH_QUANT=fp8`, `GRAPH_BS=2` (the only concurrency this pair booted at) |
| `RedHatAI/Muse-Glimmer-30B-NVFP4` | `dflash` | `meta-models/Muse-Glimmer-30B-assistant` | 15 | `MEM_RATIO=0.96`, `MINISGL_DFLASH_QUANT=nvfp4` |
| `sakamakismile/Qwen3.8-27B-MTP-NVFP4` | `dflash` | `RadixArk/Qwen3.8-27B-DSpark` | 6 | `MEM_RATIO=0.84`, `MINISGL_DFLASH_FULL_CAP=2048` |

The `--spec-*` flags go in `MINISGL_EXTRA_ARGS`; `MEM_RATIO` and `GRAPH_BS` are compose variables;
the `MINISGL_*` entries need a line in the compose `environment:` list (§5.3) — as does
`MINISGL_SPEC_SAMPLED=1`, which every row above assumes.

Two things those recipes encode:

- **A separate drafter is replicated on every rank**, beside the target and its captured graphs.
  That is why most DFlash rows raise `MEM_RATIO` and quantize the drafter: at the normal ratio the
  KV cache sizes to nothing and the engine refuses to boot. `MINISGL_DFLASH_QUANT` is
  weight-only and cannot affect correctness — the target verifies every draft token, so at worst it
  costs acceptance.
- **DSpark is not a separate `--spec-algorithm` value.** It is a `dflash` run with a DSpark draft
  checkpoint; the proposer detects its Markov and confidence heads from the checkpoint's own tensors.

### It is a single-stream optimization

Verify costs `batch × (width + 1)` rows where plain decode costs `batch`, and speculation measured
net-negative at concurrency on every model tested here. So the scheduler falls back to plain decode
above a per-algorithm batch size — `dflash` and `eagle3` above batch 1; `mtp`, `ngram` and `tidar`
uncapped; anything else above 4 — and logs the decision once. Override with `MINISGL_SPEC_MAX_BS`.

---

## 7. Running a model bigger than VRAM

The engine can place MoE expert weights outside VRAM: pinned host RAM streamed over PCIe, an
optional AVX-512 CPU expert tier, and a last-resort tier that reads experts off the checkpoint each
forward. **This is a capacity feature, not a speed one** — it trades throughput for models that
otherwise would not load at all.

All five flags go through `MINISGL_EXTRA_ARGS`, and all are off or derived by default, so a model
that fits pays nothing for their existence. The first three are per-rank budgets in **GiB** (2³⁰,
not decimal GB); the last two are layer counts.

| Flag | Meaning |
|---|---|
| `--weight-offload-device-gb` | VRAM per rank the MoE expert tier may occupy. `0` = derive it from the card's total memory. Lowering it moves whole MoE layers into the pinned host arena; each GiB surrendered is roughly 200k KV tokens. Applies to MoE expert stacks only — on a dense model it is reported as a no-op. |
| `--expert-cache-gb` | VRAM per rank for a per-expert residency cache. `0` = off. It **substitutes** for device-resident MoE layers rather than adding to them, so surrender the same GiB from `--weight-offload-device-gb`: at one budget, layer placement keeps whole layers, while the cache keeps the recently routed experts across all layers. |
| `--weight-offload-gb` | Per-rank pinned host arena budget. `0` = keep the measured default. Never an enable switch. |
| `--weight-offload-cpu-layers` | How many of the deepest offloadable MoE layers are **computed** by host AVX-512 cores instead of being streamed to the card. `0` = off. Needs neither VRAM nor pinned memory, and only the per-token activation/route crosses PCIe; it costs physical cores and int8 activations. An over-budget core request is refused, not clamped. |
| `--weight-offload-stream-layers` | How many of the last MoE layers stream their routed experts off the checkpoint every forward. `0` = off. The capacity tier of last resort — a disk read per layer per step. **Requires `GRAPH_BS=0`** (the gather cannot be captured) and a checkpoint with per-expert granularity. |

Example shape — cap the device expert tier so the remaining layers live in host RAM:

```bash
MODEL=/models/my-huge-moe TP=2 MEM_RATIO=0.85 \
MINISGL_EXTRA_ARGS="--weight-offload-device-gb 8" \
  docker compose -f docker-compose.example.yml up
```

Two things to know before you start:

- **Once offload is needed, `--weight-offload-device-gb` is effectively mandatory.** Left at `0` the
  engine derives the tier from the card's total memory — the entire KV budget — and because the tier
  is billed against that same budget, a non-empty plan under it can never leave a KV cache. The
  engine refuses such a boot in milliseconds with `UnconfiguredDeviceTierError` instead of failing
  minutes later.
- **The CPU expert tier needs an AVX-512 host.** The bundled `libcpumoe.so` is compiled
  `-march=znver4`. It is loaded only when that tier is enabled, but on a CPU without those
  instructions it would fault — leave `--weight-offload-cpu-layers` at 0 unless you are on Zen 4/5
  or an equivalent AVX-512 part.

---

## 8. Serving a model from local disk

Mount your model directory and point `MODEL` at the in-container path. Edit
`docker-compose.example.yml` to uncomment the models bind mount, then:

```bash
# in the compose file, under volumes:
#   - /path/to/your/models:/models:ro
MODEL=/models/my-finetune docker compose -f docker-compose.example.yml up
```

Local paths skip Hugging Face entirely (no network, no `HF_TOKEN` needed).

---

## 9. Troubleshooting

- **`RuntimeError: No HIP GPUs are available`** — the container cannot see the GPU. Confirm
  `/dev/kfd` and `/dev/dri` exist on the host and that the compose `devices:` /
  `group_add: [video]` block is intact.
- **The same error, but only after you edited the compose file** — check how `HIP_VISIBLE_DEVICES`
  is declared. It must stay in the **list** form (`- HIP_VISIBLE_DEVICES`), which forwards the
  variable only when it is actually set in your shell. In the mapping form
  (`HIP_VISIBLE_DEVICES: "${HIP_VISIBLE_DEVICES:-}"`) an unset variable becomes an
  empty-but-**set** one, and an empty value **selects zero GPUs**: torch then reports no HIP device,
  the ROCm check goes false, and the engine tries to pick a CUDA attention backend on an AMD card.
  The same trap applies to any other device variable you add.
- **Out-of-memory during startup or graph capture** — on a model that fits comfortably, lower
  `MEM_RATIO` (e.g. `0.7`), lower `GRAPH_BS`, or set `GRAPH_BS=0` to disable capture. On a tight
  checkpoint the fix is usually the opposite — see the next item.
- **`AssertionError: Not enough memory for KV cache after reserving ...`, or a KV cache that is
  absurdly small** — the weights and reserves have eaten the whole budget. **Raise** `MEM_RATIO` to
  the value §3.1 lists for that checkpoint, and drop `GRAPH_BS`. Large checkpoints need `TP=2`.
  Note that the assertion's own text suggests *lowering* `--memory-ratio`; on a checkpoint that is
  merely too big for the budget that makes it worse, because the budget is `ratio × free memory` and
  the KV cache is what survives the subtractions (§5.1).
- **`Model architecture ... not supported`** — the checkpoint's architecture is not in the registry
  (§3.2). Pick a supported family.
- **Model will not download** — a fresh install needs online Hugging Face access; the example
  compose sets `HF_HUB_OFFLINE=0` (the image itself bakes `1`, for pre-cached hosts). For gated
  repos, set `HF_TOKEN`.
- **Speculative decoding is configured but nothing got faster** — check `MINISGL_SPEC_SAMPLED=1` is
  in the compose `environment:` list. Without it, only all-greedy batches speculate and every
  `temperature > 0` request quietly takes the plain decode path (§6). Also check the batch size: it
  is off above batch 1 for `dflash`/`eagle3`.
- **Only a handful of requests run at once, whatever `--max-running-requests` says** — `GRAPH_BS`
  caps admission to the largest captured batch (§5.1). Raise both together.
- **An engine environment variable appears to do nothing** — compose forwards only the variables
  named in its `environment:` list. `MINISGL_*` variables exported in your shell do not reach the
  container until you add them there (§5.3).
- **Slow first request** — the first run downloads weights and captures CUDA graphs; later starts
  reuse the cached weights (the `hf-cache` volume) and are much faster.

---

## 10. What's inside the image

One stage, on `rocm/dev-ubuntu-24.04:7.2.1-complete`:

- **ROCm 7.2.1 runtime and toolchain**, plus **torch for ROCm 7.2** with gfx1201 in the fat binary,
  in a venv at `/opt/venv` that is already on `PATH` — there is no activate step.
- **The engine**, baked to `/opt/minisgl/python`: OpenAI-compatible server, tokenizer/detokenizer
  workers, scheduler, radix prefix cache, and xgrammar for structured output.
- **14 HIP kernel packages** compiled for gfx1201 at image-build time and collected under
  `/opt/kernels`: `gdn_hip`, `zaya_cca`, `mla_hip`, `attn_hip`, `attn_decode`, `attn_prefill_paged`,
  `dense_gemm`, `fp8_wmma`, `moe_hip`, `custom_ar`, `swiglu_hip`, `sampler_hip`, `tail_hip`,
  `qsa_index`.
- **`libcpumoe.so`**, the CPU expert tier for weight offload (§7), compiled `-march=znver4` — it
  uses AVX-512, so a host without those instructions must leave that tier off.
- `PYTHONPATH=/opt/kernels:/opt/minisgl/python`, `HF_HUB_OFFLINE=1`, `EXPOSE 1919`.

**No vLLM, no Triton, no flashinfer** — the native-HIP serve path needs none of them. Attention
backends that appear in `--help` but depend on those stacks (`trtllm`, `fi`, `fa`, `rdna4`,
`triton_rdna4`) cannot run on this image; use `auto` or `hip`.

See [`docs/IMAGE.md`](IMAGE.md) for the image internals and how to build it yourself, and
[`README.md`](../README.md) for the project overview.
