# minisglang-rdna4

**An OpenAI-compatible LLM server for AMD Radeon RX 9070 / 9070 XT (RDNA4, gfx1201).**
One `docker compose up`. No CUDA, no ROCm userland install, no Triton, no vLLM.

```bash
git clone https://github.com/patcarter883/minisglang-rdna4.git
cd minisglang-rdna4
MODEL=Qwen/Qwen3-4B docker compose -f docker-compose.example.yml up
```

That pulls a prebuilt image, downloads the model on first run, and serves
`http://localhost:1919/v1` — point any OpenAI client at it.

---

## Why it exists

RDNA4 has the silicon for modern inference: native fp8 and int4 WMMA instructions, and 16 GB
of fast memory on a card you can actually buy. What it hasn't had is a serving stack that uses
them. The mature engines target datacentre parts — CDNA and NVIDIA — and the portable fallbacks
that do run on a Radeon reach the GPU through generic paths that dequantize quantized weights
back to 16-bit before multiplying, spending the memory bandwidth that quantization was supposed
to save and leaving the fp8/int4 units idle.

This engine is written the other way round: **nothing dequantizes to fp16.** Weights stay 4-bit
in memory, activations are quantized to per-token fp8, the WMMA units multiply them natively,
and accumulation is f32. Every GEMM, attention and MoE path on the serving route is a
hand-written HIP kernel compiled for gfx1201 — about 35k lines of them, in a
[separate kernel repo](https://github.com/patcarter883/rdna4-hip-kernels) — under an 85k-line
engine forked from [mini-SGLang](https://github.com/sgl-project/mini-sglang).

The practical result: a 35B mixture-of-experts model serves across two consumer Radeons, and
models far larger than VRAM serve by streaming expert weights from host RAM.

## What it serves

Architecture and quantization are both read from the checkpoint — there is nothing to configure.

| | |
|---|---|
| **Dense** | Llama, Mistral, Qwen2 / 2.5 / 3 |
| **MoE** | Qwen2-MoE, Qwen3-MoE, Qwen3.5 / 3.6 (gated delta-net hybrid), GLM-4.7-Flash (MLA), ZAYA1 (CCA), Nemotron-H (Mamba-2) |
| **Other** | Laguna, Gemma-4 / DiffusionGemma (block diffusion), Muse-Glimmer, Qwen3.8-Flash-Next |
| **Checkpoint formats** | AWQ, GPTQ, compressed-tensors, MXFP4, NVFP4, fp8 — plus ModelOpt and AMD Quark headers |

18 architecture strings map to 16 model implementations; anything else fails fast at load.
4-bit checkpoints run **W4A8** by default (4-bit weights × per-token-fp8 activations); fp8
checkpoints run W8A8. Weight-only 16-bit-activation execution is available opt-in.

A single 16 GB card comfortably serves a 4–8B dense model. Large quantized MoE checkpoints run
across two cards with `TP=2`.

## What it does

- **Speculative decoding** — n-gram, MTP, EAGLE3, DFlash and TiDAR, with propose *and* verify
  under graph capture, and an adaptive verify width that picks a rung per step from measured
  acceptance. Verify is bit-exact, so sampled requests keep the target model's exact distribution.
- **Prefix caching** — radix cache, including a recurrent variant that reuses linear-attention
  state across prefix hits for gated-delta-net and CCA hybrids.
- **Bigger than VRAM** — a four-tier weight placement system: expert weights stream from pinned
  host RAM over PCIe, with an SLRU residency cache on the GPU and an optional AVX-512 CPU expert
  tier. This is a capacity feature, not a speed one — it trades throughput for models that
  otherwise would not load at all.
- **Structured output and tool calling** — JSON-schema and grammar constraints via xgrammar, with
  the tool-call format derived from the checkpoint's own chat template rather than hardcoded.
- **Reasoning models** — `reasoning_content` is split out automatically, with the delimiter pair
  derived from the chat template (so non-`<think>` markup works with no code change).
- **Parallelism** — tensor parallel, data parallel and expert parallel, with a one-shot P2P
  all-reduce over PCIe when both cards can peer.
- **fp8 KV cache**, Prometheus metrics at `/metrics`, and a web control panel for starting and
  stopping serve configurations.

## Going further

- **[docs/SERVING.md](docs/SERVING.md)** — the full serving guide: prerequisites, model recipes,
  two-card setups, every knob, troubleshooting.
- **[docs/IMAGE.md](docs/IMAGE.md)** — what is inside the image and how to build it yourself.
- **[rdna4-hip-kernels](https://github.com/patcarter883/rdna4-hip-kernels)** — the HIP kernels,
  and the one-core-per-shape policy that governs them.
- **[control-panel/](control-panel/)** — the web UI for managing serve configurations.
- **[docs/journal/](docs/journal/)** — dated engineering records: measurements, bring-up notes and
  design arguments, kept as evidence. Not maintained as guides.

## Status

A research engine that serves real workloads daily, not a hardened release. It moves fast, and
interfaces change. Two caveats worth stating plainly: Nemotron-H loads but runs its Mamba-2
recurrence on an unoptimized reference path, and the engine is built and validated on gfx1201
only — other RDNA4 parts are untested.

Issues and pull requests welcome.

## License

MIT. Forked from [mini-SGLang](https://github.com/sgl-project/mini-sglang) (`9a91cfa`); the
upstream copyright notice is preserved in [LICENSE](LICENSE).
