# The THIRD weight tier — 48 layers through the real `Engine`. Measured 2026-09-03, card 0, TP=1.

**Headline.** `weights/stream_tier.py` makes a streamed expert tier shipped code, and with it
`RadixArk/Qwen3.8-Flash-Next-NVFP4` boots **all 48 layers through `LLM` -> `Scheduler` -> `Engine`**
with weight offload engaged. The previous ceiling on that path was **32 layers**; 34, 36 and 48 were
`HostArenaCapacityError` refusals off a live `MemAvailable` (`stage_b_serve/serve_L*.refusal.txt`).

Raw JSON: `stage_b_serve/L48_stream31_serve.json`, `stage_b_serve/L8_stream3_smoke.json`.
Harness: `tests/qwen4exp_offload_serve_test.py --stream-layers N`.

---

## 1. Why a third tier, in integers

`plan.py` chooses between two homes for an expert stack: VRAM, or the pinned host arena. On this
checkpoint neither assignment fits, and the shortfall is arithmetic:

| | |
|---|---|
| routed experts, 48 layers | **70.31 GiB** (1.465 GiB/layer) |
| card | 15.92 GiB, of which the non-expert body takes 9.216 GiB |
| device tier therefore caps near | ~5 GiB |
| host tier needed | **65.92 GiB/rank** |
| live `MemAvailable` on this box | 59–65 GiB |
| `host_capacity` floor | 12.00 GiB |
| refusal at 48 layers, 3 GiB chunks | needs 69.00 GiB + 12.00 GiB floor, **short by 15.52 GiB** |

TP=2 does not close it either (35.16 GiB/rank of experts, ~58 GiB of host pinning across two ranks
against the same ~53 GiB of usable RAM) and `_load_qwen4_exp_weight` has no shard rules for this
family anyway. So the third tier is not an optimisation here; it is the only assignment that exists.

## 2. What it is

`ExpertStreamTier` aliases N layers' expert containers onto **one** shared op-buffer set and refills
the rows the layer's tokens actually route to, immediately before that layer's MoE kernel runs.
`num_experts_per_tok` is 10 of 512, so a decode step reads ~29 MiB of a 1.465 GiB stack.

Ownership is split so no model knowledge leaks into `weights/`:

| | |
|---|---|
| `weights/stream_tier.py` | the tier: aliasing, poisoning, the route, the hook, the counters |
| `models/weight.py::Qwen4ExpExpertRowSource` | "read expert *e* of layer *L* and return it in this container's op layout" |
| `models/weight.py::expert_row_source` | the dispatch twin of `load_weight` / `chunked_weight_source` |

The tier reads the container's component list off `MoELayer.granule_specs()` — the same descriptors
the seam captured — so it is format-agnostic rather than NVFP4-specific, and a container with
non-per-expert (`replicated`) tensors is **refused** rather than streamed with another layer's copy.

## 3. Where it plugs into the existing window, and what is shared

Nothing about the placement plan, the arena, the bake or the capacity gate is bypassed. The streamed
layers are **removed from the question**:

* `StageASession.begin(stream=, stream_layers=)` passes the complement to
  `resolve_weight_plan(layer_indices=...)`, so the arena is reserved and `raise_if_infeasible()` is
  asked for the layers that actually occupy a tier.
* `SeamLayerSink` — the SAME sink Stage B already used — binds the streamed layer's seam `DEVICE`
  first and then adopts it, so every discovered layer still carries a seam, `prove_seam_residency`
  still walks all 48, and the `engaged()` ledger stays honest. Adoption is **incremental** (as each
  chunk finalizes), so the tier's peak device cost is one layer's op buffers, not the tier's.
* `bind()` **refuses** a one-shot load under a configured stream tier: adoption has to happen inside
  the load window or the peak is the whole tier.

Three guards, all unconditional, against the failure mode that matters — a streamed layer reading
another layer's experts is numerically silent (right shapes, plausible magnitudes, fluent wrong text):

1. every row outside the live set is NaN, so an unstaged expert produces NaN logits;
2. `staged_layer` is recorded at the copy and re-checked at the use;
3. the route comes from `quant.kernels._route_align` — the op the served path routes with — unioned
   with the aligner's own block->expert map, not from a torch `softmax().topk()` (measured tie
   disagreement at the k-th boundary; see `stream_tier.py`).

## 4. MEASURED — 48 layers, card 0 (RX 9070 XT), TP=1, `--cuda-graph-max-bs 0`

`--layers 48 --device-gb 0.5 --host-gb 34 --stream-layers 31 --memory-ratio 0.85`,
`MINISGL_WEIGHT_ARENA_CHUNK_MIB=3072`. All 32 harness gates green, `failures: 0`.
Raw: `stage_b_serve/L48_stream31_serve.json`.

| | |
|---|---|
| tiers | 0 device / **17 pinned-host** / **31 streamed** MoE layers |
| Stage B | 49 chunks, 1140 keys, **79.53 GiB staged**, 209.0 s; min device free 568 MiB |
| arena | pinned 27.00 GiB (9 x 3 GiB), carved 24.90 GiB, **0 hipMalloc fallbacks** |
| seam proof | 48 MoE layers on the LIVE model, 17 host (24.90 GiB **inside the arena**), 31 device (1.465 GiB — ONE shared set) |
| KV pool | 108,256 tokens, 2.48 GiB (6,766 pages) |
| boot | 284.0 s |
| decode | **1.136 tok/s** — 200 sampled tokens in 176.1 s |
| stream work | 6,324 stagings, 70,737 expert-rows, **195.6 GB read**, 165.4 s of the 176.1 |
| `engaged` | `stage_a_sealed`, `stage_b_chunked_load`, `stream_tier`, `moe_resolve[host]`, `moe_resolve[device]`, `verify_after_capture` |

**The tier is 94% of the decode time** (165.4 s of 176.1 s), reading 195.6 GB off NVMe at ~1.18 GB/s.
That is the price of booting at all on this box, and it is where any future work goes.

### Quality — SAMPLED, never greedy

Sampler taken verbatim from the checkpoint's `generation_config.json`: temperature 1.0, top_k 20,
top_p 0.95. Chat-templated prompt: *"Explain how a heat pump moves heat from cold outdoor air into a
warm house, and why its efficiency drops as it gets colder outside."* First 200 tokens (the model's
reasoning trace; the budget ran out before the final answer):

> We need answer user: "Explain how a heat pump moves heat from cold outdoor air into a warm house…"
> Mention refrigerant cycle: evaporator outdoor absorbs heat even from cold air (because refrigerant
> colder), compressor raises pressure/temp, condenser releases to indoor, expansion valve lowers
> pressure/temp. Heat pump uses work W to move Q_c from cold to hot, Q_h=Q_c+W, COP = Q_h/W roughly
> Carnot T_hot/(T_hot-T_cold), efficiency decreases as temp lift increases. Also real issues:
> frosting/defrost cycles, reduced refrigerant density/mass flow, lower volumetric capacity,
> compressor work increases, auxiliary resistance heat may turn on…

Coherent, on-topic and technically correct (Carnot COP, defrost, volumetric capacity, auxiliary
resistance heat). **None of the four degeneration signatures** — no loop, no letter-spelling, no
mid-word switch, no token noise. Separately, greedy `"The capital of France is"` -> `" Paris. Paris
is"`, which is the cheap determinism check and NOT a quality read.

## 5. Defects found and fixed on the way

| where | what | fixed |
|---|---|---|
| `plan.build_planned_layers` | `layer_indices` was accepted, forwarded, and then **silently dropped** whenever a `model` was passed — i.e. on every production call, since `engine.py` always has one. A caller restricting the plan got the full layer set, an arena reserved for layers it never intended to place, and a KV pool sized against that reservation. | yes |
| `models/weight.py` (new code) | `_MODELOPT_GLOBAL_SCALE` already carries its leading dot; the row source concatenated another one and every gather raised `File does not contain tensor ...gate_proj..weight_scale_2`. | yes |
| `moe_interpose.prove_seam_residency` | byte totals were summed per LAYER, so aliased containers were counted once per layer. Under the stream tier the boot banner read **46.9 GiB "device-resident" on a 15.9 GiB card** — an impossible number presented as a measurement. Now attributed per `data_ptr()` once, with the alias count reported. | yes |
| `tools/run_offload_serve.sh` | args were interpolated into `bash -lc`, so `--quality-prompt "…with spaces"` re-split and argparse rejected it. Now forwarded through `bash -s --`. | yes |
| harness | the tier's counters were snapshotted before the first generate and read as all-zero, i.e. exactly like a tier that never engaged. | yes |
| `stream_tier.stage` (new code) | the gather stacked EVERY routed expert at once. At decode that is `top_k`=10; at PREFILL it is the union over the chunk's tokens — **measured 253 of 512** on this run — so the transient was ~740 MiB and a 48-layer boot whose decode steps ran fine OOM'd on the first long prompt (278 MiB wanted, 192 MiB free). Now read/convert/write in batches of `GATHER_BATCH=32`, which bounds it regardless of prefill width. This device transient is the one cost the tier adds that `_determine_num_pages` cannot see. | yes |
| harness | the quality leg was unguarded, so its OOM threw away the JSON of a five-minute boot — every gate result and byte count with it. Contained and RECORDED now (`quality_error`), never swallowed. | yes |

## 6. Caveats — read before quoting anything here

* **GRAPH CAPTURE IS UNDISCHARGED, and for this tier it is not a formality.** The hook does file
  I/O, allocation and a sync inside `MoELayer.forward`; all three are illegal under HIP graph
  capture, and capturing it would bake one batch's expert rows into every replay — plausible logits,
  wrong weights, no error. `Engine._build_weight_stream_tier` therefore **refuses** unless
  `--cuda-graph-max-bs 0`. Making it capturable is a design problem (the read is data-dependent on
  the route), not a patch.
* **Throughput is a capacity trade, not a win.** A streamed layer costs a disk read per step. The
  tier exists so a model that cannot boot does; it is never the fast configuration.
* **EP is refused.** The tier reads GLOBAL expert ids and writes them at the same index into a
  container whose dim 0 is this rank's shard. That id remap is unwritten, so `adopt()` raises.
* **A segfault at interpreter exit** follows this harness at 8 layers — *after* PASS and after the
  JSON is written — and it is **pre-existing**: the `--stream-layers 0` control leg segfaults
  identically. Not introduced here, not diagnosed here. The 48-layer run did NOT segfault; it took
  ~2 minutes after PASS to exit, which is `hipHostFree` returning 27 GiB of pinned pages. The box
  recovered fully (65 GiB MemAvailable, both cards idle) either way.
* **`stream_max_experts_per_staging` is the number to watch.** It is `top_k` at decode and the
  per-chunk union at prefill (253 of 512 here). Any future change that widens the prefill chunk
  widens the gather with it; `GATHER_BATCH` is what keeps the transient flat, not luck.
