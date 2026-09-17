# M1-E / Stage B — chunked load. Measured 2026-09-03, card 0 (RX 9070 XT), TP=1.

**Headline.** The shipped one-shot load tops out at **4 of 48 layers** of
`RadixArk/Qwen3.8-Flash-Next-NVFP4`; 5 OOMs. The chunked load reached **22 layers** — **5.5×** — and
what stops it there is no longer VRAM but the pinned host arena. At 48 layers the failure is now a
**capacity refusal at plan time, in milliseconds**, instead of an OOM inside `load_state_dict`.

All figures below are measured on this box on this day. Nothing here is projected.
Raw JSON: `stage_b/*.json`. Harness: `tests/qwen4exp_stage_b_test.py`.

---

## 1. The two legs

| | one-shot (shipped path) | **chunked (Stage B)** |
|---|---|---|
| layers loaded | **4** (5 OOMs) | **22** |
| device: peak `memory_allocated` | 15.22 GiB (reserved 15.60 of a 15.92 GiB card, i.e. at the wall) | 39.34 GiB *incl. arena rows* |
| device: peak allocated **excluding arena rows** | 15.22 GiB | **12.97 GiB** |
| device: free VRAM after | 4.64 GiB | 0.37 GiB |
| host: peak RSS | 6.65 GiB | **41.35 GiB** (36 GiB of it is the pinned arena) |
| wall, load only | 9.5 s | **118.9 s** |
| wall, incl. 36 GiB pin | — | 250.5 s |
| tensors filled | 106 | 528 |

The one-shot leg is given **exactly the shards the chunked leg reads** (a symlink directory), not the
whole checkpoint. That matters: pointed at the real `/model`, `load_weight` globs all 196 shards and
materializes all 48 layers' expert stacks whatever `num_layers` says — it OOMs at the identical byte
for `--layers 6` and `--layers 48`. That measures a missing layer filter, not a missing chunking, so
the honest control has to be shard-matched. (Both were run; `stage_b/oneshot_L{6,8}.json` are the
unfiltered ones, `oneshot_filtered_L*.json` the control.)

**Per chunk**, chunked leg: body 5.55 GiB in 15.4 s, then 1.465 GiB per layer in 3.9–4.2 s. Device
free is flat at 0.37 GiB across the last host layers — i.e. a host-placed layer costs **zero
additional VRAM**, which is the point of the whole feature and is here measured rather than argued.

## 2. The bake is real, and clean

This is the **first time `weights/moe_interpose.py` has ever run against a real model** (M1-A: "the
arena has never been attached to a REAL model's MoE layers"). On the 22-layer run:

| | |
|---|---|
| arena pinned | 36.00 GiB (18 chunks × 2 GiB) |
| arena carved | 26.37 GiB = 18 host layers × 1.465 GiB, exactly |
| `moved_bytes` == `pool.served_bytes` == carved | 28,311,552,000 B |
| **`torch_fallbacks`** | **0** — not one host row fell back to `hipMalloc` VRAM |
| `ArenaMemPool.assert_clean(expect_served_bytes=moved)` | passed |

Each layer's rows were bitwise read-back-compared against their device sources while the sources were
still alive (`_bake`'s step 4), and every device original was proved released by weakref, per layer.

## 3. What now blocks 48 layers on ONE card — and it is not chunking

`--layers 48 --device-gb 5 --host-gb 40` refuses **from integers, before a page is pinned**:

```
[plan] 3 device / 45 host layers, host payload 65.918 GiB, device tier 4.395 GiB
PlacementError: weight offload cannot fit this model: 90.00 GiB of pinned host arena across
1 rank(s) (65.92 GiB/rank of weights, pinned as whole 2.00 GiB chunks) exceeds the usable
ceiling 40.00 GiB
```

65.92 + 4.39 = **70.31 GiB**, which is the expert total to three decimals — the resolver and the
checkpoint agree. Two things follow:

1. **48 layers at TP=1 is arithmetically impossible on this box** (70.31 GiB of experts, ~16 GiB of
   VRAM, ~60 GiB of pinnable RAM shared with everything else). It needs TP=2/EP=2 at 35.16 GiB/rank.
   **`_load_qwen4_exp_weight` refuses TP>1** — no shard rules for the NVFP4 routed experts, the gated
   `q_proj`, the GDN head split or the replicated hyper-connections. **That is the next hard wall,
   and it is a loader-sharding problem, not a chunking one.**
2. **The 2 GiB chunk costs 36 %.** A layer's rows are 1.465 GiB and may never straddle a chunk, so one
   layer occupies a whole 2 GiB chunk and 0.535 GiB is abandoned — 45 host layers pin 90 GiB for
   65.92 GiB of weights. A **3 GiB** chunk fits two layers (2.93 GiB) → 1.5 GiB/layer → 67.5 GiB, a
   **25 % reduction in pinned bytes for a one-line settings change**. Untested: nothing above 2 GiB
   has ever been pinned on this box (P3b), and `chunk_plan` refuses past 4 GiB. Cheapest remaining
   lever on the capacity problem; measure it before believing it.

## 4. The swap tripwire — fixed, and the fix is not "raise the constant"

`DEFAULT_SWAP_TRIPWIRE_PAGES` was a flat 16,384 pages (64 MiB). That is **0.2 % of a 30 GiB pin**, so
every large arena aborted. The first 22-layer attempt died exactly there.

But raising the constant is the wrong fix twice over: it blinds the tripwire as arenas grow, and a
64 MiB pin that drives 64 MiB of eviction really is pathological. Two changes, both in
`weights/host_capacity.py`:

* **The threshold is now a RATE** — `max(flat_floor, 1 % of the bytes being pinned)`. 1 % of 30 GiB is
  78,643 pages, still under the 114,813 P3b measured *at* the ceiling, so P3b's fault is still caught.
* **The tripwire is ARMED only when `MemAvailable` is within 2× the floor.** `pswpout` is box-wide and
  carries no attribution. Measured here: pinning 10 GiB with **71 GiB free** coincided with 432,173
  pages (1.65 GiB, 16 % of the pin) of swap-out that our pinning provably did not cause — this box
  runs zram and its cumulative `pswpout` is in the hundreds of millions of pages. Headroom is the
  discriminator that needs no attribution: swap-out matters only when memory is actually scarce, and
  P3b's rank 1 *was* at its floor when it thrashed. `attach()` passes the `MemAvailable` sample it has
  already taken, so the arming decision and the floor decision come from one reading.

Residual: 1 % and 2× are chosen, not derived. They are the two numbers to revisit if a real thrash
ever gets past this.

## 5. Caveats — read these before quoting anything above

* **No forward pass was run.** This measures that the weights LOAD and that the arena rows are
  bitwise-equal to their sources. It does not claim the model computes correctly at 22 layers. That is
  `qwen4exp_fulldepth_test.py --validate` and the A1.x gates.
* **Not wired into `Engine.__init__`.** `ChunkedWeightLoader` is driven by the harness, not by the
  serve path. The engine hunk is `load_state_dict` + `post_load` + `StageASession.bind()` → the
  chunked driver with `SeamLayerSink`; the phase machinery in `bake.StageASession` needs a chunked
  mode first. **Stage B is proven, not deployed.**
* **Graph capture is undischarged**, as for all of M1. Stage B itself is boot-time file I/O and
  `copy_` — capture-N/A by construction — but the seams it binds carry the same unproven capture
  argument every other M1 module does.
* **The box is shared.** The `--layers 8` one-shot leg died with "0 bytes free" while only 7.19 GiB
  was ours; another agent's container held the card. The 4/5/6/7-layer control points are the clean
  ones and they bracket the ceiling at exactly 4.
* **`min_device_free` = 0.37 GiB** on the 22-layer run is torch's caching allocator holding reserved
  segments it never returns, not live weights: `peak_device_allocated_excl_arena` is 12.97 GiB.
  `memory_allocated()` COUNTS arena rows (they are handed out as ordinary `device='cuda'` tensors —
  that is the P5b mechanism), which is exactly what `bake.model_memory_correction()` exists to remove
  before the KV pool is sized. Never quote torch's raw device figure on an offloading serve.
* **Layer-granular chunking only.** Per-expert-range chunking is deliberately not built: a chunk
  smaller than a container would sample `quant/method.py::ct_packed_sign_convention` from a *prefix*
  of the stack and could decide the sign convention differently for two halves of one GEMM — right
  shapes, no crash, every weight off by 8 quanta. Layer-granular samples exactly what the one-shot
  load samples, so `_ct_sign` needs no threading. If per-expert-range is ever needed, that decision
  must be made once and threaded, and `weights/stage_b.py`'s module docstring says so.
* **Stage B's arena writes are still DEVICE-issued.** M1-A warned that Stage B introduces the first
  CPU store into arena pages. It does not: the chunk is staged on the card and `copy_`d into the arena
  from there, so the visibility discipline is unchanged and Phase 0's unknown #6 does not have to be
  re-answered. A future variant that reads safetensors straight into arena pages on the CPU **would**
  be the first genuine CPU store and would need that argument made.

## 6. Files

| | |
|---|---|
| `python/minisgl/weights/stage_b.py` | the driver: `LoadChunk`, `ChunkedWeightLoader`, `DeviceLayerSink`, `SeamLayerSink` |
| `python/minisgl/models/weight.py` | `qwen4_exp_chunked_source` / `qwen4_exp_chunk_files` / `qwen4_exp_nvfp4_prepass`; `_load_qwen4_exp_weight(files=, nvfp4_sets=)` |
| `python/minisgl/layers/base.py` | `missing_ok=` partial fill, `_post_load_done` guard, `load_nn_bridge_state` (was three copies) |
| `python/minisgl/weights/moe_interpose.py` | `attach_seam` / `bind_seam` extracted from `attach_seams` / `bind_plan` — one implementation of the move and its accounting, shared by both paths |
| `python/minisgl/weights/host_capacity.py` | scaled + headroom-armed `SwapTripwire` |
| `tests/core/test_weight_stage_b.py` | 11 CPU tests: totality ledger, once-only finalize, overlap refusal, partial fill |
| `tests/qwen4exp_stage_b_test.py` | the GPU harness, both legs |

**Test state:** 818 passed / 1 pre-existing failure in `tests/core`
(`test_kv_budget_keeps_exactly_five_subtrahends` — six subtrahends in the uncommitted M1-B
`engine.py`, untouched here).
