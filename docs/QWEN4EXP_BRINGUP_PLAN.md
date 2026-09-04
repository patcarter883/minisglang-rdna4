# Qwen3.8-Flash-Next (`qwen4_exp`) bring-up plan

Target: `RadixArk/Qwen3.8-Flash-Next-NVFP4`, HF arch `Qwen4ExpForConditionalGeneration`,
model_type `qwen4_exp`. Worktree `/home/pat/code/minisgl-rdna4-offload`, branch
`feat/weight-offload`, base `b1f52659`.

Status of this document: **`[RUN-2026-09-03]` SUPERSEDED IN PART — the code has run.** The original
header read *"planning only. Nothing here has been run on a GPU. No gate below is discharged"*; that
is now the most wrong line in the document. All 48 layers run on real weights on one RX 9070 XT and
emit coherent sampled text to 200 tokens. **GATE-2, GATE-5, GATE-6 and KILL-1(a) are discharged;
GATE-10 was added because the plan did not anticipate it.** Claims still labelled from static
reading are marked as such. The first-run report, including the list of predictions this document got
WRONG, is [`docs/measurements/QWEN4EXP_FIRST_RUN.md`](measurements/QWEN4EXP_FIRST_RUN.md) — read that
alongside this.

`[RUN-2026-09-03]` **The costliest wrong prediction is T1.2** (see §3 Tranche 1): its prescribed fix
— rename `.weight_scale_2` to `.weight_global_scale` — *is the bug*. The two spellings carry opposite
arithmetic conventions, and normalizing the name silently inherits DIVIDE, overflowing 97% of the
per-group fp16 scales to `inf` and NaN-ing every logit 48 layers later. Do not follow that row.

---

## 0. Executive summary

1. **Capacity gates SERVING, and it was wrongly believed to gate CORRECTNESS too. It does not.**
   ~~Shortfall ~21 GiB~~ — MEASURED (bring-up round 3, `tests/qwen4exp_fulldepth_test.py`, one
   RX 9070 XT): routed experts **70.312 GiB** (the arithmetic below was exactly right), non-expert
   body **9.216 GiB** (the "~15 GiB" was the on-disk bf16 shard bytes, which include the `mtp.*`
   head and the vision tower this build does not construct), total **79.53 GiB** against ~63.7 GiB
   reachable — shortfall **15.8 GiB**, not 21.
   The correctness consequence is the important one: **only ONE layer's experts are live at a time**
   (1.465 GiB), so a full-depth forward needs 9.216 + 1.465 = **11.377 GiB resident, measured**, and
   fits one card with 4.5 GiB to spare. Depth was never the blocker; holding all 48 layers' experts
   AT ONCE was. All 48 layers now run with real weights (see the Tranche 3 status box), so no
   correctness question below is still gated on this row. A **serve** is, and still is.
2. **Hyper-connections are no longer an unknown.** A complete reference implementation is on disk at
   `/home/pat/Projects/sglang-upstream/python/sglang/srt/layers/hyperconnection.py`. I read the
   compute functions directly; the survey's math is confirmed exactly, including all five silent-
   numerics scalars. What remains unverified is whether sglang matches HF, and any numeric check.
   Step T0.3 closes that on CPU for free.
3. **The QSA indexer is a port, not a research project — and it is deferrable behind a real
   milestone.** A 3,176-line reference exists upstream (`srt/layers/attention/qsa/`), mostly Triton.
   Critically, `sparse_attn.py` clamps `row_topk = min(topk, visible)`, so **at seq_len <= 2048
   (`indexer_budget`) sparse selection is exactly dense causal attention**. A first coherent decode
   at short context therefore needs no indexer at all. The tilelang MQA path is NV-only and has no
   AMD equivalent to copy.
4. **Hyper-connections are the same ORDER as the whole activated expert set — but they do not
   "dominate", and the original claim had the comparison backwards.** ~~more decode traffic than all
   480 activated experts combined~~. MEASURED (round 3, `tests/qwen4exp_sizing_probe.py`): HC is
   **1.193 GiB = 1.281 GB** per step (the 1.28 GB figure was right), the 480 activated experts are
   **1.373 GiB = 1.475 GB**, so HC is **86.9% of** expert traffic, not more than it. At the 706.6
   GB/s ceiling that is a 552 tok/s HC-only ceiling against a 479 tok/s expert-only one — the
   experts are the larger term. The startling part survives and is the actionable part: an
   unquantizable 1.28 GB of pure plumbing costs nearly as much decode bandwidth as all 480 routed
   experts, so T7.2/T7.3 stay real. Measure before promising throughput.
5. Sequencing is **CPU-verifiable first**: HC numeric parity, loader key-set parity and config
   plumbing all gate on tests that need no GPU and no model body.
6. Deliberately deferred, and **not** pretended done: vision, MTP, graph capture, the offload arena,
   TP=2, and the QSA indexer. Graph capture is a repo merge requirement and appears as T6.
   `[CAPTURE-2026-09-04]` **Three of those six have since LANDED** — the offload arena, TP=2, and
   graph capture (decode, at full 48-layer depth, on the offloaded TP=2 serve; see
   `docs/measurements/QWEN4EXP_GRAPH_CAPTURE.md`). Still deferred: **vision, MTP, and the QSA
   indexer** above `indexer_budget`.
7. `[RUN-2026-09-03]` **"Reuse `GDNLinearAttn` verbatim" was true structurally and false on dtype.**
   Reuse was justified from shapes and key names, which match exactly; stored dtypes were never
   checked. This checkpoint ships `linear_attn.A_log` **and** `dt_bias` as **BF16** where the
   Qwen3.5 checkpoints ship fp32, and `GDNLinearAttn` loads with `assign=True`, so a bf16 `A_log`
   goes straight into the parameter — first real-weight prefill died in `gdn_prefill_wmma` with
   `RuntimeError: expected scalar type Float but found BFloat16`. The compensating upcast existed
   and was correct but lived as a **closure private to `Engine`**, invisible to every other consumer
   of `load_weight`. Extracted to `models.weight.cast_checkpoint_tensor`. **Shape-and-name parity
   does not imply dtype parity — check the shard header.**
8. `[RUN-2026-09-03]` **The next blocker is not model code.** The routed gather that makes full depth
   fit is a **monkeypatch inside `tests/qwen4exp_fulldepth_test.py`** (`mlp.forward = routed_forward`,
   `:497`); nothing under `python/minisgl/` knows about it. The real `Engine` has no gather, so it
   still needs the whole 79.53 GiB tier resident and `tests/qwen4exp_engine_test.py` runs on a
   **4-layer subset**. See §4 and `docs/measurements/QWEN4EXP_FIRST_RUN.md` §4.

---

## 1. What the two big unknowns actually are, after checking

### 1.1 Hyper-connections — math VERIFIED from source, numerics UNVERIFIED

Read directly from `sglang-upstream/python/sglang/srt/layers/hyperconnection.py`
(`_mix_compute`, `_combine_compute`, `GroupedGemmaRMSNorm.forward`), and the instantiation at
`srt/models/qwen4_exp.py:1257-1263` and `:1611-1617` (both `hc_per_branch_norm=True`):

```
# grouped norm: 4 groups of 2560 inside a 10240 vector, fp32 variance, Gemma (1+w) gain
hn = x.f32().reshape(..., 4, 2560); hn = hn * rsqrt(hn.pow(2).mean(-1,keepdim) + eps)
hn = (hn.flatten(-2) * (1.0 + w.float())).to(dtype)          # w is [10240], init ZEROS

# mix: 10240 -> 2560
g  = silu(linear(hn, W_down) / hc)          # W_down [320,10240], hc = 4
g  = sigmoid(linear(g, W_up))               # W_up   [10240,320]
out = (g.unflatten(-1,(4,2560)) * hn.unflatten(-1,(4,2560))).mean(dim=-2)

# combine: 2560 -> 10240, adds to the UNNORMED stream
R = residual.unflatten(-1,(4,2560))                          # unnormed
b = 2 * sigmoid(linear(normed_residual, W_inject) / hc)      # W_inject [4,10240] -> [T,4]
out = (R + block_output.unsqueeze(-2) * b.unsqueeze(-1)).flatten(-2)
```

Load-bearing scalars, none of which fails loudly if dropped: the `(1.0 + w)` Gemma gain (weights are
centred on zero — drop the `+1` and the stream scales to ~0), the `/hc` divisor **twice**, the
`.mean(-2)` (not sum), the `2 *` on the inject gate, and the fact that the gate multiplies the
**normed** streams while combine adds to the **unnormed** one.

**Still unverified:** that sglang matches HuggingFace's `modeling_qwen4_exp.py` (not readable here),
and any numeric result. T0.3 makes this step one.

### 1.2 QSA indexer — reference exists, cost is real, deferral is legitimate

`srt/layers/attention/qsa/`: `qsa_indexer.py` 652, `sparse_attn.py` 456, `mqa.py` 422, `kernel.py`
354, `metadata.py` 334, `graph_metadata.py` 241, `config.py` 201, `dsa_indexer.py` 369, `glue.py`
107 = 3,176 lines. Dependencies: Triton (portable in principle), **tilelang** in `mqa.py` (NV-only,
no AMD path upstream), `sgl_kernel.top_k`.

The deferral is justified by code, not optimism: `sparse_attn.py:70` `row_topk =
tl.minimum(topk, query_relative + 1)` and `:212` `row_topk = tl.minimum(topk, visible)`. With
`indexer_budget=2048`, any sequence at or below 2048 visible tokens selects everything, so dense
causal attention is **bit-equivalent**, not merely an approximation. That makes "coherent text at
<= 2048 context with dense full-attention layers" a real, defensible milestone that does not
pre-suppose the indexer.

It also means the indexer's blockers are unchanged and still large: a second KV pool for index keys,
and an extension of `BaseAttnBackend.forward` (`python/minisgl/attention/base.py:19`), which has no
parameter for a per-query selected-page set, across fa/fi/hip/rdna4/mla/trtllm and `HybridBackend`.

---

## 2. Capacity — compute this before writing model code

Per expert, this repo's post-`post_load` op layout (`_GroupedNvFp4Experts`, `layers/moe.py:390`):

| component | shape | bytes |
|---|---|---|
| `w13` packed | (1280, 320) int32 | 1,638,400 |
| `w13` scales | (160, 1280) fp16 | 409,600 |
| `w2` packed | (2560, 80) int32 | 819,200 |
| `w2` scales | (40, 2560) fp16 | 204,800 |
| **total** | | **3,072,000** |

Checkpoint on disk per expert is 2,764,800 B (scales are F8_E4M3). x 512 experts x 48 layers:
**70.31 GiB resident vs 63.28 GiB on disk, +7.03 GiB of pure fp16-scale inflation.**

Reachable: device tier ~5.85 GiB/rank x 2 = 11.7 GiB, pinned host 26 GiB/rank x 2 = 52 GiB
(the 62 GiB two-rank ceiling in `weights/config.py:55-70` is measured under a 12 GiB floor; box has
91 GB total / 58 GB available). Total ~63.7 GiB.

### 2.1 MEASURED (bring-up round 3) — the expert arithmetic holds, the body figure did not

`tests/qwen4exp_fulldepth_test.py` builds the real 48-layer model and reads the resident bytes off
the device, so these replace the estimates above:

| tier | estimated | **measured** | note |
|---|---|---|---|
| routed experts, all 48 layers | 70.31 GiB | **70.312 GiB** | the per-expert table above is exact |
| routed experts, **one** layer | — | **1.465 GiB** | this is the number that matters, see below |
| non-expert body | "~15 GiB" | **9.216 GiB** | 948 params, 0 unfilled |
| **total** | ~85 GiB | **79.53 GiB** | shortfall vs 63.7 reachable is **15.8**, not ~21 |

The "~15 GiB" was the on-disk size of the four `model-bf16-*` shards. Those shards also carry the
`mtp.*` head and the `model.visual.*` tower, neither of which this build constructs, so it
over-counted the body by ~5.7 GiB. Read the resident bytes off the model, not the shard sizes.

**A second, sharper correction: this row never gated correctness.** A decoder layer's experts are
dead the moment the layer returns, so the *live* expert working set is one layer — 1.465 GiB — and
the weights of a full-depth forward are 9.216 + 1.465 = **10.681 GiB**. Measured free-memory delta
across the build was **11.377 GiB on a 15.922 GiB card**; the extra 0.696 GiB is torch's caching
allocator holding the staging transients (the loader materializes 512 per-expert tensors and then
`torch.stack`s them), not weights, and it is reported rather than netted out because it is real
occupancy a serve would also have to carry. Either number leaves >4 GiB of headroom.
Round 3 ran all 48 layers that way (§3 Tranche 3 status). The layer-subset restriction that every
earlier test carries in its docstring was a consequence of loading the expert tier ALL AT ONCE, and
is not a property of the architecture or of this box.

Mitigation that is legal under the kernel-core policy: keep `_scales_op` in **e4m3 + a folded
global** instead of fp16. That is a WLoad/scale *policy* on the existing e2m1 core, not a fork — but
it is real kernel work and recovers 7.03 GiB of the 15.8 GiB serving gap (79.53 -> 72.5 vs 63.7).
It is no longer on the critical path for *correctness*; it is on the critical path for a serve.
See KILL-1.

### 2.2 The serving number this architecture actually turns on: 1.373 GiB/token

`num_experts_per_tok` is **10 of 512**, so a decode step reads
`3,072,000 B x 10 experts x 48 layers = ` **1.373 GiB of expert weights per token**, not 70.312.
That is the figure any residency decision for this model has to be argued against, and it happens to
land on top of a number `docs/WEIGHT_OFFLOAD_PLAN.md` §9 has already MEASURED: NVMe at the granule,
O_DIRECT, QD>=4 is **6.9-9.0 GB/s**, which it converts to "5-7 tok/s at 1.33 GB/token". Same traffic,
so the same conclusion transfers without re-measuring: **an NVMe-resident expert tier for this
checkpoint is a ~4.7-6.1 tok/s capacity feature** (projection, from their measurement and this
model's shapes — not measured here), and mmap is disqualified for it at 0.64-0.87 GB/s.

For contrast and to keep the round-3 harness in its place: it measures **178 s/token**, ~1,700x off
that ceiling, because it reads all 512 experts (not the routed 10) through the Python weight loader
(6,144 `get_tensor` calls + 1,536 scale folds per layer). None of that gap is new physics; it is the
difference between a correctness vehicle and the gather `weights/row_table.py` already implements
for the PLE table.

### 2.3 MEASURED (bring-up round 4): the routed gather is BUILT, and the traffic is real

`tests/qwen4exp_fulldepth_test.py --routed` stages only the experts a layer's tokens route to,
against the real checkpoint, at full depth. It replaces the estimate above with a measurement:

| quantity | round 3 (whole tier) | **round 4 (routed)** |
|---|---|---|
| experts staged per decode layer | 512 | **10.0** (= `num_experts_per_tok`, exactly) |
| expert bytes per decode token | 70.312 GiB | **1.236 GiB** |
| decode latency, this harness | ~178 s/token | **~1.3 s/token** |
| 48-layer greedy run, 12 tokens, wall | ~40 min | **52 s** |

**1.236 GiB/token, not 1.373.** Both are right and they measure different things: 1.373 uses the
3,072,000 B RESIDENT per-expert footprint (fp16 scales), 1.236 uses the 2,764,800 B ON-DISK footprint
(e4m3 scales), and it is the on-disk figure that an NVMe gather actually reads. The +11% is the same
fp16-scale inflation §2.1 charges against residency; it is not paid by a gather that folds after the
read. Prefill routes more, also measured: **32.5 experts/layer** for a 5-token prompt.

What is NOT measured: serving throughput. This gather reads through `safetensors` mmap and sustains
**1257 MiB/s**, which is the mmap path §2.2 already disqualified (0.64-0.87 GB/s there, ~1.2 here
warm) — so the ~1.3 s/token above is an I/O-bound harness number, not a serve. The projection in
§2.2 survives unchanged and is now the only remaining unknown in the chain: at the O_DIRECT granule
rate `WEIGHT_OFFLOAD_PLAN.md` §9 measured (6.9-9.0 GB/s), 1.236 GiB/token is **5.7-7.4 tok/s**. The
*traffic* and the *correctness* of the routed gather are measured; its *rate* still needs the
O_DIRECT granule reader.

**The route may not be re-derived — take it from the kernel.** The first version of this gather asked
`MoELayer._ep_route` which experts to stage and produced NaN logits, because the served e2m1 path
routes with `moe_hip.moe_route_align` (softmax + top-k INSIDE the kernel) and the two break EXACT
bf16 TIES at the k-th boundary opposite ways. See GATE-10.

---

## 3. Build order

Sizes: S ≈ <1 day, M ≈ 2-4 days, L ≈ 1-2+ weeks.

### Tranche 0 — prerequisites and gates (no model code, no GPU except where stated)

| # | Step | Size | Files | Verified by |
|---|---|---|---|---|
| T0.1 | **Capacity sizing.** Script the arithmetic above against `weights/sizing.py` / `host_capacity.py`; produce a resident-bytes figure per placement plan, including an e4m3-scale variant. | S | `tests/core/test_qwen4exp_sizing.py` (new), reads `weights/sizing.py`, `weights/config.py` | A number, checked in. Pass = a plan exists that fits; fail = KILL-1 fires and the scope drops to a layer-subset correctness vehicle. |
| T0.2 | **Download the 84 GB body.** It is NOT on disk — `/home/pat/.cache/hf-ple/` holds only the 51.2 GB PLE table + `model-bf16-00010.safetensors`; 337 GB free. | S | none (cache) | `du` + index tensor count 296,475 across 206 files; header of `model-bf16-00001.safetensors` read to close the inferred mixer shapes. |
| T0.3 | **HC numeric parity, CPU-only.** Port `GroupedGemmaRMSNorm` + `GatedResidual` to a standalone torch module; feed the 8 real HC tensors from layer 10 (already on disk) plus a fixed random input; diff against sglang's `_mix_compute`/`_combine_compute` executed in-process. | S | `python/minisgl/layers/hyperconnection.py` (new), `tests/qwen4exp_hc_parity_test.py` (new) | **max abs diff vs the reference < 1e-3 bf16 / exact in fp32, for mix and combine separately.** No GPU, no lease, no model body. This is the gate that retires the "inferred math" risk. |
| T0.4 | **Config readability.** Load the real `config.json` through `utils/hf.py` + `ModelConfig.from_hf`; assert `num_experts=512`, `num_experts_per_tok=10`, `moe_intermediate_size=640`, `shared_expert_intermediate_size=640`, `is_gdn_hybrid`, `gdn_layer_ids` = the 36 with `idx%4!=3`, `full_attn_layer_ids=[3,7,...,47]`, `num_kv_layers=12`. | S | `tests/qwen4exp_config_test.py` (new), `models/config.py` | Asserted values. Catches the silent-zero spelling traps (`num_experts_per_tok=0` routes nothing). |

**Tranche 0 is the whole "verify before the model works" answer.** T0.3 and T0.4 each fail or pass
on their own, with no engine, no GPU and (for T0.3) no downloaded body.

### Tranche 1 — loader and config, still CPU-verifiable

| # | Step | Size | Files | Verified by |
|---|---|---|---|---|
| T1.1 | **modelopt quant arm.** Normalize modelopt/NVFP4 to `method="compressed-tensors", ct_format="nvfp4-pack-quantized"` so everything downstream is untouched. `[RUN-2026-09-03] DONE, but the row scoped ONE defect and the real config had THREE.` (1) the format lives in `quant_algo: "NVFP4"`, not a `format` key — untranslated, `weight_is_e2m1` (MXFP4) becomes True instead of `is_nvfp4` and every routed expert routes into the **group-32 MXFP4 kernel over group-16 scales** with an orphaned `weight_scale_2`. (2) **the `ignore` entries are fnmatch GLOBS** — 10 of 13 (`*.self_attn.*`, `*.linear_attn.*`, `*.mlp.gate*`, `*.mlp.shared_expert.*`, `*.mlp.shared_expert_gate*`, `*hyper_connection*`, `*.ple.*`, `mtp.*`, `model.mtp.*`, `model.visual.*`). Fed to the historical **substring** matcher, `'*.self_attn.*' in name` is False for EVERY module name, so the whole ignore list evaporates and the bf16 attention / GDN / hyper-connection / shared-expert / PLE / gate modules all build **quantized against unpacked tensors**. Translated to anchored `re:` patterns via `_glob_to_ignore`. (3) headers that ship `quant_algo` with no `config_groups` need the weights/input_activations block synthesized, or NVFP4 silently takes the CT default group_size 32 instead of 16. Closed table: an unlisted algo returns None so `unparsed_quant_method` names it instead of guessing. | S | `python/minisgl/quant/config.py` | **DONE.** `tests/qwen4exp_quant_test.py`, `tests/qwen4exp_config_test.py`. |
| T1.2 | **Leaf-name remap.** `[RUN-2026-09-03] THIS ROW WAS WRONG AND IT WAS THE MOST EXPENSIVE ERROR IN THE PLAN.` ~~the repo expects `.weight_global_scale`; rename before the `nvfp4_bases` pre-pass~~ — **do NOT normalize `.weight_scale_2` to `.weight_global_scale`.** modelopt's `weight_scale_2` is the **RECIPROCAL** of compressed-tensors' `weight_global_scale`: the fold MULTIPLIES by one and DIVIDES by the other. Renaming makes the fold inherit the wrong direction, which is not a no-op and does not fail loudly — MEASURED on layer-0 experts (`weight_scale_2 = 2.078102e-4`, e4m3 blocks 7..256), dividing yields per-group scales to 4.2e6 and **99,554 of 102,400 overflow fp16 to `inf` -> ALL LOGITS NaN at `L0.mlp`**. The spelling is load-bearing *information*. `fold_nvfp4_scale` now takes a **required** `global_field=` naming the suffix the global was read from and looks the direction up in `NVFP4_GLOBAL_SCALE_IS_RECIPROCAL`; an unlisted spelling raises rather than guessing, and the folded fp16 result is finiteness-checked so this bug class is a loud load-time error. The `.weight` -> `.weight_packed` half of the row was correct. | S | `python/minisgl/quant/nvfp4.py`, `python/minisgl/models/weight.py` | **DONE.** `tests/qwen4exp_loader_test.py` §[3b]: dequantized real layer-0 experts within 10x of the same layer's real bf16 shared-expert anchor — measured ratio **1.475**. |
| T1.3 | **`_load_qwen4_exp_weight` branch, placed BEFORE the `is_gdn_hybrid` test at `weight.py:1423`.** qwen4_exp *is* a GDN hybrid, so with no branch it silently routes into the Qwen3.5 loader. Must skip `model-plefp8-*.safetensors` (51.2 GB — served by `row_table.py`, never streamed) and `model.visual.*`. | M | `python/minisgl/models/weight.py` | Loader emits zero `ple.ple_embedding.*` keys and zero `visual.*` keys. |
| T1.4 | **Loader key-set parity test** — the `tests/gemma4_loader_test.py` pattern: build the model on `meta`, take `state_dict()`, stream `load_weight()`, assert emitted keys match **exactly** on name + shape + dtype. | M | `tests/qwen4exp_loader_test.py` (new) | Exact set equality. CPU-only, no GPU. This is the acceptance test for T1 and T2 together. |

### Tranche 2 — model skeleton (meta build)

| # | Step | Size | Files | Verified by |
|---|---|---|---|---|
| T2.1 | **`HyperConnection` BaseOP** — promote T0.3's module into the layer tree; `hc_norm` as a grouped RMSNorm, three `LinearReplicated` (never TP-sharded: the wide stream is replicated, the only allreduce is on the 2560-wide block output). Grouped norm decomposes as `_rms_norm(x.view(T,4,2560), None, eps) * (1 + w.view(4,2560))` using the existing `weight=None` path (`layers/norm.py:9-25`). | S | `python/minisgl/layers/hyperconnection.py`, `python/minisgl/layers/norm.py` | T0.3 parity test, re-run through the BaseOP. |
| T2.2 | **`models/qwen4_exp.py`** — decoder stack with the 4x-wide (10240) stream. Stream init is `cat([embed]*4)`. **No `input_layernorm`/`post_attention_layernorm`, no final `norm`** — `hyper_connection_mixer` (3 tensors, no `block_inject_weight`) *is* the final norm, and `lm_head` consumes `mixer.mix(h)[0]`. Reuse `GDNLinearAttn` and `Qwen3_5Attn` verbatim; reuse `Qwen3_5MoeSparseBlock`'s shape. | L | `python/minisgl/models/qwen4_exp.py` (new) | Meta-build test (`tests/gemma4_build_test.py` twin): the model constructs on `meta` and its `state_dict()` key set is exactly the checkpoint's native key set. Adding any final `norm` fails here. |
| T2.3 | **ModelConfig fields + registry entry.** `hc_count`, `hc_lowrank`, `ple_layer_ids`, `indexer_*` (5). One dict line in `models/register.py`. | S | `python/minisgl/models/config.py`, `python/minisgl/models/register.py` | T0.4 + T2.2. |
| T2.4 | **`renormalize` decision.** Config has no `norm_topk_prob`; `from_hf` defaults it False, but `Qwen3_5MoeSparseBlock` hardcodes True. Read it out of the reference, do not default. | S | `python/minisgl/models/qwen4_exp.py` | A cited line in the reference implementation. Wrong choice = degenerate text, no crash. |

### Tranche 3 — first coherent forward (the milestone)

> **STATUS (bring-up round 3, 2026-09-03): ALL 48 LAYERS RUN ON REAL WEIGHTS AND THE OUTPUT IS
> CORRECT.** `tests/qwen4exp_fulldepth_test.py`, one RX 9070 XT, 11.377 GiB resident, the real
> 51.2 GB NVMe PLE table live, 12 greedy decode steps:
>
> > `The capital of France is` **`Paris. The capital of Germany is Berlin. The capital of`**
>
> Grammatical, factually right twice, no degeneration signature, and the next-token distribution is
> SHAPED rather than spiked (` Paris` 15.72, then `\n\n` 14.26 / ` a` 14.16 / ` London` 12.72 — the
> runner-up city is a city). That is the first statement anyone has been able to make about this port that is
> about the PORT rather than about a truncation, and it retires the whole "is the architecture
> transcribed right" question — the hyper-connection stream, the 36 GDN layers, the sigmoid output
> gate, the NVFP4 expert fold, the n-gram PLE block and the dense full-attention layers are all
> jointly right, because none of them can be wrong and still yield that token.
>
> **How, given KILL-1:** only ONE layer's experts are live at a time (1.465 GiB), so the resident
> cost is body + one layer, not body + 48 layers. See §2.1. The expert tier is restreamed from its
> own four shards before each layer runs. **This is a correctness vehicle, NOT a serve** — it
> re-reads 70 GiB per forward pass (~3.7 s/layer, ~178 s/token) and is streaming-as-caching, which
> `docs/WEIGHT_OFFLOAD_PLAN.md` rejects for serving. Nothing here is a throughput result.
>
> **The streamer is not trusted on inspection.** `--validate` builds the same 4-layer subset twice,
> resident and streamed, and requires **bit-identical** logits; measured `max|delta| = 0` over three
> steps with identical greedy ids. A mis-mapped layer would be fluent garbage with no error, so this
> A/B, not the reading of the code, is why the full-depth number above is quotable.
>
> **SAMPLED too, because greedy never settles quality (repo rule).** Second full-depth run,
> `--temperature 0.8`, 14 decode steps, same harness:
>
> > `In a small village nestled between two mountains, there` **`lived a young girl named Elara. She
> > had hair like spun silver`**
>
> Fluent, on-topic narrative prose with none of the four degeneration signatures (no loop, no
> letter-spelling, no mid-word switch, no token noise). **GATE-6 is discharged at <= 2048 context**,
> which per §1.2 is bit-equivalent to the sparse path, so it is a real result and not a stand-in.
> 720 layer-stagings, 1054.7 GiB streamed, 0 non-finite logits across 15 forwards.
>
> STILL OPEN from this tranche: T3.3 (mrope) is argued, not measured. T3.2 (`gate_act`) is now
> positively confirmed (see the T3.2 row).
>
> (Round 2, retained: `LLM`/`Engine`/`Scheduler` boot, chunked prefill, sampler, detokenize, two
> concurrent requests — through the real `Engine`, which the round-3 harness deliberately is not.)

Scope: **text-only, single card, dense full-attention (no indexer), context <= 2048, eager, TP=1.**

| # | Step | Size | Files | Verified by |
|---|---|---|---|---|
| T3.1 | **GDN + MoE + HC forward, prefill only.** **DONE (round 3, full depth).** | M | `models/qwen4_exp.py` | `"The capital of France is"` -> `" Paris. The capital of Germany is Berlin. The capital of"`, all 48 layers, real weights, real PLE table, 11.377 GiB resident on one card. Sampled re-check is T3.4. |
| T3.2 | **`gate_act` positive confirmation.** **DISCHARGED (round 3).** | S | `python/minisgl/gdn/layer.py:311-319`, image `minisgl-rdna4:m1b-20260903` | POSITIVE, and it is the checkpoint's own value that makes it so. `_gate_args()` returns `(1,)` for `sigmoid` — non-empty, unlike the `silu`/`swish` case which returns `()` precisely so an old .so still works — and it is splatted at all SIX gated-norm call sites (`:333, :364, :394, :640, :654, :679`). A `gdn_hip` without the parameter fails the op schema match on argument count; 48 layers x 13 forwards ran with no such error, so the .so's schema HAS the argument. That is confirmation the arg is accepted, not inherited assumption — the residual (a kernel that takes it and ignores it) is a kernel bug, not a plumbing one. |
| T3.3 | **mrope check.** `mrope_interleaved: true`, `mrope_section: [11,11,10]`; `models/config.py` deliberately drops mrope. | S | `models/config.py:739-742`, `layers/rotary.py:38-41` | **ARGUED, NOT MEASURED — treat as open.** The arithmetic lines up: `partial_rotary_factor 0.25 * head_dim 256` -> `rotary_dim = 64` -> 32 frequency slots, and `sum([11,11,10]) == 32`, so the sections partition exactly the frequencies plain rope would build. For TEXT-ONLY the t/h/w position components are all equal, so any mrope section assignment (interleaved or blocked) collapses to the same angles. Corroborated end to end by the round-3 result — 48 layers of wrong positions do not emit `" Paris."` — but no per-element diff against the reference was run. |
| T3.4 | **Decode loop + short-context coherence. DONE (round 3, full depth, greedy AND sampled).** | M | `models/qwen4_exp.py` | 12 greedy steps -> `" Paris. The capital of Germany is Berlin. The capital of"`; 14 steps at `--temperature 0.8` -> `" lived a young girl named Elara. She had hair like spun silver"`. Per §1.2 dense at <= 2048 is bit-equivalent to the sparse path, so this is a genuine correctness result, not a stand-in. **ROUND 4 extends this to 200 sampled tokens (see GATE-6), which also covers the multi-TURN gap: the generation crossed `<|im_end|>` into an `<|im_start|>assistant` turn with a well-formed `<think>` block on its own.** Still not covered: anything past the 2048 indexer budget. |

### Tranche 4 — PLE n-gram block (layer index 1)

> **STATUS (bring-up tranche 1b, 2026-09-03): T4.1 / T4.2 / T4.3 LANDED, plus the T6.2 staging
> design.** The reference implementation was located — `transformers/models/qwen4_exp/` on
> `huggingface/transformers` `main` — so the hash and the block are **transcribed and verified**,
> not inferred. `build_layer_multipliers(248320, 3, 0, seed=1234)` reproduces the checkpoint's own
> `layer_multipliers` tensor `[23703573157769, 20109073645365, 8052911324071]` exactly, and the 16
> derived head primes reproduce `ngram_heads_vocab_sizes`; both equalities are asserted in
> `tests/qwen4exp_ple_hash_test.py` against a checked-in fixture
> (`tests/fixtures/qwen4exp/ngram_meta.json`), so they hold with nothing downloaded.
>
> Code: `python/minisgl/ple/` (hashing / per-sequence state / NVMe staging / per-batch runtime) and
> `Qwen4ExpPLE` + `GroupedRMSNorm` in `models/qwen4exp.py`, which no longer refuse. Tests:
> `tests/qwen4exp_ple_hash_test.py` (21) and `tests/qwen4exp_ple_test.py` (15), the latter including
> a cudagraph capture+replay of the decode path and a run against the real 51.2 GB table.
>
> Corrections to the table below, found by enumerating the checkpoint rather than reading docs: the
> projections are named `key_proj` / `value_proj`, the norms are `norm_key` / `norm_query` /
> `norm_conv`, and the short conv is **dilated by `ngram_size` = 3**, so its per-sequence state is
> `(4-1)*3 = 9` columns, not `k-1 = 3`. Also: the block's output is ADDED to the wide stream before
> the attn mix, and the conv is fed the NORMED gated value while the residual adds the UNNORMED one.
>
> **UPDATE (bring-up round 2, 2026-09-03): the engine wiring LANDED.** `Engine` now builds the
> `PLERuntime` next to the GDN state cache (same `max_running_req + 2` slot count, since the PLE
> state is indexed by the GDN slot id) and reserves its device bytes before the KV pool is sized;
> `Scheduler._stage_ple` stages every batch in `_finish_prepare` and `_forward` commits the n-gram
> history after the forward. Two constraints came with it and are enforced in code, not prose:
> `resolve_prefix_cache` forces **naive** for any model with `ple_layer_ids` (the recurrent-radix
> snapshot store clones GDN buffers only — it would restore GDN state at a prefix boundary and leave
> the PLE conv window and token history at their zero/EOS seed), and `run_forever` refuses the
> **overlap loop** (the n-gram hash reads host token ids *before* the forward, which the overlap
> order has not committed yet). Verified end to end on a 4-layer subset with real weights and the
> real 51.2 GB table: `tests/qwen4exp_engine_test.py`.

| # | Step | Size | Files | Verified by |
|---|---|---|---|---|
| T4.1 | Depthwise conv1d [10240,1,4] + key/value projections + three 10240-wide norms, operating on the **wide** stream (this is why `ple.conv1d` is 10240 rows), added before the attn mix. The 51.2 GB gather is DONE — use `open_qwen4exp_ngram_table`, do not reimplement. | M | `python/minisgl/models/qwen4_exp.py`, `python/minisgl/weights/row_table.py` (read-only) | Block output diffed against the reference on one token batch. |
| T4.2 | **n-gram hashing** (ngram_size 3, `layer_multipliers`, 16 per-head prime vocab offsets) feeding `NgramHeads.row_ids`. | S | `models/qwen4_exp.py` | Hash values diffed against the reference for a fixed token sequence. |
| T4.3 | **Off-by-one guard:** `ple_layer_ids: [2]` is 1-BASED -> layer index **1**. Putting it on layer 2 loads cleanly and degrades quality silently. | S | test | Assert the block is attached to index 1. |

### Tranche 5 — QSA sparse indexer (long context)

Unblocks context > 2048. Largest single item; previously scoped as the dominant cost and nothing
found here contradicts that, though a Triton reference lowers it from "research" to "port".

| # | Step | Size | Files | Verified by |
|---|---|---|---|---|
| T5.1 | **`BaseAttnBackend` ABI extension** for a per-query selected-page set, across fa/fi/hip/rdna4/mla/trtllm + `HybridBackend`, plus `prepare_metadata`/capture/replay. | M | `python/minisgl/attention/*.py` | Existing models still pass their tests with the widened signature (default = dense). |
| T5.2 | **Index-key KV pool** — a second pool the engine does not have. | M | `python/minisgl/kvcache/` | Allocation + eviction tests mirroring the paged pool's. |
| T5.3 | **Index score + top-k kernels** (Triton port of `qsa_indexer.py`/`sparse_attn.py`; the tilelang `mqa.py` path has no AMD equivalent and must be rewritten or routed around). | L | `python/minisgl/attention/qsa/` (new), possibly `rdna4-hip-kernels` | Selection parity vs a dense reference at seq <= 2048 (must select everything), then quality at 4k/8k/32k. |
| T5.4 | Sparse paged attention consuming the selection. | L | as above | Long-context coherence, sampled. |

### Tranche 6 — graph capture (**required before "complete"** per repo policy)

`[CAPTURE-2026-09-04]` **T6.1 and T6.2 are DONE and verified** on the full 48-layer TP=2 offloaded
serve. T6.3 stays open with T5 — there is no indexer to capture yet. Write-up + raw artifacts:
[`docs/measurements/QWEN4EXP_GRAPH_CAPTURE.md`](measurements/QWEN4EXP_GRAPH_CAPTURE.md).

| # | Step | Size | Files | Verified by |
|---|---|---|---|---|
| T6.1 | ✅ **DONE.** Capture the decode step. Every hidden buffer is 4x wide (10240) — size before capture, not after an OOM. | M | `python/minisgl/engine/graph.py`, `models/qwen4exp.py` | ~~Captured vs eager logits match **bit-exactly**~~ — **that verification criterion was wrong and was corrected, not quietly relaxed.** `attn_decode`'s split-K policy is keyed on the page-table WIDTH, and the captured table is the full `aligned_max_seq_len` while eager's is the batch's own `max_seqlen_k` — different `num_splits`, different fp32 reduction order, and at `num_splits == 1` a different kernel. That is pre-existing and shared with every captured model in this engine. The gate is **identical greedy token ids**: 12/12 at bs=1 and 24/24 at bs=2, both ranks, 48/48 layers, measured as a one-boot A/B (`--parity-steps`, `max_graph_bs=0` as the eager leg). Replay stability: 595 replays/leg with `repro_engine_is_reproducible=true`. Also needed the `hip` attention backend — `rdna4` raises on capture. |
| T6.2 | ✅ **DONE.** **PLE staging.** | M | `python/minisgl/ple/graph_capture.py` (new), `engine/graph.py` | The premise was half right: the *gather* is host I/O, but it already sat outside the forward — `Scheduler._stage_ple` runs in `_finish_prepare`, `PLERuntime.commit_staged` after the forward, and **neither had to move**. What was missing was that the capturer had no way to STAGE a batch at all, so `Qwen4ExpPLE.forward` raised on the capture-time warmup and capture was structurally impossible. `PLEGraphCapture` stages the synthetic batch on the reserved NULL slot 0, DISCARDS it after capture (never commits), and on every replay asserts both the row count (`padded_size`, i.e. over `padded_reqs` including cudagraph padding rows) and the `data_ptr()` identity of embeddings / slot index / conv state. Commit-exactly-once PROVEN arithmetically via prepare/commit/discard/commit_noop counters: boot 2/0/2/0 at L48, per-leg 12/12 with `commit_noops=0`. |
| T6.3 | ⬜ **OPEN, gated on T5.** Selected-page set in persistent static buffers refreshed in place (the `GDNGraphCapture` discipline for `state_indices`). | M | `attention/qsa/`, `engine/graph.py` | Capture + replay with the indexer live. **Meanwhile**, capture introduced a silent correctness hazard that T5 will inherit: the QSA indexer-budget refusal is host Python inside `Qwen4ExpAttn.forward`, so under capture it ran once — at capture, over `dummy_req` rows with `device_len=1` — and never again. Restated per replay through a new general `BaseLLMModel.prepare_for_replay` seam. |

### Tranche 7 — performance

| # | Step | Size | Files | Verified by |
|---|---|---|---|---|
| T7.1 | **Measure HC decode bandwidth** — 1.281 GB BF16 read every step (**measured resident**, round 3), ~1.81 ms/token at the 706.6 GB/s ceiling ≈ a **552 tok/s hard ceiling from HC alone**; the 480 activated experts are 1.475 GB ≈ 479 tok/s, so HC is **86.9% of** expert traffic (~~more than~~), and the two together cap decode at ~256 tok/s before anything else is counted. The BYTES are now measured; the tok/s are still a CALCULATION against the roofline — measure them. | S | profiling | A trace. Not a projection. |
| T7.2 | **Fused HC mix/combine HIP kernels.** Eager is ~9 dispatches x 96 blocks ≈ 850-900 extra launches/step. Do **not** port `hc_mix_triton.py`'s persistent-CTA spin barrier — a spinning kernel is exactly this box's GPU-wedge signature (100% use, ~0% memory traffic, never returns). Its fast path is `sm_100`-gated anyway. | L | `rdna4-hip-kernels`, `layers/hyperconnection.py` | A/B with provenance asserted per repo rule; diff the `engaged()` ledgers per leg. |
| T7.3 | **Grouped RMSNorm as a `group_size` parameter on the existing tail_hip rms_norm core** — a policy on the shared core, never a new kernel. Removes one 10240-wide read+write per HC block x 96. | M | `rdna4-hip-kernels`, `layers/norm.py` | Parity vs the 2-op decomposition + a timing A/B. |

### Tranche 8 — deferred features, in order of likely want

MTP head (T8.1, M), TP=2 (T8.2, M), expert offload arena integration (T8.3, M), vision (T8.4 — not
planned). See §4.

---

## 4. Deliberately DEFERRED — not done, not pretended done

| Item | Status | Why it is safe to defer | What it costs later |
|---|---|---|---|
| **Vision** `model.visual.*` (27 blocks) | **Not planned.** | `_QWEN35_SKIP_PREFIXES` already drops it; every other multimodal ckpt here serves text-only. | N/A — out of scope. |
| **MTP** `mtp.*` | **Deferred to T8.1. NOT implemented.** | `config.py:556-558` forces `mtp_num_hidden_layers=0` unless `spec_algorithm=='mtp'`, and the remap returns None for `mtp.*` keys, so a text-only build never builds or loads it. | New shapes: experts are **fused** (`mlp.experts.gate_up_proj` single tensors, fp8-block) unlike the backbone's per-expert files; HC inside the head; the seed is the **10240-wide pre-mixer** stream, not the post-mixer hidden. `mtp.*` is top-level — a `model.` prefix filter misses it. Needs `force_no_ep=True`. |
| **Graph capture** | `[CAPTURE-2026-09-04]` **NO LONGER DEFERRED — DECODE capture is DISCHARGED** at 48/48 layers, TP=2, arena engaged. ~~`[RUN-2026-09-03] UNDISCHARGED and never attempted for this model` … the standing guidance is **serve eager until it has been**~~ — both of those, and the build banner that said so, are superseded. Still deferred: **prefill and spec-verify** capture (prefill is eager everywhere in this engine; spec-verify is moot while `--spec-algorithm mtp` is refused for this architecture) and **T6.3** (the selected-page set, gated on T5). | It cost +2.5% (81.87 → 79.86 ms/decode step), which is the honest answer to "what did it buy": decode here is **PCIe-bound** — 568.32 MB/token/rank of host-resident experts, 39.25 ms = 49% of the step at card 1's Gen4 x8 — so there was ~2 ms/step of launch overhead to remove and no more. Keep capture because it is a merge requirement and free, not as a throughput lever. | The T4 design bet paid off in an unexpected direction: the PLE *device* arithmetic was always shape-static, so nothing had to be restructured — the gap was purely that the capturer could not STAGE a batch. What was NOT anticipated: the `rdna4` backend cannot capture at all (use `hip`), and host-Python guards inside `forward()` silently stop guarding under capture, which needed a new general `BaseLLMModel.prepare_for_replay` seam. Details: [`docs/measurements/QWEN4EXP_GRAPH_CAPTURE.md`](measurements/QWEN4EXP_GRAPH_CAPTURE.md). |
| **Expert offload arena** | **Deferred to T8.3 — and it is now the #1 item, because it is the ONLY thing between this port and a serve.** | `attach_seams` needs only stock `MoELayer` instances at unique dotted paths — no per-model registration. So building the model correctly is sufficient preparation, and round 3 showed the model IS built correctly at full depth. | Gated on KILL-1(b) — 79.53 vs ~63.7 GiB — which the pinned-host arena alone does NOT close. The shape that fits this model is a **routed gather**, not a whole-stack placement. **ROUND 4 BUILT THAT GATHER AND IT WORKS** (§2.3): 10.0 experts/layer, 1.236 GiB/token, bit-identical logits, all 48 layers, 200 sampled tokens. `[RUN-2026-09-03]` **But it is a TEST-HARNESS MONKEYPATCH, not engine code** — `tests/qwen4exp_fulldepth_test.py:497` does `mlp.forward = routed_forward`, and `grep -rn 'route_align' python/minisgl/` returns only the kernel wrapper. So T8.3 is TWO items, not one: **(i) promote the gather into a real `MoELayer` seam** (`weights/moe_interpose.py` has the seam shape; `attach_seams` needs only stock `MoELayer` instances at unique dotted paths, so no per-model registration), taking its route from `_route_align` per GATE-10; **(ii)** "move the measured mechanism off `safetensors` mmap (1257 MiB/s) onto the O_DIRECT granule reader `WEIGHT_OFFLOAD_PLAN.md` §9 measured at 6.9-9.0 GB/s", plus prefetch (the gather is currently synchronous inside the layer, so its latency is fully exposed; the route for layer L is knowable only after L-1, but the ~1 ms of GEMM per layer is a real overlap window). Also still true: `convert_nvfp4_moe`'s per-expert Python loop runs 49,152 iterations at this scale — round 4 cut it to 10 experts/layer, which is why the harness went from 365 MiB/s to 1257 MiB/s, but it is still a Python loop on the critical path. |
| **TP=2** | **Deferred to T8.2. `[RUN-2026-09-03]` NOT implemented and now REFUSED AT CONSTRUCTION**, in both the model and `_load_qwen4_exp_weight` (no shard function for this key set) — a silently-wrong TP=2 build is no longer reachable. Pinned by `tests/qwen4exp_build_test.py` §[9]. | All shapes divide 2 (nq 24, nkv 2, GDN 16/48, 512 experts). Not a shape blocker. | The HC wide stream and the rank-320 mixer have **no established sharding rule**; replicate is the safe default at 4x residual allreduce cost. |
| **QSA indexer** | **Deferred to T5.** | Verified bit-equivalence to dense at seq <= 2048 (§1.2). | Caps usable context at 2048 out of 262,144 until done. |

---

## 5. KILL / GATE table

| ID | Gate | Measurement | Threshold | If it fails |
|---|---|---|---|---|
| **KILL-1** | ~~Does the model fit at all?~~ **SPLIT IN TWO — the gate was asked at the wrong granularity.** (a) does a full-depth FORWARD fit? (b) does a SERVE fit? | Round 3 measured both off the real model (§2.1). | (a) **DISCHARGED: 11.377 GiB resident of 15.922 GiB**, because only one layer's experts (1.465 GiB) are live at a time. (b) **STILL FAILS: 79.53 GiB vs ~63.7 GiB reachable** (gap 15.8 GiB, was stated as ~21). | (a) needs nothing — all 48 layers run (Tranche 3 status). (b) **ROUND 4: the gate was still asked at the wrong granularity.** "Does 79.53 GiB fit in 63.7 GiB" is the question for a RESIDENT tier, and this model does not need one: a decode step reads 1.236 GiB of experts (§2.3, measured), so the serving question is a BANDWIDTH question, not a capacity one — "can the gather sustain 1.236 GiB/token", against the 6.9-9.0 GB/s O_DIRECT rate already measured. e4m3 scales remain worth doing (7.03 GiB, a scale *policy* on the shared core) but they are now an optimisation of a working shape rather than the thing that unblocks a serve. Do **not** cite this row to justify a layer subset in a CORRECTNESS test any more; that inference was wrong and it is what kept every earlier test at 4 layers. |
| **GATE-2** | **Is the HC math right? `[RUN-2026-09-03]` DISCHARGED — on REAL tensors.** | T0.3 CPU parity of mix/combine vs the sglang reference on the real layer-10 `attn_hyper_connection.*` tensors (`weights=real`; the random fallback was not taken). | max abs diff < 1e-3 (bf16). | **PASSED, 12/12.** fp32 is **bit-exact — `max\|d\| = 0.000e+00`, 0.00 ulp — for `hc_norm`, `mix` and `combine` separately.** bf16 differs by exactly **2.00 ulp** (`mix` 1.562e-02, `combine` 3.125e-02, `hc_norm` 6.250e-02), which is the one extra rounding this repo's shared `_rms_norm` core takes returning the normalized value in `x.dtype` before the `(1+w)` gain. The test `ast`-extracts the reference source itself rather than comparing to a transcription, and a companion test perturbs each of the 5 silent-failure scalars individually to prove the parity is not passing by insensitivity. **GATE-3 (sglang vs HF) remains open** — this is parity against sglang only. |
| **GATE-3** | **Does sglang match HF?** | Diff a single-layer torch forward against `transformers` `modeling_qwen4_exp.py` on CPU. | Agreement to bf16 tolerance. | Re-derive from HF; sglang is corroboration, not ground truth. Cheap, GPU-free — run it early. |
| **GATE-4** | **Loader key-set parity.** | T1.4 meta-build state_dict vs streamed `load_weight` keys. | Exact match on name+shape+dtype. | Fix remap before any GPU time. Failure here is loud and cheap; failure downstream is neither. |
| **GATE-5** | **`gate_act` actually consumed. `[RUN-2026-09-03]` DISCHARGED — and it was a NON-RISK, discharged for free.** | T3.2. No probe was needed: `_gate_args()` returns `(1,)` for `sigmoid` — **non-empty**, unlike the `silu`/`swish` case which returns `()` precisely so an old .so still loads — and it is splatted at all six gated-norm call sites, so a `gdn_hip` without the parameter fails the op schema match on **argument count**. | Positive confirmation, not inherited assumption. | **PASSED.** 48 layers x 13 forwards ran with no schema error, so the .so HAS the argument. The residual (a kernel that takes it and ignores it) is a kernel bug, not a plumbing one. |
| **GATE-6** | **Coherence at <= 2048, SAMPLED. DISCHARGED (round 3).** | Full-depth generation at `--temperature 0.8`, 14 steps, real weights + real PLE table. | Grammatical, on-topic, no letter-spelling / loop / mid-word-switch signatures. | **PASSED:** `"In a small village nestled between two mountains, there"` -> `" lived a young girl named Elara. She had hair like spun silver"`. Worth recording that it passed DESPITE the served scheme being **W4A8, not the declared W4A4** (gfx1201 has no FP4 math; the checkpoint's `input_scale` calibration is dropped) — that deviation is evidently not quality-fatal at this length. Re-open if longer generations degrade; triage with the degeneration-signature table before blaming the port. **ROUND 4: re-run at 200 tokens (the routed gather made it affordable — 5 min instead of ~10 h) and it holds, far more strongly than "no degeneration".** Same prompt/seed/temperature: the model finished the story, emitted `<|im_end|>`, opened `<|im_start|>assistant` + `<think>`, and inside the think block correctly summarised *its own* preceding text ("a village between mountains, a young girl with silver hair and curious eyes, her habit of watching sunsets") before closing `</think>` and resuming in-style. That is long-range self-reference, chat-template structure and special-token handling, not just local fluency; 200/200 decode steps finite, none of the four degeneration signatures. Still bounded at <= 2048 by GATE-9/T5. |
| **GATE-7** | **Is HC affordable?** | T7.1 measured decode bandwidth. | Compare against the calculated 552 tok/s HC-only ceiling. | If measured throughput sits near the HC ceiling, T7.2/T7.3 become mandatory, not optional, and any throughput promise made without them is wrong by ~2x. |
| **GATE-8** | **MoE tile choice at inter=640.** | A/B of `_moe_block_m` at the decode point. | Within ~10% of the analytic oracle. | `kernels.py:365-380` already records the current rule is 0.76-0.85x the oracle at M<=32. inter=640 (320 at TP=2) is much narrower than the Qwen3.6-35B shapes the surface was tuned on — expect wrong and budget the A/B. |
| **GATE-9** | **QSA selection parity.** | T5.3 at seq <= 2048. | Selects **all** visible tokens, i.e. bit-equal to dense. | The indexer is wrong; dense is the known-good oracle here, use it. |
| **GATE-10** | **Anything that PREDICTS which experts the kernel will read must take the route from `quant.kernels._route_align`, never from `MoELayer._ep_route`.** Binds the routed gather (§2.3), and any future offload prefetch or expert-residency policy. | Round 4, `--routed --validate`: a gather driven by `_ep_route` produced NaN; driven by `_route_align` it is bit-identical to the resident build. | Logits bit-identical to a build with the whole tier resident. | `_ep_route` does `torch.softmax().topk()`; the served e2m1 path routes with `moe_hip.moe_route_align` (softmax+top-k in-kernel). They break **exact bf16 ties at the k-th boundary** opposite ways — kernel keeps the LOWER expert index, `torch.topk` the higher. Ties are common, not exotic: bf16 gate logits have 8 mantissa bits spread over 512 experts. MEASURED, layer 0 of an 8-token prefill: experts 324 and 366 both scored -5.09375 at ranks 9/10; 2 of 8 MoE calls in that one forward diverged by one expert. Self-consistent for EP (which feeds its own ids to the kernel), so this is **not** an EP bug — but it does mean an EP and a non-EP serve can pick a different expert on a tie row. `_ep_route`'s docstring asserted the opposite and named a function (`w4a8_moe._route`) that no longer exists; corrected in round 4. |

---

## 6. Standing repo rules that bind this work

- **No env-gating on merge** — the worktree is the isolation.
- **Graph capture required** before the feature is complete (T6).
- **One kernel core per shape** — e4m3 scales and the grouped norm are *policies* on existing cores
  (T7.3, KILL-1 mitigation), never forked kernels.
- **Never quality-test at temp=0**; greedy fakes degeneration that mimics a quant bug.
- **A/B must assert provenance**, and diff the `engaged()` ledgers per leg (T7.2).
- **Measure, don't project** — every number in §2 and §7.1 is arithmetic and is labelled as such.
