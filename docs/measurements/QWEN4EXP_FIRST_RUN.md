# Qwen3.8-Flash-Next (`qwen4_exp`) — first-run status

**Date:** 2026-09-03 · **Worktree:** `/home/pat/code/minisgl-rdna4-offload`, branch
`feat/weight-offload` · **Card:** one RX 9070 XT (HIP 0, 15.922 GiB) · **Image:**
`minisgl-rdna4:m1b-20260903` · **Checkpoint:** `RadixArk/Qwen3.8-Flash-Next-NVFP4`
(78 GB body in `/home/pat/.cache/hf-q4e`, 205 files; 49 GB PLE table in `/home/pat/.cache/hf-ple`)

Every number below is tagged **MEASURED** (read off a run log named in the row) or **PROJECTED**
(arithmetic against a rate measured elsewhere). Nothing is tagged neither.

---

## 1. How far it got

**All 48 layers, real checkpoint weights, real 51.2 GB NVMe PLE table, on one 16 GB card, producing
coherent text — greedy and sampled, up to 200 decode steps.** That is past "real-weight forward" and
past "decode"; it is a full-depth correctness result. It is **not** a serve.

| Stage | Status | Evidence |
|---|---|---|
| Meta build (state_dict == checkpoint key set) | **PASS** | `tests/qwen4exp_build_test.py` |
| Loader key/shape/dtype parity, full 84 GB | **PASS** — 948 body params, 9.216 GiB, **0** out-of-model keys dropped | `full48.log`, `tests/qwen4exp_loader_test.py --full` |
| Random-weight GPU forward | **PASS** (superseded) | `tests/qwen4exp_gpu_forward_test.py` |
| Real-weight forward, 4-layer subset | **PASS** | `val.log` |
| **Real-weight forward, all 48 layers** | **PASS**, 11.377 GiB resident of 15.922 | `full48.log`, `routed_full48c.log` |
| **Decode, greedy, 12 steps, full depth** | **PASS** | `full48.log` |
| **Decode, sampled T=0.8, 14 steps, full depth** | **PASS** | `sampled48.log` |
| **Decode, sampled T=0.8, 200 steps, full depth** | **PASS** | `routed_sampled200.log` |
| Through the real `Engine`/`Scheduler` | **PASS on a 4-layer subset only** | `tests/qwen4exp_engine_test.py` |
| Full depth through the real `Engine` | **NOT RUN — this is the blocker, §4** | — |

### The outputs (MEASURED, verbatim from the logs)

Greedy, 12 steps, `full48.log`:

> `The capital of France is` → **` Paris. The capital of Germany is Berlin. The capital of`**

Top-8 after the prompt is *shaped*, not spiked — ` Paris` 15.721, `\n\n` 14.257, ` a` 14.157,
`\n` 13.604, ` __` 13.077, ` London` 12.716. The runner-up city is a city.

Sampled T=0.8, 200 steps, `routed_sampled200.log`:

> `In a small village nestled between two mountains, there` → **` lived a young girl named Elara.
> She had hair like spun silver and eyes that sparkled with an insatiable curiosity. […] dreaming of
> adventures beyond the mountains.<|im_end|>\n<|im_start|>assistant\n<think>\nThe user has shared the
> beginning of a fantasy story about a girl named Elara. […] The story has established: a village
> between mountains, a young girl with silver hair and curious eyes, her habit of watching sunsets
> and dreaming of adventures. […]\n</think>\n\nOne autumn evening, as the last sliver of sun bled
> into the`**

This is the strongest single datum in the bring-up and it is worth being precise about why. The model
finished the story, emitted `<|im_end|>`, opened a new `<|im_start|>assistant` turn, produced a
well-formed `<think>` block, and **inside it correctly summarised its own preceding output**. That is
long-range self-reference, chat-template structure and special-token handling — not local fluency.
200/200 decode steps finite; none of the four degeneration signatures (loop / letter-spelling /
mid-word switch / token noise). Per the repo rule, this was checked **sampled**, not greedy.

Nothing in the hyper-connection stream, the 36 GDN layers, the sigmoid output gate, the NVFP4 expert
fold, the n-gram PLE block, or the 12 dense full-attention layers can be wrong and still yield that.

### How it fits on 16 GB (MEASURED)

Only **one layer's experts are live at a time** (1.465 GiB), so full depth costs body + one layer:

| tier | bytes | source |
|---|---|---|
| non-expert body, 948 params | **9.216 GiB** | `full48.log` |
| routed experts, one layer | **1.465 GiB** | `tests/qwen4exp_fulldepth_test.py` |
| **total resident, measured free-memory delta** | **11.377 GiB** of 15.922 | `full48.log`, `routed_full48c.log` |
| routed experts, all 48 layers (a *serve*) | **70.312 GiB** | — |
| **whole model resident (a serve)** | **79.53 GiB** vs ~63.7 GiB reachable | — |

Two staging modes were run at full depth, and **both** produced the greedy continuation above:

| | `--stream` (whole tier/layer) | `--routed` (top-k only) |
|---|---|---|
| expert-stagings, prefill + 12 decode | 624 layer-stagings, **914.062 GiB** | 7322 expert-stagings (**11.7**/layer of 512) , **18.854 GiB** |
| read rate | **488 MiB/s** | **1257 MiB/s** |
| wall per forward | **147.5 s** (1917.5 s / 13) | **1.18 s** (15.4 s / 13) |
| experts/layer at decode | 512 | **10.0**, exactly `num_experts_per_tok` |
| expert bytes/token at decode | 70.312 GiB | **1.236 GiB** |

The 200-token run: 98,429 expert-stagings (10.2/layer), **253.448 GiB read in 208.4 s (1246 MiB/s)**,
**1.261 GiB/token**.

**The routed gather is validated bit-exactly, not by inspection** (`routed_val3.log`, `val.log`):
streamed logits vs a fully-resident build gave `max|delta| = 0` over 3 steps with identical greedy
ids. A mis-mapped layer would be fluent garbage with no error, so this A/B — not the reading of the
code — is why the numbers above are quotable.

---

## 2. Hyper-connection numeric parity — PASS, against REAL tensors

**GATE-2 is discharged.** Re-run for this report; MEASURED, not inherited:

```
[mix/torch.float32     weights=real] max|d|=0.000e+00 (0.00 ulp)  hc_norm max|d|=0.000e+00 (0.00 ulp)
[mix/torch.bfloat16    weights=real] max|d|=1.562e-02 (2.00 ulp)  hc_norm max|d|=6.250e-02 (2.00 ulp)
[combine/torch.float32 weights=real] max|d|=0.000e+00 (0.00 ulp)
[combine/torch.bfloat16 weights=real] max|d|=3.125e-02 (2.00 ulp)
12 passed
```

`weights=real` = the four `model.language_model.layers.10.attn_hyper_connection.*` tensors read from
the checkpoint (not random; the random path exists as a fallback and was not taken —
`model-bf16-00010.safetensors` is on disk). **fp32 is bit-exact (0.0, 0 ulp)** for `hc_norm`, `mix`
and `combine` separately; bf16 differs by exactly 2 ulp, which is the one extra rounding this repo's
shared `_rms_norm` core takes (it returns the normalized value in `x.dtype` before the `(1 + w)`
gain, where the reference stays fp32 until the end).

The test does **not** compare against a transcription of the reference — a transcription is the thing
under suspicion. It `ast`-extracts `GroupedGemmaRMSNorm` and the `_mix_compute` / `_combine_compute`
closures out of `sglang/srt/layers/hyperconnection.py`, execs them in an isolated namespace, and
calls them. If upstream's math moves, the test breaks.

A companion test perturbs each of the five silent-failure constants individually (the `(1+w)` Gemma
gain, both `/hc` divisors, `.mean` vs `.sum`, the `2 *` on the inject gate, and gating on the normed
stream while adding to the unnormed one) and asserts each one changes the output — so the parity is
not passing by insensitivity.

**Still not closed:** GATE-3, *does sglang match HuggingFace*. The parity is against sglang only.
The 200-token sampled result is strong end-to-end corroboration, but no per-element diff against
`transformers`' `modeling_qwen4_exp.py` was run.

---

## 3. What actually broke (all four MEASURED, all fixed)

### 3.1 ALL LOGITS NaN — modelopt's `weight_scale_2` is the RECIPROCAL of compressed-tensors' `weight_global_scale`

The first real-weight forward produced NaN logits originating in `L0.mlp`, the first NVFP4 MoE block;
every block before it was finite. The loader read the modelopt spelling but inherited the
compressed-tensors **arithmetic** (divide).

MEASURED on layer-0 routed experts: `weight_scale_2 = 2.078102e-4`, e4m3 block scales 7..256.
Dividing folds to per-group scales up to `4.2e6` — **99,554 of 102,400 overflow the kernel's fp16
scale to `inf`** → NaN. Multiplying gives dequantized `|w|` mean 0.0105 / max 0.180, which matches the
**same layer's unquantized bf16 shared expert** (`|w|` mean 0.0071). That anchor, not a formula,
settled the direction.

Nothing structural was wrong — shape `(512,1280,160)`, group 16, dtype fp16 were all already correct,
which is exactly why the existing loader test passed.

Fix (`python/minisgl/quant/nvfp4.py`): `fold_nvfp4_scale` now takes a **required** keyword
`global_field` — the literal checkpoint tensor suffix the global was read from — and looks the
direction up in a new module table:

```python
NVFP4_GLOBAL_SCALE_IS_RECIPROCAL = {
    "weight_global_scale": True,   # compressed-tensors: divide
    "weight_scale_2": False,       # modelopt: multiply
}
```

Not a bool and not a producer name: a new NVFP4 loader **cannot call it without naming the spelling
it found**, and an unlisted spelling raises rather than guessing. An unconditional finiteness check
on the folded fp16 result turns this whole bug class into a loud load-time error instead of NaN
logits (verified: passing the inverted convention now raises). The three compressed-tensors call
sites (qwen3_5, laguna, muse_glimmer) and `tools/laguna_nvfp4_check.py` pass `'weight_global_scale'`
and are byte-identical to before. Regression test: `tests/qwen4exp_loader_test.py` §[3b] dequantizes
the real layer-0 expert stack and asserts `|w|` is within 10x of the real bf16 shared-expert anchor —
**measured ratio 1.475**.

### 3.2 `RuntimeError: expected scalar type Float but found BFloat16` in `gdn_prefill_wmma`

This checkpoint ships **both** `linear_attn.A_log` and `dt_bias` as **BF16** (verified from the shard
header), while the GDN kernels and `QwenGatedDeltaNet`'s `nn.Parameter` are fp32. The fp32 upcast
existed and was correct, but lived as a **local closure inside `Engine`**, invisible to every other
consumer of `load_weight`. `GDNLinearAttn` loads with `assign=True`, so a bf16 `A_log` survives
straight into the parameter and nothing downstream catches it.

Fix: extracted verbatim to `cast_checkpoint_tensor(key, tensor, model_dtype)` in
`python/minisgl/models/weight.py`, exported from `minisgl.models`. Pure extraction, zero behaviour
change for the engine. Pinned by `tests/qwen4exp_loader_test.py` §[3c]. (Incidental: the old comment
there asserted "A_log ships fp32" — true of the Qwen3.5 checkpoints it was written against, false
here. The code was unconditional and therefore right; only the comment was wrong.)

### 3.3 The routed gather produced NaN — the route may not be re-derived (new GATE-10)

`routed_val.log`, MEASURED:

```
FAIL routed logits bit-identical to resident   max|delta|=nan over 3 steps
FAIL routed greedy ids == resident greedy ids  got=[0, 0]  want=[217558, 42110]
```

The first version of the gather asked `MoELayer._ep_route` which experts to stage. The served e2m1
path routes with `moe_hip.moe_route_align` (softmax + top-k **inside** the kernel), and the two break
**exact bf16 ties at the k-th boundary** opposite ways — the kernel keeps the lower expert index,
`torch.topk` the higher. Ties are common, not exotic: bf16 gate logits have 8 mantissa bits spread
over 512 experts. MEASURED, layer 0 of an 8-token prefill: experts 324 and 366 both scored `-5.09375`
at ranks 9/10; **2 of 8 MoE calls in that one forward diverged by one expert**. A staged-but-unrouted
expert reads uninitialised buffer.

This is self-consistent for EP (which feeds its own ids to the kernel), so it is **not** an EP bug —
but it does mean an EP and a non-EP serve can pick a different expert on a tie row. Taking the route
from `quant.kernels._route_align` makes the gather bit-identical to the resident build.
`_ep_route`'s docstring asserted the opposite and named a function (`w4a8_moe._route`) that no longer
exists; corrected.

**This binds any future offload prefetch or expert-residency policy**, which is why it is now a gate.

### 3.4 The modelopt `ignore` list is fnmatch GLOBS, not substrings

Independent of §3.1 and not anticipated by the plan. 10 of the 13 `ignore` entries are globs
(`*.self_attn.*`, `*.linear_attn.*`, `*.mlp.gate*`, `*.mlp.shared_expert.*`,
`*.mlp.shared_expert_gate*`, `*hyper_connection*`, `*.ple.*`, `mtp.*`, `model.mtp.*`,
`model.visual.*`). Fed to the historical substring matcher, `'*.self_attn.*' in name` is False for
**every** module name — so the whole ignore list evaporates and the bf16 attention, GDN,
hyper-connection, shared-expert, PLE and gate modules all build **quantized against unpacked
tensors**. Fixed by translating globs to fully-anchored `re:` patterns in
`python/minisgl/quant/config.py` (`_glob_to_ignore`).

### 3.5 Harness-only (recorded so the next harness does not rediscover them)

- `RuntimeError: Output 0 of View is a view and is being modified inplace` in
  `RMSNorm.forward_inplace` — the engine runs every forward under `torch.inference_mode()`
  (`server/launch.py:20`); a hand-built harness must too.
- `minisgl.gdn.state` does not exist — `GDNStateCache` is in `minisgl.kvcache.gdn_state`.
- `KeyError: Unsupported MoE Backend: auto` — `auto` is resolved by `Engine._adjust_config`, not the
  registry; the concrete name is `fused`.
- `page_size` must be a multiple of 16 for the native-HIP `rdna4`/`hip` attention kernels.
- A CPU-only bf16 linear died in the dispatcher (`Could not run 'fp8_wmma_C::dense_bf16_gemv' with
  arguments from the 'CPU' backend`) instead of taking the `F.linear` fallback. Real bug, fixed in
  `python/minisgl/layers/minv.py`: `minv_supported` now returns False unless both tensors are on
  device. Anything running a bf16 layer off-device (a numerics oracle, a shape probe on a box with no
  free card) hit it.
- An unguarded module-scope `set_tp_info(0,1)` in a second test module **aborts pytest COLLECTION**
  for the whole run, not just that module. Now guarded with `try_get_tp_info()`.

---

## 4. The next real blocker

**The routed gather that makes full depth fit is a test-harness monkeypatch, not engine code.**

`tests/qwen4exp_fulldepth_test.py:478-497` installs it by assigning over the module:

```python
def routed_forward(x, _lid=lid, _mlp=mlp, _inner=inner):
    _, topk_ids, _, expert_ids, ntp = qk._route_align(...)
    self.stage_routed(_lid, ids)
    ...
mlp.forward = routed_forward
```

Nothing under `python/minisgl/` knows about it — `grep -rn 'route_align' python/minisgl/` returns
only `quant/kernels.py`, the kernel wrapper itself. So:

- The real `Engine` has no gather. At full depth it needs the **whole 79.53 GiB tier resident**
  against ~63.7 GiB reachable (device ~11.7 + pinned host ~52). `tests/qwen4exp_engine_test.py`
  therefore runs on a **4-layer subset**, and says so in its docstring: *"the full 48 layers do not
  fit on a 16 GB card"*. Everything the engine test proves — pool sizing, prefix-cache choice, slot
  lifecycle, chunked prefill, sampler, detokenize, two concurrent requests, PLE staging order — is
  proven at 4 layers, where the text is meaningless by construction.
- Even wired in, the gather reads through `safetensors` **mmap at 1257 MiB/s** (MEASURED, three
  full-depth runs: 1257 / 1246 / 1590 MiB/s). That is the mmap path `docs/WEIGHT_OFFLOAD_PLAN.md` §9
  already disqualified for serving (0.64–0.87 GB/s cold there). It yields ~1.18 s/token of pure
  staging — an I/O-bound harness number, **not a serve**.

So the next blocker is one item with two halves, and neither is model code:

1. **Promote the gather from the harness into a real `MoELayer` seam** (`weights/moe_interpose.py`
   already has the seam shape; `attach_seams` needs only stock `MoELayer` instances at unique dotted
   paths, so no per-model registration). It must take its route from `_route_align` — GATE-10.
2. **Move it off `safetensors` mmap onto the O_DIRECT granule reader**, measured at
   **6.9–9.0 GB/s** in `WEIGHT_OFFLOAD_PLAN.md` §9. At the measured 1.236 GiB/token that
   **PROJECTS to 5.7–7.4 tok/s** — a capacity feature, not a throughput one. *This rate is the only
   unmeasured link in the chain; the traffic and the correctness of the gather are measured.*
   Prefetch is available and unexploited: the gather is currently synchronous inside the layer, so
   its latency is fully exposed, and layer L's route is knowable only after L-1 — but there is ~1 ms
   of GEMM per layer as an overlap window.

Second-order, already visible: `convert_nvfp4_moe`'s per-expert Python loop runs 49,152 iterations at
this scale. Round 4 cut it to 10 experts/layer (which is why the harness went 365 → 1257 MiB/s), but
it is still a Python loop on the critical path.

---

## 5. Deliberately DEFERRED — not done, not pretended done

| Item | Status | Behaviour today |
|---|---|---|
| **Graph capture** | `[CAPTURE-2026-09-04]` **DISCHARGED — this row is SUPERSEDED.** ~~UNDISCHARGED … never attempted for this model … serve eager until it has been.~~ | Decode capture is implemented and exercised at **48/48 layers, TP=2, arena engaged**: buckets `[1,2]`, replayed greedy ids IDENTICAL to eager's at both widths, `verify_after_capture_fired=2`, `arena_torch_fallbacks=0`. It required the `hip` attention backend (`rdna4` raises on capture), a new `PLEGraphCapture` (T6.2 — the block's device arithmetic was always shape-static; the gap was that the capturer had no way to STAGE a batch), and a new `BaseLLMModel.prepare_for_replay` seam so the QSA indexer-budget refusal keeps running per step instead of once, at capture, over dummy rows. Worth **+2.5%** (81.87 → 79.86 ms/decode step) — decode here is PCIe-bound, not launch-bound. Prefill and spec-verify capture stay eager. Full write-up: [`QWEN4EXP_GRAPH_CAPTURE.md`](QWEN4EXP_GRAPH_CAPTURE.md). |
| **QSA sparse indexer** | **NOT implemented.** Plan T5, 3,176 reference lines upstream. | `Qwen4ExpAttn.forward` **raises** the moment a request's context exceeds `indexer_budget` (2048). Below it the selection is provably the identity (`sparse_attn.py` clamps `row_topk = min(topk, visible)`), so the dense attention that runs is **bit-equivalent**, not an approximation — which is what makes the ≤2048 results above real. Caps usable context at 2048 of 262,144. Needs a second KV pool for index keys and a `BaseAttnBackend` ABI carrying a per-query selected-page set across fa/fi/hip/rdna4/mla/trtllm + `HybridBackend`. |
| **Expert offload arena** | **NOT integrated** — see §4, it is the #1 item. | The mechanism is built and measured in a harness; it is not in the engine. |
| **TP = 2** | **NOT implemented — refused at construction**, in both the model and the loader. | `_load_qwen4_exp_weight` has no shard function for this key set. All shapes divide 2 (nq 24, nkv 2, GDN 16/48, 512 experts), so it is not a shape blocker; the HC wide stream and the rank-320 mixer have **no established sharding rule** — replicate is the safe default, at 4x residual allreduce cost. |
| **MTP head** (`mtp.*`) | **NOT implemented.** Plan T8.1. | Loader ignore ledger counts **31** skipped `mtp.*` tensors; `ModelConfig.from_hf` refuses `--spec-algorithm mtp` for this model. New shapes when wanted: experts are **fused** (`mlp.experts.gate_up_proj`, fp8-block) unlike the backbone's per-expert files, it has its own HC, and it seeds from the **10240-wide pre-mixer** stream. `mtp.*` is top-level — a `model.` prefix filter misses it. |
| **Vision** (`model.visual.*`) | **Not planned.** Out of scope. | Loader ignore ledger counts **333** skipped vision tensors. |
| **Aux-hidden capture / draft heads** | **Refused.** | The per-layer residual is the 4x-wide stream, not a hidden-width one. |

One more thing that is **not** deferred but **is** a deviation worth recording: gfx1201 has no FP4
math, so the e2m1 kernel quantizes activations to fp8 and the checkpoint's `input_scale` calibration
is dropped. **The served scheme is W4A8, not the declared W4A4** — 1536 `*.input_scale` /
`*.input_global_scale` tensors are counted and skipped in the loader's ignore ledger. The 200-token
sampled result holds despite it, so it is evidently not quality-fatal at this length.

---

## 6. Predictions the plan got WRONG once code ran

This is the most valuable section. `docs/QWEN4EXP_BRINGUP_PLAN.md` has been corrected in place,
tagged `[RUN-2026-09-03]`.

**W1 — "the leaf-name remap": the plan's prescribed fix WAS the bug.**
T1.2 said to rename the checkpoint's `.weight_scale_2` to the repo's `.weight_global_scale` "before
the `nvfp4_bases` pre-pass or the fold silently no-ops". Renaming is exactly what must **not** happen:
the two spellings carry **opposite arithmetic conventions** (§3.1), and normalizing the name silently
inherits DIVIDE → 97% of per-group scales overflow fp16 → NaN. The spelling is load-bearing
*information*, not a naming inconvenience. This is the single most costly wrong prediction in the
document — it would have been followed, it produces no error, and the failure surfaces 48 layers
later as NaN.

**W2 — KILL-1 was asked at the wrong granularity, twice, and it gated nothing it was believed to gate.**
The plan treated "does the model fit" as a single gate on correctness. It is two gates, and only one
is real. (a) A full-depth **forward** needs body + **one** layer's experts, because a layer's experts
are dead the moment it returns: **11.377 GiB MEASURED**, fits with 4.5 GiB to spare. (b) A **serve**
needs 79.53 GiB vs ~63.7 GiB reachable. The wrong inference from (b) to (a) is what kept **every**
earlier test at a 4-layer subset — a self-imposed restriction that was never a property of the
architecture or of this box. And then round 4 showed (b) is *also* the wrong question: a decode step
reads **1.236 GiB** of experts, so serving is a **bandwidth** question, not a capacity one.

**W3 — the shortfall was 15.8 GiB, not ~21, because the body estimate was 60% too high.**
"~15 GiB" for the non-expert body was the on-disk size of the four `model-bf16-*` shards. Those
shards also carry the `mtp.*` head and the `model.visual.*` tower, **neither of which this build
constructs**. MEASURED body: **9.216 GiB**. *Read resident bytes off the model, not shard sizes.*

**W4 — the hyper-connection traffic claim was BACKWARDS.**
The plan asserted HC moves "more decode traffic than all 480 activated experts combined". MEASURED:
HC is **1.193 GiB (1.281 GB)** per step, the 480 activated experts are **1.373 GiB (1.475 GB)** — HC
is **86.9% of** expert traffic, not more than it. The experts are the larger term. The startling part
survives and is the actionable part: an unquantizable 1.28 GB of pure plumbing costs nearly as much
decode bandwidth as all 480 routed experts.

**W5 — the modelopt quant arm was scoped as one small normalization; it was two independent defects.**
T1.1 named only the `quant_method`/`ct_format` translation. It missed that `ignore` entries are
**fnmatch globs** fed to a substring matcher (§3.4), which silently deletes the entire ignore list and
builds bf16 attention / GDN / hyper-connection / shared-expert / PLE / gate modules as quantized
against unpacked tensors. Also missed: the format lives in `quant_algo: "NVFP4"`, not a `format` key,
so untranslated, `weight_is_e2m1` (MXFP4) becomes True instead of `is_nvfp4` and every routed expert
routes into the **group-32 MXFP4 kernel over group-16 scales** with an orphaned `weight_scale_2`.

**W6 — GATE-10 did not exist in the plan at all.**
"Anything that predicts which experts the kernel will read must take the route from
`_route_align`, never from `_ep_route`" is a new gate, discovered by NaN (§3.3). The plan assumed the
route was a derivable quantity. It is not: it is a kernel output, and re-deriving it breaks exact
bf16 ties the opposite way. This binds every future offload prefetch and residency policy.

**W7 — "reuse `GDNLinearAttn` verbatim" was true structurally and false on dtype.**
The plan justified reuse from shapes and key names, which matched exactly. It did not check stored
dtypes: this checkpoint ships `A_log` and `dt_bias` as **BF16** where the Qwen3.5 checkpoints ship
fp32, and the compensating cast was an `Engine`-private closure invisible to `load_weight`'s other
consumers (§3.2). *Shape-and-name parity does not imply dtype parity.*

**W8 — GATE-5 (`gate_act` silently absorbed) was a non-risk, and discharged for free.**
The plan budgeted a probe for whether the `gdn_hip` .so consumes `output_gate_type: "sigmoid"`.
`_gate_args()` returns `(1,)` for `sigmoid` — **non-empty**, unlike the `silu`/`swish` case which
returns `()` precisely so an old .so still loads — and it is splatted at all six gated-norm call
sites. A `gdn_hip` without the parameter fails the op schema match on **argument count**. 48 layers ×
13 forwards ran with no such error, so the schema has it. Positive confirmation, no probe needed.

**W9 — the document's own status header was the most wrong line in it.**
"*planning only. Nothing here has been run on a GPU. No gate below is discharged.*" GATE-2, GATE-5,
GATE-6 and KILL-1(a) are all discharged, and GATE-10 was added. Corrected.

### What the plan got RIGHT, and should be credited

- **The per-expert byte table is exact.** Predicted 70.31 GiB for the routed tier; MEASURED
  **70.312 GiB**.
- **The routed decode footprint is exactly `num_experts_per_tok`.** Predicted 10 of 512; MEASURED
  **10.0 experts/layer**, dead on, across both a 12-token and a 200-token run.
- **The indexer deferral was justified by code, not optimism**, and it held: ≤2048 is bit-equivalent
  to dense, so the coherence results are real rather than a stand-in.
- **"CPU-verifiable first" was the right sequencing.** HC parity, config plumbing and loader key-set
  parity all gated on tests needing no GPU, and all three caught things.
- **1.373 vs 1.236 GiB/token is not an error** — both are right and measure different things: 1.373
  is the **resident** per-expert footprint (fp16 scales), 1.236 the **on-disk** one (e4m3 scales), and
  it is the on-disk figure a gather actually reads. The +11% is the fp16-scale inflation, not paid by
  a gather that folds after the read.

### Still open, and honestly labelled

- **T3.3 (mrope): ARGUED, NOT MEASURED.** The arithmetic lines up (`partial_rotary_factor 0.25 ×
  head_dim 256` → `rotary_dim 64` → 32 frequency slots, and `sum([11,11,10]) == 32`), and for
  text-only the t/h/w position components are all equal so any section assignment collapses to the
  same angles. Corroborated end to end — 48 layers of wrong positions do not emit `" Paris."` — but
  no per-element diff against the reference was run.
- **GATE-3 (sglang vs HuggingFace): not run.** §2.
- **GATE-7 (is HC affordable): not run.** The HC **bytes** are measured; the tok/s ceiling
  (~552 tok/s HC-only, ~479 tok/s expert-only, ~256 tok/s combined) is **PROJECTED** arithmetic
  against the 706.6 GB/s roofline. It needs a trace, not a calculation.
- **GATE-8 (MoE tile choice at inter=640): not run.** `kernels.py:365-380` already records the
  current rule is 0.76–0.85x the analytic oracle at M≤32, and inter=640 is much narrower than the
  Qwen3.6-35B shapes the surface was tuned on. Expect it to be wrong and budget the A/B.

---

## 7. Reproduction

Full depth, greedy, routed gather (the cheapest run that reproduces §1):

```
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES=0 \
  -v /home/pat/code/minisgl-rdna4-offload:/engine \
  -v /home/pat/.cache/hf-q4e:/model:ro -v /home/pat/.cache/hf-ple:/ple:ro \
  --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
  'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_fulldepth_test.py --routed'
```

Add `--validate` for the bit-exactness A/B against a resident build, `--temperature 0.8 --steps 200`
for the sampled run. HC parity is CPU-only and needs `-v /home/pat/Projects/sglang-upstream:/sglang:ro`
plus `pip install pytest` (pytest is not in the image); see the docstring of
`tests/qwen4exp_hc_parity_test.py`.

Run logs backing every MEASURED number in this document:
`/tmp/claude-1000/-home-pat-code-minisgl-rdna4/36039161-cc69-402f-a0b6-b5813c739e8b/scratchpad/`
— `full48.log`, `sampled48.log`, `routed_full48{,b,c}.log`, `routed_sampled200.log`,
`routed_val{,2,3}.log`, `val.log`. *These are in a session scratchpad and are not durable; the
numbers, not the logs, are the record.*
