# CONTINUANCE — minisgl garbles quantized-MoE reasoning (LOCALIZED 2026-07-08: the bug is the **MoE tensor-parallel (TP≥2) path**, NOT quantization, NOT attention)

Paste into a fresh session. Repo `/home/pat/code/minisgl-rdna4`. Follow this repo's CLAUDE.md for ALL
GPU work (bare `gpu-lease -n <cards> -- …`, worktree source-isolation, container recipe). Both
gfx1201 cards are shared across the box — lease, don't poll.

**Lesson that governs this whole hunt:** validate REAL free generation (a clean, correct, terminating
answer), not isolated components. Every isolated primitive here PASSES while the assembled model fails
— so isolated "it works" claims are the trap.

---

## UPDATE 9 — 2026-07-08 (cont.): generation_config + jinja templates now LOADED (multi-EOS fix); degeneration STILL open

Per the standing directive "make sure chat templates and default configs are loaded for any model":
minisgl historically **never loaded `generation_config.json`** and honoured only a **single**
`tokenizer.eos_token_id` → multi-EOS models (GLM `[154820,154827,154829]`) **never terminated**, and
the author's recommended sampling was ignored. Fixed (branch `feat/rxf-glm-serving`):
- `utils/hf.py`: `load_generation_config`, `resolve_stop_token_ids` (union of gen_config eos-list ∪
  tokenizer ∪ config), and a `chat_template.jinja` fallback (was `.json`-only).
- `scheduler.py` stop check + `detokenize.py` trailing-drop now use the full stop-id SET.
- `api_server.py`: request `temperature/top_p/top_k` default to None → `_resolve_sampling` fills from
  generation_config then neutral (bare requests inherit GLM's top_p 0.95 / top_k 50).
- **GPU-verified:** GLM now TERMINATES (165 tok, finish_reason=stop) vs never; Qwen3.5-4B unaffected
  (still 17+25=42, stops). See memory `minisgl-generation-config-loading`.

**This fixes the never-terminate symptom but NOT the reasoning degeneration.** Sampling was RULED OUT
directly: GLM at temp=1.0/top_p=0.95/top_k=50 AND temp=0.6 still produce garbled/looping reasoning
("Wait, the user's query... 7?"). The chat template is CORRECT (transformers 5.13 auto-loads the
`.jinja`; prompt ends in `<think>` as expected). So the degeneration is a genuine compute issue,
unrelated to config/sampling/template.

**Remaining root-cause suspect = RoPE convention.** Every per-layer component was recomputed vs the
checkpoint (UPDATE 8), but ALL recomputes used minisgl's OWN post-rope q/k (rope was NEVER verified) —
a config/convention-consistent bug (wrong rope style, or a scaling) would make every recompute match
yet differ from the true model. `glm4_moe_lite.py` builds `get_rope(rotary_dim=64, base=1e6,
rope_scaling=None)`; verify the interleave convention (NeoX vs GPT-J) and partial-rotary layout against
a ground-truth (transformers GLM rope, or vLLM). If clean, do the full layer-by-layer reference diff.

---

## ★★★ UPDATE 8 — 2026-07-08 (cont.): MoE **and** MLA-attention EXONERATED on REAL data; bug is upstream/discrete

Applied a dump+recompute technique to REAL GLM-4.7-Flash-AWQ at TP=2 (serve, dump a component's real
I/O during DECODE via env-gated one-shot hooks now in `glm4_moe_lite.py`, recompute that component
offline in full precision from the real checkpoint, diff). Two more exonerations:

1. **The real TP=2 MoE BLOCK is correct** (`moe_recompute.py`, CPU): dumped layer-1 MoE I/O during a real
   decode (`MINISGL_MOEDUMP`); offline recompute from the real AWQ experts. Route sanity: dumped
   `topk_ids` == recomputed noaux_tc ids. **Routed: cos=0.99970** (fp8-matched ref) / 0.99921 (fp16).
   **Shared expert: cos=0.99996.** → routed+shared block output is correct GIVEN its input.
2. **The real TP=2 MLA DECODE attention core is correct** (`mla_recompute.py`, CPU): dumped layer-1
   absorbed-MLA decode I/O both ranks (`MINISGL_MLADUMP`); recomputed `softmax(q·latent/√256)·latent[:kv]`.
   **cos=1.00000 on BOTH ranks, all 10 heads/rank, 0 heads <0.9.** Latent cache **identical across ranks**
   (max_diff 0.0); `q_full` differs only by the correct per-rank head shard.
3. **The MLA PROJECTIONS are correct** (`proj_recompute.py`, CPU): recomputed `q_nope` and `c_kv` from the
   attention input `x` using the real per-rank projection weights (q_a/q_a_ln/q_b, kv_a/kv_a_ln) + the
   `W_UK` absorption. **q_nope cos=1.00000 (per-head all 1.0), c_kv cos=1.00000, absorb cos=1.00000, both
   ranks.** → the WHOLE MLA attention (x → q_full/latent → o_latent) is verified end-to-end; only the
   standard row-parallel `o_proj` all-reduce (proven in dense TP=2) is unrecomputed.

**So on REAL data at TP=2, the ENTIRE compute path is now exonerated:** weights, MoE kernel (full/half/
decode-gemv), TP composition, routing, MoE block (routed+shared), MLA decode attention core, latent-cache
consistency, GDN attention, activation quant (W4A16 still loops) — all correct; ranks byte-identical.

### LAYER 1 IS FULLY VERIFIED at an early decode step → suspect is STEP-CUMULATIVE (KV) or the head
Every per-layer operation (embedding→L0→L1-attention→L1-MoE) is now proven correct on real data, and all
layers are structurally identical — so a per-layer discrete bug is nearly excluded. It is NOT accumulated
small error either: block contributions are tiny (routed 0.27, shared 0.20) vs the residual (~11→61), so
per-block errors land <0.1% on the residual. **Crucially, all dumps were from an EARLY decode step
(context L=13, all prompt tokens).** The degeneration STARTS coherent and loops after several generated
tokens — the signature of a **STEP-CUMULATIVE bug: the MLA latent KV-cache STORE of GENERATED tokens at
TP=2.** If a generated token's latent is stored to the wrong slot / with wrong position / not visible to
later steps, subsequent decode attends to corrupt/incomplete KV → repetition/looping, while the early
step (attending only to correctly-prefilled prompt KV) is perfect — exactly what we observe.

### NEXT (in order)
1. **Dump at a LATE decode step (step ~20, mid-loop), not the first.** Change the one-shot guards to a
   call-counter (fire at the Nth layer-1 decode). Then: (a) re-verify the attention math still holds
   (it should — the math is fine); (b) **verify the STORED latent for GENERATED positions** — dump each
   recent token's post-`store_latent` latent AND its source hidden, recompute the latent from the hidden
   (kv_a_proj→norm→rope), and diff. A store/slot/position bug shows as a wrong latent at generated rows.
   Also dump `batch.positions`, `out_loc`, and the page-table slots per step to check slot advancement.
2. **Ground-truth layer-by-layer diff** (if #1 is clean): extend `MINISGL_RANKDUMP` to DECODE; reference =
   HF `transformers` on `zai-org/GLM-4.7-Flash` (unquantized, in cache) if the arch loads, else a full
   offline bf16 forward. First divergent layer names the module.
3. Also check the **final norm + lm_head + sampling** (large-magnitude, drives logits) — untested; and
   confirm on **Qwen3.6-35B** (GDN, so attention is exonerated → its bug is unambiguously in the MoE-model
   assembly OR the same KV-store class of bug in GDN's recurrent-state cache across steps).

Instrumentation left in place (env-gated, inert by default): `MINISGL_RANKDUMP` (per-layer residual,
GLMModel.forward), `MINISGL_MOEDUMP` (MoE block I/O, GLMSparseBlock.forward), `MINISGL_MLADUMP` (MLA decode
I/O, GLMMLAAttention.forward). Recompute scripts (`shard_recon_check.py`, `kernel_halfwidth_check.py`,
`decode_check.py`, `compose_check.py`, `moe_recompute.py`, `mla_recompute.py`) in the session scratchpad.

---

## UPDATE 7 — 2026-07-08 (cont.): every MoE-TP2 COMPONENT proven correct; act-quant re-killed; PARADOX

Continued UPDATE 6 with a battery of decisive experiments. Net: the localization `MoE ∧ TP≥2` HOLDS,
but **every isolated component of the quantized-MoE-TP2 path is now provably CORRECT**, and
**activation quant is re-ruled-out rigorously** — leaving a real paradox (correct components,
degenerate assembly). The remaining bug is a DETERMINISTIC, precision-independent, subtle logit
corruption that no component test reproduces.

### Experiments run this session (all reproducible; scripts in the session scratchpad)
1. **Real TP=2 weight-shard reconstruction (CPU, no GPU):** `shard_recon_check.py` exercises the REAL
   loader (`_shard_tensor` + AWQ gate/up merge + `awq_to_op_layout` + dequant) for GLM layer-1 expert-0
   at rank0/rank1, verifies `concat(rank0, rank1) == full checkpoint expert`. **max_abs_err = 0.000e+00**
   for gate_up AND down. → TP=2 expert weights are byte-exact correct. (`_shard_tensor` takes rank/size
   as args, `awq_to_op_layout` is pure torch — so this runs CPU-only in the lean container, no lease.)
2. **Kernel correctness at half-width (`kernel_halfwidth_check.py`, 1 GPU):** `w4a8_moe` vs an
   fp8-act-matching fp16 grouped-GEMM reference. cos=**0.99999 at BOTH inter=1536 (TP1) and 768 (TP2
   half)**. Kernel is correct at the sharded width.
3. **Decode-path kernel (`decode_check.py`, 1 GPU):** swept M∈{1,2,4,16} (M≤2 → gemm1 uses the `"gemv"`
   kernel, the DECODE path where generation degenerates). cos=**1.00000 at every M, both widths**. The
   decode gemv path is correct too.
4. **TP=2 kernel COMPOSITION (`compose_check.py`, 1 GPU):** does `w4a8_moe(full) == w4a8_moe(rank0-half)
   + w4a8_moe(rank1-half)` (the exact TP=2 all-reduce op)? cos=**0.999925, rel_err 1.2%** (the 1.2% is
   the fp8-act per-token scale computed over 768 vs 1536 — a numeric DIFFERENCE, not a bug; see #7). The
   kernel composes across the TP split.
5. **Per-rank per-layer dump on REAL GLM TP=2 (`MINISGL_RANKDUMP` hook, now in `glm4_moe_lite.py`
   GLMModel.forward — env-gated, inert by default):** dumped residual-stream norm/mean/last-token for
   every layer on rank0 & rank1 during a real prefill. **rank0.txt and rank1.txt are BYTE-IDENTICAL**
   (diff empty); norms grow normally (5.9→61, no NaN/blowup). → the all-reduce IS bit-identical,
   routing IS identical across ranks; the degenerate output is DETERMINISTIC (identically wrong on both
   ranks), NOT a cross-rank-divergence bug. (Degeneration confirmed live: `17+25`→ loops
   "The user is asking…The user is asking…", never 42.)
6. **GDN attention at TP=2 is HEALTHY (rigorously):** unquantized dense `Qwen/Qwen3.5-4B` (GDN, no MoE)
   at TP=2 solved a hard relative-motion algebra problem — set up `60(t+2)=80t`, solved `120=20t`, 1479
   chars coherent, `finish_reason=stop`. → for **Qwen3.6-35B (GDN+MoE+TP2, confirmed still degenerate
   this session** — `127×8` loops "27×8…27×8…", and drops tokens: "Ident" for "Identify") the bug is
   MoE-side, NOT GDN. GDN sharding fully exonerated.
7. **W4A16 (fp16 acts, ZERO act-quant) GLM @ TP=2 STILL DEGENERATES** (fresh build at
   `/home/pat/code/rdna4-hip-kernels-w4a16`, `MINISGL_MOE_W4A16=1`, `/pkg` mount): identical looping.
   → **activation quantization is DEFINITIVELY not the cause** (re-confirms UPDATE 5); the 1.2% from #4
   is not it either (W4A16 makes composition exact, yet still loops).

### The paradox (state to resolve next)
Weights ✅, kernel (WMMA + gemv, full + half width) ✅, cross-rank composition ✅, routing ✅,
ranks byte-identical ✅, act-quant ruled out ✅, GDN attn ✅ — yet GLM & Qwen3.6-35B degenerate
deterministically at TP=2 with subtle token-level corruption (looping, dropped chars). The gap is
between "synthetic components correct" and "REAL assembled serve broken." Untested = exactly that gap.

### NEXT — the definitive end-to-end check (highest value)
**Run the REAL MoE block on REAL data at TP=2 and compare against a full-precision recompute.** Extend
the `MINISGL_RANKDUMP` hook (already in `glm4_moe_lite.py`) to also dump, for layer 1: the MoE-block
INPUT hidden, the `topk_ids`/`topk_weights`, the routed-expert output, the shared-expert output, and
the block output — during DECODE (not just prefill; extend past the `_dumped` first-forward guard).
Offline, dequant the REAL layer-1 experts to bf16 and recompute the MoE block in pure torch; diff vs the
dumped real output. This closes the synthetic↔real gap in ONE shot:
- If the real MoE block output DIVERGES from the bf16 recompute → the bug is in the real MoE integration
  (candidates my synthetic tests did NOT cover: E=64/128 scale, real sparse routing distribution through
  `moe_align`, the real `post_load` stack actually fed to the kernel, or the **shared-expert / router**
  add at TP=2 — none exercised end-to-end on real weights through the kernel).
- If the real MoE block MATCHES → the MoE is exonerated and the bug is in **MLA decode at TP=2** (GLM;
  never isolated — no servable dense-MLA model) or the lm_head/final-norm; refocus there. (Note the
  looping "repeat recent output" symptom is classically an attention/KV-decode signature.)

Fast reproducers (this session): degenerate = GLM-AWQ TP2 (fp8 or W4A16) and Qwen3.6-35B-AWQ-4bit TP2;
healthy = dense `Qwen/Qwen3.5-4B` TP2 and `Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4` TP1. All `--dtype bfloat16`
(attn_hip is bf16-only). Serve scripts + validation scripts in the session scratchpad.

---

## UPDATE 6 — 2026-07-08: BUG LOCALIZED TO `MoE ∧ TP≥2` (biggest narrowing yet)

Ran a 2×2 factorial over {dense, MoE} × {TP=1, TP=2} using SMALL models that fit, and probed REAL
free generation (17+25→42 clean+terminating is the pass bar; the failing models can't even do that).

| Model (served on minisgl, lean image)     | Attention | MLP  | TP | Result |
|--------------------------------------------|-----------|------|----|--------|
| Qwen3-4B-Thinking (prior)                  | MHA       | dense| 1&2| ✅ healthy |
| **Qwen/Qwen3.5-4B** (unquantized)          | **GDN**   | dense| **2** | ✅ healthy (17+25=42, correct column-mult CoT, terminates) |
| **Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4**       | MHA       | **MoE (GPTQ int4)** | **1** | ✅ healthy (17+25=42 clean; 3×24=72, 72−17=55 over 1000-char CoT, `finish_reason=stop`) |
| GLM-4.7-Flash-AWQ                          | MLA       | MoE  | 2  | ❌ degenerate |
| Qwen3.6-35B-A3B-AWQ (compressed-tensors)   | GDN       | MoE  | 2  | ❌ degenerate |

**The degeneration lives UNIQUELY in the intersection `MoE ∧ TP≥2`.** Neither factor alone triggers it:
- MoE at **TP=1** is healthy (Qwen1.5-MoE-GPTQ: correct multi-step arithmetic, coherent 1000-char CoT,
  clean termination — long-generation stress ruled out "length/context-dependent degeneration").
- **TP=2 attention** (both MHA *and* GDN) with a dense MLP is healthy. GDN-TP being healthy makes the
  **Qwen3.6-35B failure airtight = MoE-at-TP=2** (GDN sharding exonerated; its only remaining delta vs the
  healthy dense GDN Qwen3.5-4B is the MoE block). GLM's MLA-TP wasn't independently isolated (no servable
  dense-MLA model), but MoE-TP is the parsimonious single cause covering both failing models.

**Why this overturns Updates 1–5's framing:** it is NOT "quantized-MoE assembly, precision-independent."
It is specifically the **MoE _tensor-parallel_ path**. That's exactly why W4A16 didn't fix it — it was
never a precision bug. (W4A16 GLM at TP=2 degenerates under the `mmq_regdirect_w4a16_moe` kernel, which is
ENTIRELY different code from the w4a8 `mmq_fp8_moe_gemm` kernel — two independent kernels failing
identically points AWAY from the kernel and toward their SHARED input: the TP-sharded expert weights / the
routed all-reduce.)

### What the static audit already EXONERATED (hand-verified this session, do not redo)
- **AWQ/GPTQ/compressed-tensors expert weight-load + shard AXES** at TP=2: gate/up split output-N (dim 1),
  down split input-K (dim 0); scales/qzeros follow; the gate_up→`awq_to_op_layout` interaction lays gate
  into rows [0:N), up into [N:2N) — consistent with silu_and_mul's `d=out1.shape[1]//2` split. The TP=2
  intermediate-block indexing MATCHES between gate_up output and down input (rank r owns block r,
  consumes block r). All divisibility aligns for GLM (inter=1536 → 768/card, %128==0, %8==0). GPTQ shards
  identically to AWQ under `_shard_tensor` (both put output on axis 1, input on axis 0).
- **Routing is identical across ranks**: gate + `e_score_correction_bias` are REPLICATED; hidden entering
  the MoE is post-attention-all-reduce (bit-identical across ranks) → identical topk on both ranks. GLM
  `_noaux_tc` constants match config (n_group=1→group path skipped, norm_topk_prob, scale 1.8).
- **The reduce STRUCTURE**: routed all-reduced in `MoELayer` (moe.py:554), shared all-reduced via
  `LinearRowParallel` (Qwen) or replicated (GLM), summed ONCE — no double-count, no missing reduce.

### The bug is CONTENT-level (not shape/axis) — next steps must instrument at RUNTIME, TP=2
1. **Runtime shard-reconstruction check on REAL weights (TP=2 GLM):** dump rank0 & rank1
   `layers.1.mlp.experts.gate_up_proj._w_op`/`_scales_op`/`_zeros_op` (and down_proj), dequant expert 0
   on each rank, and verify `concat(rank0_half, rank1_half)` reproduces the checkpoint expert's dequant
   EXACTLY. Axes are right; this catches a subtle CONTENT corruption (stride/contiguity/off-by-a-group).
2. **Cross-rank routing + partial-output check (TP=2 GLM):** confirm `topk_ids` are byte-identical on
   rank0 vs rank1 for layer 1; dump each rank's routed output BEFORE the all-reduce and verify
   `rank0_partial + rank1_partial` is sane (no NaN/scale blowup) and matches a from-scratch two-rank sum.
3. **Kernel-at-sharded-width validation:** the w4a8_moe / w4a16_moe kernels were validated at FULL expert
   width. Re-validate at the HALF-WIDTH (TP=2) expert shapes (GLM inter_per_partition=768) vs a true-fp16
   grouped GEMM — a tiling/padding assumption that only holds at full width would show here.
4. **Fix, then validate REAL free generation** (17+25→42 clean, 127×8→1016, terminates) at TP=2.

**Fast reproducers banked (all on the lean image, standard `w4a8_moe` path, no /pkg mount needed):**
- Healthy MoE TP=1: `Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4` `--dtype bfloat16` (attn_hip is bf16-only;
  fp16 crashes `attn_hip v0 is bf16-only`). Note it CANNOT go TP=2 (moe_inter 1408→704/card, 704%128≠0).
- Healthy dense GDN TP=2: `Qwen/Qwen3.5-4B` `--dtype bfloat16 --tensor-parallel-size 2 --disable-pynccl`.
- Serve scripts + logs in this session's scratchpad. Endpoint is `/v1/chat/completions` (no
  `/v1/completions`). Do NOT pass an `rsa` field (defaults are opt-in; a bare request is a plain completion).

---

## THE HEADLINE (what is now certain)

minisgl serves **quantized-MoE models with grammatical-but-degenerate output** — it *understands* the
prompt but loops / gets arithmetic wrong and never terminates. Confirmed on **GLM-4.7-Flash-AWQ**
(MLA + AWQ g128) and **Qwen3.6-35B-A3B-AWQ-4bit** (GDN + compressed-tensors g32). The SAME checkpoints
are coherent on vLLM. Symptom, greedy: `17+25 → "27"/loops` (want 42), `17×25 → "170"` (want 425),
`127×8 → loops "8. 8. 8…"`, thinking never closes `</think>`.

### The one thing PROVEN (via a validated W4A16 build): **it is NOT activation quantization.**
We built + wired a native **W4A16 (fp16-activation, no act-quant) MoE + dense path** into minisgl,
**numerically validated cos=1.00000 vs a true fp16 GEMM on real int4 weights**, served GLM-AWQ with it
— and it **degenerates identically**. fp16 activations do not fix it. The entire fp8/int8-activation
hypothesis is CLOSED. (Earlier RXF int8-act port also degenerated — same reason.)

### Therefore the bug is in the **assembled quantized-MoE-MODEL path**, independent of precision.

---

## RULED OUT (each GPU-verified — do NOT re-investigate)

- **minisgl engine / decode / sampling / attention / all-reduce** — dense unquantized `Qwen3-4B-Thinking`
  reasons perfectly at **TP=1 AND TP=2** (30+12=42, 100×8=800/160/56→1016, 17×25=425→383).
- **The kernels** — `w4a8_fp8_wmma` gemm (gemv/wmma/scalar) parity-exact incl block_m=16 gemv;
  `moe_hip.moe_align` valid; `awq_to_op_layout` byte-identical to vLLM's `_awq_to_op_layout_single`;
  `w4a8_moe` end-to-end cos~1.0 (M=1 & M=32, g=32 & g=128); TP=2 packed gate/up shard reconstruction exact;
  **W4A16 moe + dense both cos=1.0 on real weights**.
- **Activation quantization** (fp8 AND int8 AND now proven by W4A16-doesn't-fix-it).
- **KV_FP8** (degenerates same on/off); **chat template / tokenization** (byte-identical to reference);
  **MLA math + mla_hip kernel** (kernel byte-identical to validated; GLM attention is bf16/exact anyway);
  **noaux_tc routing math** (matches DeepSeek/GLM reference on paper); **config parsing**
  (rope_theta 1e6, g=128 correct); **group_size handling** (kernel derives it at runtime).
- **vLLM caveat:** vLLM serves these via **awq_marlin (W4A16)** by default — so "vLLM forced-fp8 works"
  was NOT a clean fp8-act test. What's solid: vLLM(marlin/W4A16) correct, minisgl(any quant) broken.

## The bug MUST be in what synthetic tests BYPASS (the active hypothesis)

All the above tested *components* with *synthetic* weights + *random/precomputed* routing. The two
things never exercised are exactly where a "correct kernels + correct engine + broken assembled model"
defect lives:

1. **Real routing** — does the model select the RIGHT experts on real hidden states? Dump GLM's
   **actual `topk_ids` per layer** during a real prefill and compare against a from-scratch recompute
   of noaux_tc from the gate weights + hidden. A wrong route (wrong experts fire) → garbage that all
   the numerics-correct kernels faithfully compute.
2. **Weight-load name mapping** (`python/minisgl/models/weight.py`) — the streaming loader's
   expert-stacking / gate-up merge / TP-shard name mapping. If a real checkpoint tensor lands in the
   wrong slot (shapes still match → loads silently), experts are wrong. Synthetic tests fed weights
   directly, bypassing this entirely.

Also worth a look: the **per-layer activation diff vs a correct reference** (see harness below — the
vLLM reference capture was blocked by 35B not fitting a hand-rolled `LLM()`; a compose-plugin dump or
a smaller reproducing model would unblock it and pinpoint the first divergent layer).

## NEXT STEPS (in order)

1. **Instrument minisgl to dump per-layer `topk_ids` + hidden** for GLM-AWQ on a fixed short prompt
   (the residual-stream dump already exists — see `qwen3_5.py` MINISGL_DUMP_LAYERS hook; add topk).
   Recompute noaux_tc offline from the gate weights; diff. Divergence → routing bug.
2. **Audit `weight.py` GLM expert-load mapping** end-to-end: for one real GLM layer, verify the loaded
   `_w_op`/`_scales_op`/`_zeros_op` (dequant) equals the checkpoint expert's dequant, per expert id.
   (Synthetic AWQ conversion passed; the REAL load path is untested.)
3. **Per-layer activation diff** vs a correct reference (compose-plugin vLLM dump, or HF/a small model)
   → first divergent layer localizes the module.
4. Whatever it yields, validate with REAL free generation (17+25→42 clean, 127×8→1016, terminates),
   then land the fix.

---

## STATE OF THE CODE

- **Active branch/worktree:** `feat/rxf-glm-serving` at `/home/pat/code/minisgl-rdna4-rxfglm`
  (off rdna4 tip 687db94). Contains ALL the new work:
  - `python/minisgl/quant/kernels.py`: **`w4a16_moe`** (grouped fp16-act MoE via
    `mmq_regdirect_w4a16_moe` + scatter epilogue), **`w4a16_linear`** (dense), `_w4a16_wide` helper,
    `MOE_W4A16` env; plus the RXF precomputed-topk wiring in `rxf_moe`.
  - `python/minisgl/layers/moe.py`: W4A16 branch in `MoELayer.forward` + `_w_rep` build in
    `_GroupedAWQExperts.post_load` (gated on `MINISGL_MOE_W4A16 != "0"`; builds w_rep_wide, frees _w_op).
  - `python/minisgl/quant/method.py`: `W4A8LinearMethod` W4A16 path (shared expert / dense) —
    `process_weights_after_load` builds `_w_rep_wide`, `apply` uses `w4a16_linear`.
  - (RXF: `rxf_moe` precomputed-topk; canonical rxf package lives in rdna4-hip-kernels/rxf.)
  - NOTE: worktree also shows deletions of vendored `*_hip/` dirs (from a cleanup) — noise; ignore/clean
    before any commit. Real diff = kernels.py + moe.py + method.py + qwen3_5.py.
- **Kernel repo:** `/home/pat/code/rdna4-hip-kernels` HEAD has the grouped W4A16 MoE kernel
  (`mmq_regdirect_w4a16_moe` + `_scatter`), dense wide (`mmq_regdirect_w4a16_wide`), and the weight
  repacks (`repack_int4_to_w_rep`, `repack_int4_to_w_rep_moe`, `repack_w_rep_wide`,
  `repack_w_rep_wide_moe` in `w4a8_fp8_wmma/torch-ext/w4a8_fp8_wmma/weight_repack.py`). Also `rxf/`
  (my canonical RXF port, parity-green) — untracked/uncommitted in that repo.
- **Fresh w4a8_fp8_wmma build** (has the W4A16 ops; the LEAN IMAGE's baked /opt/kernels copy is STALE):
  built into a clean worktree `/home/pat/code/rdna4-hip-kernels-w4a16` via `local/build_local.sh` in
  the lean image. The `.so` + package sit at
  `/home/pat/code/rdna4-hip-kernels-w4a16/w4a8_fp8_wmma/torch-ext`. Mount it FIRST on PYTHONPATH.

---

## INFRA RECIPES (all validated this session)

**Images:** serve on `minisgl-rdna4:lean` (baked canonical kernels at /opt/kernels; but its
`w4a8_fp8_wmma` is STALE — no W4A16 MoE ops, so mount the fresh build ahead). Scratchpad for all temp
files: `/tmp/claude-1000/-home-pat-code-minisgl-rdna4/<session>/scratchpad` (recreate per session).

**Serve GLM-AWQ with W4A16 (the fp16-act path) — reasoning probe:**
```
gpu-lease -n 2 -- docker run --rm --name glm_w4a16 \
  --device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined \
  --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
  -v /home/pat/code/rdna4-hip-kernels-w4a16/w4a8_fp8_wmma/torch-ext:/pkg \
  -v /home/pat/code/minisgl-rdna4-rxfglm:/engine \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  --entrypoint bash minisgl-rdna4:lean -lc '
    export PYTHONPATH=/pkg:/opt/kernels:/engine/python:/engine
    MINISGL_KV_FP8=0 MINISGL_MOE_W4A16=1 python -m minisgl \
      --model QuantTrio/GLM-4.7-Flash-AWQ --tensor-parallel-size 2 --host 0.0.0.0 --port 1919 \
      --disable-pynccl --cache-type radix --cuda-graph-max-bs 0 --memory-ratio 0.80'
```
Drop `MINISGL_MOE_W4A16=1` and `-v …:/pkg` (leave /opt/kernels) for the fp8 baseline. Probe via
`/v1/chat/completions` greedy: "What is 17 + 25? Answer briefly." (want 42),
"What is 127 multiplied by 8? Think step by step." (want 1016).

**Kernel/function validation harnesses (scratchpad, reuse):** `w4a16_moe_validate.py`,
`w4a16_moe_fn_e2e.py`, `w4a16_dense_validate.py`, `w4a8_moe_e2e.py`, `wmma_g128_parity.py`,
`tp2_shard_check.py`, `moe_align_check.py` — pattern: mount `/pkg` (fresh w4a8) + `/engine`,
`PYTHONPATH=/pkg:/opt/kernels:/engine/python:/engine`, real int4 weights vs true fp16 GEMM ref.

**Per-layer dump harness:** `qwen3_5.py` has an env-gated `MINISGL_DUMP_LAYERS=<path>` residual-stream
dump (prefill, last token). vLLM reference: `scratchpad/vllm_layer_dump.py` (module-level decoder-layer
monkey-patch for spawn workers) — BLOCKED because 35B OOMs a hand-rolled `LLM()` on 2×16GB even at
fp8-KV/max_num_seqs=1/max_num_batched_tokens=1024; only the compose serve fits. To unblock: a vLLM
plugin dump under `docker compose --profile serve`, or a smaller reproducing model.

**Build a fresh canonical kernel package for the lean image** (source-isolate first — kernel repo may
be mid-edit by another agent): `git worktree add --detach <wt> <commit>` then
`docker run -v <wt>:/work -e GPU_ARCHS=gfx1201 -e ROCM_INCLUDE=/opt/rocm-7.2.1/include \
 --entrypoint bash minisgl-rdna4:lean -lc 'cd /work/<pkg> && bash local/build_local.sh'`.

---

## GOTCHAS

- The **W4A16 wide kernel needs `group_size ≥ 64`** (g=128 → wide 8; g=64 → 4; **g=32 unsupported**).
  So the W4A16 path works for **GLM-AWQ (g128)** but NOT **Qwen-CT (g32)** as-is. `_w4a16_wide()` raises.
- `MINISGL_MOE_W4A16=1` **frees `_w_op`** in post_load (only `_w_rep` kept) → the fp8 path can't run;
  it's all-M W4A16. A fp8-prefill/W4A16-decode "decode" mode would need both (doubles expert memory → OOM).
- Degenerate output is **chaotic under greedy** — don't over-read a single incidental correct token
  ("127×8=1016" appeared inside a loop once). The real test is: does `17+25` cleanly give `42` and stop?
- Lean image `/opt/kernels/w4a8_fp8_wmma` is **stale** (no W4A16 MoE ops) — always mount the fresh
  `/pkg` build FIRST on PYTHONPATH.
- Don't `pkill -f` broad patterns from the Bash tool — it kills the tool's own shell (exit 143/144).
  Kill containers by name (`docker kill <name>`).

## BYPRODUCTS BANKED (not the bug fix, but real & kept)
- **RXF ported to canonical rdna4-hip-kernels** (`rxf/`, parity-green) + `rxf_moe` precomputed-topk wired.
- **RXF quantizer sensible-default layer skips** (dense-replace + MTP, config-driven) in the paroquant
  registry (`vllm-gfx1201/.rxf-inspect/model_registry.py` `runtime_ignore`).
- **GLM-4.7-Flash-RXF checkpoint** built (21.88 GB, valid) at `/home/pat/.cache/huggingface/GLM-4.7-Flash-RXF`
  (serves degenerate — same model-assembly bug, not RXF).
- **minisgl W4A16 MoE + dense path** — working, numerically validated (cos=1.0); the real fix once the
  assembly bug is found may still want it (vLLM's correct path is W4A16/marlin).
- EOS/stop-token fix (branch `fix/eos-stop-tokens`, worktree `-eos`) — still unlanded; separate.

## MEMORY
Full diagnosis (with the Update 1→5 correction trail) is in the auto-memory:
`glm-awq-fp8act-decode-degrades.md`. Read it first.
