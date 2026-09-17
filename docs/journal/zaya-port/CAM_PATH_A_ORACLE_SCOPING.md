# CAM Path A — the from-scratch co-trained Titans-MAC ORACLE (cost + build scoping)

**What this is.** A scoping document (CPU-side; no model/training code written here) for building a
**minimal, from-scratch, co-trained Titans-MAC** model whose ONLY purpose is to **measure whether
trained-in, co-trained memory can do multi-hop edit-propagation** — the one capability our frozen-base
bolt-on program proved a frozen base cannot deliver above the in-context ceiling. It is a **yardstick /
upper-bound probe, not a product.** We build the smallest model that could still exhibit the capability,
run our existing edit-ripple disambiguation eval on it, and read a single GO/NULL.

**Why now (it is the sanctioned next question).** The whole frozen-base track has CONVERGED
([[cam-titans-distance-scorecard-results]], [[cam-why-injection-fails-diagnosis]],
[[cam-cartridge-editripple-build]]): every activation-write form (additive / KV-attendable /
recurrent-seed / cartridge / lightly co-trained LoRA) tops out at **≈ RAG** on bridge-routed hops and
**below RAG** on competing-path hops — the **F2 ceiling** (frozen hop-2 attention). The ROUND-2 fork
(brief §6) said: run Path D first, then pursue **Path A "if D matches RAG and we want to test whether
trained-in can EXCEED it."** D matched RAG and did not exceed it, exactly as predicted. That precondition
is now met. Path A is the only route left that could exceed the in-context ceiling, because it is the
only one that lets the reasoning circuit be **trained to consume the memory** (F3's converse).

---

## 0. The single most important finding of this scoping pass

**The hard part of a Titans build already exists in our tree and is validated.**
`/home/pat/code/memory-organ-tapmetric/cam/deep_memory.py` is a **graph-free, differentiable,
bit-parity-validated Titans deep-memory cell**:

- depth-2 **MemoryMLP** whose weights `W1,W2` are *per-sequence test-time state*;
- **per-token surprise gradient** as closed-form matmuls (`cam/deep_mem_analytic.py`, `analytic_surprise`
  = explicit MLP backprop, no `torch.func` `create_graph` — this is the OOM fix that killed lucidrains
  `titans-pytorch`);
- **α/θ/η per-chunk gates** (decay / adaptive-lr / momentum) from chunk reps, exactly the Titans/LMM cell;
- **momentum + decay parallel scan** (`cam/store_recurrence.py`, `_gated_scan`);
- **causal short conv** on ingest, `retrieve()` = `gelu(q@W1)@W2`, `forward()` = the surprise store update.

Crucially it is **pure torch** (plain matmuls) and **differentiable wrt the outer params via first-order
graph only** — i.e. it is exactly the shape an **outer-loop pretraining loss can backprop through**, and
it depends on **neither fla nor gdn_hip**. This reframes the brief's "Path A = custom kernels, no code,
high effort" line: the *inner loop* (the genuinely hard, novel-numerics part) is **done**. What remains is
comparatively ordinary engineering (§2). It also **decouples Path A from the fla RDNA4 segfault** — that
segfault only ever bit the frozen *Qwen3.5-GDN* base via fla's gated-delta layer
([[cam-native-gdn-slower-than-torch]]); a from-scratch MAC never imports fla at all.

---

## 1. Architecture — the minimal oracle config

Titans-MAC specifically (Memory-As-Context): the memory module's `retrieve()` output is **prepended as
context tokens that every attention layer reads**, and the whole thing — backbone attention, the memory's
k/v/q projections, the α/θ/η gate nets, the persistent tokens — is **co-trained from init** (Titans outer
loop). Inner loop updates only the MemoryMLP weights, at test time, via surprise (already implemented).

**Design tension for an oracle.** Smaller = cheaper and "smallest that still exhibits the capability" is
the goal — BUT the eval **filters to cases where the base composes the multi-hop chain true without any
edit** (you can only test whether an edit *ripples* through a chain the model can already reason over). Too
small a base → that filter collapses (the frozen-base cartridge runs already hit **filtered-n = 5**;
[[cam-cartridge-editripple-build]]) → the test is underpowered, and a null becomes **uninterpretable**
(can't tell "MAC didn't ripple" from "base can't reason"). So the oracle must be the smallest model that
still clears a **base-competence gate**, not the smallest model outright.

**Recommended oracle: ~230M co-trained Titans-MAC.**

| Component | Pick | Note |
|---|---|---|
| d_model | **1024** | |
| n_layers | **16** | depth matters — Biran "Hopping Too Late" shows hop-2 fires in *later* layers; too shallow can't compose the second hop at all |
| n_heads | **16** (head_dim 64) | standard |
| FFN | ratio 4 (→4096), GELU or SwiGLU | |
| norm / pos | RMSNorm + RoPE | |
| context length | **4096** | matches all Titans/Miras headline runs; keeps 4K activations small on 16 GB |
| vocab / embed | **32k** (Llama/Mistral tokenizer), **tied** | small vocab keeps embedding params modest at small d_model |
| persistent memory tokens | **32** | learned, prepended (Titans "persistent memory") |
| **memory module** | `cam/deep_memory.py` **as-is**: heads 4, **L_M = 2**, expansion 4, chunk_size 16, causal-conv k=4, mem_dim = d_model | depth-2 is what the analytic surprise supports; matches Miras "2-layer MLP memory" |
| MAC insertion | memory `retrieve()` output + persistent tokens **prepended at every attention layer** | the faithful MAC "every attention layer reads it"; reuse the `cam/cartridge.py` per-layer prepend pattern |

**Param budget (verified):** attn+FFN 12.6M/layer × 16 = 201M; tied embeddings 33M; memory module ~3M →
**≈ 237M**.

**Fallback down-scale (only if throughput-bound):** d_model 896 / 14L ≈ **166M**, or 768 / 12L ≈ **111M**.
Do **not** drop below ~150M without expecting the base-competence gate (§4) to fail. The oracle is a
diagnostic, not a headline model — but it cannot be so weak the diagnostic reads noise.

**Stretch knob if the depth-2 base delivers the edit but does not ripple:** bump the MemoryMLP to L_M=3–4
(Miras attributes generalization to memory depth). This requires generalizing `analytic_surprise` past
depth-2 (or accepting a `torch.func` fallback for the deeper memory only) — flag as a contingency, not the
default.

---

## 2. Kernel / module inventory — HAVE vs NEED

### HAVE (validated, in-tree)
- **The Titans memory cell** — `cam/deep_memory.py` + `cam/deep_mem_analytic.py` + `cam/store_recurrence.py`.
  Graph-free, pure-torch, first-order-differentiable, bit-parity vs autograd. Runs on CUDA **and** gfx1201,
  no fla, no gdn_hip. *This is the hard/novel part and it is done.*
- **A per-layer attention context-prepend** — `cam/cartridge.py` (`_prepend`: prepend timeless K/V into
  every attention layer with an extended mask). Written for a frozen HF model, but the *mechanism* is the
  MAC context-prepend; port the pattern onto our own backbone.
- **Titans-MAC/LMM co-train scaffolding** — `cam/cotrain_lora.py` (the "circuit learns to route on the
  injected memory read" experiment).
- **The whole edit-ripple eval** — `cam/mquake_ripple.py` (multi-hop ripple, MQuAKE-CF-3k schema),
  `cam/cartridge_ripple.py`, the **disambiguation protocol** (`CAM_RIPPLE_TRAIN_HOP` / `CAM_RIPPLE_EVAL_HOP`,
  the language→{family,script,country} 2-hop), the base-compose validity filter, and
  no_edit / RAG / memory conditions. Data present: `data/counterfact.json`; MQuAKE-CF-3k loaded by path.
- **BABILong harness** with Titans reference numbers — `cam/babilong_eval.py` (Titans-MAC ~62% vs GPT-4
  ~45%) — a secondary sanity axis if we want to confirm the oracle reasons at all.
- **Proven cloud path** — cloud-lease + RunPod recipe ([[cloud-cam-ripple-cuda-fla-recipe]],
  [[cam-cartridge-cloud-recipe]]); `gpu-lease`/`gpu-status` arbiter.
- **gdn_hip fwd+bwd** ([[gdn-hip-backward-validated]]) — available but **NOT on Path A's critical path**
  (that's for GDN/frozen-Qwen work). Only relevant if we later want a GDN-flavored memory.

### NEED (not in tree)
1. **A from-scratch decoder-only backbone.** No standalone LM exists in `cam/` — `DeepMemory` is a bolt-on.
   Vanilla: RoPE + SDPA + RMSNorm + tied embeddings + SwiGLU/GELU FFN + the 32 persistent tokens. ~a few
   hundred lines, entirely standard.
2. **The MAC wiring on OUR backbone.** `cartridge.py` prepends a *static trainable* KV into a *frozen* model;
   MAC prepends the **dynamic memory `retrieve()` output** into a **trainable** model, co-trained end-to-end.
   New wiring, but the prepend/mask bookkeeping is a direct adaptation of `cartridge.py`, and the memory
   module plugs in unchanged.
3. **An outer-loop pretraining harness.** The existing trainers fit an adapter against a *frozen* base; none
   does full-model pretraining. Need: corpus pipeline, AdamW + cosine schedule, gradient checkpointing, DDP,
   periodic eval. Standard boilerplate — the inner-loop surprise update lives *inside* `deep_memory.forward`
   and the outer loss backprops through it automatically (first-order, graph-free).
4. **Eval repoint** to the from-scratch model + the memory-write edit path (§4). Moderate; the harness exists.
5. **(Optional, NOT critical-path) a Triton/HIP surprise-scan kernel.** `deep_mem_analytic` notes "the math
   the Triton kernel will implement." Only a throughput optimization; the pure-torch path answers the
   question. **No new kernel is required to get the number.**

### On fla specifically (the brief asked)
fla ships **DeltaNet / GatedDeltaNet (== Qwen3.5's GDN) / delta_net / gated_deltanet** and the linear-attn
family — it does **not** ship a Titans-MAC test-time-surprise **MLP** memory as a clean drop-in. The Titans
reference impl is **lucidrains `titans-pytorch`**, which `deep_memory.py` was written to **replace** (its
`create_graph` surprise OOMs past ~6 segments). So for Path A, fla is **neither necessary nor sufficient**:
not necessary (our memory cell is pure torch), not sufficient (fla lacks the MLP-surprise memory). Net: the
Path A cloud image does **not** even need `flash-linear-attention` installed — it is *simpler* than the
frozen-base fla recipe (plain CUDA torch + our `cam/` code + a train loop).

---

## 3. Where to train — the decisive cost lever

**The fla segfault is a red herring for Path A.** The reason cloud-CUDA-via-fla was the path of least
resistance for the *frozen-base* experiments was that they ran the frozen **Qwen3.5-GDN** base, whose HF
gated-delta layer is fla and **segfaults in stage-2 backward on RDNA4**
([[cam-native-gdn-slower-than-torch]]). A from-scratch MAC uses **vanilla SDPA attention + the pure-torch
`deep_memory` cell** — it never touches fla. So **local gfx1201 is genuinely viable for Path A** (unlike the
frozen-base runs), and the decision comes down to **throughput and shared-box contention**, not correctness.

### Token budget (and its risk)
Papers use **15–100B tokens**; an oracle can be far smaller, but there is a **floor**: the base must know
enough world knowledge that the eval's *base-composes-true* filter yields a usable n. Chinchilla-optimal for
230M is ~4–5B tokens — enough for language competence but **thin on the factual composition** (language →
country → head-of-state) the disambiguation eval probes. **Mitigation (legitimate for an oracle):** train on
a **Wikipedia-heavy corpus augmented with the CounterFact/MQuAKE relation schema** (languages, countries,
capitals, heads-of-state, families, scripts) so the base learns the *exact composition structure* the eval
tests at far fewer tokens than general web.

- **Floor: ~8–10B tokens.** Below this, expect the base-competence gate (§4) to fail → filtered-n collapse →
  underpowered/uninterpretable, **NOT a clean negative.** This is the single biggest scientific risk.
- **Target: 15B tokens.** Comfortable competence for a schema-augmented 230M base.
- **Comfortable upper: 30B.** Diminishing returns for a diagnostic.

### Cost (FLOPs ≈ 6·N·D, MAC ≈ 2× plain-transformer FLOPs for ingest/retrieve + prepended context)

| Config (15B tok, MAC 2×) | GPU-hr | GPU-days | Cloud $ |
|---|---|---|---|
| **1× A100-80GB** (~180 TF eff, $1.5/hr) | **65** | **2.7** | **~$100** |
| 1× H100 (~350 TF eff, $2.7/hr) | 33 | 1.4 | ~$90 |
| 4× RTX 3090 ($0.88/hr total, cheapest $/hr, slow) | 585 | 24 (→~6 wall-days on 4×) | ~$130 |
| **Local 2× gfx1201** (~15–20 TF eff each, shared box) | ~500–600 | ~20–25 (→~10–14 wall-days on 2×) | electricity + **1–2 weeks of shared-box monopoly** |

Token-budget sensitivity on A100: **8B → ~35 GPU-hr / ~$50**, 15B → ~65 / ~$100, 30B → ~130 / ~$195.

### Recommendation: **cloud CUDA, single A100 or H100.**
- **A100/H100 finish the pretraining run in 1.5–3 wall-days for ~$90–200** (add restarts/hparam sweeps and a
  little eval → **budget $150–250**).
- It **frees the two contended gfx1201 cards** — a 20–25 GPU-day run would monopolize a shared card for
  **1–2 weeks**, a heavy tax on every other agent/repo on the box (the GPU-lease protocol exists precisely to
  avoid this kind of long hold).
- It **reuses the proven cloud-lease + RunPod recipe**, minus the fla install (Path A doesn't need it).
- **Local gfx1201 is the fallback** if cloud is unavailable — now technically correct (no fla dependence),
  just slow and box-hostile. Use it for **eval** (cheap, short) even when training on cloud.

Pick **A100** as the default (best $/GPU-day for this size; H100 only if wall-clock is urgent). 3090 is the
lowest absolute $ but ~6 wall-days and not worth the babysitting for a one-off.

---

## 4. Eval wiring — how the existing edit-ripple eval attaches, and the GO/NO-GO

The disambiguation harness (`mquake_ripple.py` / `cartridge_ripple.py`) was built to score a **frozen base +
injected memory vs RAG vs no_edit**. For a from-scratch MAC there is **no separate frozen base — the MAC
model IS the thing under test.** The three conditions re-map cleanly:

| Condition (frozen-base harness) | Path A re-map |
|---|---|
| `no_edit` (question only) | MAC, question only |
| `rag` (edit prepended as text) | MAC with **edit-in-context** (prepended text) — the in-context ceiling |
| tap/cartridge (injected memory) | **MAC memory-write**: ingest the edit statement so the **test-time surprise update** folds it into the MemoryMLP, then ask the question — the MAC's *own* trained-in memory path |

**What stays (do not change these — they are the load-bearing controls):**
- **Base-compose validity filter** — score ripple only where the from-scratch base composes the true
  multi-hop chain *without any edit*. This doubles as the **base-competence gate**: if filtered-n is small
  (say < ~30), the model is too weak → the token budget was too low → **re-train with more tokens before
  reading any MAC result.** (This is the §3 floor risk made operational.)
- **Disambiguation train-hop ≠ eval-hop** (`CAM_RIPPLE_TRAIN_HOP` / `CAM_RIPPLE_EVAL_HOP`): write the edit as
  ONE relation, score ripple on a **held-out** relation. This is what separates a genuine *installed belief*
  (composes into any downstream relation = reasoning) from a *trained shortcut* (the failure mode that sank
  every frozen-base variant, [[cam-b4-ripple-shortcut-not-belief]]). **TRANSFER ripple is the headline number.**

**Sanity gates before the headline is meaningful:** (a) filtered-n ≥ ~30; (b) MAC **control** (train-hop =
eval-hop) ripples high (≥ ~0.8) — proves the memory delivers at all; (c) single-hop delivery intact.

### The GO / NO-GO read

Reference points (all our own, same metric family):
- RAG / in-context ≈ **0.5** (the ceiling every frozen-base method reached);
- frozen-base **tap** transfer ≈ **0.31** (mid-layer) / **0.455** (late-layer);
- frozen-base **cartridge** ≈ **0.5** (≈ RAG single-hop-derived, [[cam-cartridge-editripple-build]]);
- frozen-base co-trained-LoRA transfer ≈ **0.25** (control 0.923 — the shortcut signature);
- MAC **control** (trained hop) target ≈ **0.85** (what "the memory delivers" looks like).

- **GO — co-training IS the lever.** MAC **memory-write TRANSFER ripple materially EXCEEDS RAG**
  (> ~0.6, ideally trending toward the 0.85 control). This would be the **first positive evidence** that
  trained-in, co-trained memory composes an edit across a held-out hop where *every* frozen-base method
  plateaus at the in-context ceiling. It says: build the real thing (a larger co-trained memory model).
- **NULL — co-training is not sufficient at this scale.** MAC transfer ≈ **RAG (~0.5)** or ≈ the tap's
  **0.31**. A strong, publishable negative: even from-scratch co-trained MAC does not exceed in-context for
  edit-ripple at this scale — the F2 diagnosis extends even to a trained base, or the capability needs more
  scale than an oracle carries.
- **INVALID — underpowered.** filtered-n small or control ripple low → fix the base (more tokens / larger
  config) and re-run. **Not** a negative.

---

## 5. The honest caveat (stated plainly, per brief §5)

**No paper — Titans included — has ever tested multi-hop edit-propagation.** Titans' "reasoning" headline is
**BABILong multi-fact aggregation** (reason across facts *present in a long context*), which is adjacent to
but **not** an edit rippling through a chain the model composes *itself*. Our metric is unaddressed across the
entire literature we read (Titans, Miras, Nested Learning/HOPE, LongMem, Cartridges, AtlasKV — brief §5, the
ROUND-2 "UNIVERSAL finding").

Consequences that must not be oversold:
- A faithful Titans-MAC oracle is an **open experiment, not an answer-key.** It measures **whether
  co-training the memory-read is the right lever** (by exceeding the frozen-base ceiling, or not). A **GO** is
  a *new* result; a **NULL** falsifies only "*this* minimal co-trained MAC at *this* scale does it," **not**
  "no Titans variant could."
- **Scale is a confound on a null.** Titans' flagship is ≤760M; our oracle is ~230M. The base-competence gate
  (§4) controls for "too weak to reason at all," but it does **not** rule out that edit-ripple simply needs
  more scale. Report a null as scale-bounded.
- The oracle **does not hand us a correct design to copy.** It tells us whether to invest in a trained-in
  memory direction at all — it is a **yardstick / upper-bound probe**, exactly as commissioned.

---

## Bottom line

**A Path A oracle costs roughly 3–5 GPU-days / ~$150–250 of cloud CUDA (a single A100 or H100, ~15B
schema-augmented tokens; the pretraining run itself is ~1.5–3 wall-days and ~$90–200, plus eval and a
restart or two), because the hard, novel part — the graph-free, outer-loop-differentiable Titans memory
cell with test-time surprise and α/θ/η gates — already exists and is bit-parity-validated in
`memory-organ/cam/deep_memory.py`, so the build reduces to a small from-scratch decoder backbone, MAC
context-prepend wiring adapted from `cam/cartridge.py`, a standard AdamW pretraining loop, and repointing
the existing edit-ripple disambiguation eval; it depends on neither fla nor gdn_hip and so is unaffected by
the RDNA4 fla-backward segfault (local gfx1201 is a viable-but-box-monopolizing fallback, cloud is
recommended purely for throughput and to free the shared cards). It answers exactly ONE question: does a
co-trained Titans-MAC ripple a HELD-OUT edit through a self-composed multi-hop chain ABOVE the RAG /
in-context ceiling (~0.5) that every frozen-base method — tap 0.31, cartridge 0.5, co-trained-LoRA 0.25 —
plateaus at (GO if transfer > ~0.6 toward the ~0.85 trained-hop control; NULL if it sits at RAG/0.31) —
with the standing caveat that no paper, Titans included, has ever tested edit-propagation, so the oracle is
an open upper-bound probe, not a design to copy, and a null must be read as scale-bounded rather than as a
falsification of the whole idea.**
