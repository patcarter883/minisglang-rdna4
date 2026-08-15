# CAM — Path D: Cartridge edit-ripple experiment (spec)

Chosen 2026-07-12 after the Titans/Miras/NL/LongMem/Cartridges/AtlasKV research
([[titans-miras-nested-longmem-research]], docs/zaya-port/CAM_TITANS_MIRAS_RESEARCH_BRIEF.md). Goal:
test whether a **Cartridges-style trained KV prefix** (arXiv:2506.06266) distilled from *the base-with-
edit-in-context* integrates an edit into MULTI-HOP reasoning and TRANSFERS to held-out hops on a FROZEN
base — where our injected/trained tap learned a shortcut ([[cam-b4-ripple-shortcut-not-belief]]).

## Hypothesis
In-context/RAG reasoning composes (our RAG ripple ~0.50–0.64); a static KV inject does not (0.000). A
cartridge = the frozen model's in-context behavior AMORTIZED into a trained KV prefix (KL-distill teacher=
base+edit-in-context ‖ student=base+cartridge). So the cartridge should RECOVER RAG-level ripple as a
persistent memory, and — the open question — TRANSFER it to hops absent from self-study. Ceiling = RAG
(it distills RAG); the win is persistence/efficiency/composability + being the first frozen-base method to
recover in-context ripple transferably (our tap couldn't even match RAG).

## Design calls
1. **Base = standard-attention Qwen3-4B** (clean per-layer KV prefix; hybrid Qwen3.5-4B has only 8 full-
   attn layers — revisit later). Eval is base-agnostic (base-oracle), so just re-establish RAG/no_edit.
2. **Minimal focused trainer** (new cam module), reuse fla/cloud recipe + plug into mquake_ripple
   disambiguation. Not the Hazy repo, not a bend of recall_mag.
3. **Self-study curriculum IS the experiment.**

## Method (Cartridges, faithful)
- Cartridge Z = trainable K/V vectors per layer, shape [n_layers, p, n_kv_heads, head_dim] ×2. p ∈
  {128, 512}. Init from the KV cache of the first p tokens of the edit text (random unstable per paper).
  Prepended to every layer's K/V (post-projection, post-RoPE-safe — position 0 block, sink-frozen).
- Teacher = frozen base with the edit as a text prefix in-context. Student = frozen base + Z (no edit
  text). Loss = Σ_i KL( teacher(·|edit ⊕ x[:i]) ‖ student(·|Z, x[:i]) ) over next-token dists, on the
  self-study conversations x. ONLY Z trains; base in inference_mode.
- Self-study data: generate diverse Q&A about the edit (generic seeds: structuring/summary/question/
  use-case/creative), teacher-generated. One epoch, no reuse.

## Experiment matrix (disambiguation: distill on X → eval held-out hops; RAG=ceiling, tap 0.31=floor)
- **D0 generic**: generic seeds, NO hop-specific Q → eval country/family/script. STRONGEST if ripple
  emerges with zero hop supervision.
- **D1 single-hop**: distill only the direct edit cloze → does it still transfer to 2-hop? (control)
- **D2 one-2-hop**: distill `country` 2-hop Q → eval `family`/`script` (our exact disambiguation).
Per variant compare: cartridge vs RAG (teacher) vs no_edit vs old tap. SUCCESS = cartridge transfer
ripple ≈ RAG on held-out hops (ideally D0, no hop supervision).

## Build steps
1. `cam/cartridge.py` — Cartridge (trainable per-layer KV prefix) + attach hooks (prepend K/V at each
   attn) + init-from-first-p-tokens.
2. `cam/cartridge_train.py` — self-study gen (teacher) + KL-distill trainer (only Z).
3. Wire into a runner that, after training Z, calls eval_indist_ripple with the cartridge attached
   instead of the tap bank (reuse the disambiguation env CAM_RIPPLE_TRAIN/EVAL_HOP for the eval hop).
4. Cloud run (fla, Qwen3-4B) via campaign-style launcher; baselines RAG/no_edit auto in eval.

## Risks
- Cartridge self-study distribution = single-hop → can still shortcut (that's why D0/D1/D2 vary it).
- Ceiling is RAG (won't EXCEED it) — if we want to exceed RAG, that's the from-scratch Path A, later.
- Qwen3-4B RAG ripple baseline must be re-measured (donor sweep had it ~0.57–0.80 noisy).

---

## D3 — "entailment self-study" (base writes its own multi-hop consequences) — added 2026-07-12

Motivation (user insight): use the FROZEN BASE ITSELF to write the memory — no separate donor/encoder, so
the memory is automatically in the base's own geometry (removes the translator gap; explains why the donor
bake-off was moot — a donor is just differently-foreign). The strong form: the base WITH the edit in
context genuinely composes (that's why RAG ripples), so let it write its own ENTAILMENTS of the edit into
memory, not just the bare fact.

**D3 self-study = multi-relation 2-hop-through-the-bridge Q&A about the SUBJECT**, generated+answered by
teacher (base + edit-in-context): "currency/capital/continent/head-of-state/population/neighbour of the
country where {subject-language} is spoken?" across 6–10 downstream relations. Distil all. **EXCLUDE the
eval relation** (family/script) from the self-study set → a positive transfer = genuine generalization.

Mechanistic bet = SHARED BRIDGE: satisfying many downstream relations via a per-relation output shortcut is
expensive, but ONE installed bridge ("the country is now <CF object>") serves them all → the parsimonious
distillation solution flips toward installing the belief, which then generalizes to the HELD-OUT relation.
Prediction: D3 transfer > D2 (single trained hop) on family/script; matching RAG = expected ceiling,
exceeding RAG = the strong result. This is the "multi-task the second hop forces bridge installation" lever
from [[cam-why-injection-fails-diagnosis]], now via base-written distillation instead of a trained tap.

Runner: `--variant D3` (reuses D2 machinery; only the self-study QUESTION SET changes). Compare D3/D2/D0/D1
vs RAG, eval family+script, ~50-60 edits. Queued behind the D0/D1/D2 family+script runs.
