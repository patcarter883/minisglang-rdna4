# CAM — Titans / Miras / Nested Learning / LongMem research brief (pre-planning)

Deep read of the Google memory line + the one frozen-base precedent, done 2026-07-12 to inform the
"trained-in memory" build decision. Follows the frozen-base falsification
([[cam-b4-ripple-shortcut-not-belief]], [[cam-why-injection-fails-diagnosis]]). Sources:
Titans arXiv:2501.00663 · Miras arXiv:2504.13173 · Nested Learning/HOPE arXiv:2512.24695 (blog:
research.google/blog/introducing-nested-learning-...) · LongMem arXiv:2306.07174.

## 1. The decisive fact: the entire Google line is FROM-SCRATCH, never frozen-base
- **Titans**: trained end-to-end from scratch. Nested loops — OUTER loop (ordinary pretraining) trains
  attention W_Q/W_K/W_V/W_O, the memory's k/v/q projections, the per-token gate nets (α,θ,η), persistent
  tokens; INNER loop updates ONLY the memory MLP weights at test time (a forward-pass surprise update).
  Attention is co-trained WITH memory from init. **Zero** frozen-pretrained-base experiments. "Frozen"
  never appears for a base model.
- **Miras**: same — "we REPLACE attention modules with Miras variants in Llama's macro architecture,"
  trained from scratch. Remark 2: learning key→value is a META-LEARNING problem where "all other
  parameters of the network (projections, convolutions) are optimized in the outer loop."
- **Nested Learning / HOPE**: a NEW from-scratch architecture (self-modifying + Continuum Memory System,
  multi-frequency updates). Solves forgetting architecturally, not by bolting onto a frozen LLM.
⇒ Our frozen-base failure was never in these papers' regime. They sidestep it by co-training. Confirmed.

## 2. Miras diagnosis of OUR bug (the sharpest result of the research)
Our design (matrix/MLP store · MSE/dot-product attentional bias · simple gate · GD) = the DeltaNet/
Titans-LMM cell of Miras Table 1 — already the "standard" cell every strong model occupies. But:
- **The 4 Miras axes DO NOT govern multi-hop composition.** They tune single-pass memorization —
  (i) memory architecture, (ii) attentional bias/objective, (iii) retention/forget gate, (iv) optimizer —
  all about robustness/recall-capacity/forgetting. The non-Euclidean objectives (Huber/p-norm/simplex =
  Yaad/Moneta/Memora) are motivated by noise/outliers, NEVER by cross-hop transfer. Changing MSE→Huber
  would not make a stored fact compose into a 2nd hop. **The axis the paper is ABOUT (ii) is the one it
  gives least reason to expect helps us.**
- **The real lever is Remark 2's OUTER loop, which a frozen base severs.** The memory minimizes a LOCAL
  key→value loss against a read-out path that (frozen) has only a single-hop route — so it's free to learn
  a hop-A surface shortcut nothing pressures into a composable form. "Delivers single-hop, hop-A shortcut
  that doesn't transfer" is exactly the signature of an inner-loop store optimized against a frozen outer
  loop. No axis choice repairs this; the composition failure lives in the frozen outer loop we don't train.
- **Secondary lever (hypothesis): axis (i) memory DEPTH.** Miras attributes generalization to the
  expressive 2-layer MLP memory (linear-memory ablation is the 2nd biggest loss). A vector/linear store can
  only memorize a lookup (→ shortcut); a deeper MLP could fit a rule. Only shown for capacity/long-context,
  never multi-hop — treat as hypothesis. (Our pointer/id-bank = Miras's "value-less memory," Lp p→1.)

## 3. LongMem — the ONE proven frozen-base recipe, and its ceiling
- Frozen backbone (407M GPT-2, KV source only) + trainable **SideNet** (151M, L/2 layers, ~27% trainable),
  26B tokens. Memory = 65k-token KV cache from backbone layer 18, top-64 retrieval, joint-attention +
  sigmoid gate — ALL fusion in the side path. Backbone never modified to read memory.
- **Works because it does NOT fine-tune the frozen attention** (which is what catastrophically forgot in
  our Rung 2). Its stated motivation is our exact bug: "directly adapting the entire LLM ... suffers from
  catastrophic forgetting." Decoupled side-net avoids stale KV + preserves knowledge.
- **Ceiling: retrieval/long-context/many-shot ICL ONLY** — ChapterBreak, PG-22 ppl, 5-task ICL. NO
  multi-hop / compositional reasoning demonstrated. Past its envelope for our goal.

## 4. Feasibility — small and affordable
- Titans/Miras headline results all **≤760M–1.3B params**, **15–100B tokens**, **4K training context**,
  memory = **2-layer MLP** (deeper helps more at small scale). Titans-MAC beats GPT-4 on BABILong multi-fact
  reasoning at ≤760M. A 170–400M from-scratch Titans/Miras is trainable on 2×16GB + modest cloud (~single-
  digit GPU-days; 4K ctx keeps activations small). No paper reports $ / GPU-hours. HOPE at 1.3B.

## 5. The critical GAP across the WHOLE literature (risk + opportunity)
NONE of Titans / Miras / Nested Learning / LongMem tests **multi-hop EDIT-propagation / knowledge
composition** (MQuAKE/CounterFact-style). Titans' "reasoning" = BABILong multi-fact *aggregation*
(reason across facts in a long doc) — adjacent to, but NOT, edit rippling through a reasoning chain.
⇒ Even a faithful from-scratch Titans is NOT proven to do what we want. This is (a) a real risk for any
build and (b) a genuinely open question / potential contribution — no one has tested trained-in memory on
edit-ripple, which is precisely our metric.

## 6. Candidate paths (for planning — not yet chosen)
- **A. From-scratch small Titans-MAC (170–400M), then run OUR disambiguation edit-ripple test.** The only
  clean way to positively answer "does trained-in, co-trained-read memory transfer cross-hop." Faithful;
  a few GPU-days; reuses fla/GDN infra. Risk: Titans may not do edit-propagation (untested) — but that IS
  the novel question. Strongest scientific payoff.
- **B. LongMem-shaped trainable read-path on a FROZEN base, aimed at reasoning.** Miras-Remark-2 fix
  (make the read/consume path trainable) done LongMem-correctly (separate residual side-net, DON'T fine-tune
  frozen attention → avoids the Rung-2 forgetting). Cheaper (adapter-scale, base frozen); reuses our stack.
  Risk: LongMem only showed retrieval; side-net composition unproven. Answers "can a frozen base be made to
  integrate via a decoupled trained reader."
- **C. Minimal: deeper-MLP memory (axis i) + small trainable read adapter** on current harness. Cheapest,
  most incremental; tests the axis-(i) rule-vs-lookup hypothesis. Weakest as a definitive answer.

Recommendation lean: **A** for the definitive answer (it's the faithful contrast and a novel result),
with **B** as the pragmatic option if we want to keep the frozen base and reuse infra. Decide before spec.

---

## ROUND 2 (2026-07-12) — Cartridges, AtlasKV, Nested-Learning primary read

Two papers the user surfaced from a LocalLLaMA thread + a rigorous primary read of Nested Learning.

### Cartridges (arXiv:2506.06266, Stanford Hazy) — the best-aligned frozen-base route
- **Method:** frozen base, train ONLY a small KV-cache prefix (`p` virtual tokens, ~0.15–1 GB) via
  **context distillation**: KL( teacher = frozen model WITH real context in-window ‖ student = frozen
  model WITH cartridge ), over next-token dists. Data = "self-study": diverse self-generated Q&A from
  GENERIC seeds (structuring/summary/question/use-case/creative — not task-specific). Init cartridge from
  first-p-tokens' KV (random unstable). Prefix > LoRA (LoRA's MMLU collapses).
- **Directly attacks our F2/F3:** paper's thesis = a STATIC/compressed KV cache loses multi-part
  "structural awareness" but a TRAINED one RECOVERS it — cartridges MATCH/BEAT ICL (LongHealth beats full
  ICL at 10× less mem) where lossy KV-compression collapses at 2×. = "amortize in-context reasoning
  (which composes, our RAG ~0.5) into a trained KV prefix." Our untrained KV-inject was 0.000; the
  TRAINING is the missing piece.
- **Cost:** ~30 min/8×H100 for 8B (unoptimized); only KV prefix trainable, base in inference mode →
  a 4B on 2×16GB+cloud is feasible (hours). Cartridges are COMPOSABLE (concat w/o retraining).
- **Gap = exactly ours:** NO counterfactual/edit test; "composition" = parallel UNION of corpora, not
  CHAINED A→B→C. Self-study only distills its question distribution → single-hop seeds can still shortcut.
  ⇒ actionable: teacher = base+edit-in-context (we KNOW it ripples), distill, test edit-ripple on HELD-OUT
  hops via our disambiguation protocol. CEILING = RAG's ripple (it distills RAG); win = persistence/
  efficiency/composability + being the FIRST frozen-base method to recover in-context ripple.

### AtlasKV (arXiv:2510.17934) — RULED OUT (confirms our wall)
KBLaM-at-billion-scale: KG triples → attendable KV branch, top-k retrieval, base frozen, only 3 proj
heads trained. Authors' OWN Limitations: flattened independent-triple design "will still block multi-hop
reasoning." Single-hop grounding only (52.7% vs KBLaM 5.5%). = confirmation of attendable≠integrated.

### Nested Learning / HOPE (arXiv:2512.24695) primary read — + a middle path I'd missed
- Confirmed FROM-SCRATCH, co-trained (Titans lineage). HOPE = self-modifying Titans + Continuum Memory
  System (chain of MLP memories each updating every C^(ℓ) steps). Canonical nums: 1.3B/100B → 14.39 ppl,
  58.04 avg (my earlier 57.23/15.11 were wrong). Anti-forgetting = multi-frequency redundancy.
- **★ §7.3/§9.1 "Ad-hoc Level Stacking" = a real middle path:** take a PRETRAINED model's FFN/MLP blocks,
  use them as initial states of multi-frequency CMS copies (η→0 recovers frozen exactly), then LIGHTLY
  continual-pretrain (Llama3-8B/3B, 15B tokens). NOT bolt-on-frozen-attention, NOT full-FT. Beat
  EWC/InCA/ICL on class/knowledge-incremental. BUT only tested on CLASSIFICATION continual learning, needs
  custom kernels, NO official code.
- **Same gap:** NO multi-hop edit-propagation/ripple (nearest = CTNL continual translation). Feasibility
  bound = TOKENS not params (smallest 760M/30B = many GPU-days; sub-400M on few tokens = weak base).

### UNIVERSAL finding across ALL 6 papers
NONE (Titans, Miras, Nested-Learning, LongMem, Cartridges, AtlasKV) tests multi-hop EDIT-propagation /
knowledge-editing ripple. Our metric is unaddressed literature-wide ⇒ novel regardless of path.

### REVISED FORK
- **Path D — Cartridges self-study, adapted for edit-ripple (FRONT-RUNNER).** Frozen base; teacher =
  base+edit-in-context (ripples); distill into a KV cartridge with generic self-study; test transfer to
  HELD-OUT hops (our disambiguation). Cheapest, most on-thesis, reuses infra+RAG-teacher+metric, novel.
  Honest ceiling = RAG-level reasoning (win = persistence/efficiency/composability + first frozen-base to
  recover in-context ripple transferably; our tap couldn't even match RAG).
- **Path A — from-scratch small Titans/HOPE.** Only route that could EXCEED RAG (trained-in belief) but
  expensive (30B tokens = GPU-days), custom kernels, no code, weak at small scale, AND still needs us to
  build the edit-ripple eval. High risk/effort.
- **Path E — NL §9.1 CMS pretrained-init + light continual-pretrain.** Middle path; continual-learning
  consolidation; unproven for reasoning; custom kernels, no code. Park as a later contrast.
- Recommendation: **Path D first** (cheap, decisive, on-thesis, novel), with A/E as follow-ons if D
  matches RAG and we want to test whether trained-in can EXCEED it.
