# CAM — "Distance to Titans/HOPE without training a model" scorecard (program)

Goal (user, 2026-07-12): measure **how close a FROZEN-BASE approach (no training a model — only a
lightweight cartridge/memory) gets to Titans/HOPE's HEADLINE performance**, across all four of their
headline axes, then judge whether that level is **functionally useful**. RAG/ICL is a yardstick tool,
NOT the target — the target is Titans' capability. Individual injection tricks were never expected to win
alone; they are facts for assembling a best-of stack (two negatives → a positive).

## Two tracks
- **METHOD track (improve the frozen-base method):** cartridge (V0, matches RAG on edit-ripple) → D3
  entailment self-study → deep memory-layer ([[/docs/zaya-port/CAM_MEMORY_LAYER_SPEC.md]]) → BEST-OF STACK
  (cartridge + suppress-true delivery [best gate 0.833] + deep memory on hard hops). The winner is what
  gets run on the scorecard.
- **YARDSTICK track (measure distance to Titans across 4 axes):** run {frozen-base + best-stack} vs
  {RAG/ICL} vs {Titans/HOPE published numbers} on each axis below.

## The 4 axes (all selected)
1. **BABILong multi-fact reasoning** — Titans FLAGSHIP (Titans-MAC > GPT-4 / Llama-8B+RAG at ≤760M).
   Multi-fact reasoning distributed in a long doc. Direct Titans comparison. github.com/booydar/babilong.
   Metric = QA accuracy vs context length; compare cartridge / RAG / Titans-reported.
2. **Long-context recall (NIAH / passkey)** — Titans' capacity headline. Cartridges already ~ICL parity;
   less discriminating but completes the picture. Synthesize passkey at increasing depth/length.
3. **Knowledge-edit ripple (OUR axis)** — novel (Titans never tested). Scale up: MQuAKE OOD, multi-edit,
   longer chains. Cartridge ≈ RAG on family; underperforms on script. Already built (cam/cartridge*.py).
4. **Continual knowledge / no-forget** — HOPE headline (CTNL, class-incremental). Add facts/skills
   SEQUENTIALLY without forgetting. Tests the "living memory" product angle + cartridge COMPOSABILITY
   (independently-trained cartridges concatenate — Cartridges paper). Compare vs sequential-FT (forgets)
   and vs RAG.

## Deliverable = the scorecard table
For each axis: {frozen best-stack | RAG/ICL | Titans/HOPE published} + the GAP. The headline number =
"a frozen base + cartridge reaches X% of Titans' capability at Y× less cost, no model training."
Expected shape (from everything so far): frozen ≈ RAG on reasoning axes (matches in-context, doesn't
exceed); the Titans margin over RAG = the residual distance a frozen base cannot close without touching
the base. NIAH ≈ parity. Continual = the most open (composability could shine or reveal interference).

## Then: functional usefulness
Characterize what "≈in-context quality, but persistent / ~38× cheaper inference / composable / editable /
no-forget" is good for: long-doc QA, per-user memory, editable knowledge bases, continual accrual. Maps
back to the CAM production targets ([[cam-production-targets]]).

## Sequencing (GPU-aware: 2 local gfx1201 + cloud; cartridge runs local on Qwen3-4B)
1. Edit-ripple scale-up — extend the running cartridge harness (cheapest; in progress via D3).
2. Continual-knowledge — leverages cartridge composability; medium build; product-relevant.
3. NIAH — quick synthetic; easy.
4. BABILong — biggest build (external benchmark); highest value for the direct Titans number; start early.
Best-of stack + memory-layer feed in from the METHOD track as they mature.
