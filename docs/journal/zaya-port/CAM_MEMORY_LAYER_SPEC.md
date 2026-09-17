# CAM — Deep memory-layer + distillation build (spec)

Spec'd 2026-07-12 after: frozen-base injection falsified ([[cam-why-injection-fails-diagnosis]]); the
cartridge (static trained KV prefix, distillation objective) MATCHES RAG on family, underperforms on
script ([[cam-cartridge-editripple-build]]); Titans/Miras research ([[titans-miras-nested-longmem-research]]).
Goal: test whether a MORE EXPRESSIVE / more NATIVE injected memory module, trained with the cartridge's
distillation objective, transfers better than the static KV cartridge on HARD hops — combining Miras
axis-i (deep memory → rule not lookup) with the recompute-forcing distillation objective.

## The realization that shapes this
Our residual tap is ALREADY an injected cross-attention head (query from hidden, K/V from bank, OV-style
additive write). So "extra head" / "memory layer" / "additive tap" are ONE continuum of
"trained module that writes to the residual", differing in EXPRESSIVENESS and NATIVE-NESS:
  additive-vector  <  cross-attn head (current tap)  <  attention BLOCK (attn+MLP)  <  deep-MLP memory  <  recurrent memory
The "native module" principle: a module shaped like the base's OWN blocks (attention+MLP) writes
outputs the frozen downstream layers were trained to consume → more in-distribution than an arbitrary
vector. This build sweeps that axis, objective held fixed at distillation.

## What's held FIXED (the lever that already worked)
The DISTILLATION objective (cartridge D1 worked even with single-hop self-study ⇒ the objective, not the
curriculum, defeated the shortcut). Teacher = frozen base + edit-in-context; student = frozen base +
memory module; loss = KL over next-token dists on diverse self-study; ONLY the module trains. This is the
essential ingredient — the memory-layer variants change only the MODULE, not the objective.

## The module variants to sweep (expressiveness / native-ness axis)
- **V0 — static KV prefix (the cartridge, baseline, done):** matches RAG family, weak script.
- **V1 — deep-MLP memory (Titans-LMM):** per position, `q=Wq·h`; read a value from a 2–4-layer MLP
  memory keyed by q (init from the base's KV of the edit); write `h += g·Wo·read`. Tests Miras axis-i.
- **V2 — attention BLOCK (the "native module" / head idea):** a full trainable attention block inserted
  at a mid layer that attends over {the sequence ⊕ the memory bank} and writes attn+MLP to the residual —
  the LongMem-SideNet / Titans-MAC-minimal shape; the most in-distribution writer.
- **V3 — V1 or V2 + the winning write-op** from the delta/rotation/two-sided probes (fold in whatever, if
  anything, beat additive).
All trained by distillation, base frozen. Insert at a mid-early layer (bridge-formation band, ~L4–8) OR
every-layer like the cartridge — start with a single mid insert + compare to all-layer.

## Eval
Same edit-ripple disambiguation. PRIMARY comparison: each variant vs the KV-prefix cartridge (V0) vs RAG,
on FAMILY and especially SCRIPT (the hop where V0 underperformed — the discriminating test). Success =
a variant beats V0 on script and/or approaches RAG there. Report cartridge/module vs RAG vs no_edit,
filtered-n, delivery, cost.

## The scientific question it answers
Does EXPRESSIVENESS (deeper/more-native memory) buy rule-over-lookup transfer on hard hops, or does the
static KV prefix already capture everything a frozen base can compose? 
- If a variant beats V0 on script → expressiveness matters; deep memory is the lever for hard hops.
- If all variants ≈ V0 ≈ RAG → expressiveness is NOT the lever; RAG is the hard frozen-base ceiling
  (F2: frozen attention), and the terminal finding is "you can amortize in-context reasoning into a
  persistent memory but cannot exceed it without touching the base." Either outcome is publishable.

## Honest ceiling
Even the most expressive frozen-base memory likely tops out at ≈RAG, because F2 (frozen hop-2 attention)
is untouched. To EXCEED RAG needs co-training attention (forgetting → needs the preservation anchor) or
from-scratch (Path A). This build establishes the ceiling / finds the best frozen-base object; it is not
expected to beat RAG.

## Build
- Extend the cartridge trainer (cam/cartridge*.py) OR the tap (cam/gated_tap.py) with the module variants
  behind a `--mem-module {kvprefix,mlp,attnblock}` flag; reuse the distillation trainer + eval verbatim.
- Standard-attention Qwen3-4B (clean, local, no fla) as for the cartridge; every-layer vs single-insert
  as a secondary knob.
- Sequence: launch AFTER the write-op probes (delta/rotation/two-sided) report — they decide whether V3
  (write-op) is worth including and whether the write algebra has any headroom at all.
