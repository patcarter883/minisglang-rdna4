# CAM ← Titans reorientation — the bet, the state, the levers

**The bet (project aim, 2026-07-09):** achieve the Titans *end-goal* — a memory the model
**reasons with**, not just retrieves from — via a **bolt-on to a FROZEN pretrained base** (the tap),
**without** training Titans memory layers into the model. If it works, you get Titans-style
capability on *any* pretrained model with no retraining ("Titans for everyone" in the truest sense).
The success test is **not** single-fact recall (RAG's turf) — it is **multi-hop reasoning
integration** (ripple effects): does an edit propagate through a chain the model composes itself?

## State (measured)
- **Single-hop belief-override WORKS** (frozen base + tap): overrides a 0.99-confident prior to the
  counterfactual 0.64–0.79, LOCAL + GENERALIZES, zero context tokens. MAC (Titans Memory-as-Context)
  walls at 0.000; the MAG **tap** is the working organ. ([[cam-tap-belief-override-measured]])
- **Multi-hop ripple (MQuAKE-CF-3k):** the current tap **LOSES to generation-time RAG** — TAP
  ~0.18–0.21 vs RAG ~0.83–0.88 on base-compose ∩ tap-eligible. The single-hop edit delivers but does
  **not propagate through the reasoning chain**. (Single-hop control pending to confirm this is
  genuine *integration* failure, not a tap that isn't firing in free-gen.)

## The degeneration wall (MQuAKE free-gen, 2026-07-09) — the sharpest finding
Testing ripple in FREE GENERATION exposed the real blocker: the single-layer residual tap injected
EVERY step **degenerates into repetition** ("Finnish Finnish Finnish…"), so free-gen delivery craters
to 0.02–0.07 (vs 0.64–0.79 in the teacher-forced belief-override eval). This is the known
[[cam-tap-needs-seed-once]] problem. It reveals the CORE TENSION:
- **single-hop delivery** wants the tap ON briefly (emit fact, stop) → **seed-once** fixes it;
- **multi-hop reasoning integration** wants the edit present THROUGHOUT the chain → **sustained**
  injection — which the residual tap CANNOT do without degenerating.
So the current tap can DELIVER a fact but cannot SUSTAIN influence over a reasoning chain. The ripple
question is unanswerable with a single-layer always-on residual tap. The core research problem is
therefore a **NON-DEGENERATING SUSTAINED INJECTION** — which is exactly what levers 1a (multi-layer,
lower per-layer strength) / 1c (KV injection) are for. Router/logit injection does NOT qualify
(output-only, doesn't touch reasoning). This wall, not "does the edit propagate," is the thing to break.

## The gap, stated precisely
The bolt-on can WRITE to the base (single-hop) but the write does not yet PARTICIPATE IN REASONING.
This is a **training-objective / injection-geometry** problem on the bolt-on — NOT evidence that you
must bake memory layers into the model. So the bet stays alive; the levers below all keep the base
frozen and train only the tap(s).

## Levers (to make the bolt-on integrate into reasoning)
1. **BROAD/EARLY availability of the edit — the Titans placement principle (user idea 2026-07-09,
   refined against the paper).** PAPER FINDING: Titans does NOT tap many backbone layers — it inserts
   ONE memory module, but makes its output available EARLY/BROADLY so downstream reasoning uses it:
   MAC *prepends retrieved memory as context tokens every attention layer reads*; MAL places memory
   *before* attention (whole stack computes on top of it); MAG gates over the full sequence. Depth in
   Titans = the memory MLP's INTERNAL depth (L_M 1–4) + test-time learning, NOT multi-point insertion.
   OUR tap injects ONCE at L24 (75% depth) — the OPPOSITE: the intermediate hop is computed earlier and
   never sees the edit. So the fix is to make the edit available early/broadly:
   - **1a. Earlier + multi-layer RESIDUAL taps** (`--tap-layers` weighted early, e.g. `8,16,24`).
     Cheap; infra already supports the list (this run was `[24]`, `multi=False`). Approximates
     "throughout depth." Do first if multi-hop is poor.
   - **1c. Multi-layer KV injection (STRONGEST candidate; boltA-endorsed, untried at multi-layer
     scale):** inject the memory into attention KEYS/VALUES at multiple layers, so it's available to
     reasoning like context is (Titans-MAC gets this via attention) BUT at a level a frozen base can
     actually use. `recall_boltA.py`'s own diagnosis named "multi-layer KV" as the escalation for
     exactly this failure. More work than 1a; likely the real fix.

   PROJECT HISTORY (why NOT prefix/context injection — decision context, DIARY Phase 1):
   The original CAM "MAC" (`recall_boltA.py`) supplied the memory as a PREFIX in INPUT-EMBEDDING space
   → `memory ≈ no_memory`, the FROZEN base IGNORED it (an injected latent prefix is out-of-distribution;
   Titans-MAC works only because Titans is TRAINED FROM SCRATCH to use its memory-context). MAG (zero-
   init gated residual tap) was chosen because it starts as a no-op and learns in without destabilizing
   the base. ⇒ "most literal Titans MAC = latent prefix" is a RE-TREAD of an already-FALSIFIED path;
   do NOT pursue prefix injection. The frozen-base-compatible version of "memory available throughout
   depth" is multi-layer residual (1a) or multi-layer KV (1c).
2. **Multi-hop / ripple TRAINING OBJECTIVE.** The tap was trained for next-token DELIVERY, never for
   reasoning-consistency. Add a ripple objective (train the tap so downstream multi-hop answers match
   the edited chain), still training only the bolt-on.
3. **Addressing / routing.** The bank is set globally, keyed by subject, not routed to where the
   subject appears in the query. Position-/content-addressed injection so the edit fires at the right
   reasoning step.
4. **Toward real test-time memory (the actual Titans core).** The tap is trained-then-frozen with an
   explicit associative store written at test time. Titans' core is a memory whose weights UPDATE at
   test time via surprise-gradient. Moving the bolt-on toward online surprise-update is the deepest
   (and riskiest) lever — the thing that would make CAM genuinely Titans, not "RAG + a frozen tap."

Priority if multi-hop is poor: **1 → 2 → 3 → 4** (cheapest structural fix first).

---

# PLAN B — a broad, reasoning-integrated tap (the real investment)

Gated by experiment A (`--indist-ripple`): only pursue B if the tap ripples IN-DISTRIBUTION. If it
doesn't ripple even where it delivers, the residual-tap mechanism is the wrong vehicle → jump to
KV-injection (1c) / test-time learning (4) instead of scaling this tap.

B is grounded in the FOUR failure modes we MEASURED (each component fixes a specific one), and it
stays entirely within "train the bolt-on" — the base is never touched, so the bet holds.

## The four measured failures → the four B components
| # | Measured failure (this session) | B fix |
|---|---|---|
| F1 | **Distribution-coupling**: on OOD edits the tap emits its TRAINING object-types ("Finnish"/"English" languages) — delivery 0.02 vs 0.64 in-dist. NO_CONFGATE didn't help. | **B1: broaden training** — all CounterFact relations (+ zsRE), BALANCED across object-types, so the tap delivers what the BANK encodes, not a memorized object class. |
| F2 | **Single-token only**: real/MQuAKE objects are multi-token; the tap can't emit them (value-capacity wall). | **B2: pointer/tap split** — the POINTER (#100, lossless) EMITS the exact object tokens; the TAP only installs the BELIEF that ripples. Decouples delivery from integration — the tap's job gets *easier* (a belief signal, not a token emitter). |
| F3 | **Degenerates under sustained injection**: single-layer always-on tap → "Finnish Finnish Finnish…" in free-gen; delivers-once (seed-once) but can't SUSTAIN influence over a chain. | **B3: multi-layer, lower per-layer strength** (lever 1a) — distribute the injection early+mid+late so no single layer over-steers; sustains without degenerating AND enters before the reasoning (Titans placement). |
| F4 | **Delivers, doesn't reason**: tap trained on next-token DELIVERY only, never on reasoning-consistency. | **B4: ripple training objective** — train the multi-layer tap so post-edit MULTI-HOP answers match the edited chain, not just the single-hop token. This is the reasoning-integration that was never trained. |

## Staging (cheap → expensive, each with a gate)
- **B0 (gate): PASSED (2026-07-09).** Experiment A (in-dist language→country ripple, gate 0.782 valid):
  TAP ripple 0.352 vs RAG 0.426 (Δ−0.074), shortcut 0.037, **other 0.611 (largely degeneration)**.
  Ripple EXISTS in-distribution and nearly matches RAG with zero context tokens → proceed to B. The
  loss is degeneration ("other"), NOT failure to reason → **B3 (multi-layer, non-degenerating) is the
  highest-leverage next step** (attacks the 0.611 directly + tests the Titans placement principle).
- **B1 — broaden delivery.** Retrain the tap on all relations, balanced object-types. GATE: does it
  deliver HELD-OUT relations (OOD-within-CounterFact) in free-gen ≥ ~0.5? (kills F1). ~1 training run.
- **B2 — pointer/tap split.** Wire the pointer to emit the object; the tap injects the belief bank
  alongside. GATE: multi-token edits deliver (pointer) + single-hop belief installed (tap). Reuses
  #100 pointer + the tap.
- **B3 — multi-layer injection. FALSIFIED (2026-07-09).** Retrained taps at 8,16,24 together
  (`--multi`), re-ran experiment A. Result: **WORSE** — TAP ripple 0.352→**0.220**, degeneration
  ("other") 0.611→**0.780** (gate still 0.772, so not a delivery failure). More residual injection
  points = more perturbation = MORE degeneration. Titans' "memory throughout depth" does NOT transfer
  to "more residual taps on a FROZEN base" (Titans co-trains the base to absorb it). ⇒ residual-stream
  injection has a delivery-vs-degeneration tradeoff and is CAPPED below RAG; single-layer L24 (0.352)
  is the sweet spot. Pivot OFF residual-multilayer.
  **DEPTH SWEEP (2026-07-09, single-layer L6/L12/L18/L24, no `--multi`, 60-case subset, n≈34/layer):**
  ripple is NON-MONOTONIC in depth, PEAKING mid: L6 0.235 / **L12 0.412** / L18 0.286 / L24 0.314. ⇒
  "residual capped ~0.35" was PARTLY a WRONG-DEPTH artifact (all prior work sat at L24=75%). Depth
  MATTERS (early L6 worst); sweet spot ~L12 (37%). CAVEAT: n≈34 → L12-vs-L24 within noise; L12-vs-L6
  solid. NEXT: confirm L12 vs L24 on the FULL 101-set (tight CIs) BEFORE 1c — a well-placed single tap
  may be a cheaper win than KV. If L12 holds near RAG → B4 ripple-objective AT L12. 1c KV now has a
  principled target depth (the L12 band).
- **B4 — ripple objective.** Add the multi-hop consistency loss (MQuAKE-style edit→multihop→rippled
  answer, held-out). GATE: MQuAKE OOD ripple: TAP vs generation-time RAG (the make-or-break). (kills F4)

## Open questions / risks (resolve as we stage)
- Does broadening PRESERVE per-relation delivery or does capacity force a floor (the value-capacity
  wall)? Measure delivery-vs-#relations.
- Can multi-layer injection sustain WITHOUT degenerating? Untested — B3's gate.
- Does the ripple objective teach integration or just MEMORIZE 2-hop answers? Held-out multi-hop only.
- Is "tap installs a belief that ripples WITHOUT emitting the token" coherent? B2's premise — if the
  belief-only tap can't ripple, the token-emission and the belief may be inseparable (→ 1c KV).
- Cost: multi-layer + ripple objective + broad data = a bigger training run per stage. Budget the
  lease time; each stage is one bind+train+eval (~10–20 min) but there are 4.
- DEEPEST frontier (not in B, flagged): lever 4 (test-time surprise update on the tap's memory) — the
  actual Titans core. Only if B3/B4 plateau.
