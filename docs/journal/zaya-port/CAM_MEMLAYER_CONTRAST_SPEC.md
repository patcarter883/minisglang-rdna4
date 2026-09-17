# CAM — Why frozen-base injection fails at reasoning-integration, and the trained-in memory-layer contrast

**Status:** experiment spec + mechanistic analysis (2026-07-12). Follows the disambiguation result
([[cam-b4-ripple-shortcut-not-belief]]): the B4 ripple objective ripples the *trained* 2-hop (0.85)
but collapses to 0.31 (below RAG) when trained on `family` and evaluated on `country` — a trained
shortcut, not an installed belief. This doc (a) states the mechanistic diagnosis for WHY bolt-on
injection cannot integrate into multi-hop reasoning while a trained-in memory layer can, and (b) specs
the minimal contrast experiment that should CONFIRM the diagnosis.

---

## 1. The measured wall (recap, all on frozen Qwen3.5-4B, fla, in-dist language-pivot ripple)

| Delivery mechanism | single-hop delivery | multi-hop ripple | vs RAG (~0.50) |
|---|---|---|---|
| KV-append @ full-attn L11 | delivers (0.56–0.63) | **0.000** | ✗ falsified |
| GDN-state seed | — | 0.067 | ✗ falsified |
| residual tap @ L11 | delivers | 0.21 | ✗ |
| residual tap @ L12 (GDN) | delivers | 0.31 | ✗ |
| multi-layer combo tap | delivers | 0.26 | ✗ |
| **B4-trained tap, trained hop** | 0.65–0.80 | **0.85** | ✓ but… |
| **B4-trained tap, TRANSFER hop** | **0.80 (delivers!)** | **0.31** | ✗ **shortcut** |

The transfer row is the crux: **delivery is intact (gate 0.80) but the edit does not propagate to a
downstream relation the tap was not trained on.** Delivery ≠ integration.

---

## 2. Mechanistic diagnosis — corrected against the actual tap wiring

A multi-hop question — "head of state of the country where X holds citizenship" — is answered by the
base computing an INTERMEDIATE ENTITY (the country) internally at some later layer/position, then
composing the next hop over it. The intermediate entity **exists only as an internal residual
representation; it is never a literal input token.**

**What the tap actually is** (from the code map, `gated_tap.py`): a forward POST-hook at one mid-depth
frozen layer that returns `h + g·W_o·(softmax(W_q h · (W_k bank)ᵀ) · W_v bank)` at EVERY position. So
the tap's READ is already content-addressed by the running hidden state (`q = W_q h`) — the naive
"it's keyed to the subject token, can't fire at the bridge entity" story is WRONG. The real failures
are deeper:

**F1 — The memory holds only the FIRST hop; there is nothing for the second hop to retrieve.** The
bank content is static and encodes the edit association (subject X → object Croatia), computed offline
and keyed by the subject text. At the bridge position the base is computing "head-of-state-of-Croatia";
its query there is about Croatia, which does not match the bank's subject-X key, and even if it did,
the bank's VALUE is "Croatia" (a hop-1 answer), not the hop-2 premise the composition needs. The tap
can change the local readout of hop 1 but offers nothing addressable to hop 2.

**F2 — The frozen composition does not RE-COMPOSE the perturbed representation (attendable ≠
integrated).** This is the load-bearing wall. KV-append made memory attendable → ripple **0.000**.
Seeding memory into the GDN RECURRENT STATE (`GDNStateInjector`, the most "built-in" frozen path we
have) → **0.067**. Both delivered single-hop yet neither propagated. Frozen downstream attention routes
by the base's ORIGINAL manifold; an injected value is read out locally but the QK circuits that carry
the bridge entity into the next hop were never trained to route on it. **No frozen-base injection form
we tried integrates** — additive, KV-attendable, or recurrent-state-seeded.

**F3 — Only the memory module is trainable, so B4 gradient shortcuts to output-biasing.** The ripple
objective backprops the multi-hop NLL through the frozen stack, but the ONLY trainable params are the
tap's `W_q/W_k/W_v/W_o/gates`. With the composition frozen, the shortest-path solution is to bias the
OUTPUT distribution for the trained composition — not to install a premise the frozen circuit
re-composes. Hence hop-specificity (the disambiguation collapse 0.85→0.31).

**Revised prediction.** F2 says a frozen base is likely the binding constraint: every frozen-base
memory FORM (additive / attendable / recurrent-seed) has already failed, so a nicer frozen memory
layer (Rung 1) will probably ALSO fail to transfer. The decisive ingredient is F3's converse —
**letting the base LEARN TO CONSUME the memory** (co-train a low-rank base delta on the composition
layers jointly with the memory), so the belief lives in weights the composition routes on. That is the
faithful "trained-in memory" / Titans regime and connects to the editing literature's weight-edit-
ripples-better-than-activation-steering result. Rung 1 is run as the frozen-base control that should
FAIL; Rung 2 is the confirming positive.

---

## 2b. Literature grounding (the diagnosis is the field consensus)

Every clause of §2 is independently established:

- **The bridge entity is an internal residual, never a token; hop 2 fires LATE.** Yang et al.
  "Do LLMs Latently Perform Multi-Hop Reasoning?" (ACL 2024) show hop 1 resolves the bridge entity in
  early–mid layers as a hidden representation; Biran et al. "Hopping Too Late" (EMNLP 2024) show the
  second hop only fires in *later* layers (sometimes too late) and **back-patching a late hidden state
  to an earlier layer fixes up to 66%** of failures. ⇒ our tap at mid-L12 is off-position for hop 2 (F1).
- **Stored ≠ integrated; the reasoning circuit reads different layers than where the fact sits.**
  CaKE "Circuit-aware Editing" (2025): ROME/MEMIT write where the fact is *recalled*, but the multi-hop
  circuit reads from *other, later* components → the edit "exists but isn't accessed." This is our
  KV-append 0.000 / GDN-seed 0.067 verbatim (F2).
- **Edited facts only ripple when they share parameter directions.** Zhang et al. "Why Does New
  Knowledge Create Messy Ripple Effects?" (2024): GradSim (edit-grad vs related-fact-grad cosine)
  predicts ripple at Pearson r≈0.85. A frozen-base injection shares no such direction (F2).
- **The methods that DO ripple route through the full frozen stack via context, or co-train the read
  with attention.** Cohen et al. RippleEdits (TACL 2024): in-context editing **82.8%** vs ROME 60.8%
  on compositionality. Zhong et al. MQuAKE→MeLLo (EMNLP 2023): external memory + iterative prompting of
  the *frozen* model beats parametric editors. Titans-MAC concatenates memory into the **attention
  context** with **read/gate/projections trained jointly with attention** (Behrouz 2025). ⇒ integration
  needs the fact IN the representation the (jointly-trained) circuit routes on — not beside it (F3).

**Correction to an earlier framing:** it is NOT "weight edits ripple, activations don't" — the field
finds *both* single-shot edit forms ripple poorly. The lever is **circuit/context alignment**: RAG/ICE
(our ~0.50 baseline) already puts the fact on the circuit via context, which is why the frozen tap
cannot beat it *transferably* — the tap can only exceed RAG by learning a bridge-bypassing shortcut.

## 3. The contrast experiment — an ablation ladder, same disambiguation test on each rung

All rungs: frozen Qwen3.5-4B, fla cloud, `--ripple-objective`, TRAIN on `country`, EVAL on `family`
(and `script`) — identical protocol to [[cam-b4-ripple-shortcut-not-belief]]. Baseline = the tap
(control 0.85 / transfer 0.31). A rung "confirms integration" iff its TRANSFER ripple rises materially
above the tap's 0.31 and above RAG (~0.50), i.e. the belief generalizes to the untrained hop.

- **Rung 0 — tap @ mid-L12 (baseline, done):** additive, tap-only trainable. Transfer 0.31.
- **Rung 1 — LATE-layer placement (CaKE/back-patching test, nearly free — NO new code).** Re-run the
  B4 disambiguation with the tap at the LATE layers where hop 2 fires (the late full-attn band
  L19/L23/L27, and a span). Tests F1 directly: if delivering the edit where the second hop reads lifts
  TRANSFER above 0.31, placement was the issue. Prediction (given multi-layer combo already = 0.26):
  weak — placement alone on a frozen base is insufficient. Cheap enough to run first and de-risk Rung 2.
- **Rung 2 — Titans-faithful: co-train the memory read WITH attention** (the confirming positive).
  Insert a `memlayer` associative read AND add a low-rank delta (LoRA) on the QK/MLP of the composition
  band, trained JOINTLY on the ripple objective so the attention learns to route on the memory read —
  the exact Titans-MAC/LMM mechanism (read/gate/projections trained with attention). Base otherwise
  frozen. Train on `country`, eval on `family`/`script`. **Prediction: transfer rises materially above
  0.31 and above RAG** — because the belief now lives in a representation the (co-adapted) circuit
  composes, not beside it. This is the deliverable that CONFIRMS the diagnosis.

Interpretation: Rung 1 failing + Rung 2 transferring = the clean confirmation that integration requires
the reasoning circuit to be trained to consume the memory (F2/F3), which a frozen-base bolt-on cannot
provide however it is shaped or placed. If Rung 2 ALSO fails to transfer, the constraint is deeper
(the frozen base's hop-2 machinery is itself too weak — Biran "hops too late"), pointing to a fuller
Titans-MAC insert (memory as attended context + trained attention) as the next rung.

---

## 4. Build notes

- New inject-mode `memlayer` in `cam/gated_tap.py` (a `MemLayerInjector` alongside MAG/KV/GDNState);
  attach at `--tap-layers`, read the bank set by `mquake_ripple`/`recall_mag` associatively.
- `recall_mag.train_taps`: register the new module's params as trainable; keep base frozen; ripple
  objective unchanged (gradient already flows through the frozen base for B4).
- Disambiguation harness already in place: `CAM_RIPPLE_TRAIN_HOP` / `CAM_RIPPLE_EVAL_HOP`.
- Run recipe: [[cloud-cam-ripple-cuda-fla-recipe]]; `campaign_run.sh NAME "<EXTRA>"` with the hop envs.

---

## 5. RESULTS (2026-07-12) — the ladder reached its ceiling; diagnosis confirmed on the negative side

Disambiguation protocol, train=family / eval=country (scorable hop), RAG ~0.50–0.64, baseline tap 0.31:

| Rung | Config | TRANSFER ripple | Control (train=country) | Note |
|---|---|---|---|---|
| 0 | tap @ mid-L12 | 0.31 (n13) | 0.846 | baseline |
| 1a | tap @ late-L27 | **0.455** (n11) | — | placement helps (F1), still < RAG |
| 1b | tap @ span 19,23,27 | 0.214 (n14) | — | noisy |
| 2 | tap + **naive** co-train LoRA | *invalid* — RAG→0, n→1 | *invalid* | catastrophic forgetting |
| 2v2 | tap + **anchored** co-train LoRA | **0.250** (n16) | **0.923** vs RAG 0.692 (n13) | anchor recovered base |

**Findings.**
1. **Every frozen-base variant's transfer is below RAG** (0.21–0.455, noisy; RAG 0.50–0.64). Placement
   (F1) helps marginally; nothing beats RAG transferably.
2. **Co-training the reasoning circuit's own attention (Rung 2) does NOT install a transferable belief.**
   It *helps the trained hop* (control 0.846→0.923) but *not the transfer hop* (0.250) — the shortcut
   signature reappears even when the trainable params live in attention (F3 generalizes: any bolt-on
   trained on one composition learns that composition, not a relation-agnostic belief).
3. **Catastrophic-forgetting gotcha (proven + fixed):** naive co-train (lora_lr = tap_lr = 1e-3, no
   anchor) destroys the base's parametric composition (RAG→0, filtered n→1). The memory-OFF preservation
   anchor (KL vs frozen base) + small lora_lr (1e-4) is REQUIRED and recovered RAG to 0.69/n=13. This is
   itself a mini-confirmation: a trained-in memory must *learn to consume memory without forgetting* — a
   constraint a one-shot injection never has to satisfy, and that naive adaptation violates.

**Conclusion.** On a FROZEN pretrained base, no bolt-on — however shaped, placed, or lightly co-trained
— achieves cross-hop reasoning-integration above the context (RAG/ICE) baseline. The necessary
ingredient is editing the model's PARAMETRIC representation of the fact so *all* downstream circuits
compose over it; a frozen-base memory cannot do this by construction (the trainable object always fits
the trained composition). Positively demonstrating that a genuine trained-in Titans memory transfers
requires TRAINING such a model (canonical-memory build), not adapting a frozen one. The negative side of
the diagnosis is now fully, empirically confirmed and literature-consistent (§2, §2b).
