# Path R: move the composition into a TRAINABLE agent — the frozen base reads, it does not believe

**Status:** PROPOSAL, unbuilt. No env gate yet. Sits between Path D
(`CAM_CARTRIDGE_EDITRIPPLE_SPEC.md`, completed, matched RAG) and Path A
(`CAM_PATH_A_ORACLE_SCOPING.md`, sanctioned, expensive). Entry criterion is a **disambiguation
gate that must be run FIRST** (§5, Gate 0) — this spec is NO-GO until it passes.

> **Why this doc exists.** Re-reading RecursiveMAS's released trainer (upstream `38f7da4`,
> 2026-06-28 — see [[recursivemas-published-method]]) falsified a hypothesis that CAM had skipped
> their end-to-end objective ([[cam-link-may-be-warmstart-only]]). It had not: CAM Stage 2 already
> backprops answer-token CE through the frozen base (`memory-organ-p/cam/recall_mag.py:401→406`).
> The useful finding was different and is the premise of this doc: **their objective is appropriate
> for their topology and insufficient for ours, and the fix is topological.**

---

## 1. What this is

Today CAM is `memory → tap → FROZEN BASE`, and the base is asked to *believe* a fact well enough
that its own downstream circuits compose over it. That is the measured wall.

Path R is:

```
memory  ↔  small TRAINABLE composer agent  →  RecursiveLink  →  frozen base (solver)
```

The memory and the composer are co-trained. The composer does retrieval and multi-hop composition.
The link carries a latent into the frozen base as **context-equivalent input embeddings**. The frozen
base does fluency and world knowledge and is never asked to integrate anything.

**The claim is not "belief installation now works."** It does not, and nothing in this doc changes
[[cam-why-injection-fails-diagnosis]]. The claim is that the *composition step* is relocated out of
the frozen circuit into one we are allowed to train, which raises the ceiling from the activation-
write band to the context band.

## 2. Why the wall does not bind here

The three failure modes from `CAM_MEMLAYER_CONTRAST_SPEC.md`, taken one at a time:

| | Failure mode | Why Path R is not subject to it |
|---|---|---|
| **F1** | bridge entity is an internal residual, hop 2 fires in late layers; our tap sits at mid-L12 | The bridge lives inside the **composer**, which is trainable. Read position is a learned choice, not a frozen constraint. |
| **F2** | frozen downstream attention routes on the base's original manifold; injected value is read but not re-composed (**load-bearing**) | Composition finishes **before** the frozen base sees anything. The base receives an already-composed latent in context. It re-composes nothing because it is asked to re-compose nothing. |
| **F3** | with only the tap trainable, gradient takes the shortest path and biases the output, bypassing the bridge | Still live. See §6 — this is the kill risk, and Gate 0 exists for it. |

And the weak-gradient result that forced CAM's local Stage-1 objective
(`memory-organ-p/cam/recall_boltA.py:4-14`: through-base gradient "too weak/indirect to learn the
binding from scratch" across 32 frozen layers) also stops binding: the store is now **one short hop**
from its loss, inside the composer, not 32 frozen layers away.

**The ceiling this targets.** Every activation-write variant we ever ran clusters **0.27–0.46**
(every write algebra × every layer; `CAM_MEMLAYER_CONTRAST_SPEC.md`). RAG/ICE sits at **~0.50–0.62**,
and in-context editing in the literature reaches **82.8%** vs ROME's 60.8%. RecursiveMAS's only
channel is input embeddings — the maximally in-context position. So the honest target band is
**RAG-or-better**, not Titans. Say so in every write-up.

## 3. Architecture

Names reused verbatim where they already exist.

### 3.1 Memory — reuse, do not rebuild

`_ProductKeyStore` / `_PKAdapter` from `python/minisgl/cam/memory.py`, or the training-side store in
`memory-organ-p/cam/pk_store_adapter.py`. The Titans deep-memory cell already exists, graph-free and
bit-parity-validated, at `memory-organ-tapmetric/cam/deep_memory.py` (+ `deep_mem_analytic.py`) —
`CAM_PATH_A_ORACLE_SCOPING.md` identified it as the hard part that is already done. Path R inherits
that for free.

### 3.2 Composer — small, trainable, reads the memory natively

A 1.7B–4B model with the memory read **co-trained with its attention**, not bolted on. This is the
one ingredient the frozen-base track could never have. Candidate: Qwen3-1.7B (matches RecursiveMAS's
`sequential_light` planner, so their released inner adapter is a sanity reference).

The composer's job is the multi-hop chain. It emits `S` latent positions. Nothing decodes to text.

### 3.3 Link — RecursiveMAS `CrossModelAdapter`, unchanged

```
LN(in) → Linear(in → 2·out) → GELU → Linear(2·out → out) + residual_proj(in → out) → LN(out)
```

Position-wise, no cross-position mixing. `3·in·out + 2·out² + 2·in + 6·out` params — ~14–21M for the
dims in play. This is the module `ACKNOWLEDGMENTS.md` already credits as CAM's originating spark;
Path R uses it for its **original** purpose (carrying one model's hidden states into another's space)
rather than for memory delivery.

### 3.4 Injection into the frozen base — `inputs_embeds`, slot placed LATE

The `S` latents occupy ordinary positions with `attention_mask=1`, spliced at a placeholder in the
solver's templated prompt. **Place the slot as late in the prompt as the template allows.** The
reference puts it mid-prompt, which strands the entire instruction tail; for a served path the prefix
before the slot is radix-cacheable and everything after it is not. This is a template choice with no
effect on the method and a large effect on serve cost.

Note this is a *different channel* from the current serve tap (`stage_cam` at
`python/minisgl/models/qwen3_5.py:455`). Path R does not use the tap seam. It needs the mixed
token-id / embedding prefill path, which minisgl does not have today — see §7.

## 4. Training recipe

Two stages, borrowing the *shape* both projects independently converged on (local objective for the
deep component, through-frozen task CE for the small delivery module), with one deliberate deviation.

**Stage R1 — memory + composer, joint, local.**
Task CE inside the composer only. The frozen base is **not** in this graph. Rationale is measured, not
aesthetic: `recall_boltA.py:4-14` (gradient too weak through the frozen stack) and
`titans/warmstart/train_mem_canonical.py:175-181` (where it *is* strong enough, LM loss **bypasses the
store** — which is why the InfoNCE addressing term exists). Keep the addressing supervision.

**Stage R2 — link only, CE through the frozen base.**
Freeze memory + composer. Train only the `CrossModelAdapter` by next-token CE on the final answer,
backpropagated through the frozen solver. This is exactly RecursiveMAS `train/outer/common.py:614` →
`:935`, and exactly the shape of CAM Stage 2. `gradient_checkpointing` on.

**Deviation — train the rollout, do not teacher-force it.**
RecursiveMAS trains teacher-forced and rolls out autoregressively at inference; their adapter is never
trained on its own outputs (there is no `latent_steps` in their trainer at all — the training latent is
a trim of real assistant hidden states to `--max_latent_tokens 80`). That is textbook exposure bias.
They cannot afford to fix it because their rollout is `use_cache=False` and quadratic. **Ours will be
KV-cached, so roll out in the loop.** This is the one place Path R should knowingly diverge from the
reference, and it is worth a controlled A/B (§8) because it may also speak to
[[cam-tap-needs-seed-once]] — "always-on degenerates" has the same shape as an adapter never trained
under its own multi-step influence.

## 5. Staged plan with hard gates

### Gate 0 — DISAMBIGUATION FIRST. Nothing else starts until this passes.

This is the entry criterion, not a later validation. B4 looked like the only lever that beat RAG
(0.759 vs ~0.50) and was a trained shortcut; the disambiguation test is what caught it, and it must
catch it here *before* we spend on a composer.

Harness: `memory-organ-tapmetric/cam/mquake_ripple.py`, `_select_twohop` at lines 448 / 533, driven
via `CAM_RIPPLE_TRAIN_HOP` / `CAM_RIPPLE_EVAL_HOP`.

Run a minimal composer (no link, no frozen base — score the composer's own output) on:
- **Control:** train `country`, eval `country`
- **Transfer:** train `family`, eval `country`

**GO iff** transfer ≥ RAG on the same filtered set, with control ≥ transfer by no more than the
control/transfer collapse B4 showed (0.846 → 0.308). A composer that only ripples its trained hop is
B4 at a larger scale and buys nothing.

**NO-GO iff** transfer < RAG. Stop. Report it. Path R is then falsified for the same reason the
frozen-base ladder was, and Path A remains the only route.

**Respect the reporting contract.** `mquake_ripple.py` withholds the verdict entirely when single-hop
delivery `ctrl_gen < CAM_MQUAKE_CTRL_MIN` (default 0.40) — "the tap is NOT firing … fix delivery
before reading a ripple verdict." Do not read a ripple number through a broken delivery gate. Report
UNFILTERED, then FILTERED, then the like-for-like HEADLINE, as the harness already does.

### Gate 1 — link carries the composition

Add the link + frozen solver. Compare end-to-end against the composer's own score from Gate 0.
**GO iff** the linked system retains ≥90% of the composer's transfer score. A large drop means the
link is the bottleneck and the latent is lossy; that is a link-capacity question, not a topology
failure — widen and retry once.

### Gate 2 — beat RAG end-to-end

**GO iff** end-to-end transfer > RAG on the filtered set, at equal or lower token cost. This is the
whole claim. RecursiveMAS Proposition 1 gives the cost side: deleting the `m·|V|·d_h` vocab
projection, which at our flagship's |V|=248,320 / d_h=2048 is a 121× ratio on that term.

### Gate 3 — serve it

Only after Gates 0–2. Requires §7's engine work. Reuse `CAMMemory` / `BackendCAMRuntime` and the
`/cam/*` control plane; the delivery path is new.

## 6. Risks & kill-criteria (honest)

1. **The shortcut reappears in a new costume (LIKELIEST).** The composer learns to answer directly
   and use the frozen base as a fluency wrapper. Looks like success in-distribution, collapses off it.
   *Kill:* Gate 0. This is exactly why it is Gate 0.
2. **The link becomes a lossy text codec.** If the latent only ever encodes what a text handoff would,
   we spent a GPU to avoid a tokenizer. *Detect:* logit-lens the latent through the solver's
   unembedding and compare against the text the composer would have emitted; if nearest-token decode
   round-trips cleanly, there is no latent advantage. *Not fatal* — the efficiency claim survives —
   but it downgrades the result to "compressed RAG".
3. **Exposure bias, inherited.** Mitigated by rolling out in the loop (§4), but that is unproven and
   is itself an A/B.
4. **We are optimising a system whose headline we did not verify.** RecursiveMAS's 86.7% AIME25 from
   ~11B total params is a single run and unreplicated (all four `*-Outerlinks` HF repos show 0
   downloads). We are borrowing the *topology*, not the numbers, and nothing here depends on their
   accuracy — but do not cite their results as support.
5. **Cost is off-box.** No sharding path exists in RecursiveMAS's outer loop (no accelerate, no DDP,
   no FSDP); all agents resident on one device. Their light config needs ~24 GB. This is a cloud job
   — the MI300X path in [[cloud-lease-rocm-torch-env]] — not a 2×16 GB job. Their CE is also taken
   over the full sequence at full vocab with a `.contiguous()` copy and no `logits_to_keep`
   (`outer/common.py:615`); if we reimplement, use chunked/fused linear-CE or that term OOMs first.

## 7. Engine work this needs (not covered by existing CAM serving)

- **Mixed token-id / embedding prefill.** minisgl's prefill takes ids and does an embedding lookup.
  Path R needs "here are ids, and at positions [a,b) use these vectors." This is the main integration
  cost and it does **not** reuse the `stage_cam` seam.
- **Radix interaction.** The prefix before the slot is cacheable and shareable; the latent span is
  not; the suffix is position-shifted per request. Radix reuse terminates at the slot — hence §3.4's
  late-slot rule.
- **A captured, KV-cached latent rollout.** Structurally this is our spec-decode **propose** loop with
  the sampler replaced by the link and the LM head skipped, so it should be a port of
  `docs/SPEC_PROPOSE_GRAPH_CAPTURE.md`, not a new subsystem. Do not replicate the reference's
  `use_cache=False` rollout — it survived their June update
  (`inference/inference_utils/inference_mas.py:968,978`) and is O(S × prompt_len).
- **Two resident models with independent KV pools.** Engine memory accounting assumes one model plus
  an optional draft; see [[serve-vram-accounting-35b-16gb]] for why the pool is the small remainder
  after weights.

## 8. How to A/B

Gate 0, no engine work required, runs on the existing harness:

```bash
# control
CAM_RIPPLE_TRAIN_HOP=country CAM_RIPPLE_EVAL_HOP=country \
  python -m cam.recall_mag --mquake-eval <path> --indist-ripple ...
# transfer  (the one that decides)
CAM_RIPPLE_TRAIN_HOP=family  CAM_RIPPLE_EVAL_HOP=country \
  python -m cam.recall_mag --mquake-eval <path> --indist-ripple ...
```

Confirm the arm actually ran: the harness prints `ctrl_gen` and withholds the verdict below
`CAM_MQUAKE_CTRL_MIN`. Compare `tap_bc` / `rag_bc` (base-compose ∩ eligible), never the unfiltered row.

Rollout-vs-teacher-forcing A/B (Stage R2, after Gate 1): same link, same data, one arm teacher-forced
on real hidden states, one arm rolled out in the loop. Score both on Gate 2's metric **and** on
sustained generation, since [[cam-tap-needs-seed-once]] is the symptom we suspect it explains.

## 9. What is NOT claimed

- Not belief installation. `cam-why-injection-fails-diagnosis` stands: no frozen-base variant achieves
  cross-hop transfer above RAG, and Path R does not make the frozen base believe anything.
- Not Titans. Trained-in memory that composes remains Path A's question
  (`CAM_PATH_A_ORACLE_SCOPING.md`), and no paper in the Titans/Miras/NL/LongMem line tests multi-hop
  edit-ripple at all — that gap is unchanged and is still our open contribution.
- Not a contradiction of Path D. Cartridges matched RAG with a *static* trained KV; Path R's bet is
  that a *trainable composer* clears it. If Gate 2 fails, D and R tie, and the honest read is that the
  context band is simply where frozen-base memory tops out.
- Consistent with `memory-organ-p/ROADMAP.md`, which already states that reasoning-integration is "a
  measured wall … explicitly not the target." Path R does not reopen that target; it relocates the
  composition so the wall is not in the path.

---

Related: [[recursivemas-published-method]] · [[cam-link-may-be-warmstart-only]] ·
[[cam-why-injection-fails-diagnosis]] · [[cam-b4-ripple-shortcut-not-belief]] ·
[[cam-1c-kv-injection-falsified]] · [[titans-miras-nested-longmem-research]] ·
[[cam-tap-needs-seed-once]] · [[cloud-lease-rocm-torch-env]]
Docs: `CAM_PATH_A_ORACLE_SCOPING.md` · `CAM_CARTRIDGE_EDITRIPPLE_SPEC.md` ·
`CAM_MEMLAYER_CONTRAST_SPEC.md` · `CAM_TITANS_MIRAS_RESEARCH_BRIEF.md`
