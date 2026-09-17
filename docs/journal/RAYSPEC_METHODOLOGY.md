# RaySpec — a speculative-decode drafter whose native inference op is a hardware ray-cast

Status: **SHELVED (2026-07-19) — measured-negative on both axes; kept as a design record.**
Stage-0a (Torch, no RT) ran on the real DFlash corpus + a 23-agent adversarial council. Result:
(1) RT axis red — the hard tube-hit readout the RT silicon accelerates fails and *worsens* with
R (hard1 4.2→1.4%); root cause = ρ double-duty as CE-temperature and hit-radius; even the
council's decoupled-ρ/τ fix has an honest ceiling ~15–35% < EAGLE. (2) GEMV axis dead — at
matched V×3R budget a plain low-rank factored head BEATS the ray/metric geometry everywhere
(64.7 vs 62.4 top1 @ rank96), so the ray adds nothing a factored head doesn't. The headline
"93% of ceiling" was an artifact of an underfit denominator (true oracle ≈100%; ray ≈64% abs).
See `memory/rt-hardware-vq-assignment-launch-bound.md` for the full arc. The mechanism/plan below
is preserved as-is for the record; it is NOT a live workstream.

---

Author context: this follows two measured negatives that *shape* the design rather than
contradict it — see `memory/rt-hardware-vq-assignment-launch-bound.md`:
1. RT gives nothing for FP4 requant, small-codebook VQ assignment (launch-bound GEMM,
   ~19µs flat), or the existing GEMV/suffix-match drafters.
2. An **off-the-shelf** dense-retrieval drafter (semantic embedding NN → copy nearest
   continuation) *loses* to the free `propose_ngram` everywhere (code 1.09 vs 1.86,
   prose 1.01 vs 1.29 accept-len). Mechanism: "nearest context" ≠ "same next token."

RaySpec's premise is the inverse of #2: **do not borrow a semantic geometry — learn a
geometry, end-to-end, so that a hardware ray-cast *is* next-token prediction.** This is
the "diffusion move": not "run our drafter on the RT unit," but "train a drafter whose
inference primitive the RT unit implements natively," the way diffusion-LMs trained a
model whose native op is denoising (rather than porting AR decoding onto raster hardware).

---

## 0. What the RT core actually is (the primitive we build on)

RDNA4 (gfx1201) ray accelerator, one per CU (~64 on RX 9070 XT), invoked via
`image_bvh8_intersect_ray`:

- Input: a **ray** (origin `O`, direction `D`, `tmin`/`tmax`) + a BVH node pointer.
- Descends a static BVH doing **8 box tests per traversal step**, returns hit primitives
  **sorted by `t`** (distance along the ray). With an any-hit shader you collect *all*
  primitives the ray pierces, front-to-back.
- Runs **concurrently** with VALU/WMMA. During bs=1 decode we are dispatch/overhead-bound
  (`memory/serving-overhead-bound-not-bandwidth.md`) — the WMMA units and the entire RT
  array sit idle between launches.

So the hardware gives us, for free-of-VALU, a **ranked top-m retrieval over a large static
point set, per ray**. That is precisely a tree-drafter's candidate-generation step. The
two durable value propositions, ranked:

1. **Idle-silicon offload.** Candidate generation moves off VALU/WMMA onto the RT array
   and overlaps the target's compute → in the overhead-bound regime the draft approaches
   *free*, which collapses the speculative break-even accept-length toward ~1
   (`memory/spec-decode-breakeven-gdn-moe.md`: our drafters die because their forward pass
   is expensive; a near-free drafter changes the economics).
2. **Native ranked output → native tree.** Sorted hits map 1:1 onto a DDTree frontier
   (`memory/ddtree-dflash-tidar-plan.md`: best-first heap + ancestor mask, drafter-agnostic).

Constraint the hardware imposes: rays/boxes are **3D**. We embrace it (learn 3D views +
ensemble) rather than fight it (project high-D and lose).

---

## 1. Geometry & the model

### 1.1 Objects (per view `r`, `r = 1..R`)

- **Token codebook** `C_r ∈ ℝ^{V×3}`: a learned 3D position for every vocab token. Static
  ⇒ one BVH per view, each token an axis-aligned box of half-width `ρ` (the "tube" radius).
  Built once at load. `V ≈ 150k` — big enough that BVH traversal (~`log8 V ≈ 6` node
  tests/ray) beats brute force, the regime the 256-codebook microbench lacked.
- **Ray head** `g_r`: maps a drafter feature vector `f_t ∈ ℝ^d` to a ray:
  `O_r = A_r f_t + a_r ∈ ℝ³`, `D_r = normalize(B_r f_t + b_r) ∈ ℝ³`,
  with `A_r, B_r ∈ ℝ^{3×d}`. Tiny (`6d + 6` params/view).

`R` views total. All parameters: `R·(V·3 + 6d+6)`. For `V=150k, d=2560, R=12`:
≈ `12 · (450k + 15k) ≈ 5.6M` params — a small head, distillable cheaply.

### 1.2 Hit semantics (what the hardware returns)

Ray `(O_r, D_r)` "hits" token box `c` iff the ray pierces the box, i.e. the token is within
`≈ρ` (L∞) of the ray *line*. Hit `t = (c − O_r)·D_r` (along-ray distance). RT returns the
hit set sorted by `t`. Define the per-view **candidate list** for step `t`:
`H_r = [ tokens pierced by ray r, ordered by increasing positive t ]`, truncated to top-`m`.

### 1.3 Aggregation → draft score

A token's draft strength aggregates evidence across views. Baseline rule (tunable):
`score(v) = Σ_r  w(rank of v in H_r)`  with `w` a decreasing rank weight (e.g. `1/(1+rank)`);
tokens hit near-front in many views win. Top of `score` = the draft candidates; feed the
ranked set as a DDTree frontier layer.

### 1.4 Depth (multi-token drafts) — EAGLE-shaped loop

At propose time we lack future hidden states, so depth is autoregressive, exactly as EAGLE:
the drafter maintains its own feature recurrence. **RaySpec = EAGLE's autoregressive draft
loop with the vocab projection replaced by a ray-cast.** The trained trunk `T_θ` (§2, depth a
lever) produces `f_t`; instead of `logits = W_U f_t` (a `d×V` GEMV over the whole unembedding)
we cast rays into the token-BVHs. Draft step:

```
f ← draft_recurrence(h_t, emb(prev_draft_token))     # EAGLE-style, on WMMA
for r in views:  ray_r ← g_r(f)                       # R tiny matmuls, on WMMA
hits ← RT_cast(rays, token_BVHs)                       # R concurrent casts, on RT array
cand ← aggregate(hits)                                 # ranked candidates
push cand into DDTree; pick next_draft_token; repeat K times
```

The op RT *replaces* is the draft head's `d×V` projection at each of the `K` draft steps —
moved off WMMA onto the idle RT array so it overlaps the target's verify pass.

---

## 2. Training — a purpose-built drafter, trained properly (NOT a bolt-on head)

The drafter is a **model we train**, sized and dataed to be competitive with EAGLE, not a
tiny head hung off frozen features. The target stays the **teacher** — in spec decode the
drafter must predict the model we actually serve, so distilling from it is correct practice,
not a shortcut. What is *not* a shortcut we accept: freezing a cached backbone, a 5M-param
map, or an afternoon of compute. Capacity, data scale, and compute are **levers we sweep**.

**Drafter architecture.** A trained trunk `T_θ` (its own transformer, depth/width a lever —
from EAGLE-scale 1–2 layers up to a small standalone LM if the geometry needs it) consuming
the target's hidden state + token embeddings, producing the feature `f_t` that drives the R
ray heads. The trunk is trained end-to-end **with** the ray heads and the token codebooks
`C_r`, so the entire representation is optimized to *live in the R-view 3D ensemble geometry*
— the effective drafting capacity is `R×3` behind a trained encoder, not a lossy shadow of a
fixed high-D feature. Everything below the target is trainable; only the served target is
frozen (see the optional co-shaping track).

**Data & compute.** Large, diverse corpus (code, prose, agentic/tool-call, math, multiling.)
— not the tiny gate corpora. Teacher-forced target hidden states + full next-token
distributions as supervision. This is a **real training run on cloud multi-GPU**
(`memory/cloud-lease-rocm-torch-env.md`, `memory/cloud-cam-ripple-cuda-fla-recipe.md`), sized
like training an EAGLE/MTP drafter (order GPU-days, not hours), because the whole point is to
find out whether a properly-resourced RT-native drafter *beats* EAGLE — a starved one tells
us nothing.

**Losses.**
  - *Ranking/hit loss* per view: differentiable surrogate for "target's top-k tokens are
    front hits." For token `c`, tube-membership `μ_r(c) = σ((ρ² − d⊥(c, ray_r)²)/τ)` with
    `d⊥` the perpendicular line distance; along-ray order via `t`. Push target top-k to high
    `μ` and small positive `t`; push distractors out of the tube; regularize codebook spread
    (anti-collapse).
  - *Aggregate distillation:* KL( target top-k ‖ softmax(aggregate score) ) so the unioned
    ensemble reproduces the teacher's ranked head — the objective that *forces* the geometry
    to be next-token-predictive (the inversion of the failed off-the-shelf-embedding gate).
  - *Draft-recurrence loss:* EAGLE-style feature-regression + CE so multi-step autoregressive
    drafts stay on-distribution.

**Optional co-shaping track (heavier, flagged).** A light LoRA/adapter fine-tune of the
*target* to make its residual stream more linearly-3D-projectable per view. Higher payoff
ceiling for the geometry, but it changes the served weights — a separate, gated commitment,
not on the critical path to the Stage-0 answer.

**Per-target.** Like EAGLE, a RaySpec drafter is target-specific: the pipeline produces one
per served model. That is expected cost, not overhead.

---

## 3. Serving / inference path (where each op runs)

Per decode step at bs=1:

| step | work | unit | notes |
|------|------|------|-------|
| target forward | 1 layer stack | WMMA/HBM | already run; `h_t` is free |
| draft recurrence ×K | trunk `T_θ` (depth a lever) | WMMA | trained drafter, §2 |
| ray heads ×K×R | `3×d` matmuls | WMMA | tiny |
| **candidate gen ×K×R** | **ray casts into static BVHs** | **RT array** | **the offloaded, overlappable op** |
| aggregate + DDTree | rank/heap | VALU | small |
| verify | target on draft tree | WMMA/HBM | existing `accept_greedy_ondevice` |

RT casts overlap the target's verify pass (pipeline one draft step behind — drafters are
staleness-tolerant; reuse the `FutureMap` spec-overlap machinery,
`memory/perf-push-futuremap-overlap.md`). Verification, tree accept, EOS truncation:
unchanged, via `python/minisgl/spec/accept_gpu.py`. RaySpec is a new `Proposer`
(`python/minisgl/spec/proposer.py`) producing a DDTree frontier — the rest of the spec
stack is untouched.

---

## 4. Staged plan with hard gates (cheap-before-expensive, RT last)

**Stage 0 — Feasibility gate (Torch only, NO RT, NO HIPRT).** *This is the go/no-go, and it
is a real training run, not a probe.* Train the **full purpose-built drafter** (§2: trunk
`T_θ` + R ray heads + token codebooks, end-to-end, distilled from the target on a large
corpus, cloud multi-GPU). Do it at a scale where a negative result is *trustworthy* — a
starved head failing proves nothing. Two sub-gates:
- **0a (fast, small target):** dev on `Qwen3-0.6B` to debug the geometry/losses and find the
  `(R, trunk-depth, ρ, m)` that carry signal at all, before spending on the big run.
- **0b (real):** train against the actual serving target (`Qwen3.5-4B` / whatever we serve),
  full corpus and compute.
- Simulate the ray-cast in Torch (brute force: perp-distance of all `V` points to each ray,
  tube-filter, sort by `t`, top-`m`) — bit-behavior identical to what RT will do.
- Measure **per-step top-k hit rate** and **autoregressive accept-length** vs (a) `ngram`
  (code 1.86 / prose 1.29 / agentic 6.36) and — the real bar — (b) a **fully-trained EAGLE/
  MTP drafter of comparable trunk cost** (`memory/spec-decode-mvp.md`). Sweep `R`, trunk
  depth/width, `ρ`, `m`.
- **GO iff:** accept-length is **competitive with or beats EAGLE** (not merely beats ngram)
  at a draft cost the RT-offload makes cheap — i.e. RaySpec's *drafting quality* stands on
  its own and RT then makes it *cheap*. **NO-GO iff** a properly-resourced `R×3` geometry
  can't reach EAGLE-class accept-length — then RT cannot rescue it and we stop, before HIPRT.

**Stage 1 — RT kernel prototype (HIPRT, only if Stage 0 GO).**
- Build static token-BVHs (`V` boxes × `R` views); implement ray-cast candidate-gen as a
  HIPRT kernel returning top-`m` sorted hits; **bit-parity vs the Stage-0 Torch brute force.**
- Microbench: latency of `K×R` concurrent casts vs the WMMA candidate-gen it replaces;
  **profiler-confirm** the casts actually run on the RT array and overlap target compute
  (the "free" claim must be measured, per `memory/serving-overhead-bound-not-bandwidth.md`).

**Stage 2 — Engine integration.**
- `RaySpecProposer` implementing §3; DDTree frontier from aggregated hits; verify via
  existing accept path; end-to-end tok/s vs ngram/EAGLE/MTP at bs=1 and small batch on the
  target's own (free) hidden states.
- **GO iff:** net tok/s win over the best existing drafter on ≥1 real workload.

**Stage 3 — Ensemble & parallel scaling.** Sweep `R` vs RT-array occupancy (bushy DDTree,
the multi-RT parallelism); tune `ρ`, distillation depth, tree width; graph-capture the
propose loop (`memory/graph-capture-required-for-complete.md` — eager-only ≠ done).

---

## 5. Risks & kill-criteria (honest)

- **3D capacity (biggest).** An `R×3` ensemble geometry — even with a trained trunk feeding
  it — may lack the rank to reach EAGLE-class accept-length; a plain small transformer head
  may dominate on accept-per-cost. *Killed at Stage 0 if so — that's the whole point of the
  properly-resourced Stage 0: a starved drafter failing would be an uninformative negative.*
- **Training cost / EV.** Stage 0 is now a real (cloud, GPU-days) training run per target,
  not an afternoon. That is the deliberate cost of a *trustworthy* answer — but it means the
  bet is only worth opening if we're prepared to fund the honest test. Sequenced 0a (small
  target, cheap debug) → 0b (real) so we spend the big budget only after the geometry shows
  signal at all.
- **"Free" depends on real overlap.** If the RT cast can't be scheduled concurrently with
  target compute on this stack, the offload value evaporates. *Profiler-gated at Stage 1.*
- **HIPRT ↔ HIP/Triton interop.** No Triton path to RT; the cast is a HIPRT kernel that must
  interop with our decode kernels and hand results back for aggregation. Real integration
  cost — but only paid after Stage 0/1.
- **Learned along-ray metric.** We sidestep MIPS-vs-L2 by *learning* the geometry so that
  "front hit along the ray" equals "top logit," but must verify the learned metric behaves
  (no degenerate collapse of `C_r`). Regularize codebook spread; monitor at Stage 0.
- **Distillation drift on OOD text.** Drafter trained on a corpus may under-hit novel
  domains; acceptable (verification is lossless — misses just cost speed, not correctness),
  but measure the floor.

## 6. What reuse buys us / what must be built

- **Reuse (infra/teacher only, not the drafter):** the served target as distillation teacher;
  fully-trained EAGLE/MTP drafters as the *baseline to beat* (`memory/spec-decode-mvp.md`);
  DDTree tree-verify; `accept_greedy_ondevice`; `FutureMap` overlap; the spec `Proposer`
  interface; cloud-lease training env (`memory/cloud-lease-rocm-torch-env.md`).
- **Train / build (the drafter is a first-class trained model, not a bolt-on):** the trunk
  `T_θ` + R ray heads + token codebooks and their end-to-end distillation trainer, at real
  scale per target (Stage 0); the HIPRT static-BVH build + `image_bvh8` cast kernel with
  top-m collection (Stage 1); `RaySpecProposer` + aggregation → DDTree glue (Stage 2);
  optional target co-shaping adapter (§2, separate commitment).

Bottom line: RaySpec is the first RT idea in this repo that clears the three bars the
earlier ones failed — **search set large enough to amortize (V≈150k), geometry *trained* to
be next-token-predictive (not borrowed), and the offloaded op sits on genuinely idle silicon
in an overhead-bound regime.** It is a real bet with a real cost: Stage 0 is a proper,
per-target training run (cloud, GPU-days), because a trustworthy go/no-go needs a
properly-resourced drafter — but it is still settled entirely in Torch, before one line of
HIPRT is written. The order of spend is: debug-scale training (0a) → real training (0b) →
only-then HIPRT (Stage 1) → engine (Stage 2). We pay for the geometry to prove itself; we do
not pay for RT integration until it has.
