# CAM — production targets (real-world-effective)

**Source:** distilled from the spine integration + live serving E2E (2026-07-08/09, minisgl
`cam-production`, Qwen3.5-4B, softsteer export) by the agent testing CAM as a real client would.
This is the **master production roadmap**; it supersedes the ad-hoc "start with #10" sequencing.

**Framing.** Today a client consuming CAM gets behavior *largely replicable by a small fact dict +
prompt injection*. These targets are the deltas that make CAM the thing nothing else does.
Base-coupling is **excluded** from the bar — adapters are in progress; that gap is artifacts, not
mechanism (tracked as T8).

**How this reconciles with the two priors:**
- **Bake-off** ([[cam-substrate-bakeoff-pointer-wins]]): pointer (GTE-whitened key + exact object
  ids) is the lossless, base-agnostic **default** delivery. Satisfies T5 (multi-token) and the
  "zero prompt tokens" half of T1 — pointer delivery is *in-forward*, not prompt-prepend.
- **Titans review** + **hybrid spec** (`CAM_HYBRID_DESIGN.md`): the **tap** is the only path to
  soft-steering, reasoning-integration, and **prior-override (T7)** — the one capability prompt
  injection fundamentally cannot replicate. So the tap is not a niche opt-in; T7 is *the demo* that
  justifies the mechanism over RAG. Hybrid = pointer default + tap where it earns its cost.

---

## Priority order (working agent)

**T6 → T1 → T5 → T2 → T3 → T7 → T4 → T9 → T8**

- **T6 + T1 first** — they restore CAM's two differentiators (honest gate; in-forward read) on the
  path clients actually hit (ambient `/v1/chat`, not the explicit `/cam/ask`).
- **T5** — carry the #100 pointer path into the serving store so real dev facts (multi-token
  objects) are first-class.
- **T2** — semantic GTE keys (`#10`); needs T1's read path live to measure retrieval quality on
  real traffic.
- **T3, T7, T4** — capacity, the RAG-beating editing demo, and the real-knowledge evidence.
- **T9** — client ops polish. **T8** last only because it waits on external artifacts (adapters).

**Definition of "real-world effective":** a client can make CAM its **primary** fact store (side
index demoted to backup) when **T1, T2, T3, T5, T6** hold simultaneously on real traffic. **T7** is
the demo that justifies the mechanism over RAG; **T4** is the evidence; **T8** is the founding
promise.

---

## Issue mapping

| Target | Where it lives | Tracked as |
|---|---|---|
| T1 tap-native transparent read | minisgl serving | this doc (serving) |
| T2 semantic subject keys + retrieval harness | minisgl serving + memory-organ | "minisgl #10" (semantic key) + hybrid-spec §5 |
| T3 capacity → thousands | memory-organ | **#4** (N-scale), **#17** (editing @ N-scale) |
| T4 real-knowledge validation | memory-organ | **#1** (validate on real knowledge) |
| T5 multi-token objects @ serving parity | minisgl serving + memory-organ | **#2** (multi-token transfer), **#100** (pointer id-bank, merged) |
| T6 write-gate fidelity (real `base_p`) | minisgl serving | this doc (serving) |
| T7 prior-override as a served capability | minisgl serving + memory-organ | **#16** (prove editing on real benchmark) |
| T8 transfer leg in serving (adapters) | minisgl serving + memory-organ | **#18** (reusable translator), **#13** translator-loader |
| T9 client-facing ops | minisgl serving | this doc (serving) |
| T10 `/v1/completions` endpoint coverage | minisgl serving | this doc (serving); pairs with T1/T9 |

Serving-side targets (T1, T6, T9, the serving halves of T2/T5/T7/T8) live in this repo's CAM docs
because minisgl's GitHub is the public upstream sglang fork — not a place for CAM issues. The
research-side halves are tracked as memory-organ issues (standing instruction: memory-organ is
GitHub-managed).

---

## The targets (current → target → accept)

### T1 — Tap-native transparent read (retire the RAG fallback)
- **Current:** ambient `/v1/chat/completions` AND `/generate` reads share ONE helper
  (`_cam_auto_augment`, `MINISGL_CAM_AUTO=1`, `cam-production` lines 486/584) that cosine-matches
  facts and **prepends them as prompt text**. In-forward delivery (pointer *or* tap) only fires on
  explicit-subject paths (`/cam/ask`, `mem_subject`). The differentiator — zero prompt tokens,
  mid-generation — is bypassed on exactly the path real traffic uses. **Because both endpoints call
  the same helper, the fix is one change, not per-endpoint** (and it lands on T10's new endpoint too).
- **Target:** ambient requests resolve subject(s) → read → stage bank(s)/ids → **in-forward
  delivery** (no injected text). Multi-subject: pick and validate either a merged-bank read or
  per-segment staging.
- **Accept:** a plain chat question answered from memory with the **served prompt byte-identical to
  the user prompt** (verify server-side), at ≥ the delivery rate of the equivalent `/cam/ask`, and
  inert on unrelated prompts (conf-gate/router holds false-delivery at baseline). Holds on
  `/v1/chat/completions`, `/generate`, and `/v1/completions` (T10).

### T2 — Semantic subject keys + retrieval-quality harness (minisgl #10)
- **Current:** auto-read matches capitalised proper-noun spans only; pooled base-embedding key
  cannot separate paraphrase (~0.58 cos) from different-entity-same-name (~0.62); auto-write leans
  on a regex fallback. Precision/recall on real traffic: unmeasured.
- **Target:** semantic key (GTE-ModernColBERT / `CAM_GTE_KEYS`, whitening MANDATORY) for
  addressing; labelled eval harness for write precision/recall and read hit-rate/false-delivery on
  **real** (not synthetic) traffic.
- **Accept:** paraphrased queries for a stored subject hit **≥0.85** while false-delivery on
  near-miss entities stays ≤ current baseline; harness runs in CI against a frozen labelled set.

### T3 — Capacity: ~128 facts → thousands (memory-organ #4 / #17)
- **Current:** ~4–9 subjects/bank crowding knee → ~128 comfortable facts/namespace; past the knee
  delivery degrades *silently*; LRU eviction can drop a durable fact with no client signal.
- **Target:** thousands of coexisting facts with flat delivery (per-position/disjoint-bank scaling,
  online re-shard as a first-class op) and **announced** degradation (eviction + crowding pushed to
  clients, not just in stats).
- **Accept:** **≥0.9 delivery at N=1,000** held-out reads in serving; a subscribed client sees every
  eviction; re-shard (raise B) completes online with zero lost facts.

### T4 — Real-knowledge validation (memory-organ #1)
- **Current:** load-bearing numbers are the synthetic name→cargo probe + curated/CounterFact
  single-relation edits. Real phrasing / entity distribution / fact diversity unproven.
- **Target:** ingest a real corpus slice (e.g. a project's docs/decisions — the spine distillation
  output is a ready supply) and hold delivery/locality/generalization on natural queries.
- **Accept:** on **≥500 real facts**: valid ≥0.95, delivered ≥0.85, locality Δ ≤0.05, paraphrase
  generalization ≥0.7 — **through HTTP**, not the offline harness.

### T5 — Multi-token objects at serving parity (memory-organ #2, #100)
- **Current:** the validated serve path stores short/single-token objects; the pointer id-bank
  (#100) solved span-exact delivery offline. Real dev facts have multi-token objects
  (`CODENAME.md`, `feature/spine-direct-graph`, version strings).
- **Target:** carry the #100 pointer path into the **serving** store so arbitrary short phrases
  (2–6 tokens) are first-class objects.
- **Accept:** span-exact delivery **≥0.9 for 2–6-token objects** via `/cam/ask` AND the T1
  transparent path; `/cam/remember` stops 422-ing/truncating multi-token objects.

### T6 — Write-gate fidelity on the model-share path
- **Current (frontend/model-share mode, the deployed one):** the base-uncertainty gate is OPT-IN
  (`MINISGL_CAM_WRITE_GATE=1`; default `/cam/remember` force-writes); the gate is a *generation*
  probe; `base_p` is never computed (always 0.0), so clients can't rank/audit/calibrate. The gate
  is CAM's most defensible primitive — serving ships a diluted version of it.
- **Target:** **logit-level gate** on the shared backend model (the seam exists — the scheduler
  owns the logits): compute real `base_p` at prefill, gate by `remember_tau`, return it on every
  write and probe.
- **Accept:** served gate decisions match the offline `eval_persistent` gate on a probe cohort;
  `base_p` populated and monotone with the offline measurement; **gate on by default** for ambient
  writes, force-write only as an explicit mode.

### T7 — Expose prior-override (editing) as a served capability
- **Current:** the strongest validated result — overriding a frozen base's confident prior
  (counterfactual 0.996 same-base and cross-family) — is not reachable through normal serving:
  ambient chat gets the RAG note (T1); nothing in the API demonstrates "the model now *believes*
  the edit." Prior-override is the one capability prompt injection fundamentally cannot replicate;
  it should be the serving demo, not an offline table.
- **Target:** a served edit visibly flips the base's confident answer in plain chat (no subject
  param, no prompt note), neighbors unaffected.
- **Accept:** CounterFact-style eval through HTTP: mem-on flips the prior on edited subjects
  (**≥0.9**), true-prior suppressed, locality Δ ≤0.05 on neighbor prompts, all with T1's
  byte-identical-prompt guarantee.

### T8 — Transfer leg in serving (adapters; artifacts pending)
- **Current:** cross-base transfer validated offline (translator, 0.94→0.66 by distance); serving
  has never attached one store to two bases. Coupling is an artifact shortage, not a mechanism gap —
  so serving should be *ready* when artifacts land.
- **Target:** serve the SAME store against a second base via translator/adapter: checkpoint format
  carries the translator, load-time donor-match relaxed to translator-match, namespace state
  base-agnostic.
- **Accept:** curated store + base swap + zero re-ingestion → delivery on base-2 **≥0.8× base-1**
  through HTTP; a client (spine) replays nothing.

### T9 — Client-facing ops (small, keeps trust)
- **Current:** persistence, namespaces, auth, audit/undo/rebuild, stats land (P0/P1 done). Two
  client-visible gaps: no batch health probe (spine probes one fact at a time via `/cam/ask`), and
  evictions are pull-only (stats counter).
- **Target:** `POST /cam/probe {subjects:[...]}` → per-fact delivered/conf in one call;
  eviction/crowding webhooks or an events endpoint.
- **Accept:** spine's readback verification and eviction-reconciliation each collapse to one request.

### T10 — Completions-API coverage (`/v1/completions`)
- **Current:** the server exposes only `/generate` (native raw-prompt) and `/v1/chat/completions`
  (chat). There is **no OpenAI-style `/v1/completions`** (legacy text-completions) route — an
  OpenAI-SDK client using the completions API (not chat) **404s**, CAM aside. CAM ambient
  read/write is wired into the two endpoints that DO exist (via the shared `_cam_auto_augment` /
  `_cam_auto_write` helpers), so this is an API-surface gap, not a CAM-mechanism gap.
- **Target:** add `/v1/completions` (OpenAI text-completions request/response shape) reusing the
  `/generate` request path, so it inherits the shared CAM helpers automatically (and the T1
  in-forward upgrade for free).
- **Accept:** an OpenAI-SDK `completions.create(...)` call is served with a conformant response;
  with `MINISGL_CAM_AUTO=1` it reads from memory identically to chat (same helper), and the T1
  byte-identical-prompt guarantee holds on it too.
- **Priority:** pairs with T1 (shared-helper fix covers it) / T9 (client-facing surface). Cheap;
  do it alongside T1.
