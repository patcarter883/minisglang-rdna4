# q4e: the defect is CONTEXT RETRIEVAL, and it is q4e-exclusive — 2026-09-22

Second thread of the day. [`Q4E_DEGENERATION_2026-09-22.md`](Q4E_DEGENERATION_2026-09-22.md) chased
answer DELIVERY (turns returning `content: ""`) and carries a correction at its head. This one chases
the defect that actually costs work: the model getting CONTENT WRONG.

    Hermes 5d26424a4be3 asked for a review of `@docs/ui-plan @ui`
    and rendered those as `docs/22909-74486-01` and `ui-0000446015-000046`.

Landed: the PLE once-per-forward ledger is now EVALUATED (it never was). Narrowed: GDN exonerated
across 1,379 turns on five models; the defect lives in q4e-exclusive machinery. Open: which of QSA
and PLE, with a dose ladder running to reproduce on demand.

---

## 1. The split: parameters fine, context broken

Pat's observation, from the degraded serve's own transcripts:

| task | needs | degraded serve |
|---|---|---|
| "What is 17 + 26?" | knowledge | **43 — correct** |
| "capital cities of France, Japan, Italy, Egypt, Canada" | knowledge | **Paris/Tokyo/Rome/Cairo/Ottawa — correct** |
| raw `/v1/completions` "The capital city of Italy is" | knowledge | **Rome, Cairo, Canberra, Ottawa, Athens, New Delhi — all correct** |
| "Repeat this exactly: the quick brown fox…" | **retrieval** | "Are all objects non-homogeneous?" |
| `@docs/ui-plan @ui` | **retrieval** | `docs/22909-74486-01` |

The weights are fine. What breaks is the machinery that reads the PROMPT. That is a large narrowing
and it invalidates the earlier probe design: `answer_delivery_probe` asks for capital cities, a
prompt with NO identifier in it, so there is nothing to conflate — its confabulation detector fired
ONCE in 30 turns on a serve that was catastrophically broken. **Presence of a detector is not
coverage.** `tools/q4e_context_vs_knowledge.py` replaces it: a knowledge CONTROL arm against a
retrieval arm whose answer is a random code planted in the prompt and existing nowhere else.

## 2. GDN is exonerated (p = 0.002)

Pat's suggestion: the other Qwen models share GDN but not QSA/PLE, so their sessions are a control.
Confirmed in code — `PLERuntime`/`ple_embedding` and the QSA indexer appear in `qwen4exp.py` ONLY,
while GDN is shared with `qwen3_5` and `nemotron_h`; Qwen3.6 is GDN too.

Census over Hermes `state.db`, ground-truthed: an identifier counts as confabulated only if it
carries ≥10 digits and appears NOWHERE the model was shown (all prior messages in that session).
Two false-positive classes had to be cut first, and both are instructive:

1. a flat regex with no context check scored *dates, URLs and CVE ids* — cloud models came out
   HIGHER than q4e, reproducing the exact trap journal §6 already found ("a turn-level detector
   cannot distinguish invented ids from reused ones");
2. with the context check but no id-field exclusion, the model's own harness-minted tool-call ids
   (`fc_…`, `call_…`, `toolu_…`) scored 80–96% on every cloud model.

With both cut, and **every surviving event read by hand** (n=17, small enough to read):

| | turns | genuine confabulations |
|---|---|---|
| **Qwen3.8-Flash-Next** (GDN + QSA + PLE) | 201 | **3 — 1.49%** |
| Qwen3.8-27B · 27B-MTP-NVFP4 · 3.6-35B-A3B · 3.6-35B-AWQ · 3.6-27B-AWQ (GDN) | 1,379 | **0 — 0.00%** |

Fisher two-sided **p = 0.00203**. All 12 GDN-group hits were benign: part-number enumerations
(`SEN0708/SEN0709/SEN0706`, `OPA320/OPA227/TLV9001`), hex register lists (`0x00/0x40/0x80/0xC2`),
real URLs, and one genuine path. q4e's three are the real thing — a **malformed UUID** with four
groups instead of five (`c828746f-2e02-67d2-ffffeb505287`), the `_40hex_NNN` fused form
(`dev_08e6d564…a3e0025` beside `e6d5642…a3e0025_057`), and invented `mc_…` / `@cmd_…__` ids.

This independently reproduces journal §6's census (q4e 1.5% vs 0.0% over 3,034 control turns) with a
completely different detector. Two instruments, same rate.

## 3. QSA cannot explain the reported failure

Session 5d26424a4be3 had `history=0` and ~70 prompt tokens. QSA's own contract:

> Below the budget they are equal BY CONSTRUCTION (the selection contains every visible token —
> that is the free correctness gate)

At r=4 a 70-token prompt is ~17 blocks, far under budget, so QSA runs **dense** there. It is not
sparsifying anything and cannot drop context it is not dropping. Unless the ≤budget bit-exactness
claim is itself violated — which `MINISGL_QSA_DENSE_CHECK=1` tests in place — QSA is not the
mechanism for the short-prompt failures.

Nor could it be convicted by reading. The hypothesis was that `QSAIndexCache` has **no invalidation
of any kind** (no free, no reset, nothing in the scheduler calls one) while being addressed by
recycled physical KV slots. It does not hold: `compressed_slot = physical_kv_slot // r` inherits the
allocator's decisions verbatim, only COMPLETE groups are scored, and `require_page_size` stops a
group straddling a page — so a stale entry is always overwritten before it can be read. Against
sglang upstream, our compression order (`mean → Gemma norm → rope at the group's OLDEST member`)
matches; upstream additionally passes an MRoPE axis map and `is_neox_style`, but for text-only
prompts MRoPE collapses to 1-D.

## 4. LANDED: the PLE ledger is evaluated

PLE runs on every token regardless of context length, which is what makes it the live suspect. And
`PLERuntime` carries this, verbatim, above its counters:

    prepares == commits + discards      and      commit_noops == 0
    A non-zero commit_noops is a forward that ran with nothing staged (or a batch committed twice)
    — the failure that freezes the n-gram context with no error anywhere.

**Nothing in the engine ever read those counters.** The module documents the exact arithmetic for its
own worst failure and never performs it — the same shape as a metric missing one plumbing hop and
exporting a flat zero, or a kernel gap closed with no caller.

It is a bad one to leave silent, because of the shape of the damage: a frozen n-gram history does not
crash and does not touch the weights. It corrupts the model's representation of ITS OWN CONTEXT, so
the reply comes back fluent, confident, and wrong about what it was shown. Knowledge intact,
retrieval broken — §1's signature exactly.

`ledger_fault()` now evaluates it and the scheduler reports after every `commit_staged`, once at full
volume then every 1000th (a fault repeats every forward and would otherwise bury the serve).
Mid-forward suppression is load-bearing and pinned by test: `commit_staged` runs AFTER the forward,
so between `prepare` and it the counters are unequal BY DESIGN, and a checker without that would fire
on every healthy step and get switched off. `tests/ple_ledger_test.py`, 13 checks, run in the serve
image CPU-only.

**This does not prove PLE is the cause.** It means that if it is, the engine says so instead of
serving confident nonsense.

### 4a. A latent bug the ledger would catch

`commit_staged()` has exactly ONE call site: inside `_forward`. The four speculative VERIFY sites
call `engine.forward_verify` directly and, by the scheduler's own comment, "never pass through
`_forward`" — yet they stage PLE with `defer_commit=True` immediately before the verify forward. So
on that path nothing parks the batch, `commit_verified` has nothing to resolve, and the next
`prepare_finish` silently overwrites `self.batch`: **the n-gram history freezes.**

Latent for q4e today — it serves `spec_algorithm=none` and mtp is refused for this architecture — so
it is not the bug under investigation. Recorded because it is real, because the ledger now makes it
loud, and because the production decode path is separately guarded (PLE forces the SYNCHRONOUS loop
precisely so the n-gram context cannot lag a token under overlap scheduling).

## 5. Open, ranked

1. **Reproduce on demand.** `tools/q4e_context_vs_knowledge.py` dose ladder, x-axis COMPUTED prefill
   (target ~730k, first failure historically at 189,627). Baseline point 0 clean: K 0/10, R 0/12,
   ident 0/2.
2. **Then bounce onto the ledger build and re-run.** A `PLE LEDGER FAULT` line convicts PLE outright;
   silence sends the search to QSA.
3. **`MINISGL_QSA=0`** runs the full-attn layers dense — the QSA ablation, no code change needed.
   `MINISGL_QSA_DENSE_CHECK=1` separately tests the ≤budget bit-exactness claim in place.
4. **PLE ablation** (Pat: people ablate PLE and the model still works) — there is no off switch today
   and `serve.sh` hard-errors without the shards, so it needs a gate on
   `hidden = hidden + self.ple.forward(hidden)`.
5. Parked, NOT merged: `fix/think-eos-guard` rewrites a mid-think EOS to the span close so a turn
   cannot deliver nothing. It fixes DELIVERY, not wrong content. Do not merge it believing otherwise.
