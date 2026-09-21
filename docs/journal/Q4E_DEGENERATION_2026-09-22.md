# Qwen3.8-Flash-Next: the degeneration is a SERVE-UPTIME defect — same-code reboot A/B — 2026-09-22

Follow-on to [`Q4E_DEGENERATION_2026-09-21.md`](Q4E_DEGENERATION_2026-09-21.md). One production
incident (Hermes session `5d26424a4be3`), one live reproduction, one same-code reboot A/B that
settles the question yesterday's five investigations could not: **the q4e serve loses answer delivery
over its own uptime.** A bounce restores it completely, with nothing else changed.

Headline: **chat-lane `no_answer` 28/30 → 0/30 across a reboot on the same commit.**

---

## 1. The incident: 25 minutes, zero delivered content

Session `5d26424a4be3`, 23:02:03 → 23:27:04 AEST 21 Sep, workspace `motor-control`, prompt
"Perform a review of the UI. @docs/ui-plan @ui". `agent.log` records `history=0` — a fresh ~70-token
context, not a deep one.

Degenerate from its **first reasoning tokens**, not mid-stream: it rendered `@docs/ui-plan @ui` as
`docs/22909-74486-01` and `ui-0000446015-000046` — §6's `id_noise` signature, but in *reasoning*
rather than tool args. 72,622 chars of reasoning, zero content, tailing into `Hmm ...` ×90. Hermes
injected its truncation-continuation banner at 23:20, re-fed the fragment, and killed the turn at
23:27 with its own repetition guard (`_turnDuration` 1,493 s).

This is a **different shape** from yesterday's `ebbc1dd0903b`: no radix HIT at depth, no thousands of
clean tokens first, no latched tool-call opener, no held block. It was bad from token one on an
almost-empty context.

## 2. Live reproduction on the still-running container

`lease-hitdiv-serve` — the *same* container that booted 17:59 and served yesterday's §4/§6
verification — was still up the next morning, and still broken:

| prompt | result |
|---|---|
| `What is 17 + 26? Answer with just the number.` | `17`, 2 tokens, content `""` |
| `Write one sentence about the ocean.` | reasoning `Internal reasoning step 1: 51424964`, content `""` |
| `Repeat this exactly: the quick brown fox…` | drifted into "Are all objects non-homogeneous?" |
| `List the capital cities of France, Japan…` | 24 reasoning tokens, `<\|im_end\|>` mid-think, content `""` |

Streaming confirms the shape: **24 reasoning deltas, 0 content deltas, `finish_reason: "stop"`.**

**The failure is delivery loss, not gibberish.** The cleanest specimen — the model gets the answer
right and then throws it away:

    reasoning: "The user wants the sum of 17 and 26… 17 + 26 = 43. So the answer is 43. I'll just put 43."
    content:   ""

**And it is invisible to monitoring.** The engine reports `finish_reason="stop"` and counts the turn
in `minisgl_requests_success_total` (155/157 at capture). `minisgl_empty_completions_total` does NOT
count it, because the reasoning tokens *are* output. Same blindness as the `empty_stop` metric in §2
of the 2026-09-21 journal, one layer down.

### Why nothing in the engine catches it

`_resolve_think_budget` (api_server.py) returns `THINK_BUDGET_UNBOUNDED` for any template that
consumes the reasoning LEVEL — the whole Qwen3 family; q4e's template renders `xhigh` for an unset
`reasoning_effort`. `ThinkGate.suppress_eos` then returns False *by contract*:

> NEVER under an unbounded budget. Holding EOS is only legitimate because the β backstop is
> guaranteed to release it… Unbounded thinking has to mean the model may also stop on its own.

So on this arm there is no EOS hold **and** no β backstop: nothing guarantees an answer is produced.
This is §7's "one live wedge component" and §9's open lead #1 — confirmed in source and live. Note
§7 describes the unbounded case as an EOS *hold* that risks a wedge; for EOS specifically it is the
opposite — the hold is *disabled*, which is the direction this incident hit.

`reasoning_max_tokens=200` re-arms both mechanisms: 9 of 10 runs then ran 201–324 tokens and forced
`</think>`. **But the answers were still junk** (`'<|answer'`, `'```'`, `' or any other'`, Chinese
text, `01-54230110121,24`), and one run returned `completion_tokens=0` — the gate arms in the decode
loop *after* the first token is sampled, so an EOS on step 1 is ungated. **The per-arm think budget
is containment, not the fix.**

## 3. Controls run BEFORE concluding anything

| alternative explanation | test | verdict |
|---|---|---|
| the model/weights are broken | same questions via `/v1/completions` (no template, no think span) | **healthy** — Rome, Cairo, Canberra, Ottawa, Athens, New Delhi all correct |
| prompt shape (bare prompt vs agent traffic) | Hermes-shaped system prompt + tools offered, `ptok=359` | still degenerate: malformed in-span `<tool_call><function=read_file><parameter=nonexistent-h>`, content `""` |
| special tokens / `<think>` structure | raw completions WITH `<\|im_start\|>` wrapper and with `<think>` | all three coherent — **not** the locus |
| GPU fault / driver | `dmesg` over the whole window | clean; no amdgpu fault or reset |
| a dead TP rank | `docker top`, `rocm-smi` | both ranks alive, both cards present |
| prefix / recurrent-radix cache serving wrong state | nonce-prefixed prompts that cannot share a prefix | equally degenerate — exonerated, consistent with §3 |
| context length (short vs deep) | session `268e9ff30b69` a1 (fresh, short) at 18:05 vs `5d26424a4be3` (fresh, short) at 23:02 | healthy vs catastrophic — **length controlled, uptime differs** |

That last row is the one that pointed at uptime. Note `268e9ff30b69`'s `content: ""` turns are **not**
delivery failures — every one carries `ncalls: 1`, i.e. reason-then-call, which is correct. Checked
before using that session as a healthy baseline; the counts alone would have misled.

## 4. The reboot A/B — the finding

**Method.** `tools/answer_delivery_probe.py` (new): 5 prompts × 6 reps × 2 lanes = 60 requests per
arm, `top_k=1, temperature=0`, `max_tokens=400`. Six countable detectors; every turn's full text
written to the fixture. Determinism is deliberately NOT a detector — §3 measured this engine's greedy
floor at a near-tie argmax flip within a handful of tokens, so scoring "the arms diverged" would fire
on a healthy serve.

**One variable.** Both arms: commit `6494f18f`, image `minisgl-rdna4:lean`, byte-identical `[serve]`
banner *and* full `ServerArgs` line (`tp=2 conc=2 mem=0.85 graph_bs=0 expert_cache_gb=2.5
weight_offload_gb=33 page_size=16 gdn_radix=True spec=none`), same `/model` + `/ple` + `/ghost`
mounts, same checkpoint, same prompt set and reps. The serve-path source is byte-identical between
the boot commit `cb165420` and `6494f18f` — `d678e5f0` and `84b31161` touched only `tools/` and
`docs/`. Relaunched through the gpu-control panel (`POST /api/up`, which wraps `gpu-lease -n 2
--detach --name …`), then the new banner diffed against the old before trusting anything.

| | before (up 15 h) | after (fresh) |
|---|---|---|
| **chat / no_answer** | **28/30** | **0/30** |
| chat / eos_in_think | 18/30 | 0/30 |
| chat / loop | 4/30 | 0/30 |
| chat / digit_noise | 1/30 | 0/30 |
| raw / no_answer | 13/30 | 6/30 |
| raw / loop | 6/30 | 1/30 |

Every chat-lane detector went to zero. Per-prompt, `no_answer` before→after: arith 5→0, capitals
5→0, count 6→0, echo 6→0, ocean 6→0.

**Verdict: the q4e serve degrades over its own uptime.** Not a code regression, not the checkpoint,
not prompt shape, not the client.

Read the clean arm's text, per the probe's own rule:

    reasoning: "We need answer user's simple arithmetic… 17+26=43. Ensure no extra."
    content:   "43"

**Honest caveat on the control lane.** `raw/capitals` is 6/6 empty in *both* arms — a base-LM
completion declines an instruction-shaped prompt with no chat template. That is prompt design, not a
defect, and it is why the raw lane is a valid within-prompt before/after control but its *absolute*
`no_answer` rate is not a quality measure. `raw/arith` and `raw/count` (0/6 in both arms) are the
trustworthy control cells.

## 5. What this does to yesterday's conclusions

- **Every q4e quality measurement must now record serve uptime.** Any A/B whose arms sat at different
  uptimes is confounded. §4's gated-vs-raw arms ran at 17:27 and 17:52 on separate young boots
  (probably fine); §5's replay arms and §6's live verification ran at various uptimes on a boot that
  was provably broken by 23:02.
- **§5's "greedy never confabulates" holds only on a fresh serve.** On the degraded serve at
  `top_k=1, temperature=0` the probe produced `Internal reasoning step 1: 51424964`,
  `01-54230110121,24` and `<parameter=nonexistent-h>`. §5's 0/10 was measured on a young boot.
- §3 (state-restore exonerated) and §7 (the unbounded think budget) stand, and §7 gains a second,
  opposite-facing consequence: the unbounded flag disables the EOS hold as well as the backstop.
- §6's `id_noise` detector is args-only and scores fused forms (`hex::`, `_40hex_`), which free prose
  never produces. The free-text signature needed its own rule — a real identifier decorated with an
  invented digit run. A naive `\d{6,}` misses every real sample (`docs/22909-74486-01`'s longest run
  is 5), so the new `digit_noise` rule is "bare run ≥6 OR one token carrying ≥8 digits", scored only
  on digit-free prompts, calibrated 10/10 against five observed samples plus five controls.

## 6. Open, ranked

1. **Find the mechanism.** Unknown. It was already broken 5 h in on light traffic (106 requests), so
   neither pure request volume nor pure long idle. Candidates, all things that accumulate over a
   boot: the 2.5 GiB / 1923-slot expert cache (`free=0`, ~48.7k promotions / 46.8k evictions at
   capture), the 27.94 GiB/rank pinned host weight arena, radix/KV fragmentation, the host-tier GDN
   snapshot store (`MINISGL_REC_SNAP_HOST=1`).
2. **Bisect INSTALLED and running** (`tools/answer_delivery_timeline.py`, `ccec304f`): systemd user
   timer `q4e-delivery-timeline.timer` fires every 30 min — 10 chat requests per tick, CPU-only, no
   lease — recording the failure rate beside serve uptime, request/token totals,
   `empty_completions`, KV-pool use, prefix-hit ratio and the `[expert-cache]` counters.

       column -t /home/pat/fixtures/minisgl-answer-delivery-timeline/timeline.tsv
       journalctl --user -u q4e-delivery-timeline | grep ONSET
       systemctl --user disable --now q4e-delivery-timeline.timer      # stop it

   30 min brackets the transition to well under an hour (healthy 18:20, catastrophic 23:02 → a
   ~4.7 h window) without the probe becoming a load on the serve it measures. The tick resolves its
   container per run (`--container auto`), because the lease label changes on every relaunch and a
   hardcoded name would quietly keep measuring a container that no longer exists. A tick against a
   booting or absent serve exits non-zero and writes NO row, so a gap in the TSV means a failed tick,
   never a stopped timer. Unit files: `~/.config/systemd/user/q4e-delivery-timeline.{service,timer}`.

   **Already excluded as the signature:** `fill=0.987`, `inflight=25`, `free=0` read IDENTICALLY on
   the fresh healthy serve and on the broken one. Look for what diverges, not for what looks
   alarming. First three ticks (up 801/902/985 s): `no_answer` 0/10 throughout.
3. **Fix the step-1 arming hole** regardless of the above: `_maybe_arm_think_gate` runs in the decode
   loop after the first token is sampled, so a bounded-budget request can still end with
   `completion_tokens=0`.
4. **Give the unbounded path an EOS guard that is not the β budget.** These are two mechanisms behind
   one flag. An EOS sampled inside a span the template itself opened is never a legitimate turn end —
   there is no answer. The shape already exists: `ToolCallGate` refuses the turn-ending token while a
   block is open, bounded by a large budget. The reasoning span wants the same treatment.
5. **Bounce the serve before any quality run**, and before blaming a checkpoint or a commit.

## 7. Landed

| commit | what |
|---|---|
| `6494f18f` | `tools/answer_delivery_probe.py` — two-lane answer-delivery probe, `digit_noise` calibrated 10/10 |
| `922dc870` | `tools/answer_delivery_compare.py` — arm comparison that REFUSES incomparable arms |
| `2f689c44` + `ccec304f` | `tools/answer_delivery_timeline.py` — scheduled tick walking the uptime curve; container resolved per tick |

Fixtures: `/home/pat/fixtures/minisgl-answer-delivery/` — `20260921-232215-before-reboot`,
`20260921-233626-after-reboot`, plus `before-reboot-serve-state.txt` (the degraded serve's `/metrics`,
expert-cache counters and card state, captured before teardown).
