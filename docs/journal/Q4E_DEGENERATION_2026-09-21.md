# Qwen3.8-Flash-Next degeneration — incident, root-cause chase, and the think-gated fix — 2026-09-21

One production incident (Hermes session `ebbc1dd0903b`), five investigations, three commits landed
in-tree, one detector calibrated, two leads closed by measurement. Everything below is backed by a
receipt: a fixture, a commit, a log line, or a Prometheus query. Inconclusive results are recorded
as inconclusive.

Timeline (AEST): incident 13:49–14:04 · forensics 14:30–15:30 · cross-engine probe 15:46–16:30 ·
divergence test 17:04–17:18 · think-gated fix 17:30–18:30 · first gated production session 18:05–18:27 ·
confabulation chase 18:40–19:40.

---

## 1. The incident: a mid-think collapse became an 11-minute blind wedge

Session `ebbc1dd0903b` (UI review of the motor-control repo), request `uid=14`, 13:49:09:

1. Radix prefill HIT at `cached_len=43616`, then **3,867 tokens of clean review reasoning** (the
   ConnectionPanel WebSocket bug — real, correct analysis).
2. Mid-think, the model degenerated into eval-harness-register looped text: bracketed
   meta-instruction prose ("omnibus middleware test", "[FLIRT_PROBE…]" markers, a hidden
   math-question framing) repeating whole blocks — the phrase "a continuation of the original
   conversation" occurs **41 times** in the 34,358-char retained reasoning span.
3. 13:53:06, the engine warned: *"uid=14 has 8194 chars buffered inside an unclosed tool-call
   block after 3867 completion tokens — nothing has been streamed to the client since the opener."*
   The degenerate text had emitted the XML tool-call opener (its format is embedded in the system
   prompt by the checkpoint's own template), the **raw-first tool matcher latched it**, and every
   token after it was held from the client.
4. 14:04:33 the client aborted (`Aborting request for user 14`, `minisgl_requests_aborted_total`
   +1) and sent a continuation retry, which completed normally.

### What it was NOT — closed by measurement, not assumption

| hypothesis | test | verdict |
|---|---|---|
| foreign text injected via a second request | `minisgl_requests_inflight` / `_running_requests` / `_waiting_requests` over the whole window (Prometheus :9090) | **max 1** — never a concurrent request |
| external traffic via the cloudflared tunnel | `cloudflared_tunnel_total_requests` (read via a netns-sharing throwaway container — the image is distroless) | **0 over 21 h of tunnel uptime** |
| radix KV cross-request leak | mechanism review | not an injection vector — prefix reuse only skips prefill compute for a request's own identical prefix |
| prompt-side artifact bleed | full-context scan (messages, api_content, system_prompt; UUID + hex patterns) | the ID strings appear **nowhere** in the context — the model confabulated them (§5) |
| another local eval harness | disk search (excluding the user's own Claude Code transcripts discussing the excerpt) | no source; also no public-web match for the register text |
| state-restore corruption | §3 divergence test | **exonerated** — restored state is byte-identical to fresh |

**Termination chain, attributed precisely.** No cap fired: the client sent `max_output_tokens`
32,768, the engine's derived default was 24,406 (boot banner), the runaway killer needed 65,536
held chars. The request ended at the **client's 924 s watchdog**; the flushed fragment reached the
client as `finish_reason=length`. Call this a "max_tokens cut" only after checking which of the
three terminators fired — they are distinguishable in the serve log.

**Engine behaviour during the "stall":** not stalled. `minisgl_generation_tokens_total` rose at
18–19 tok/s for the entire 15 minutes (~16.4k tokens): a full-speed reasoning-span loop, invisible
only because the latched opener held everything server-side.

---

## 2. The commit-campaign review (what was already fixed)

The 09-18 → 09-21 commits form a coherent, evidence-driven campaign; the incident intersects it:

- **Junk tool-arguments** (the production signature, 7.9% of q4e turns vs ≤0.8% on every control):
  root cause was the grammar layer forcing JSON into a checkpoint whose template mandates XML
  (25% junk args, code bodies capped at 369 chars). Fixed by `9e8f167d`/`ebdaf08c` (scope the
  grammar to structured calls only), `c502912c` (schema-typed arg coercion), `9ab2f3f7`
  (forward unknown tool names). **The post-fix probe was already clean at 0/57**
  (fixture `20260920-184258-q4e-fixshape`).
- **Turn ends mid-tool-call** (a regression the grammar removal exposed; found in production,
  session `e5b8b76e21e8`, twice in eleven turns): fixed by the `ToolCallGate` eos-guard
  (`6e50c758`, landed 13:39 — two minutes before the incident serve booted).
- **Dead agent turn at the 8,192 cap**: fixed by pool-derived max_tokens (`3c969cfb`); this serve
  resolves to 24,406.
- **preserve_thinking OFF**: tried, measured WORSE, reverted with a full write-up (`6c010a4c`).

The incident serve therefore ran with every fix live — the collapse was a signature none of them
covered: **spontaneous mid-reasoning degeneration inside an unclosed think block**, with the
containment machinery (EOS suppression + raw-first holdback) converting it into the worst possible
shape: 11 minutes of client blindness, ended only by the watchdog.

---

## 3. HIT-vs-fresh divergence test — the state-restore lead, closed

The suspicion: GDN recurrent-state restore at deep `cached_len` (the norm on this arm, because
preserve_thinking makes prefills long and shared-prefix-heavy) corrupts hidden state without
crashing — exactly what mid-stream quality collapse looks like.

**Method** (`tools/gdn_hit_fresh_divergence.py`): replay the real incident context (session
messages 0..28, the real Hermes toolset, deterministic filler) at ~69k prompt tokens; greedy decode
(`top_k=1`, the only sampling param sent); three phases per boot — `fresh` (cold cache), `hit`
(same request again; radix restores at `cached_len=69120`), `partial` (restore + delta prefill,
the production seam); two cold boots; compare token streams.

**Result** (fixtures `minisgl-hit-divergence/`, request SHA `8658065cd501934e` across all runs):

| comparison | result |
|---|---|
| fresh vs HIT, same boot (×2 boots) | **byte-identical** — reasoning, content, tool args |
| fresh vs fresh, across boots | diverges at char 3 ("Let me" vs "Let's continue") — near-tie argmax flip |
| partial vs partial, across boots | same class of flip |

**Verdict: restored GDN+PLE state IS fresh state.** The HIT path sits exactly at the engine's
kernel-atomics determinism floor; it is not above it. The state-restore hypothesis for the collapse
is closed. (Corollary used later: this arm flips near-tie decisions run-to-run on identical
prompts — the measured basis for §5's margin analysis.)

---

## 4. Think-gated tool matching — the fix (`3534b460`)

The checkpoint's own contract (read off its `chat_template.jinja`): the generation prompt opens
the reasoning span, and tool calls sit strictly **after** the span closes. The raw-first tool
matcher (which exists for Laguna-shaped checkpoints that emit calls without ever closing the span)
therefore latches on noise whenever a model emits opener-shaped text mid-think — and the system
prompt embeds the call-format example, so a degenerating model mimics it. The latched opener then
(a) holds every later token from the client and (b) arms EOS suppression on a call that does not
exist.

**The change** (`MINISGL_TOOL_MATCH=think-gated`, set on the q4e arm in `serve.sh`):

- Frontend: the reasoning split runs FIRST; the tool matcher sees only its content lane — in the
  streaming loop, the flush path, the non-streaming final parse, and the mid-flight runaway scan.
- Reasoning layer: a `tool_openers` override keeps a mid-span opener inert in `reasoning_content`
  (the client keeps seeing the stream live, even mid-collapse). The default release rule is KEPT
  for Laguna-shaped checkpoints — vLLM/SGLang parity is pinned by test.
- Scheduler: `ToolCallGate.arm(..., think_closers=...)` suspends opener matching until the span's
  close (or answer-header) pattern commits; in-span openers never latch or suppress EOS; the
  8,192-token block budget still bounds a real post-span call. The RSA lane is deliberately NOT
  gated (ZAYA-only, Laguna-shaped).
- Per-arm env; the default (`raw`) changes no other arm's behaviour.

**Tests** (`tests/tool_match_gating_test.py`, 8/8, pure/no-GPU): gate in-span inertness, budget
bounding post-span, ungated arm unchanged, gated stream order, default release rescue, gated
non-streaming parse, and an end-to-end replay of the incident transcript shape (in-span opener +
looped register text + real post-span call) proving gated mode holds nothing while the raw mode
still latches exactly as it did on the day. Existing suites (`think_gate`, `toolcall_eos_guard`,
`runaway_kill`, `continuation_guard`, `delimiter_derivation`, `reasoning_tool_continuation`,
`tool_arg_coercion`, `unknown_tool_forward`) all green unchanged.

**Same-build A/B** (fixtures `20260921-172736-q4e-thinkgated`, `20260921-175219-q4e-raw-control`,
only the env differs; verified via `/proc/1/environ` — `docker exec env` does NOT see serve.sh's
runtime exports):

| | gated (15 turns) | raw (9 turns) |
|---|---|---|
| junk_args | 0 | 0 |
| any diagnostic flag | 5/15 = 33.3% | 3/9 = 33.3% |
| empty_stop | 3 (p=0.27, Fisher; all three contain no opener → mode-independent code paths) | 0 |
| flagged flavor | refusal/empty turns | tool calls with confabulated tags leaking as content |

Identical flag rates, different flavors — the mode does not change how often the model misbehaves
on this probe, only which confusion flavor shows. Note the probe is non-streaming, so the
streaming-lane gating is verified by the unit tests and the production session (§6), not by this
A/B's traffic.

**Live verification:** the first production session on the gated build (`268e9ff30b69`, §6)
produced a 19,931-char think turn that stayed coherent (top repeated 40-gram: 1×; the incident loop
was 41×) and recovered into a tool call; zero block/EOS warnings in the serve log for the whole
session. The incident shape did not recur.

**Related fix** (`cb165420`): the compose environment block enumerates forwarded vars and
`MINISGL_TOOL_MATCH` was not among them — an outer override never reached the container (the
fourth instance of the five-hops trap). This blocked the raw control arm until fixed.

---

## 5. Arg confabulation — a model-side residual, chased to its mechanism

First gated production session (`268e9ff30b69`, local NVFP4) vs same-day cloud session
(`a88e2dfea548`, `qwen/qwen3.8-flash`, same project):

| | local q4e | cloud |
|---|---|---|
| assistant turns / calls | 20 / 19 | 201 / 250 |
| junk_args / wedge / length events | 0 / 0 / 0 | 0 / 0 / 0 |
| max coherent think | 19,931 chars (recovered) | 22,189 chars |
| **confabulated calls** | **3 of 19** (msgs 39683/39687/39690) | **0 of 250** |
| tool-result error rate | 21% | 3% |
| outcome | abandoned the re-verification | completed end-to-end |

The three corrupted calls are well-formed JSON with **confabulated content**: a real filename
decorated with invented identifier noise (`uuid::name`, `_40-hex_NNN` forms), a bridge command
line in a `path` slot, `head 161 166`. Exhaustive context scan: the ID strings exist nowhere the
model was shown — it invented them in the format of tokens it sees elsewhere (git SHAs in terminal
results). This is NOT parse-related (the args are the model's own tokens; the gated pipeline
delivered them intact) and NOT junk_args (the payloads carry apparent work).

**NVFP4 execution path, read off the checkpoint and the engine** (answers "isn't the upconvert
lossless?"): the checkpoint is ModelOpt **NVFP4 W4A4 group-16** with GDN/attention/PLE/hyper-connection
layers spared (`quantization_config.ignore`). The engine serves the e2m1 weights packed as int4
through the fp8-WMMA path — **no VRAM upconvert; the weights are bit-exact NVFP4**. The two-level
scale (e4m3 × f32) folds to one fp16 per group of 16: ~2^-11 relative rounding (~0.05%). The real
deviation is the **activations**: the checkpoint is calibrated for static per-group A4; the engine
serves per-token e4m3 — a scheme neither side of the "NVFP4 ≈ FP8" parity claims measured (those
are Blackwell-native W4A4, dense released models, benchmark contexts). KV scales verified: the
sidecar carries 24 calibrated tensors covering exactly the 12 full-attention layers.

**Replay experiments** (`/tmp/hashbleed_replay.py`, fixtures `hashbleed-replay/`): the identical
52,761-token decision point (everything before the first corrupted call; same request SHA), same
probe schema:

| arm | reps | calls made | confabulated |
|---|---|---|---|
| local q4e, temp 0 | 10 | 4 | **0** |
| local q4e, temp 0.8 (1,200 cap) | 6 | 0 (cap truncated deliberation) | — |
| local q4e, temp 0.6 / 0.5 | 6 / 6 | 2 / 0 | 0 |
| local q4e, temp 0.8, production 4,096 budget | 6 | 0 — all six thought to the cap | — |
| GGUF UD-IQ4_XS (lemonade, tail-5, temp 0.8) | 5 | 6 | 0 |

**Verdicts:** greedy never confabulates → correct-vs-corrupt arg decisions are **degraded but
ordered** (margin narrowing, not floor collapse; the engine's atomics floor was measured directly
in §3). The production corruption was **not reproducible** in 12 replay calls at the exact decision
point — it is rarer than one n=19 session implied. Honest caveat: the replay uses the probe's
system prompt and 4-tool schema, not Hermes' full prompt; this arm's act-vs-think decision is
hypersensitive to that framing (call rate flips run-to-run with no temperature pattern). If the
signature recurs, the missing variable to try is the real Hermes system prompt.

Operational note for the GGUF arm: lemonade's ~90 s proxy timeout cannot hold a 52k-token prefill
on that ~72 tok/s CPU-hybrid box — the arm ran a truncated decision point (suggestion, not proof)
and the full-context version needs `global_timeout` raised in its UI.

---

## 6. id_noise detector — calibrated (`d678e5f0`)

Added to the probe's scorer as a SECONDARY detector: fires on **fused identifier decoration**
(`uuid::name`, `hex::name`, `_40hex_NNN`, `shorthex::`) in tool args, and shell syntax in a `path`
value — forms nothing in an agent's context legitimately produces. Calibration through the replay
validator (the repo's gate for detector edits): **3/3 incident calls caught; 0 firings across
3,034 control turns** (four models) — q4e 1.5% vs 0.0% everywhere else.

Two false-positive classes were found and cut during calibration, both instructive:
1. Pattern-shape over-fires on legitimate regex search patterns (`z_tilt|bed_screws|screws_tune`,
   `python -m minisgl` — 20 turns on the 27B-MTP control). Shell-syntax checks are now `path`-only.
2. **Bare hex/SHA is usually legitimate reuse** of an id the model saw in a tool result (the
   deepseek control copies SHAs verbatim from git output; URLs carry hex segments). URLs are
   stripped; bare ids no longer fire; only fused forms score. Lesson: a turn-level detector cannot
   distinguish invented ids from reused ones — only the fusion forms are safe to score.

---

## 7. Cross-model census — who else was exposed

The parser machinery ran engine-wide; its harm did not. Length-terminated events in state.db
(strict wedge subset — `length` finish with >500 chars reasoning, no delivered content, no
calls — in parentheses):

| arm | assistant turns | length-terminated (wedge-shaped) |
|---|---|---|
| Qwen3.8-Flash-Next (+ mislabeled `model` era) | 181 + 1,112 | **8 (2)** — incl. one 293,536-char reasoning turn (msg 25280, its partial call recovered) and the incident's own `ebbc1dd` turns |
| Qwen3.6-35B-AWQ | 418 | 0 (0) |
| Qwen3.8-27B-MTP | 651 | 0 (0) |
| Laguna | 1,668 | 0 (0) |

The one documented cross-model held-block incident (session `f59a0028faa6`, Qwen3.6-35B subagent,
~45k tokens to a dead socket) was a REAL unclosed call — the shape the runaway killer exists for,
not the spurious in-span latch. ThinkGate's EOS hold inside think is unbounded on any template that
consumes reasoning-effort kwargs (the whole Qwen3 family) and 1,024-token-bounded elsewhere — that
unbounded hold is the one wedge component that remains live after the tool-gate fix, and the
per-arm think budget is its closure (open lead).

---

## 8. Commits landed this session

| commit | what |
|---|---|
| `14531a0e` | probe provenance: pin the target model to the MATCHING `/v1/models` entry (multi-model endpoints like lemonade list their whole registry) |
| `3534b460` | think-gated tool matching — in-span openers are noise, not calls (`MINISGL_TOOL_MATCH=think-gated`, q4e arm) |
| `cb165420` | compose: forward `MINISGL_TOOL_MATCH` (the override could not reach the container) |
| `d678e5f0` | `id_noise` detector, calibrated 3/3 vs 0/3,034 |

New files: `tools/gdn_hit_fresh_divergence.py`, `tests/tool_match_gating_test.py`.
New fixtures: `minisgl-hit-divergence/` (6 samples), `hashbleed-replay/` (6 arms),
`20260921-1546/1549/1727/1752-*` probe runs.

## 9. Open leads (ranked, with what would close them)

1. **Per-arm think budget** (`reasoning_max_tokens` via chat_template_kwargs; ThinkGate arms at
   3/4 of max_tokens for bounded requests): caps a genuine think-loop in-engine instead of at the
   client watchdog. The one live wedge component remaining.
2. **Detector-triggered intervention** (sampler-side repeated-n-gram / sustained top-1-prob):
   zero effect on healthy turns; closes the scorer's remaining signature; pairs with the id_noise
   detector's plumbing.
3. **Think-span-gated repetition penalty**: the Qwen card's own "endless repetition" remedy,
   region-gated so it never touches tool-call logits (global presence penalty measured to kill
   tool calls — do not set it globally).
4. **If confabulation recurs**: replay with the REAL Hermes system prompt (the missing variable);
   track production rate via `toolcall_degen_replay_validate` as sessions land.
5. **Full-context GGUF arm**: raise lemonade's `global_timeout` first (~90 s default cannot hold
   the prefill).
