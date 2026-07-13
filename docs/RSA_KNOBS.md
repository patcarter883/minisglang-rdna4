# Markovian RSA — knob reference for the SPINE agent

In-engine Recursive Self-Aggregation (RSA), implemented per **arXiv:2605.05365**. Enable it per
request by adding an `rsa` field to a `/v1/chat/completions` call. The whole
expand → aggregate → select loop runs server-side (each rollout is an internal generation), so one
HTTP request returns the final aggregated answer.

```jsonc
POST /v1/chat/completions
{
  "model": "zaya",
  "messages": [{"role": "user", "content": "…"}],
  "max_tokens": 2048,
  "rsa": { "n": 16, "k": 4, "t": 2, "tail_tokens": 4096, "think_budget": 1024,
           "agg_max_tokens": 512 }
}
```

`"rsa": true` uses the server defaults; omit it (or `"rsa": false`) for a normal single completion.
Any subset of fields patches the defaults.

## The five paper knobs

| Paper | Field | Default | What it does | Push it UP when… | Push it DOWN when… |
|------|-------|---------|--------------|------------------|--------------------|
| **N** | `n` | 16 | **Population**: candidates generated each round. More = wider search, more diverse candidates. | hard problems, want best-of-many quality | latency/cost matters; easy tasks |
| **C** | `k` | 4 | **Aggregation subset**: how many candidate *reasoning tails* each new candidate is built from (C≤N). Higher C = each aggregation sees more peers (more synthesis, longer prompts). | you want strong cross-candidate synthesis; also improves prefix-cache reuse (see caching note) | keep aggregation prompts short/cheap |
| **T** | `t` | 2 | **Rounds**: round 0 expands, rounds 1..T-1 aggregate. More rounds = more refinement passes. | iterative refinement helps (multi-step reasoning) | 1 round (just expand+select) is enough |
| **τ** | `tail_tokens` | 4096 | **Tail length**: only the last τ tokens of each candidate's **reasoning trace** are carried into the next round (the Markovian "bounded workspace"). `0` = carry the full trace. | reasoning is long and the important work is spread out | keep context small/fast; short reasoning |
| **β** | `think_budget` | `null`→1024 | **Thinking budget**: after β reasoning tokens the server force-emits `</think>` so the model **stops reasoning and produces its answer**. Prevents "thinks until max_tokens, never answers". | the task genuinely needs long reasoning (proofs, multi-step) | you want fast, decisive answers; short tasks |

## Two budget knobs (not in the paper, but you should set them)

| Field | Default | What it does |
|-------|---------|--------------|
| `max_tokens` (top-level) | 8192 | Per-**rollout** completion budget (round 0 exploration). |
| `agg_max_tokens` | `null`→`max_tokens` | Completion budget for the aggregation rounds **and the final answer**. The final generation is a single sequential decode (~23 tok/s on ZAYA), so this is the aggregation-latency floor: 256≈11s, 512≈22s, 1024≈44s. **Set this** — it is unbounded by default. |

**Guidance:** set `agg_max_tokens` to the expected *answer* length, which is usually far shorter than
a rollout. Tool calls / short JSON → 256; normal answers → 512; long-form → 1024. `think_budget` (β)
bounds the *reasoning* inside that budget, so a good combo is e.g. `think_budget: 512,
agg_max_tokens: 768` (≤512 tokens reasoning, then the answer in the remaining room).

## Thinking vs. answer (how the response is shaped)

Every rollout and the final answer are `<think> … </think> answer`. β force-closes the `<think>`
span at the budget; the reasoning lands in `reasoning_content` and the user-facing answer in
`content`. If the model re-opens `<think>` after a forced close, that is folded back into
`reasoning_content` too — `content` is always just the final answer. **Tails carried between rounds
are reasoning-only** (the last τ tokens *before* `</think>`), matching the paper's `tail_τ(y)`.

## Structured output & tools (compose with RSA)

- `response_format: {type: "json_schema", …}` → the **final** answer is grammar-constrained to valid
  JSON (rollouts stay free-form for exploration).
- `tools: [...]` with **`tool_choice: "required"`** (or a specific `{"type":"function","function":
  {"name": …}}`) → the final answer is **grammar-constrained to a complete, schema-checked tool call**
  (same treatment as `response_format`): the arguments are forced to match the tool's `parameters`
  schema and the generation terminates. Returned as OpenAI `tool_calls`, `finish_reason: "tool_calls"`.
- `tools` with **`tool_choice: "auto"`** (or default) → an xgrammar **structural tag**: the model may
  answer in prose OR call a tool, and *if* it opens a JSON tool-call wrapper (`<tool_call>` / `<tools>`)
  its arguments are forced to the tool's schema. Free text is unaffected (no regression). **Caveat for
  ZAYA:** its native tool format is `<zyphra_tool_call>` **XML** (not JSON), so the structural tag
  doesn't trigger and ZAYA `auto` calls stay best-effort (it also tends to *reason about* calling
  without committing). This path fully helps JSON-native tool models (Qwen3-style). **On ZAYA, use
  `tool_choice: "required"` for a guaranteed, schema-valid call.**
- `think_budget` still applies, so structured/tool answers can't get stuck reasoning forever.

## Caching note (why C and canonical ordering matter)

Aggregation prompts are built from canonically-ordered tails, so prompts that share tails share a
**prefix** — the radix prefix cache (incl. the CCA recurrent state) can reuse it, and the reuse grows
with **C** (at C=N every aggregation prompt is the same sorted tail block). Rollout *generations* are
not cached (they are unique and would only pollute the cache); the shared prompt prefixes are. Net:
higher C costs more compute per candidate but caches better.

## Server defaults

The **`Default` column above is the library/paper default (N=16, C=4, T=2, τ=4096, β=1024)** — the
ZAYA-report max-quality config. This 16 GB / DP=2 deployment intentionally serves a **lighter,
concurrency-safe default** (the paper config OOMs the box under load and is slow for routine work):

| Knob | This deployment | Paper/library |
|------|-----------------|---------------|
| N (`n`) | **4** | 16 |
| T (`t`) | **1** | 2 |
| β (`think_budget`) | **512** | 1024 |
| `max_tokens` (rollout) | **1024** | 8192 |
| `agg_max_tokens` | **640** | `max_tokens` |
| τ (`tail_tokens`) | **2048** | 4096 |
| `top_p` | **0.95** | 1.0 |

> **`top_p=0.95`** is deliberate: ZAYA is a verbose reasoning model and at `top_p=1.0` its rollouts
> wander into never-EOS runaways that burn the whole budget (slow + truncated). 0.95 lets them stop
> naturally. β also caps reasoning at ¾ of `max_tokens` so the answer always has room.

> ## ⚠️ For JSON / structured output, you MUST pass `response_format`
> Asking for JSON only in the system prompt ("Return ONLY valid JSON…") is **not reliable** — a
> reasoning model rambles and never commits clean JSON (this was the classifier bug). Pass
> `response_format: {type:"json_schema", json_schema:{…}}` so the **final** answer is grammar-
> constrained to your schema. RSA rollouts stay free-form for exploration; only the final answer is
> constrained. Validated: RSA + `response_format` → valid JSON; RSA + prose-only → rambling prose.

Rationale: at N=4, 25 concurrent requests = 100 rollouts, which fits the 128 decode slots
(`max_running_req=64` × 2 DP replicas, all graph-captured up to bs=64) with no queueing and no OOM.
**Override per request for hard problems** (e.g. `rsa:{n:16, t:2, think_budget:2048}`) — the
per-request `rsa` field always wins.

These are set on the serve command (`docker-compose.yml`) and are individually overridable by env:
`MINISGL_RSA_N / _T / _THINK_BUDGET / _MAX_TOKENS / _AGG_MAX_TOKENS / _TAIL_TOKENS` (and the CLI flags
`--rsa-n / --rsa-t / --rsa-k / --rsa-tail-tokens / --rsa-think-budget / --rsa-max-tokens /
--rsa-agg-max-tokens`; `MINISGL_THINK_BUDGET` is the global β fallback).

## Quick recipes

- **Fast, decisive** (chat, tool use): `{n:8, k:4, t:1, think_budget:256, agg_max_tokens:256}`
- **Balanced** (default-ish): `{n:16, k:4, t:2, think_budget:1024, agg_max_tokens:512}`
- **Max quality** (hard reasoning): `{n:16, k:6, t:3, tail_tokens:4096, think_budget:2048, agg_max_tokens:1024}`
