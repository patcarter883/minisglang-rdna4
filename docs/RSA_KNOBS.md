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
- `tools: [...]` → the final answer is an action: the model calls a tool when the request needs one
  (returned as OpenAI `tool_calls`, `finish_reason: "tool_calls"`), instead of fabricating an answer
  from the tools-free rollouts.
- `think_budget` still applies, so structured/tool answers can't get stuck reasoning forever.

## Caching note (why C and canonical ordering matter)

Aggregation prompts are built from canonically-ordered tails, so prompts that share tails share a
**prefix** — the radix prefix cache (incl. the CCA recurrent state) can reuse it, and the reuse grows
with **C** (at C=N every aggregation prompt is the same sorted tail block). Rollout *generations* are
not cached (they are unique and would only pollute the cache); the shared prompt prefixes are. Net:
higher C costs more compute per candidate but caches better.

## Server defaults

Set fleet-wide defaults with `--rsa-n / --rsa-k / --rsa-t / --rsa-tail-tokens / --rsa-think-budget /
--rsa-max-tokens / --rsa-agg-max-tokens` (or `MINISGL_THINK_BUDGET` for the global β fallback). Any
per-request `rsa` field overrides them.

## Quick recipes

- **Fast, decisive** (chat, tool use): `{n:8, k:4, t:1, think_budget:256, agg_max_tokens:256}`
- **Balanced** (default-ish): `{n:16, k:4, t:2, think_budget:1024, agg_max_tokens:512}`
- **Max quality** (hard reasoning): `{n:16, k:6, t:3, tail_tokens:4096, think_budget:2048, agg_max_tokens:1024}`
