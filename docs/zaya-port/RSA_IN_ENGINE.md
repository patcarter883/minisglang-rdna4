# In-engine Markovian RSA — per-call API (Zaya)

The whole Markovian-RSA process (expand N rollouts → aggregate K-subsets over T rounds → select)
now runs **inside the minisglang server** and is served on the **normal port** (default `1919`) at
`/v1/chat/completions`. There is **no separate shim/proxy and no extra port** — a single chat
request with an `rsa` parameter runs the entire loop server-side and returns the final answer.

## How another agent calls it

`POST http://<host>:1919/v1/chat/completions` — a normal OpenAI chat body, plus a top-level `rsa`
field that the agent **varies per call**:

- **omit `rsa`, or `"rsa": null`, or `"rsa": false`** → an ordinary single completion (the existing
  sampling knobs `max_tokens`, `temperature`, `top_k`, `top_p`, `ignore_eos` apply).
- **`"rsa": true`** → run RSA with the server's `--rsa-*` defaults.
- **`"rsa": { ...patch... }`** → run RSA, overriding these fields **for this call only**:

| key | type | default | meaning |
|---|---|---|---|
| `n` | int ≥1 | 16 | population size N (rollouts in round 0) |
| `k` | int ≥1 | 4 | aggregation set size K (size of each random subset) |
| `t` | int ≥1 | 2 | total rounds T (round 0 = expand, t≥1 = aggregate) |
| `tail_tokens` | int ≥0 | 4096 | Markov tail of each prior-round trace carried into aggregation; `0` = full trace |
| `max_tokens` | int ≥1 | 8192 | per-rollout completion budget (round 0) |
| `agg_max_tokens` | int\|null | null | budget for aggregation rounds + final selection; null → use `max_tokens` |
| `temperature` | float ≥0 | 0.8 | rollout sampling temperature |
| `selection` | enum | `auto` | `auto` (majority vote if ≥2 boxed answers extract, else a final aggregation call), or force `majority` / `final_agg` / `sample` |
| `max_concurrency` | int ≥1 | 16 | max in-flight internal generations for this run |
| `max_retries` | int ≥0 | 1 | retries per failed rollout |
| `enabled` | bool | true | set `false` to turn RSA off for this call (same as `"rsa": false`) |

RSA calls require `messages` (chat); a raw `prompt` is rejected with HTTP 400. RSA calls are
**non-streaming** (the answer isn't known until the final round) — `stream` is ignored on the RSA path.

### Example

```bash
curl http://127.0.0.1:1919/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "zaya",
  "messages": [{"role":"user","content":"Solve and put the answer in \\boxed{}: ..."}],
  "rsa": { "n": 8, "k": 4, "t": 3, "tail_tokens": 2000, "max_tokens": 4096, "temperature": 0.7 }
}'
```

### Response

A standard OpenAI `chat.completion` whose `choices[0].message.content` is the selected final answer,
plus an `rsa` metadata block:

```json
{
  "choices": [{"index":0,"message":{"role":"assistant","content":"... final answer ..."},"finish_reason":"stop"}],
  "usage": {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0},
  "rsa": {"selection_method":"majority_vote","n":8,"k":4,"t":3,"tail_tokens":2000,
          "rounds":3,"population":8,"n_requests":17,"vote_detail":{"winner":"...","tally":{...}}}
}
```

(`usage` token counts are 0 on the in-engine path — the front-end reply carries no per-call token
accounting; `rsa.n_requests` reports how many internal generations the run issued.)

## Server-side defaults

The server's default RSA params (used by `"rsa": true` and as the base for a partial patch) are set
at launch with `--rsa-n / --rsa-k / --rsa-t / --rsa-tail-tokens / --rsa-max-tokens /
--rsa-agg-max-tokens / --rsa-temperature / --rsa-selection / --rsa-max-concurrency /
--rsa-max-retries`. RSA never runs unless a request opts in, so leaving the defaults as-is is safe.

## Capacity (one 16 GB card)

A single RSA call fans out `n` concurrent rollouts through the same scheduler, so the run must fit the
KV pool: **`n × (tail_tokens + generated) ≤ 65,615 tokens` (EP off) or `≤ 114,869` (EP on)**. The
default `n=16, tail=4096` already ≈ fills the EP-off pool — lower `n`, `tail_tokens`, or `max_tokens`
for headroom (one-card sweet spot ≈ `n=8`). With `--data-parallel-size 2` / `--enable-ep` the `n`
rollouts fan out across both replicas automatically (no client changes).

## Relationship to the old shim

The standalone proxy `python -m minisgl.rsa.server` (port 2929) still exists and works, but is now
redundant: the same `rsa` request body works directly against the engine on `1919`. New integrations
should target `1919`.
