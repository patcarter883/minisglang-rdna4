# Markovian RSA shim for minisglang

Recursive Self-Aggregation (RSA, Venkatraman et al., arXiv:2509.26626) is a
test-time-compute method: expand a query into N rollouts, then repeatedly
aggregate K-subsets of the population into improved candidates over T rounds,
and select a final answer. The ZAYA1-8B report uses a **Markovian** variant.

This package (`minisgl.rsa`) implements the Markovian variant as an
OpenAI-compatible **shim proxy** that sits in front of a running minisglang
server. It is pure orchestration over the backend's `/v1/chat/completions`
API — it needs none of minisglang's ZMQ/scheduler machinery, so it runs as a
standalone front process rather than inside the engine. The vLLM reference
proxy (`/home/pat/code/vllm-gfx1201/rsa/`) is the design ancestor; this is a
minisglang-native re-implementation matching `minisgl.server.api_server`'s
FastAPI + pydantic style.

## What "Markovian" means here (assumption)

The task brief leaves the precise semantics open. We implement the most
faithful reading of *"each aggregation round conditions only on the previous
round's aggregated state (a Markov chain over rounds)"*:

- **Round 0 (expansion):** N independent rollouts of the original prompt.
- **Round t (1 ≤ t < T):** build N new aggregation prompts, each from a random
  K-subset drawn **only from round t-1's population**. No earlier round, and no
  accumulated cross-round history, is ever sampled — round t's state is a
  function of round t-1's state alone (the Markov step).
- The original query is *re-stated* as the problem statement each round (so the
  model always knows what it is solving), but the aggregation **context** — the
  candidate solutions shown — is purely the immediately preceding round's
  traces.
- **Selection:** from the final round's population, by majority vote over
  `\boxed{...}` answers (`auto`/`majority`), a final aggregation call
  (`final_agg`), or a random pick (`sample`).

This differs from *generalized* (non-Markovian) RSA, where a round may sample
across the full accumulated population. With `T=2` the two coincide (there is
only one prior round); the distinction bites at `T≥3`.

Round t-1 traces are truncated to their final `tail_tokens` tokens before being
fed into aggregation (the report's τ; set `--rsa-tail-tokens 0` to carry full
traces). Tails are token-exact when a local HF tokenizer is available, else a
character approximation; either way the cut advances to a paragraph/line
boundary so a round never starts mid-thought.

## Running

1. Start a normal minisglang server (the RSA backend), e.g. on its default
   port 1919. GPU work goes through the shared arbiter per `CLAUDE.md`:

   ```
   /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 -- \
     <your usual minisglang serve command, e.g. python -m minisgl --model ... --attn hip --graph N>
   ```

2. Start the RSA shim (CPU-only — **no GPU lease needed**, it only makes HTTP
   calls to the backend):

   ```
   PYTHONPATH=python python -m minisgl.rsa.server \
     --backend http://127.0.0.1:1919/v1 --port 2929 \
     --rsa-n 16 --rsa-k 4 --rsa-t 2 --rsa-tail-tokens 4096
   ```

3. Point any OpenAI client at the shim (port 2929 by default) instead of the
   backend. It serves `/v1/chat/completions`, `/v1/models`, and `/healthz`.

## Per-request control

Pass an `rsa` field in the request extra-body to override defaults per call:

```jsonc
{
  "model": "...", "messages": [{"role": "user", "content": "..."}],
  "rsa": { "n": 8, "k": 4, "t": 3, "tail_tokens": 2048 }
}
```

- `"rsa": false` or `{"enabled": false}` → bypass RSA, pass straight through.
- `"rsa": true` or omitted → use the server defaults.
- A request with `n > 1` is always passed through unchanged (RSA owns the
  population dimension).
- Request-level `temperature` / `max_tokens` override the RSA rollout defaults.

The response is a standard chat completion plus an `rsa` block reporting
`variant: "markovian"`, rounds run, population size, selection method, and the
vote tally.

## Parameters (defaults match the ZAYA1-8B Markovian config)

| flag / field        | default | meaning                                            |
|---------------------|---------|----------------------------------------------------|
| `--rsa-n`           | 16      | population size N                                   |
| `--rsa-k`           | 4       | aggregation set size K                              |
| `--rsa-t`           | 2       | total rounds T (round 0 = expand)                  |
| `--rsa-tail-tokens` | 4096    | tail of each prior-round trace (τ; 0 = full trace) |
| `--rsa-max-tokens`  | 8192    | per-rollout completion budget (round 0)            |
| `--rsa-agg-max-tokens` | =max | budget for aggregation rounds + final selection    |
| `--rsa-temperature` | 0.8     | rollout sampling temperature                       |
| `--rsa-selection`   | auto    | `auto`/`majority`/`final_agg`/`sample`             |
| `--rsa-max-concurrency` | 16  | max simultaneous backend rollouts                  |

## Cost

A full run issues up to `N × T` rollouts plus (at most) one final selection
call. With the defaults that is ~32 backend completions per user request — RSA
trades latency/throughput for answer quality, so size the backend's batch and
`--rsa-max-concurrency` accordingly.

## Notes on the minisglang backend

minisglang's frontend (`minisgl.server.api_server`) currently reports zero
token usage and does not implement the native `n` fan-in parameter or
`/tokenize`. The shim therefore: (a) issues one separate chat request per
rollout (no shared-prefill `n` batching — the engine's prefix cache still
amortizes the shared prompt); (b) reports whatever usage the backend returns
(zeros today, so the shim's usage totals will read low until the backend
populates them); and (c) computes tails with a local HF tokenizer (resolved
from the model id, or `--tokenizer`), falling back to a char approximation.
