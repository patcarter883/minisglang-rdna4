# replay_load.py — trace replay against a running serve

Replays a captured set of chat-completion requests against a running minisgl serve over the
OpenAI-compatible endpoint (`POST http://localhost:<port>/v1/chat/completions`). The point of
replaying a capture rather than generating synthetic traffic is SHAPE: a real agent workload is
bursty — fan-out clusters separated by idle stretches, with a wide spread of prompt lengths — and a
constant-concurrency load generator pins `running_requests` flat and exercises a scheduler regime
that never occurs in production.

No capture is committed to this repo. Point the tool at your own.

## Input format

One JSON object per line:

```json
{"offset_s": 0.0, "dur_s": 41.2, "messages": [{"role": "user", "content": "..."}], "max_tokens": 2048}
```

| Field | Required | Used for |
|---|---|---|
| `messages` | yes | The request body, passed through verbatim. |
| `offset_s` | `schedule` (hard — a record without it raises `KeyError`), `sequence` | Arrival time relative to the start of the capture. Records are sorted by it. Read as 0 when absent, so a file with no `offset_s` gives every request its own lane in `sequence`. |
| `dur_s` | `sequence` | Observed end-to-end duration in the capture; with `offset_s` it gives the interval the lane assignment is built from. Read as 0 when absent, which collapses `sequence` to one serial lane. |
| `max_tokens` | no | Per-request output cap, clamped to 4096. Defaults to 512. |

Any other keys are ignored. Every request is sent with `temperature: 0.7` and a 600 s client timeout.

## Modes

```
python tools/loads/replay_load.py <jsonl> <port> sequence [model]
python tools/loads/replay_load.py <jsonl> <port> schedule [speed=8] [model]
python tools/loads/replay_load.py <jsonl> <port> saturate <conc> <dur_s> [model]
```

| Mode | Preserves | Drops | Use it for |
|---|---|---|---|
| `sequence` | Concurrency width. The capture's `[offset_s, offset_s+dur_s]` intervals are greedily partitioned into lanes — one lane per unit of the capture's real peak concurrency — and the lanes run in parallel, each firing its next request the moment the previous one completes. | Wall-clock timing and idle gaps. | **Benchmarking.** Deterministic and comparable across kernel or scheduler changes, because no result depends on how fast the run happened to be. Streams each request, so it reports per-request TTFT and decode tok/s. |
| `schedule` | Arrival shape — fan-out bursts and idle gaps alike — by firing each request at `offset_s / speed`. `speed` compresses wall clock without distorting the pattern. | Nothing, but the run takes `span / speed` seconds and results move with ambient load. | Watching the real traffic pattern on a dashboard, and reproducing scheduler behaviour that only shows up under bursty arrival. |
| `saturate` | Nothing. `conc` workers loop over the requests back-to-back (longest `max_tokens` first) for `dur_s` seconds. | Arrival shape entirely. | Forcing sustained decode for kernel profiling. Not representative — do not quote its throughput as a serving number. |

`model` defaults to `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`; pass it explicitly for anything else. Note
the positional slot differs per mode — it is argument 4 for `sequence`, 5 for `schedule`, 6 for
`saturate`.

## Retry cleanup — `--keep-all`

A capture taken from a live agent usually contains requests that timed out client-side and were
re-sent, so replaying it raw double-counts that work and inflates prompt-token volume. The script
therefore drops a hardcoded set of offset-sorted indices (`DROP_IDX`) before replaying, and prints
how many it dropped.

**`DROP_IDX` is specific to the capture it was written for.** Against any other file those indices
remove arbitrary requests. Pass `--keep-all` to replay the file as-is, or edit `DROP_IDX` for your
own capture.

## Output

All modes print `ok` / `err` / total output-token counts and elapsed wall time. `sequence`
additionally prints p50/p90/max TTFT (time to first content token, i.e. prefill latency), p50/p90
per-request decode tok/s, and overall output throughput across the lanes.
