# Real-world serving load — agripath-server

`agripath_real_load.jsonl` — 119 chat-completion requests from a REAL LangSmith agent trace
(`019f6e2d-97e6-7a23-a116-e4138ac73aa3`, project `agripath-server`), all against the served model
`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`. Each record carries the **real schedule** so the replay
reproduces the actual traffic SHAPE, not a synthetic saturation load.

## Real envelope (why shape matters)
- span **2236s (~37 min)**, 119 calls
- **peak concurrency 7** (overlapping calls) — bursty, NOT flat
- inter-call start gap: median **6.2s**, max **226s** (0-gap clusters = parallel fan-out)
- call duration: median **40s**, max **305s**
- prompt tokens: min 454 / median 2685 / max 13850

A constant-N saturation replay pins running_requests flat and looks nothing like this on Grafana;
the schedule mode below matches the real spiky 0→7→0 pattern with idle stretches.

## Record format (one JSON/line)
`{"offset_s", "dur_s", "messages":[{"role","content"}...], "max_tokens", "prompt_tokens"}`

## Replay
```
# FAITHFUL (default) — fire at real offset_s/SPEED; SPEED compresses wall clock, preserves shape.
python tools/loads/replay_load.py tools/loads/agripath_real_load.jsonl <port> schedule [speed=8] [model]

# SATURATION — N workers back-to-back for DUR (kernel profiling only; unrepresentative shape).
python tools/loads/replay_load.py tools/loads/agripath_real_load.jsonl <port> saturate <conc> <dur_s> [model]
```

NOTE: message contents are agripath-server's real agent prompts — INTERNAL. Do not publish outside
the private serving repos.

