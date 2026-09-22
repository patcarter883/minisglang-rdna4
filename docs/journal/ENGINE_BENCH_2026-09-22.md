# minisgl vs tcclaviger/vllm on identical checkpoints — 2026-09-22

Same cards (2x gfx1201, TP=2), same harness (`tools/spec_content_bench.py`), SAMPLED
(temp 0.8 / top_p 0.95 / top_k 20), 256 decode tokens, a fresh random prefix per request so no
repeat is a radix-cache hit, four content families every run. Both engines verified on their NATIVE
W4A8 paths — ours by the `[hip-engage]` ledger, theirs by
`Using R4dMxfp4MoEExperts (libr4d grouped MXFP4/fp8 GEMM, gfx1201)`.

## Results

    M=1 tok/s                  code    math  struct   prose    accept rate
      THEIRS g32 plain         68.8    68.8    68.8    68.6      —
      THEIRS g32 MTP k=3      132.6   133.5   143.9   117.2    0.63-0.80
      THEIRS g16 MTP k=3*      95.5   115.3   123.9   106.9    0.68-0.85
      OURS   g32 plain         95.1    95.4    94.9    95.7      —
      OURS   g32 MTP k=2      111.3   112.4   111.7   106.4    0.65-0.72
      OURS   g32 MTP k=4      106.8   105.4   106.5    93.5    0.44-0.59
      OURS   awq plain         92.1    92.4    91.9    92.8      —
      OURS   awq MTP k=4      110.0   110.9   115.3    90.0    0.39-0.59

    M=4 tok/s
      THEIRS g32 plain        208.9   208.0   207.6   209.0
      THEIRS g32 MTP k=3      341.2   424.2   448.7   354.4
      OURS   g32 plain        235.7   230.6   232.1   237.0
      OURS   g32 MTP k=2      256.6   253.8   254.1   235.1
      OURS   g32 MTP k=4      227.5   228.9   234.7   212.3   <- a LOSS vs plain

    * the g16 arm ran at 4096 ctx; its weights are ~2.25 GiB/rank heavier and left 0.24 GiB for KV
      at 32768. Its M=4 column is NOT comparable (starved pool, TTFT ~1.5 s from queuing).

## 1. We are faster without speculation; they are faster with it

**1.38x ours at M=1 plain, 1.13x at M=4.** Their MTP then overtakes: their best (143.9) is 1.29x our
best (111.7). At M=4 the gap widens sharply — they gain **1.63-2.16x** from MTP where we gain
**1.09x**, and go NEGATIVE at k=4.

## 2. Why our MTP cannot scale: `_DECODE_GEMV_MAXM = 16`

`layers/minv.py:169-178` pins spec verify to the decode GEMV so that "ordinary decode and
spec-decode VERIFY (M=K+1) land on the SAME kernel", which is what keeps verify bit-matching
sequential decode. Verify's M is **batch x (k+1)**, so the ceiling is a joint cap:

    k <= 16/batch - 1        batch 4 -> k<=3;  batch 8 -> k<=1

    batch 1, k=2 -> M=3   GEMV      1.17x
    batch 1, k=4 -> M=5   GEMV      1.12x
    batch 4, k=2 -> M=12  GEMV      1.09x
    batch 4, k=4 -> M=20  CROSSES   0.96x   <- the only arm that crosses is the only arm that loses

The sign flip lands exactly on the crossing. This is NOT "spec verify is inherently expensive" — it
is a threshold we chose. The irony: on a MoE target the invariance it protects is already lost
(`minv.py:41-43`, gemm2 swaps gather-reduce for a non-deterministic atomic scatter at M<=2, so
verify at M>=3 is always on a different arm than decode at M=1 anyway).

Two options, neither started: a verify-shaped tiled arm that is M-invariant by construction, or an
accepted measured crossing on the verify path. Note `verify-bit-exactness-is-load-bearing`:
approximate verify costs +3-7% per-draft acceptance, so the second is not free.

## 3. A second, independent problem: proposal quality at depth

Our accept rate falls 0.72 (k=2) -> 0.54 (k=4); theirs holds 0.725 at k=3. That is the PROPOSER, not
the verifier. We are tuned around it rather than against a limit — but both shipped widths were
re-measured and are CORRECT today (k_mtp=2 for MXFP4, k=4 for AWQ; k=4 on MXFP4 is worse everywhere
and a net loss at M=4).

## 4. No regression, and no MXFP4-vs-AWQ kernel problem

`CONTINUANCE_fp8_kv_calibration.md:155` records 90.4 no-spec / 102.3 MTP at M=1 on AWQ. Re-measured:
**92.1 plain, 110-115 MTP** — both reproduce. Production Prometheus showing 65-70 was LONG-CONTEXT
REAL TRAFFIC, not degraded code.

The same error produced a phantom "MXFP4 is 1.74x slower than AWQ": 38.55 (MXFP4) vs 65-70 (AWQ)
were both production aggregates over different traffic. Matched, **MXFP4 95.1 vs AWQ 92.1** — MXFP4
is marginally FASTER. The `[hip-engage]` ledger independently showed the e2m1 GEMV and WMMA arms
engaged throughout, so there was never a dispatch fallback.

**METHOD RULE: a production aggregate is a measurement of the traffic, not of the engine.** Never
place `1/(rate(tpot_sum)/rate(tpot_count))` or a `max_over_time(rate(...))` peak beside a bench.

## 5. Their separate-drafter spec paths do not run on 16 GiB cards

Three drafters, three distinct failures: **DFlash** on the 35B has no VRAM window at any util or
context (drafter + compiled head + graph pool allocate OUTSIDE `gpu_memory_utilization`, so lowering
it cannot make room — four configs all landed at 14.86-15.00 GiB allocated); **DFlash2** on the 27B
boots then HANGS (100% GPU at 73W idle power, `generation_tokens_total` stuck at 0 — the log alone
cannot show this, check the counter); **DSpark** dies in capture/warmup. MTP pays none of this
because its drafter ships inside the checkpoint — a structural advantage on constrained VRAM that is
independent of acceptance rate.

## 6. His group-16 format: better drafts, worse serving

Accept rate up across every family (struct 0.855 vs 0.796) — finer quantisation, better agreement —
but 20-30% slower at M=1. Two causes, neither arithmetic: **FSE disabled on all 40 layers**
(`shared-expert FSE quantization compatibility is not implemented for MXFP4_16Config`), and ~2.25
GiB/rank heavier weights that land directly on the KV pool.

We do not fuse shared experts at all: `Qwen3_5MoeSharedExpert` is a separate UNQUANTIZED bf16 SwiGLU,
which matches the checkpoint's own ignore list (their FSE path quantizes it as expert #257). We pay
2 extra GEMMs/layer and are still 1.38x faster, so fusion is not where their advantage lives. If we
ever want it, his RFI per-expert format-tag table is the mechanism that would let a bf16 shared
expert live in a quantized routed stack — the launch saving without the precision loss.

## 7. Caveats

Both engines quantize activations to fp8 on a checkpoint declaring NO `input_activations`, with no
calibration data to have. Neither arm's QUALITY reflects what the quantizer specified. The g16 M=4
column is confounded by its starved KV pool. TTFT was broken for the first few arms (fixed mid-run;
it counts any non-empty delta field now). At M>1 the two engines' draft counters mean different
things — ours counts scheduler steps, theirs counts per-request drafts — so M=4 acceptance is not
comparable across engines and is reported per-engine only.
