# ZAYA spec-decode accept-len tests — re-measure against the batch-invariant verify fix

**Why these exist.** ZAYA spec-decode acceptance was floored (DFlash ~0.26 accept-len on minisgl vs
1.30 on vLLM; TiDAR two-forward ~0.18, fused ~0.07). Root cause identified: the **verify forward
(M=K+1) is not bit-identical to sequential decode (M=1)** — bf16 GEMM reduction order is tile-shape
(M) dependent, the extend-attention kernel differs from single-token decode, and ZAYA's top-1 MoE
router amplifies ULP drift into **expert-flips**. The verify logits the drafter is judged against
therefore diverge from the decode logits it was trained to match → drafts get rejected → low
accept-len. The fix is **batch-invariant verify kernels** (in progress in
`/home/pat/code/rdna4-hip-kernels`). These harnesses are the decisive re-measure once it lands.

## The tests

| script | target | draft | measures |
|---|---|---|---|
| `tools/run_zaya_dflash_accept.sh` | RXF `ZAYA1-8B-RXF-h32` (deployed serve) | vLLM-trained `m4dss-ep14-opd-r1` | mean accept-len, emitted/step, greedy losslessness |
| `tools/run_zaya_tidar_accept.sh` | fp8 `zaya1-tidar-opd-fp8` (self-draft) | — | accept-len, emitted/step, greedy losslessness |

Both run on `minisgl-rdna4:lean` with the **live kernels repo mounted** (`-v
/home/pat/code/rdna4-hip-kernels:/kernels`, `PYTHONPATH=/kernels/_kernels`), so they exercise
whatever verify kernels are currently checked out — the run logs the kernels `HEAD` so each result is
pinned to a kernel version. Shared engine: `tools/zaya_spec_accept.sh`.

The DFlash test deliberately uses the **untouched vLLM-trained drafter** (not a re-distill), so any
accept-len gain is attributable to the **kernel fix alone**. The re-distill was stopped precisely
because it optimised the wrong target (decode-token labels vs verify-M logits); fixing verify makes
prefill≡decode≡verify, at which point the existing drafter should serve correctly with no retrain.

## Protocol (before/after the fix)

```
# BEFORE (baseline on current kernels, HEAD without the fix): establishes the floor
gpu-lease -n 1 -- bash tools/run_zaya_dflash_accept.sh      # expect mean accept-len ~0.26
gpu-lease -n 1 -- bash tools/run_zaya_tidar_accept.sh       # expect ~0.18 (two-forward)

# AFTER the batch-invariant kernels land in rdna4-hip-kernels (same command, new kernels auto-picked-up):
gpu-lease -n 1 -- bash tools/run_zaya_dflash_accept.sh      # HYPOTHESIS: accept-len recovers toward vLLM's 1.30
gpu-lease -n 1 -- bash tools/run_zaya_tidar_accept.sh       # HYPOTHESIS: TiDAR lifts off its floor
```

- **accept-len** (`[spec] mean accept-len` / `emitted/step` in the server log) is THE number.
- **losslessness** is the correctness gate — baseline (spec off) greedy text must be a byte-exact
  prefix of the spec-on text. A MISMATCH means the verify path is wrong, independent of acceptance.
- Runs **eager** (`GRAPH=0`) by default so the only variable is the verify path. After accept-len
  recovers, re-run with `GRAPH=8` for the prod tok/s number (graph capture required before any
  throughput claim — accept-len itself is graph-independent since graph replays the same kernels).

## Knobs

- DFlash: `DRAFT_HOST=<ckpt> NUM_DRAFT=<block>` — default `m4dss-ep14-opd-r1` at 4; for the 15-wide
  drafter use `DRAFT_HOST=.../ZAYA1-8B-DFlash-CCA-5L-ns15-ep14 NUM_DRAFT=15`.
- TiDAR: `NUM_DRAFT=4` (block_size), `FUSED=1` for the single-forward fused path (only `FUSED=0`
  two-forward is lossless), `MIX<1.0` for Trust-Diffusion logit-mixing (intentionally not lossless).
- `GRAPH`, `MEMRATIO`, `GENTOK` on both.

## Interpreting the result

- **accept-len jumps toward vLLM's number** → verify-M was the dominant tax; the fix recovers
  ZAYA spec-decode with no retrain. Ship graph pass for tok/s.
- **accept-len barely moves** → the gap is cross-engine OOD / drafter quality, not verify
  determinism; re-distill against verify-M logits (reuse the preserved 1.12M on-policy seedbuf,
  re-targeted to verify logits) becomes the next lever.
