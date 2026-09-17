# Recurrent-radix prefix caching under spec-decode (PROTOTYPE)

**Status:** prototype, default-OFF. Env gate: `MINISGL_GDN_RADIX_SPEC=1`.

> **2026-07-16 RESULT — DOES NOT WORK YET.** Empirically the prototype produces **zero cache hits**
> under spec (identical-prompt probe: hit=0, flat TTFT). A no-spec serve of the same model hits
> correctly (368/379 tokens reused, TTFT 0.356→0.071s, ratio 0.65), so recurrent-radix itself is
> fine — enabling it under spec silently suppresses the rec_state capture-or-match. Output is NOT
> corrupted (just no reuse). The "all capture is on the prefill path so it should just work" analysis
> below was WRONG; treat it as the hypothesis that failed. Do NOT enable in production. Next: instrument
> rec_state attach vs match under `_spec_loop` to find where the hit is lost. (Minor confound: the two
> serves also differed in EP; design intent says EP composes and spec is the gated combo.)

## What this is

Recurrent-state hybrid models (GDN / CCA-Zaya) normally run the **`naive`** prefix cache — no
prefix reuse — whenever spec-decode is enabled. The scheduler forces it:

```python
# python/minisgl/scheduler/scheduler.py
_rec_radix_ok = self.engine.spec_config is None            # historical gate
```

so `--spec-algorithm mtp` (etc.) silently disabled all prefix caching. This prototype lifts that gate
behind an env flag, so a spec serve *also* gets recurrent-radix prefix reuse (shared system prompts,
few-shot prefixes, multi-turn history).

```python
_rec_radix_ok = (
    self.engine.spec_config is None
    or os.environ.get("MINISGL_GDN_RADIX_SPEC") == "1"
)
```

## Why it should be lossless (the analysis behind the flag)

1. **All reuse-relevant capture sites are on the PREFILL path**, which is *always* non-spec:
   - `_stash_rec_state` — only caller is the **chunked-prefill** branch of `_process_last_data`.
   - `_maybe_capture_rec_state` — at prefill-complete and at finish (`_free_req_resources`).

   The **decode** step performs **no** radix insertion. Spec only replaces the decode step, so a spec
   commit advancing `cached_len` by >1 never reaches the page-aligned stash guard — there is nothing
   for the variable commit length to trip. (This is why the prototype is a *gate flip*, not new
   capture code: plain decode does no per-step checkpointing either.)

2. **Prefill is byte-exact.** The M-invariance fix (route dense linears through `layers/minv.py::
   minv_linear`) makes chunked prefill bit-identical to single-pass across all layers, so the state
   captured at a prompt page boundary equals a fresh forward's — see
   [[cca-prefix-cache-gemm-m-dependence]] / `tools/cca_chunk_bisect.py` (0.0 all 40 CCA layers).

3. **State clone/restore is byte-exact.** `GDNStateCache.load_slot`: *"decomposition-invariant under
   the bit-exact recurrent kernel, so continuing a prefill from this restored state is byte-identical
   to prefilling the shared prefix from zero."* `install_verify_state` gathers from the verify
   per-token scratch, which is parity **0.0** vs the oracle recurrence (`tools/gdn_hip_parity.py`,
   `scratch[last]==final_state`).

4. **Ordering holds.** Under spec the scheduler runs `_spec_loop`, which is **synchronous**
   (schedule → forward → process, no forward launched ahead) — the exact clone-after-forward /
   before-next-forward ordering recurrent-radix needs. The historical gate was caution, not a
   mechanism conflict.

5. **The known GDN-spec verify-argmax gap does not corrupt reuse.** That gap changes *which* tokens a
   sequence emits (verify argmax vs decode argmax on some prompts), not the recurrent state cached for
   the tokens it *did* emit. Reuse is keyed by the exact emitted tokens, so a later request matching
   that token prefix restores the correct byte-exact state. (SPEC_DECODE.md §GDN verify.)

## What is NOT covered (future work)

- **Within-generation checkpointing.** Neither this prototype nor the non-spec path inserts
  decode-generated prefixes into the radix tree before finish. Adding it (capture page-aligned state
  from the verify scratch during decode) would raise hit-rate for long shared *generated* prefixes,
  but is a separate feature that should land for spec and non-spec together.
- **Sampled spec** (`MINISGL_SPEC_SAMPLED=1`) is orthogonal and expected to compose, but is part of
  the same A/B matrix below.

## Validation bar (MUST pass before default-on)

Baseline against **spec-WITHOUT-cache**, not plain decode — the verify-argmax gap already breaks
byte-identity vs plain decode independent of caching, so comparing to plain decode would misread it.

1. **GSM8K equivalence.** `spec + recurrent_radix` vs `spec + naive`, greedy, same seed:
   accuracy and per-item answers match (reuse the `gsm8k.*.json` baselines already in the tree; the
   M-invariance A/B harness in `tools/`). Any systematic accuracy drop = a real caching bug (this is
   exactly the 45→25 signature the M-invariance work chased).
2. **Whitebox state fidelity.** Extend `tools/cca_radix_whitebox.py` / `cca_kv_seam.py` to run with
   spec on: restored-prefix state == fresh-prefill state, 0.0, per layer.
3. **Finish-time / multi-turn.** A two-turn conversation where turn-2's prompt = turn-1 prompt+output
   restores at the turn-1 boundary and produces identical continuation to no-cache.

## How to A/B

Both need a container recreate (`docker compose --profile qwen35b-mtp up -d`).

```
# A: spec, no prefix cache (current default / baseline)
MINISGL_GDN_RADIX_SPEC=0

# B: spec + recurrent-radix prefix cache (this prototype)
MINISGL_GDN_RADIX_SPEC=1
```

Confirm B took the path: startup log shows
`recurrent-state hybrid model: using recurrent radix prefix cache` followed by the
`^ PROTOTYPE: recurrent radix + spec-decode` warning. Watch the prefix-cache metrics on Grafana
(`minisgl_prefix_cache_hit_ratio` / `..._hit_tokens_total` / `..._prompt_tokens_total`, in the
"Prefix Cache" dashboard row) and TTFT on repeated shared-prefix prompts — hit ratio reads ~0 in
config A (naive) and >0 in B.

## Rollback

Delete the `or os.environ.get(...)` clause to restore the hard gate; the flag then no-ops.
