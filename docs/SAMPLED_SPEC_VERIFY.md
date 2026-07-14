# Scope — sampled (rejection-sampling) speculative verify

**Why.** minisgl spec-decode is greedy-only: the loop gates on `spec_ok = all(req.is_greedy)`
(`scheduler.py`) and `verify_greedy` accepts a draft iff it equals the target's **argmax**. Every
RSA request decodes at `temperature=0.8, top_p=0.95` (`rsa/core.py:300`; the Markovian aggregation
rounds are rollouts too), so `is_greedy=False` → the batch falls back to plain decode → **the DFlash
drafter never engages on the RSA workload.** Sampled speculative decoding (Leviathan 2023 / Chen 2023)
makes spec lossless *for sampling*, unlocking the drafter for RSA (both the concurrent rollouts and the
single-stream aggregation decode, which is ~half the RSA wall and launch/BW-bound — the ideal spec
target).

**Losslessness redefined.** Greedy spec is *byte-identical* to plain greedy. Sampled spec is
*distributionally identical* — the emitted tokens are drawn from exactly the target's
temp/top_p/top_k distribution. The validation gate changes accordingly (KL / mean-match over many
samples at fixed seed, not a byte diff).

## Algorithm (per request, per step)

Draft tokens `x_1..x_K` are sampled from the drafter's per-position distributions `q_1..q_K`; the
verify forward gives the target distributions `p_1..p_{K+1}`. Both `q` and `p` are built by applying
the **request's own** `temperature/top_k/top_p` to the respective logits (so a greedy req, temp→0,
degenerates `p` to a one-hot and this reduces to `verify_greedy`).

```
for i in 1..K:
    r ~ U(0,1)
    if r <= p_i(x_i) / q_i(x_i):        # accept
        continue
    x' ~ normalize(relu(p_i - q_i))     # residual sample; return
    return emit x_1..x_{i-1}, x'   (i tokens, num_accepted = i-1)
# all K accepted:
x_{K+1} ~ p_{K+1}
return emit x_1..x_K, x_{K+1}     (K+1 tokens, num_accepted = K)
```

Correctness holds for **any** `q` (even a deterministic argmax proposal, where `q=onehot(x_i)`,
`q_i(x_i)=1`, accept-prob `= p_i(x_i)`, residual `= renorm(p_i` with `x_i` zeroed`)`). `q` only affects
the **acceptance rate**: sampling the draft from the drafter's softmax (shaped like `p`) accepts far
more than argmax. So drafts must be **sampled from `q` at the req's sampling params**, not argmax'd.

## What changes

1. **Proposer emits `q`.** `propose` still returns draft ids, but when sampled-spec is active the
   proposer also stashes per-req per-position draft logits `[K, vocab]` (e.g. `proposer.draft_logits[uid]`),
   and samples the drafts from those logits under the req's `temp/top_k/top_p` instead of `argmax`.
   - DFlash already computes `logits[1:]` per block position (`dflash.py` `drafts = argmax(logits[1:])`)
     — change the argmax to a sample and keep the logits. MTP/EAGLE3: same (each AR step has logits).
   - n-gram has no distribution → `q = onehot(table token)` (deterministic proposal; still lossless).
2. **`probs_from_logits(logits, temp, top_k, top_p)` helper** — temp-scale, top_k/top_p mask to `-inf`,
   softmax → the sampling distribution. Reused for both `q_i` (drafter) and `p_i` (target). The fused
   HIP sampler returns a *sample*, not the distribution, so this is a small torch helper (fp32).
3. **`verify_sampled(draft, q, p, u) -> AcceptResult`** in `spec/accept.py`, the sampled analogue of
   `verify_greedy` (same return shape: emitted list + num_accepted). Host/torch v1; on-device later.
4. **Route in `_spec_decode_step`** (the branch at the existing `verify_greedy(drafts, target)` call):
   - all-greedy batch → keep the current `verify_greedy` fast path (byte-exact, on-device-eligible).
   - any non-greedy req → build `p_i` per position via `probs_from_logits` with that req's params, read
     the proposer's `q_i`, draw `u`, run `verify_sampled`. The verify **forward is unchanged** (still
     returns `[sum(K+1), vocab]`, still graph-capturable) — only the accept changes.
5. **Spec-loop gate.** Replace `spec_ok = all(is_greedy)` with "all reqs are greedy OR sampled and
   unconstrained" — i.e. run spec for sampled batches too. Per-req params flow through, so a mixed
   greedy+sampled batch just picks the right accept per req.

## Edge cases / interactions

- **TP>1 lockstep (critical).** Sampled draws must be rank-consistent or the ranks accept differently
  → the verify batch desyncs (the fault I just fixed for greedy). Two RNG sources now: the **draft
  sample** (already covered — `_bcast_drafts_tp` broadcasts rank0's drafts) and the **accept draws `u`
  + residual/bonus samples** (NEW). Reuse the existing precedent: plain decode already broadcasts
  rank0's sampled token (`engine.py:800`). So broadcast rank0's accept **outcome** (emitted ids +
  num_accepted) to all ranks — simplest and guarantees lockstep. (`p` is identical on all ranks after
  the lm-head all_gather, so only `u`/`q`/residual RNG needs syncing; broadcasting the outcome subsumes
  all of it.)
- **Constrained (grammar) + sampled: SUPPORTED** (`_verify_sampled_constrained`). Per position the
  grammar bitmask masks the logits, `p_i` is built from the masked logits, and the draft is rejection-
  accepted (a grammar-violating draft has masked `p=0` → always rejected, exactly as the greedy
  masked-argmax rejects it); residual/bonus sample from the masked `p`; the matcher advances on every
  committed token. Same think-gate/reasoning scaffolding as `_verify_greedy_constrained`. RSA applies
  grammar only to the final answer, so this covers the structured RSA answer.
- **DDTree + sampled: SUPPORTED (lossless-commit).** The tree DISCOVERY stays greedy (a heuristic for a
  good draft path) and the lossless LINEAR COMMIT (`_spec_decode_step(ddtree_drafts=True)`) routes the
  discovered path through `verify_sampled` like any other req — so DDTree engages under sampling and is
  distributionally lossless. The tree's *higher-acceptance* benefit under sampling (multi-candidate
  rejection down the tree) needs SpecTr (`ddtree_walk_sampled`), a follow-up; today the tree buys draft
  quality, not extra sampled acceptance.
- **TiDAR fused:** already has its own β/logit-mix greedy verify; sampled TiDAR is out of scope for v1.
- **On-device accept:** `accept_greedy_ondevice` has no sampled analogue yet; sampled-spec v1 uses the
  host accept path (the on-device gate already excludes non-greedy — extend later with a vectorized
  rejection kernel, or wire flashinfer's `chain_speculative_sampling` if `sampler_hip` grows it).
- **Graph capture:** unaffected — the verify forward is the captured part; accept is host/eager (same
  as greedy today). partial-K padding (98ea775) still applies.
- **EOS truncation:** unchanged — applied to the emitted list after accept (same keep-loop).

## Staging

- **v1 (host, DFlash/MTP/EAGLE3/ngram, unconstrained):** proposer samples drafts from `q` + stashes
  logits; `probs_from_logits` + `verify_sampled`; per-req route; TP outcome-broadcast. Distributional
  validation harness (spec-on vs spec-off token histograms/KL at a fixed seed over N samples).
- **v1.1:** on-device sampled accept kernel (or flashinfer chain-spec) for the concurrency lever.
- **v2:** sampled DDTree / sampled TiDAR if the accept-len justifies it.

## Expected payoff (RSA)

Accept-len under sampling is **lower than greedy** (≈ how often a draft-sampled token survives the
`p/q` test; for a well-matched drafter at temp 0.8/top_p 0.95, roughly accept-len 2–4 vs greedy's
~4.67). The win concentrates on the **single-stream aggregation decode** (launch/BW-bound → spec
amortizes the weight read well) and secondarily the batched rollouts. Net: the recreated RXF drafter
finally accelerates the actual RSA workload instead of only the (rarely-used) rsa-off greedy lane.

## Relation to the drafter recreation

The re-distilled RXF drafter (CE-to-token training) already yields a reasonable `q` for rejection
sampling — no training change needed for v1. But for RSA the **capture should be sampled** (temp 0.8 /
top_p 0.95 on RSA-shaped prompts) so `q` is shaped for the sampled contexts it serves; see the capture
tool's header. Sequence: build sampled-spec-verify (this doc) → re-capture sampled on RSA prompts →
re-distill → measure sampled accept-len.
