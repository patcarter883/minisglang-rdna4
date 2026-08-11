# DFlash spec-verify: the capture bug is FOUND AND FIXED — plus the residual that is not

Supersedes the previous version of this file (the "narrow rungs are broken" framing), **which was
wrong about the mechanism and wrong about which fault matters**. See §5 for exactly what it got
backwards and why. Also supersedes `CONTINUE_dflash_regression_prompt.md`.

---

## 1. What was actually wrong

**`GDNVerifyGraphCapture` asked a WIDTH-DEPENDENT question once, at the widest width, and then
answered it that way for every width.**

`gdn/graph_capture.py:_replay_verify_available(qmax)` decides whether the GDN layers will take the
ReplaySSM **ring** verify (append the draft window to the ring, commit by cursor rewind) or the
**materialising** verify (stage a per-token SSM state, commit by gathering the accepted index). The
only width-dependent term is `qlen <= REPLAY_RING_LEN`, and **`REPLAY_RING_LEN = 8`**.

The capturer asked it once at `Qmax`. `gdn/layer.py:513` (`use_replay`) re-asks it per replay against
`md.verify_max_qlen` — the width actually being run. With the shipping ladder `[3, 7, 15]`:

| rung | qlen | capturer asked at Qmax=16 | layer actually did | agreed? |
|---|---|---|---|---|
| 3 | 4 | materialising | **ring** | **NO** |
| 7 | 8 | materialising | **ring** | **NO** |
| 15 | 16 | materialising | materialising | yes |

Because `_ssm` was allocated, `_metadata` published a non-empty `ssm_scratch` at *every* width, and
`scheduler.py:4604` (`if qlen and not md.ssm_scratch`) reads that non-emptiness as "commit by
materialised state". So on rungs 3 and 7 the scheduler **installed the zeros the capturer had
allocated as the SSM state of every live slot, on every verify step, for all 48 GDN layers**, and
then `reset_ring` discarded the ring that actually held the truth.

Total destruction of the recurrent state, once per step. That is why the corruption was catastrophic
rather than subtle, and why it was monotone in width.

**The fix** (`gdn/graph_capture.py`, 3 functional lines): ask per captured width
(`self._replay_by_q`), allocate the scratch if *any* width needs it, and publish `ssm_scratch` only
for the widths that actually materialise. The serve log now states the mapping:

```
[gdn] spec-verify commit per width: qlen=4->ring-rollback, qlen=8->ring-rollback,
      qlen=16->materialised (REPLAY_RING_LEN gate; ssm scratch allocated)
```

## 2. Measured — Qwen3.6-27B-AWQ-INT4, TP=2, greedy (`MINISGL_SPEC_SAMPLED=0`), fp8 drafter

Prompt: "Write a short technical explanation of how a B-tree index speeds up database lookups.",
`max_tokens 160, temperature 0`. Reference = `SPEC=none`, 649 chars, **floor `-1`** (two identical
requests in one boot agree exactly, re-verified this session).

> Compare the **full** stream: this is a thinking model and the API splits it into
> `reasoning_content` **then** `content`. Diffing `content` alone compares post-`</think>` tails and
> misses every early divergence. The previous session's numbers were taken correctly; this is a trap
> for the next harness.

| leg | code | captured widths | rung used | first diff | accept-len | output |
|---|---|---|---|---|---|---|
| `pre_adapt` | pre | [3,7,15] | adaptive | 25 | (never reached 100 steps) | **91 chars, degenerate, truncated** |
| `fix_adapt` | **post** | [3,7,15] | adaptive → 7 (97%) | 18 | **1.82** | coherent |
| `k15pin3` | pre | [3,7,15] | pin 3 | **0** | **0.26** | `<\|im_start\|>` then `<think>` loop |
| `fix_pin3` | **post** | [3,7,15] | pin 3 | 192 | **1.48** | coherent |
| `fix_pin7` | **post** | [3,7,15] | pin 7 | 18 | 1.82 | coherent |
| `fix_pin15` | **post** | [3,7,15] | pin 15 | 192 | 2.02 | coherent |
| `k3` | pre | **[3]** | 3 | 192 | 1.48 | coherent |

**`k3` is the experiment that broke the old story open, and it needs no code change**: `SPEC_K=3`
makes `verify_width_ladder` return a single-element ladder `[3]`. Same rung, same drafter, same
quantization as `k15pin3` — the *only* difference is whether widths 7 and 15 were also captured.
`0.26 → 1.48` acceptance and garbage → coherent. So the narrow rung was never broken; capturing a
**mixed** ladder was.

## 3. What is still NOT fixed — the residual, better characterized

Greedy spec is still not byte-identical to greedy plain. The residual is **numeric, not structural**:
at the divergence the two streams differ by one token at a coherent branch point (`\n` vs ` `) and
both continue as valid English. It is **width-dependent in magnitude**:

| rung | qlen | vs ring L=8 | first diff |
|---|---|---|---|
| 3 | 4 | window is half the ring | 192 |
| **7** | **8** | **window exactly fills the ring** | **18** |
| 15 | 16 | ring not used (materialising) | 192 |

Leading hypothesis, **UNPROVEN — test it, do not assume it**: the ReplaySSM ring verify is not
bit-exact, while the materialising verify is (`gdn/layer.py:534` calls it "the RECURRENT
(non-WMMA) oracle — bit-stable, the whole point of verify"). The kernel flushes **once at window
start with `reserve=q_len`** (`gdn_kernels.hip:879`, FLUSH POLICY), so at `qlen=8` that flush must
fold the *entire* ring into the checkpoint every step, at `qlen=4` only half, and at `qlen=16` the
ring is bypassed. More folding → more reassociation → more drift. That ordering matches the table.

This also explains why the old doc found eager verify byte-identical: at K=15 eager runs `qlen=16`,
which takes the **materialising** path. **Eager was never exercising the ring at all.** So "eager is
lossless" is not evidence that the ring is.

Note the consequence for the shipping default: post-fix the controller settles on **rung 7**, which
is the rung with the *worst* residual. If the hypothesis holds, the cheap mitigation is to make the
ring gate strict (`qlen < L`, pushing qlen=8 to the materialising path) or to keep the ladder off
the exactly-full width — both are one-liners, both cost VRAM/speed, and **neither should be done
before the hypothesis is measured.**

## 4. How to measure (the protocol still stands, with one addition)

Everything in the old §5 remains right and was re-validated: gate readiness on the container **log**
not the port; assert provenance from `docker inspect` and hard-fail on a key compose does not forward
at all; measure the determinism floor per model rather than assuming it. The harness used here
(`leg.sh` + `diff.py`, session scratchpad) does all of that; rewrite it, don't hunt for it.

**The addition: compare `reasoning_content + content`, not `content`.**

## 5. What the previous version of this doc got wrong

* **"The NARROW rungs are broken."** No — a narrow rung captured *alone* is fine (`k3`). The mixed
  ladder is what breaks them.
* **Its §3 leading candidate (`_vcap_swa_cache_seqlens`, per-width attention statics) is dead
  twice over.** `_fill_swa_verify_static` refills that buffer every replay from the *live*
  `_vcap_qlen`, so the value was always right — **and Qwen3.6-27B has no SWA at all** (it is a
  48-layer GDN hybrid). The bug was never in the attention backend.
* **"The width-15 residual (192) is a separate, milder fault."** Right that it is separate, wrong
  that it is secondary — post-fix it is the *only* remaining losslessness violation, and rung 7's
  version of it is the one the shipping config actually hits.
* The old §2 ladder (0 / 25 / 192) is reproduced here exactly. The observations were sound; the
  mechanism inferred from them was not.

## 6. Ranked next actions

1. **Test §3's hypothesis**: pin each rung and diff the ring verify against the materialising verify
   on identical state. If the ring is not bit-exact, that is a kernel-level correctness question for
   `rdna4-hip-kernels/gdn`, not an engine one.
2. **Re-take everything measured through the bug** — every DFlash acceptance number taken under the
   default ladder on a GDN hybrid was scored against a target whose recurrent state was being zeroed
   every step. That includes the vendor comparison (we sat at 71–90% of llama.cpp on identical
   weights) and `CONTINUANCE §11`. The 5.7× rung-3 acceptance recovery says the gap was largely this.
3. **Check the other GDN spec paths for the same width-scoping error** — `can_use_ddtree_verify` and
   the fused-TiDAR capture family were not audited here.
4. Laguna greedy non-determinism (old §5) — untouched, still open.

## 7. The meta-lesson, restated because it changed

The old doc's lesson was "a stable anomaly with a rotating explanation means the explanations are
wrong". That held. The sharper version:

**A capability gate that is width/shape-dependent must be asked at the shape being run, never once at
the widest.** The capturer and the layer asked the same question at two different widths and silently
disagreed; nothing asserted they matched, and the disagreement surfaced as "the drafter is bad".
Where two code paths must agree on a dispatch decision, make one of them *state* the decision (the
new banner) so a diff is possible at all.
