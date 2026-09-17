# DFlash acceptance retake after the GDN ring-gate fix — and what did NOT need retaking

Companion to `docs/CONTINUE_dflash_capture_bug.md` (the bug + fix) and
`docs/measurements/DFLASH_ACCEPTANCE_REAL_TRAFFIC.md` (the vendor comparison this corrects).

Fix under test: `5538fa09`, merged as `a86a2600`.

---

## 1. Scope — which numbers the fix could possibly have moved

The bug required **both** conditions, and that is narrower than the previous doc assumed:

1. the target is a **GDN hybrid** (`GDNVerifyGraphCapture` is only constructed when
   `self._gdn_state is not None`, `engine/graph.py:477`), **and**
2. the captured ladder **straddles `REPLAY_RING_LEN = 8`** — some `qlen = w+1 <= 8` while
   `Qmax = max(w)+1 > 8`. Only then does the capturer answer "materialising" at `Qmax` while the
   layer answers "ring" at the narrow rungs.

Ladder arithmetic (`spec/width.py verify_width_ladder`, pure host code):

| K | ladder | qlens | straddles L=8? |
|---|---|---|---|
| 2 / 3 | [2] / [3] | [3] / [4] | no |
| 4 (MTP) | [2,3,4] | [3,4,5] | no |
| 6 (EAGLE3) | [3,4,6] | [4,5,7] | no |
| **8** | [2,4,8] | [3,5,9] | **YES** |
| **15 / 16 (DFlash)** | [3,7,15] | [4,8,16] | **YES** |

So the affected surface is exactly **DFlash (K≥8) on a GDN-hybrid target**.

### Architecture audit — checked, not assumed

| model | `layer_types` | GDN? | affected |
|---|---|---|---|
| Qwen3.6-27B-AWQ-INT4 | `linear_attention` + `full_attention`, 64 layers | **yes** (48 GDN) | **YES** |
| Qwen3.6-35B-A3B-AWQ | `linear_attention` + `full_attention`, 40 layers | **yes** | **YES** |
| Muse-Glimmer-30B | `sliding_attention` ×39 + `full_attention` ×13 | no | no |
| Laguna-XS-2.1 | `sliding_attention` + `full_attention`, window 512 | no | no |

### Therefore these did NOT need retaking — and the previous doc was wrong to say they did

* **The llama.cpp vendor comparison** (`DFLASH_ACCEPTANCE_REAL_TRAFFIC.md` §2: vendor greedy prose
  1.941 / greedy code 2.448 / sampled prose 1.326 / sampled code 2.093, "we sit at 71–90%") ran on
  **Muse-Glimmer**, which has no GDN layer at all. `GDNVerifyGraphCapture` is never constructed, so
  `REPLAY_RING_LEN` never runs. **The 71–90% gap is NOT this bug** and remains unexplained.
  `CONTINUE_dflash_capture_bug.md`'s "plausibly just this bug, since every leg of that comparison
  ran through the broken rungs" was wrong; it is corrected there.
* **`CONTINUANCE §11`** (`CONTINUANCE_laguna_spec_and_decode_perf.md:668`, the 4.508 matrix) ran on
  **Laguna** — no GDN — **and eager** (`--cuda-graph-max-bs 0`, §11.2), i.e. no verify graph was
  captured at all. Unaffected twice over.
* **MTP (K=4)** and **EAGLE3 (K=6)** on any target, including the production `qwen35b-awq` MTP
  config: their whole ladder fits under the ring, so capturer and layer always agreed.

This is worth stating plainly because the previous framing implied a repo-wide re-measurement. It is
not: it is the Qwen DFlash pairs.

## 2. Basis — say which number you mean

`scheduler.py:4668` warns that two figures share the name "mean accept-len" and differ by exactly 1.
Every figure here is computed from the raw Prometheus counters, not the derived gauge:

```
instances = emitted - accepted        # every request emits exactly one bonus token
accepted drafts / verify = accepted / instances   <- the serve log's "mean accept-len"; the VENDOR basis
committed tokens / verify = emitted / instances   <- 1 + the above; decides whether spec PAYS vs 1.0
accept rate = accepted / drafted
```

The vendor harness used the identical basis (`ac/st`, `st = predicted_n - accepted_n`), so
drafts/verify here is directly comparable to 1.941 / 2.448 in basis — though not in model.

Counters are read from `/metrics` with labels stripped, **not** from the serve log's cumulative
line: that line only prints every 100 verify steps, and a pre-fix arm degenerates and finishes early,
so it can legitimately never reach the threshold. Where both exist they agree (leg 1: metrics 0.236
vs log 0.26).

## 3. Protocol

Qwen3.6-27B-AWQ-INT4, TP=2, `SPEC=dflash SPEC_K=15` (ladder `[3,7,15]`), `MINISGL_DFLASH_QUANT=fp8`,
CONC=2, MEM_RATIO=0.90, adaptive width (no pin — the shipping default). 4 prompts per class,
`max_tokens=400`, greedy `temperature=0`, sampled `temperature=0.8 top_p=1.0`.

**A/B is old CODE, not emulation**: the `pre` arm boots from a worktree at `e9bf7dcc`
(`/home/pat/code/minisgl-rdna4-dfpre`), the `post` arm from `5538fa09`. Provenance per leg is
asserted from `docker inspect`, and the `post` arm is additionally proven live by the engine's own
`[gdn] spec-verify commit per width:` banner, which the `pre` arm cannot print.

Note this protocol differs from the vendor legs (which used 1 prompt/class at 300 tokens). That is
deliberate — there is no vendor baseline for Qwen, so the value here is the pre/post delta, and more
prompts make that delta more robust than n=1 would.

## 4. Results

Raw per-leg counters in the session scratchpad (`acc/<leg>/metrics.json`, `acc_summary.json`).

| leg | arm | drafts/verify | committed/verify | accept rate | (req,verify) | tokens generated |
|---|---|---|---|---|---|---|
| prose greedy | pre | 0.236 | 1.236 | 0.068 | 276 | 341 |
| prose greedy | **post** | **1.918** | **2.918** | **0.279** | 547 | 1600 |
| code greedy | pre | 0.253 | 1.253 | 0.071 | 225 | 282 |
| code greedy | **post** | **3.314** | **4.314** | **0.441** | 370 | 1600 |
| code sampled | pre | 0.174 | 1.174 | 0.052 | 379 | 445 |
| code sampled | **post** | **3.302** | **4.302** | **0.481** | 371 | 1600 |

| class | drafts/verify | committed/verify |
|---|---|---|
| prose greedy | 0.236 → 1.918 (**8.1×**) | 1.236 → 2.918 (2.36×) |
| code greedy | 0.253 → 3.314 (**13.1×**) | 1.253 → 4.314 (3.44×) |
| code sampled | 0.174 → 3.302 (**19.0×**) | 1.174 → 4.302 (3.66×) |

### Reading these

* **Pre-fix, DFlash was almost certainly net-NEGATIVE here.** committed/verify of 1.17–1.25 means a
  16-row verify forward bought ~0.2 extra tokens per step over plain decode's 1.0. Post-fix it buys
  2.9–4.3. (Throughput was NOT measured — this is acceptance only. The tok/s claim needs its own run.)
* **The `tokens generated` column is itself evidence.** Every leg asked for 4×400; the pre arm
  produced 282–445 because the corrupted target degenerated and hit a stop early. The post arm
  produces the full 1600 every time.
* **Prompt-class structure is restored.** The known ~2.5× code-over-prose effect is visible post-fix
  (3.314 vs 1.918) and absent pre-fix (0.253 vs 0.236) — with the recurrent state zeroed every step,
  the target had no idea what it was reading, so class stopped mattering. A corrupted target flattens
  exactly the signal a healthy drafter exploits.
* **Sampled ≈ greedy post-fix on code** (3.302 vs 3.314), which is unusual — sampled verify normally
  costs acceptance. Worth a look, but it is a second-order question next to the 13×.
* The controller now sits on rung **7** (88–98% of steps) and reaches rung 15 on code (10%), versus
  pre-fix where it sat on rung **3** (80–87%) — the broken rung. It was narrowing because acceptance
  looked terrible, and acceptance looked terrible because the narrow rungs were broken: a feedback
  loop that pinned the serve to its worst configuration.

### Basis comparison, stated carefully

drafts/verify here shares the vendor harness's basis, so 1.918 (prose greedy) / 3.314 (code greedy)
sit alongside llama.cpp's 1.941 / 2.448 **in units** — but those are **Muse-Glimmer on llama.cpp**
and these are **Qwen3.6-27B on minisgl**. Different model, different stack. This is NOT a vendor
comparison and must not be quoted as one. The actual vendor gap is unchanged and still open (§1).

## 5. Still open

The residual losslessness violation (greedy spec ≠ greedy plain: rungs 3/15 at char 192, rung 7 at
char 18) is **not** fixed by this and is unrelated to acceptance — see
`CONTINUE_dflash_capture_bug.md` §3 for the ring-bit-exactness hypothesis and the test to run.
