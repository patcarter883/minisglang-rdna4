# Continuation prompt — the DFlash verify-capture correctness bug

Paste into a fresh session. Supersedes `docs/CONTINUE_dflash_regression_prompt.md`, **which is wrong**
— see §7 before reading it.

---

## 1. The one thing

**The engine captures spec-verify CUDA graphs at widths `[3, 7, 15]`. The NARROW rungs are broken,
and the adaptive width controller selects them ~99% of the time.** So every DFlash acceptance number
this repo has ever taken under the default config — mine, and `CONTINUANCE §11`'s entire matrix —
was measured through a corrupted verify forward.

This is a CORRECTNESS bug, not a perf one: greedy spec decode must be token-identical to greedy plain
decode (linear verify commits only tokens the target produced at its own query rows). It is not.

## 2. Proven, each by a control (do not re-derive)

Qwen3.6-27B, prompt "Write a short technical explanation of how a B-tree index speeds up database
lookups.", `max_tokens 160, temperature 0, top_p 1.0`, plain-decode baseline is DETERMINISTIC
(two identical requests in one boot agree exactly, so any divergence below is real):

| config | first char where spec output differs from plain |
|---|---|
| `MINISGL_SPEC_VERIFY_WIDTH_PIN=3` | **0** — degenerate: emits raw `<\|im_start\|>`, loops `<think>` |
| `MINISGL_SPEC_VERIFY_WIDTH_PIN=7` | **25** — *byte-identical output to the adaptive default* |
| `MINISGL_SPEC_VERIFY_WIDTH_PIN=15` | **192** — correct text, near-lossless |
| `GRAPH_BS=0` (eager verify) | **-1 — BYTE-IDENTICAL to plain** |

Monotone in width, and eager is exactly lossless. Muse-Glimmer is affected too but SILENTLY (coherent
text, diverges at char 142) — which is why this went unnoticed for so long: only a model whose
corruption lands somewhere structural garbles visibly.

Also established by control:

* **Not accept/commit.** `MINISGL_SPEC_FORCE_N0=1` (stage+verify a full block, accept ZERO drafts,
  emit only the bonus — `scheduler.py:4228` says that must be byte-identical to plain) is STILL
  garbled. So the fault is in the verify FORWARD.
* **Not sampling.** Garbles with `MINISGL_SPEC_SAMPLED=0` (greedy argmax verify), banner-confirmed.
* **Not fp8-KV.** `MINISGL_KV_FP8=0` makes it WORSE (Qwen emits a single junk char).
* **Not padding.** At bs=1 `pad_to_captured_width` never pads (the controller has already truncated
  drafts to the chosen width), so pinning changes only WHICH captured graph is replayed.
* **Not the drafter.** Drafter quality cannot affect correctness, only speed.

## 3. Leading candidate — UNPROVEN, test it, do not assume it

`engine/graph.py:448` documents this exact failure mode:

> "A backend that supports verify capture but cannot repoint its per-width statics would replay a
> narrow graph against the WIDEST width's cu_seqlens/kbound/seq_idx — silently wrong output, not an
> error."

`HIPAttnBackend.set_verify_width` (`attention/hip.py:353`) repoints THREE statics: `_vcap_cu_q`,
`_vcap_swa_out_loc`, `_vcap_swa_page_table` — each a per-qlen entry in `_vcap_by_qlen[qlen]`.

But `RDNA4Metadata` (`attention/rdna4.py:70`) documents a FOURTH width-dependent static:

    swa_verify_cache_seqlens[bs]   context_len = Wp + qlen   (bounds the kernel's key reads)

and `_vcap_swa_cache_seqlens` is allocated ONCE (`hip.py:334`), always sliced `[:bs]` (`hip.py:379`)
— it is NOT a per-qlen entry. If it carries the widest rung's value at a narrow replay, the kernel
reads `15 - w` stale ring slots, which reproduces the observed ladder arithmetically:

| rung | stale keys | observed |
|---|---|---|
| 15 | 0 | near-lossless |
| 7 | 8 | corrupt at char 25 |
| 3 | 12 | degenerate at char 0 |

**WHAT IS NOT YET CHECKED:** whether `_fill_swa_verify_static` (`hip.py:436`) refills that buffer
each step from the CURRENT `_vcap_qlen` (in which case the value is right and this candidate is dead)
or from a baked widest value (in which case this is the bug). Read that first — it is a five-minute
question and it decides the whole thing.

Second candidate if that acquits: the residual divergence at width 15 (192 on Qwen, 142 on Muse,
width-independent) is a SEPARATE, milder fault in the captured verify body.

## 4. Repro (~4 min per leg)

```bash
git worktree add --detach /home/pat/code/minisgl-rdna4-cap <this-branch>
cd /home/pat/code/minisgl-rdna4-cap
# plain baseline (run the SAME request twice — this model is deterministic, so floor must be -1)
MODEL=qwen27b SPEC=none TP=2 CONC=2 MEM_RATIO=0.90 \
  gpu-lease -n 2 --detach -- docker compose --profile serve up -d
# then spec, pinning each rung in turn:
MODEL=qwen27b SPEC=dflash SPEC_K=15 TP=2 CONC=2 MEM_RATIO=0.90 MINISGL_DFLASH_QUANT=fp8 \
  MINISGL_SPEC_SAMPLED=0 MINISGL_SPEC_VERIFY_WIDTH_PIN=3 ...
```

Ready-made harnesses in the session scratchpad pattern (rewrite, don't hunt for them): a losslessness
gate that runs plain twice + spec once and prints first-diff, and a per-rung driver.

**Mitigation available today, no code change:** `MINISGL_SPEC_VERIFY_WIDTH_PIN=15` takes served
output from garbage to near-lossless. Not a fix (192 is still a divergence), but strictly better than
the default for anyone serving DFlash right now.

## 5. Measurement protocol — the thing this session got wrong

**RUN THE LOSSLESSNESS GATE BEFORE QUOTING ANY ACCEPTANCE NUMBER.** One boot. Greedy spec must equal
greedy plain. An entire session was spent on vendor benchmarks, seed-tail sweeps, precision ladders
and sharding validation before anyone checked the invariant — and the invariant was violated the
whole time, so those numbers were scoring the drafter against a corrupted target.

**Determinism is per-model. Measure the floor, never assume it.**

| model | plain greedy, two identical requests in one boot |
|---|---|
| Muse-Glimmer | bit-identical (floor -1) |
| Qwen3.6-27B | bit-identical (floor -1) |
| **Laguna-XS** | **diverges at char 65** |

Laguna's non-determinism is why its acceptance carries a **16.3% same-config band** (five replicates
of one config: 1.142 / 1.206 / 1.263 / 1.309 / 1.328). Muse's synthetic legs reproduce EXACTLY, so
differences there are real at any size. Applying the wrong band to the wrong model invalidates the
comparison in either direction.

**Assert provenance per leg, from `docker inspect`, not from what you exported.** Five separate
failures this session, every one silently resolving to a default:

* labelled Prometheus series (`minisgl_spec_*{model_name=...}`) broke a delta parser — every metric read 0;
* `MINISGL_DFLASH_SEED_TAIL` was not forwarded by compose — tail stayed 64;
* a helper script re-assigned `WT` after it was set — the wrong worktree booted;
* `serve.sh`'s `: "${MINISGL_DFLASH_QUANT:=nvfp4}"` substitutes on EMPTY as well as unset — asking for
  bf16 silently served nvfp4 (caught only by the reserve being 1.37 GiB where bf16 needs 3.16);
* `MINISGL_SPEC_SAMPLED` was forwarded only by the `cam` service — greedy verify was UNREACHABLE on
  the serve path, which is why the invariant had never been testable.

`docker exec sh -lc` is NOT a provenance check — a login shell re-sources the profile and shows a
cleaned env. Use `docker inspect --format '{{range .Config.Env}}...'`, plus the engine's own banner.

**A serve that dies looks alive.** The scheduler subprocess can die with the container still `Up`,
port 1919 listening, `RestartCount` 0. Gate readiness on the container LOG
(`AssertionError|Traceback|out of memory|Not enough memory for KV cache`), not on the port, not on
the container existing. Also `docker compose down` may leave it — `docker rm -f` if so.

## 6. Dead hypotheses — DO NOT RE-RUN

Each was measured and killed this session:

| hypothesis | verdict |
|---|---|
| DFlash prompt-seed tail (64 -> 256/528) | **null**: interleaved 3x paired ratios 1.029 / 1.007 |
| fp8 drafter beats nvfp4 on real traffic | **no**: nvfp4 1.637 vs fp8 1.564 (both exact) |
| capture bug depresses ACCEPTANCE | **no**: eager 1.545 vs captured 1.682 on Muse |
| fp8-KV causes the capture corruption | **no**: bf16 KV is worse |
| `temperature:0` isn't greedy so the doc's control was void | **no**: measured bit-identical |
| Muse never sees the prompt (`supports_prefill_seed=False`) | **no**: `prefill_aux_tail=0` means seed-ALL |
| capture-layer indexing | **no**: HF `target_layer_ids` = "residual AFTER layer L", confirmed against
  Meta's GGUF conversion, which reads input-to-layer-(L+1) via `llama_get_embeddings_layer_inp` |

## 7. Why the previous continuation prompt is wrong

`CONTINUE_dflash_regression_prompt.md` claims a "repo-wide DFlash acceptance regression" and sends
you to bisect `spec/dflash.py`'s propose path. Do not. Its headline compares a **prose, ~25-token
prompt, 300-token, sampled** measurement (1.08) against §11's **code, 720-token, greedy** baseline
(4.508). Measured this session, with everything else held fixed:

* prompt class is worth **~2.5x** (Laguna prose 0.93-1.07 vs code 1.90-2.44, one boot);
* generation length is worth **+75%** (ONE prompt, only `max_tokens`: 150 -> 1.811, 1200 -> 3.178);
* the same-config noise band on Laguna is **16.3%**.

Its six "eliminated hypotheses" were each a single measurement against that band. Its verify-width
observation (`15:0(0%)`) is read as a symptom of low acceptance; it is actually **the controller
sitting on the broken rungs**, which is the real bug — the one thing in that document pointing at the
truth, interpreted backwards.

## 8. Also true, and useful

* **Vendor reference exists and runs on this box.** Meta ships a GGUF DFlash drafter
  (`meta-models/Muse-Glimmer-30B-GGUF`: `muse-glimmer-30B-kquant-17gb.gguf` + `dflash-kquant.gguf`),
  and llama.cpp b1309 (lemonade `rocm-nightly` pin) runs it on gfx1201 with `--spec-type draft-dflash`.
  Measured: greedy prose 1.941, greedy code 2.448, sampled prose 1.326, sampled code 2.093. **We sit
  at 71-90% of that on identical weights** — the standing unexplained gap, and plausibly just this
  bug, since every leg of that comparison ran through the broken rungs. RE-TAKE IT AFTER THE FIX.
* **Real-traffic fixtures**: `/home/pat/fixtures/minisgl-real-traffic/` — 12 real Hermes requests
  (3.6k-42.7k prompt tokens) + manifest. Production traffic through this serve has a MEDIAN prompt of
  **18,390 tokens** (378 sessions; only 5 of 52 request dumps under 1k) and emits only **40-180
  tokens** per turn. Every synthetic benchmark in this repo uses 25-99 token prompts. Use the fixtures.
* **Drafter sharding landed** (`models/draft_linear.py`): one shared core for DFlash / CCA / EAGLE3,
  replacing two diverged `_PlainLinear` copies; TP col/row sharding with self-disabling fallback.
  Validated bit-exactly (three Laguna legs byte-identical to replicated). Makes fp8 Muse bootable
  (2.48 -> 1.73 GiB); bf16 still does not fit (3.16 GiB, dies sizing the KV pool).
* **`_load_draft_weights` staged the WHOLE checkpoint on every rank** when the drafter was
  unquantized — fixed; published "bf16 drafter cannot fit" figures were inflated by that peak.

## 9. Ranked next actions

1. **Read `_fill_swa_verify_static` (`hip.py:436`)** and settle §3's candidate. Five minutes.
2. **Fix the per-width statics**, then re-run the §2 ladder. Success = `-1` at every rung.
3. **Chase the width-15 residual** (192/142) — a separate, milder fault.
4. **Re-take everything measured through the bug**: the vendor comparison (§8) and `CONTINUANCE §11`.
5. **Laguna greedy non-determinism** (§5). Excluded already: radix prefix cache, async MoE all-reduce,
   custom AG/AR collectives — all still non-deterministic with each disabled. Remaining suspects: MoE
   split-K atomic accumulation, SWA ring slot reuse across requests.
6. **Only then** the token-level drafter diff vs llama.cpp. Doing it before the fix chases a number
   produced by a known fault.

## 10. The meta-lesson, because it is the reason this took so long

DFlash has come back "borderline net-negative" across many sessions, and each session produced a
DIFFERENT locally-plausible cause: kernel occupancy, drafter quality, prompt class, context
starvation, precision. Every one was individually reasonable and individually measured.

**A stable anomaly with a rotating explanation means the explanations are wrong.** And it contradicted
a strong external prior — DFlash makes things faster for everyone else, which is why the vendor ships
the drafter at all. That prior was evidence about our implementation, not something to explain away.
The 71-90%-of-reference gap was the signal; it did not need explaining, it needed finding.

Test invariants before measuring quantities.
