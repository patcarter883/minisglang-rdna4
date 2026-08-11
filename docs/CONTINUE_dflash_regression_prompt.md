# Continuation prompt — DFlash acceptance regression (repo-wide)

Paste this into a fresh session to resume.

---

We are chasing a **repo-wide DFlash speculative-decoding regression**. DFlash currently proposes so
badly that it is a net loss on every model, and this is NOT specific to any one checkpoint. Read
**`docs/MUSE_GLIMMER_PORT.md` §10** for how it surfaced, and
**`docs/CONTINUANCE_laguna_spec_and_decode_perf.md` §11** — the acceptance audit that established what
Laguna's accept-len is *supposed* to be. Recall memories `spec-must-be-measured-sampled`,
`ab-baseline-must-be-old-code-not-emulated`, `ab-harness-must-assert-provenance`,
`diff-engaged-ledgers-on-every-serve-ab`.

## The finding

`MODEL=laguna SPEC=dflash TP=2` — the repo's own validated pair — measures **mean accept-len 1.08–1.13
at K=15**, sampled. Documented behaviour for the same pair (§11, measured, build held constant):

| prompt class | documented accept-len |
|---|---|
| real code, 720 tok | **4.508** |
| real code, 1600 tok | **6.150** |
| prose | **2.912** |
| repetitive | ~9.5 |
| recorded run, 300 reqs | **8.09** |

**1.08 is below even the prose floor.** §11 explicitly concluded "there is no regression to bisect" for
an earlier 8.1→2.58 scare (different prompt classes, both reproduced). This is a different, real one.

Muse-Glimmer measures the same 1.11–1.29, with an identical verify-width signature — it is the same
bug, not a bad port.

## Exact repro (~4 min)

```bash
git worktree add --detach /home/pat/code/minisgl-rdna4-base d506d639   # PRE-regression-hunt code
cd /home/pat/code/minisgl-rdna4-base
MODEL=laguna SPEC=dflash TP=2 gpu-lease -n 2 --detach -- docker compose --profile serve up -d
# wait for VRAM to climb (NOT the port — see gotchas), then:
curl -s -X POST http://localhost:1919/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "model":"laguna","messages":[{"role":"user","content":"Write a short technical explanation of how a B-tree index speeds up database lookups."}],
 "max_tokens":300,"temperature":0.8,"top_p":0.95}'
docker logs lease-gpu0-1-serve 2>&1 | grep "\[spec\] mean accept-len"
```

Expect: `mean accept-len=1.08 … verify-width[0:1(0%) 3:137(68%) 7:62(31%) 15:0(0%)]`.

## Established — do NOT re-derive

**It is not the Muse-Glimmer port.** `d506d639` predates all of that work and reproduces byte-identical
numbers (1.13→1.08, same widths, 37.84/71.59 tok/s). Provenance is settled; do not re-litigate it.

Six hypotheses eliminated, each with everything else held fixed:

| suspect | how tested | result |
|---|---|---|
| Muse port / new code | Laguna on `d506d639` | identical — not it |
| Capture-layer indexing | `MINISGL_DFLASH_CAPTURE_LAYERS=2,14,26,38,50` vs config's `1,13,25,37,49` | 1.12 vs 1.11 — not it |
| Verify mode / draft-dist scale | `temperature:0` (greedy verify is argmax, scale-invariant) | ~1.2 — not it |
| Adaptive verify width | `MINISGL_SPEC_VERIFY_WIDTH_PIN=15` (100% of steps at 15) | 1.15 — not it |
| Persistent prefix-KV | `MINISGL_DFLASH_PERSIST_KV=0` (recompute path) | 1.24 — not it |
| Aux feature mode | `MINISGL_EAGLE3_AUX_MODE=r` vs default `xr` | 0.89, *worse* — default is right |

**The width pin is the most informative result.** With 15 drafts genuinely offered, only ~1.15 are
accepted. The controller narrowing to 3/7 is a SYMPTOM of low acceptance, not its cause — so do not
start on `spec/width.py`.

## Remaining surface

The propose path itself, in `python/minisgl/spec/dflash.py` and `python/minisgl/models/dflash.py`:

- the block-diffusion denoise (`denoise` / `denoise_cached`, `models/dflash.py:~474/~508`)
- the `fc` + hidden-norm aux fusion (`fuse_aux`, `models/dflash.py:~418`)
- the anchor/mask block construction and `block_pos` RoPE phases (`_propose_eager`,
  `spec/dflash.py:~957-1097`, esp. the noise block `[anchor, mask, mask, …]` and `base_pos`)
- the target-context KV prefix projection (`project_ctx` / `attend_block`)
- drafter weight loading (`_load_laguna_weights`, `_load_draft_weights`)

## Recommended first move

Write a **drafter-only unit test that bypasses serving entirely**: feed known aux hidden states + a
known prefix, and compare the drafter's proposed tokens against the target's actual continuation for
the same context. This isolates propose from verify, the scheduler, the KV pool and graph capture in
one shot.

- If the drafter proposes WELL in isolation → the bug is in how the scheduler feeds it (aux
  accumulation `scheduler.py:~4518-4548`, `ctx.aux_hidden` per-uid slicing, prefix positions).
- If it proposes BADLY → the bug is in the drafter build or its weights.

`tools/dflash_window_parity.py` is precedent for the shape of such a harness. A reference for what
"good" looks like: the drafter should reproduce the target's own greedy continuation for several
tokens after a committed prefix.

Second move, if that is inconclusive: bisect. §11's audit ran on a known-good build; find that commit
and walk forward. Accept-len is a cheap, high-signal bisect metric (one boot + one request per step),
but **hold the prompt class constant** — accept-len legitimately spans 2.6→9.5 across prompt classes on
this drafter, which is exactly what made the earlier scare a false alarm.

## Gotchas that cost time in the last session

- **A dead serve looks alive.** When the scheduler subprocess dies, the container stays `Up` and port
  1919 keeps listening (the frontend survives). `docker ps` + an open port prove NOTHING. Poll
  **VRAM** (`rocm-smi --showmemuse`); 0% means the model never loaded. Same trap as CLAUDE.md's
  "a card can be leased and still be DEAD".
- **Env knobs silently do nothing unless compose forwards them.** `MINISGL_DRAFT_RESERVE{,_MARGIN}_GB`,
  `MINISGL_DFLASH_CAPTURE_LAYERS` are now forwarded; **`MINISGL_EAGLE3_AUX_MODE` is still NOT** (it was
  only patched in a throwaway worktree). Setting an unforwarded var on the host looks like it worked.
  This is plausibly part of why the regression went unnoticed — several of these levers were untestable
  through the normal launch path. Landing that passthrough is worth doing early.
- **Compose substitutes the EMPTY STRING for an unset var**, so any new passthrough needs its consumer
  to treat `""` as unset (`(os.environ.get(X) or "").strip()`), not `is not None`.
- **`gpu-lease -n 1 -- docker run … -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES`** expands in the OUTER
  shell where it is unset → "No CUDA GPUs are available". Wrap the whole docker invocation in a script
  or `bash -c` so it expands INSIDE the lease.
- Restore the user's serve when done: `MODEL=muse TP=2 CONC=4 MEM_RATIO=0.85` (plain decode, no spec).

## Muse-Glimmer specifics (only if you touch that path)

`SPEC=dflash` is wired for `MODEL=muse` but should stay OFF. Its drafter only fits at 4-bit
(`MINISGL_DFLASH_QUANT=nvfp4`, ~2.09 GiB reserve; bf16 and fp8 cannot size a KV pool at all), and it
needs `CONC=2 MEM_RATIO=0.95 MINISGL_DRAFT_RESERVE_MARGIN_GB=0.30` to boot. There is one **known
unfixed defect** there: the drafter's head calls `lm_head.logits_all_rows` directly, bypassing
`MuseGlimmerForConditionalGeneration._logits`, so draft logits carry neither `output_multiplier`
(1/sqrt(26)) nor the tanh softcap — the draft distribution is ~5x too sharp. Greedy verify is immune
(both transforms are monotone), which is why it did not move the greedy test, but **sampled rejection
verify and DDTree both compare distributions and are wrong until it is fixed**. Fix it before trusting
any *sampled* Muse acceptance number — but note it cannot explain the Laguna regression, which is the
actual target here.
