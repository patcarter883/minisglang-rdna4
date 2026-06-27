# SPEC_DECODE.md — speculative decoding for minisgl (Triton-free, native-HIP)

Status: **MVP working and GPU-validated** (n-gram / prompt-lookup, topk=1, greedy, MHA, eager,
synchronous loop). Enable with `--spec-algorithm ngram` (see args below). Validated on Qwen3-0.6B
(gfx1201): output is coherent, drafts are accepted (~11% on repetitive workloads, emitted/step up
to 1.25), and multi-token acceptance is **bit-identical to sequential decode through the verify
kernel** (`tools/spec_lossless.sh` → PASS on all prompts) — i.e. the accept/commit/KV-rollback is
provably lossless. Residual divergence from a plain-decode baseline is only the inherent
extend-kernel vs decode-kernel fp difference, not a spec defect. This doc is both the upstream
porting reference and the concrete minisgl plan; read it before extending.

Harnesses: `tools/spec_smoke.sh` (coherence + baseline diff), `tools/spec_lossless.sh` (rigorous
accept/rollback equivalence), `tests/spec_core_test.py` (CPU unit tests for proposer + acceptance).
Diagnostics (env-gated, inert by default): `MINISGL_SPEC_DEBUG=1` logs running acceptance stats,
`=2`/`=3` per-step traces; `MINISGL_SPEC_FORCE_N0=1` stages+verifies drafts but commits only the
bonus (1 token/step) — isolates verify-forward vs accept-path bugs.

---

## 1. Why this design (the two findings that set the whole shape)

**Finding A — verify needs *no new kernel* on MHA models.** Upstream SGLang's EAGLE verify does
not use the decode kernel; it reuses the *ragged paged-prefill ("extend") kernel with a custom
mask*. In minisgl, `TritonRDNA4Backend.prepare_metadata` (python/minisgl/attention/triton_rdna4.py)
already builds exactly the right metadata when a req's `extend_len > 1` with a non-empty cache: it
takes the "extend prefill with partial cache hit" branch, builds a ragged `cu_seqlens_q`, and
dispatches to `_hip_prefill_paged` → `attn_prefill_paged.flash_prefill_paged` with `causal=1`. That
**is** the topk=1 (linear-chain) verify forward: K+1 query tokens per sequence attending the paged
KV prefix under a causal mask. So the MVP verify is mechanically a "decode-like batch where each req
has `extend_len = num_draft + 1`." Zero kernel changes for MHA (Qwen2/Qwen3 dense).

**Finding B — spec-decode fights overlap scheduling, so run it synchronously first.** The scheduler's
zero-sync overlap path (`Scheduler._forward`, python/minisgl/scheduler/scheduler.py) reads the next
batch's inputs straight from a GPU `token_pool` the previous step wrote — never syncing to host. But
speculative acceptance is a **data-dependent count** (how many drafts matched) that the host must
know to lay out the next batch's positions and KV. That is inherently a sync point. Upstream solves
it with `FutureMap` overlap machinery — premature for an MVP. We run spec-decode in the synchronous
`normal_loop` (gated like `ENV.DISABLE_OVERLAP_SCHEDULING`), exactly the eager-first discipline used
to bring up GLM. Overlap/`FutureMap` is a later optimization.

**Consequence — the cheapest proposer first.** Given Finding A only frees us on **MHA** models, and
our MTP-bearing models (GLM, DeepSeek) are **MLA** (whose multi-query verify kernel does not exist
yet), the lowest-friction first proposer is **n-gram / prompt-lookup**: no draft model, no draft
head, no new weights, runs on MHA where the verify kernel already exists. Every line of the engine
machinery it exercises (verify forward mode, multi-token acceptance, KV rollback, scheduler cycle) is
**proposer-agnostic** — swapping in an MTP/EAGLE draft head later only replaces the proposer and adds
the MLA verify kernel. n-gram-first wastes nothing and de-risks everything.

---

## 2. Upstream SGLang reference (for the later MTP/EAGLE phases)

Source: `python/sglang/srt/speculative/` in sgl-project/sglang.

- **Methods**: EAGLE/EAGLE2 (feature-based draft + dynamic tree), EAGLE3 (multi-layer feature
  fusion — recommended upstream), STANDALONE (separate draft LLM), NGRAM (what our MVP ports),
  MTP/NEXTN (built-in next-token heads on DeepSeek-V3 / GLM-4.x).
- **Workers**: `EAGLEWorkerV2` owns a `_target_worker` + `_draft_worker`; draft and target are
  separate `ModelRunner`s that **share** the KV pools. Loop: `draft()` (propose) →
  `verify()` (one target forward, `ForwardMode.TARGET_VERIFY`) → `_draft_extend_for_decode()`.
- **Tree (EAGLE2)**: `build_tree_kernel_efficient` emits `draft_token`, `positions`,
  `retrieve_index/next_token/next_sibling`, and a flat `custom_mask`. `topk>1` = branching tree;
  `topk=1` = linear chain (no branching mask) — the simple port.
- **Verify attention contract**: ragged `qo_indptr` (num_draft_tokens queries/seq) + paged KV +
  a flat boolean `custom_mask` indexed `mask_indptr[seq] + qrow*(seq_len+tree) + kvcol`, AND-ed into
  the score matrix, with a `SKIP_PREFIX` fast path. **Only needed for `topk>1`.** For topk=1 the
  causal mask the existing kernel already applies is sufficient.
- **Acceptance** (`reject_sampling.py`): greedy = walk the chain while `argmax(target)==draft`;
  sampled = `coin*q < p` accept + residual-distribution bonus (distribution-equivalent to
  non-spec sampling).
- **EAGLE3 target seam**: target must expose the token embedding table **and** aux hidden states
  from 3 intermediate decoder layers (`set_eagle3_layers_to_capture`), wired before graph capture.

Backend/topk matrix takeaway: every spec-capable backend needs multi-query ragged attention vs
paged KV; topk=1 needs no custom mask. The AITER/ROCm backend is the closest AMD analog if we later
build the MLA tree-verify kernel.

---

## 3. The minisgl verify cycle (MVP: n-gram, topk=1, greedy, MHA, sync loop)

Per running req, one spec step replaces one decode step:

1. **Propose** (host): `propose_ngram(req.input_ids, num_draft=K, max_ngram=N)` → draft `d[0..K-1]`
   (possibly empty → degrades to a plain decode step). python/minisgl/spec/proposer.py.
2. **Stage** (host→GPU): the req contributes K+1 query positions
   `[confirmed_token, d0, ..., d_{K-1}]` at absolute positions `p .. p+K` (p = device_len-1).
   Write d into the GPU `token_pool` at those positions; allocate KV pages for the K new slots;
   build positions/cu_seqlens so the verify batch has `extend_len = K+1` per req.
3. **Verify forward**: reuse `prepare_metadata` + `_hip_prefill_paged` (extend branch, causal).
   One target forward over all `sum(K_i + 1)` tokens. Gather logits at each of the K+1 positions.
4. **Accept** (host): `verify_greedy(d, target_argmax)` → `(emitted, n)` where target_argmax[i] =
   argmax of logits at query position i. Emit `t0..tn` (n+1 tokens), n drafts accepted.
   python/minisgl/spec/accept.py.
5. **Commit + rollback**: append `emitted` to the req (advance device_len/cached_len by n+1). KV for
   the n accepted draft positions is valid and kept; **free the KV pages of the K−n rejected draft
   positions** (rollback to the pre-verify tail + accepted run + 1 bonus slot). The bonus token
   `tn`'s KV is recomputed next step (its position held a now-rejected draft, or n==K).

Greedy makes steps 4–5 **lossless**: output is bit-identical to non-speculative greedy decoding.
Sampling acceptance (the `coin*q<p` + residual rule) is a follow-up; production bench is greedy
(temperature 0), so greedy verify is exactly correct for the headline number.

### KV rollback detail
The cache manager (python/minisgl/scheduler/cache.py) allocates by **page**. To keep rollback
simple, the MVP forces **page_size = 1** for spec-decode runs (per-token slots → free the tail
`K−n` slots directly). page_size>1 partial-page rollback is a later refinement. The HIP decode /
prefill kernels accept any page_size, so this is purely an allocator-bookkeeping convenience.

---

## 4. Component status / plan

| # | Component | File | Status |
|---|-----------|------|--------|
| 1 | n-gram proposer | `python/minisgl/spec/proposer.py` | **done, CPU-tested** |
| 2 | greedy acceptance | `python/minisgl/spec/accept.py` | **done, CPU-tested** |
| 3 | CPU unit tests | `tests/spec_core_test.py` | **done, green** |
| 4 | `SpecConfig` + `--spec-*` args | `engine/config.py`, `server/args.py` | **done** |
| 5 | engine verify forward path | `engine/engine.py::forward_verify` | **done, GPU-validated** |
| 6 | scheduler spec cycle + rollback | `scheduler/scheduler.py::_spec_loop/_spec_decode_step` | **done, GPU-validated** |
| 7 | multi-token detokenizer streaming | `message/tokenizer.py`, `tokenizer/detokenize.py` | **done** (see note) |
| 8 | GPU coherence + losslessness harness | `tools/spec_smoke.sh`, `tools/spec_lossless.sh` | **done, PASS** |

> **Note (#7):** a step commits several tokens per req, but the incremental detokenizer keys
> streaming offsets by uid and assumed one message per uid per batch — multiple per-token messages
> clobbered the shared `surr_offset`/`read_offset` and produced a growing-prefix duplication
> (`"colors colors: colors: red"`) even though the committed token ids were correct. Fix:
> `DetokenizeMsg` carries `extra_tokens` so a step's tokens travel in ONE message; the detokenizer
> appends them together and drops a trailing EOS. The normal one-token path is unchanged
> (`extra_tokens=[]`).

### Later phases (post-MVP)
- **Sampling acceptance** — `coin*q<p` + residual bonus for temperature>0 correctness.
- **MTP/NEXTN self-spec** — stop skipping the MTP head in `models/weight.py::_is_beyond_decoder`;
  load it as a 1-layer draft; reuse this entire cycle with the head as the proposer. GLM/DeepSeek.
- **MLA multi-query verify kernel** — `mla_hip` currently has decode (qlen=1) + materialized
  prefill; MLA verify needs multi-query against the paged latent. Required for any MLA-model spec.
- **EAGLE2 dynamic tree** — `topk>1`: add `custom_mask`+`mask_indptr` to `attn_prefill_paged`.
- **Overlap scheduling** — `FutureMap`-style to hide the acceptance host-sync.
- **CUDA graph capture** of the verify forward (fixed max tree size, dynamic accept count post-replay).

---

## 5. Invariants / gotchas

- **Greedy-only correctness** today: assert all reqs in a spec batch are greedy; non-greedy reqs
  fall back to plain decode until sampling acceptance lands.
- **Sync loop only**: spec-decode forces the synchronous `normal_loop`; do not mix with overlap.
- **page_size=1** under spec-decode (rollback simplicity) — see §3.
- **MHA only** for the MVP (MLA verify kernel absent). Guard at config time.
- The verify forward must **not** be flagged `cold_prefill` (it has a cache prefix) — guaranteed by
  `cached_len > 0`, which always holds for a decoding req.
