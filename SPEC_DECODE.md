# SPEC_DECODE.md — speculative decoding for minisgl (Triton-free, native-HIP)

Status: **working and GPU-validated on both MHA and MLA** (n-gram / prompt-lookup, topk=1, greedy,
eager, synchronous loop). Enable with `--spec-algorithm ngram` (see args below).
- **MHA** (Qwen3-0.6B, TP=1): coherent; ~11% accept / 1.25 tok/step on repetitive prompts; spec
  output **bit-identical to sequential decode through the verify kernel** (`tools/spec_lossless.sh`
  → PASS) — accept/commit/KV-rollback is provably lossless.
- **MLA** (GLM-4.7-Flash AWQ, TP=2): coherent; **~45% accept / 2.2 tok/step**; SPEC == FORCE_N0
  bit-identical on all prompts (`tools/spec_glm.sh` → PASS). Uses the absorbed multi-query
  `mla_hip.mla_verify` kernel (paged latent, no prefix re-materialization), kernel parity-validated
  (`mla_hip/mla_hip_parity.py verify`, cos≈1.0).

Residual divergence from a plain-decode baseline is only the inherent verify-kernel vs decode-kernel
fp difference, not a spec defect. This doc is both the upstream porting reference and the concrete
minisgl plan; read it before extending.

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
| 9 | MLA multi-query verify kernel (absorbed) | `mla_hip/mla_kernels.hip::mla_verify[_fp8]` | **done, parity PASS** |
| 10 | MLA wiring (backend/layer/scheduler) | `attention/mla.py`, `models/glm4_moe_lite.py`, `layers/embedding.py` | **done, GPU-validated** |
| 11 | MLA spec end-to-end (GLM TP=2) | `tools/spec_glm.sh` | **done, PASS (45% accept)** |

> **MLA verify (#9–11):** the verify forward keys on `metadata.max_seqlen_q` — q_len==1 →
> `mla_decode` (single-token), q_len>1 → `mla_verify` (absorbed multi-query over the **paged
> latent**, one CTA per `(head, query-row)`, per-row causal bound `cached_len+qi+1`). No prefix
> re-materialization, so the spec speedup is preserved. Two seams this exposed: (a) the LM head's
> TP>1 fast path gated on request-count (`bs==1`) collapsed a verify batch's K+1 rows to one — fixed
> to gate on row-count (`embedding.py`); (b) KV rollback is now page-size-aware (frees whole pages
> beyond the kept run), so MLA keeps `page_size=16` while MHA stays at 1.

> **Note (#7):** a step commits several tokens per req, but the incremental detokenizer keys
> streaming offsets by uid and assumed one message per uid per batch — multiple per-token messages
> clobbered the shared `surr_offset`/`read_offset` and produced a growing-prefix duplication
> (`"colors colors: colors: red"`) even though the committed token ids were correct. Fix:
> `DetokenizeMsg` carries `extra_tokens` so a step's tokens travel in ONE message; the detokenizer
> appends them together and drops a trailing EOS. The normal one-token path is unchanged
> (`extra_tokens=[]`).

### Later phases (post-MVP)
- **Sampling acceptance** — `coin*q<p` + residual bonus for temperature>0 correctness.
- ~~**MLA multi-query verify kernel**~~ — DONE (`mla_hip.mla_verify`, absorbed, paged latent).
- **EAGLE2 dynamic tree** — `topk>1`: add `custom_mask`+`mask_indptr` to `attn_prefill_paged`
  (and an analogous per-query tree mask to `mla_verify`).
- **Overlap scheduling** — `FutureMap`-style to hide the acceptance host-sync.
- **CUDA graph capture** of the verify forward (fixed max tree size, dynamic accept count post-replay).

---

## 6. Model coverage: proposer abstraction, GDN, MTP, DFlash/EAGLE3

The proposer is pluggable (`python/minisgl/spec/base.py::Proposer`); the verify/accept/commit/KV-
rollback machinery is proposer-agnostic. Four families share one interface:

| proposer | model | `needs_last_hidden` | `capture_layer_ids` | owns | verify |
|----------|-------|---------------------|---------------------|------|--------|
| **ngram** | none | no | None | nothing | linear |
| **MTP** | target's appended head | yes | None | head + 1-layer draft KV | linear |
| **DFlash** | separate ckpt | (via aux) | [N target layers] | fc+trunk+KV, mask-block | linear block |
| **EAGLE3** | separate ckpt | (via aux) | [3 target layers] | fc+1-layer+KV, d2t/t2d | tree |

`Proposer.propose(reqs, num_draft, ctx)` returns drafts; `Proposer.on_accept(reqs, num_accepted)`
rolls back draft-owned state. `make_proposer(spec_config, engine)` is the factory. The seams MTP/
DFlash/EAGLE3 need from the engine: (a) the target's **last hidden state** per verified position;
(b) **aux-hidden capture** — `capture_layer_ids` programs a target hook to stash N decoder layers'
outputs; (c) `bind_target(embed, lm_head, d2t, t2d)` so a draft borrows the target's embed/head.

### Target hidden-state exposure (seams (a)+(b)) — DONE, GPU-validated (capture-neutral)
The foundation MTP/EAGLE3/DFlash consume. **OFF by default** — a pure n-gram serve (and normal
decode) pays nothing: capture only engages when a proposer declares `needs_last_hidden` /
`capture_layer_ids`. The exposed API (what the draft-head phases call):

- **`*ForCausalLM.forward(return_hidden=False)`** (`models/{qwen3,qwen3_5,glm4_moe_lite}.py`, base
  in `models/base.py`). Default returns just `lm_head` logits (unchanged). With `return_hidden=True`
  returns `(logits, last_hidden, aux_hidden)`:
  - `last_hidden`: `[num_tokens, hidden]` post-final-norm, pre-`lm_head` (the MTP/EAGLE seed).
  - `aux_hidden`: `[num_capture_layers, num_tokens, hidden]` — the residual stream AFTER each
    programmed decoder layer (what feeds the next layer), stacked **in the `set_capture_layers` id
    order**; `None` if no layers are programmed.
- **`*ForCausalLM.set_capture_layers(ids: list[int] | None)`** — program which decoder layers stash
  their output (`None`/`[]` = off). Wired once at proposer init from `proposer.capture_layer_ids`.
- **`Engine.forward_verify(batch, return_hidden=False)`** (`engine/engine.py`) — threads the flag to
  the model; same return contract. One forward, no extra pass.
- **Scheduler** (`scheduler/scheduler.py::_spec_decode_step`): reads the proposer's
  `needs_last_hidden` / `capture_layer_ids` at init (programs `set_capture_layers`). Per verify step,
  iff `capture`, calls `forward_verify(return_hidden=True)`, then for each still-running req clones
  the **seed row** = `block_start + len(keep)-1` (the verify row that produced the last committed
  token — the same index the GDN per-token-state install uses) into per-uid
  `_spec_last_hidden[uid]` `[hidden]` / `_spec_aux_hidden[uid]` `[num_capture_layers, hidden]`. These
  are handed to the **next** step's `ProposeContext(last_hidden=…, aux_hidden=…)` keyed by uid;
  finished uids drop out. The big `[T, …]` verify tensors are released after slicing.

Validated: `tools/spec_capture.sh` (Qwen3-0.6B, MHA, layers `0,13,27`) — the `_CaptureProbeProposer`
(env `MINISGL_SPEC_CAPTURE_PROBE=<ids>`) drafts exactly like n-gram but asserts the per-uid shapes
(`last_hidden=(1024,)`, `aux_hidden=(3,1024)`, bf16, delivered across steps) and its greedy output is
**bit-identical to plain n-gram spec** (capture is output-neutral: 3/3 prompts MATCH). Note the
oracle is plain-spec, NOT plain-decode (the verify kernel's fp differs from the decode kernel — see
the top-of-doc residual-divergence note).

### GDN-hybrid models (qwen3_5 / qwen3_5_moe) — DONE, BIT-EXACT (per-token-state verify kernel)
GPU-validated on Qwen3.5-4B (TP=1, `--attn hip`, `tools/spec_gdn.sh`): coherent, ~17% accept /
1.29 tok/step, and **SPEC == FORCE_N0 BIT-IDENTICAL on all prompts** (368/266/347/394/392-char
outputs, all MATCH). This is the GDN "completion" item: the 2× re-advance is gone and the output is
now bit-exact vs a 1-token-per-step reference (previously only "coherent, fp-drifting").

The mechanism: a verify batch carries `Batch.spec_verify=True` (phase "decode", `extend_len=K+1`/seq),
which routes the GDN layer + `build_gdn_metadata` through the varlen recurrent path. The scheduler
sets `gdn_metadata.capture_verify_state=True` + `verify_max_qlen=max(K_i+1)`, so the GDN bridge
(`models/qwen3_5.py`) calls `QwenGatedDeltaNet.forward_prefill_verify`, which runs the **dedicated
per-token-state HIP kernels** — `gdn_hip.causal_conv1d_fwd_verify` + `gdn_hip.gdn_prefill_verify`.
These are byte-identical recurrences to the plain conv/`gdn_prefill` kernels but ALSO snapshot the
conv-window + ssm state AFTER EACH token into a scratch `[max_qlen, N, ...]`, stashed per
`gdn_layer_id` on the metadata. (They use the bit-stable RECURRENT gdn kernel, never WMMA — the
chunk-size dependence is exactly the non-bit-exactness this removes, so it is independent of
`GDN_HIP_WMMA_PREFILL`.) After acceptance, `GDNStateCache.install_verify_state` gathers, per
still-running seq, the captured state at index `committed-1` (the state after the last committed
token) and writes it into the live conv+ssm slot for every GDN layer — a pure gather, no snapshot,
no second forward. Kernel parity: `tools/gdn_hip_parity.py` (`gdn_prefill_verify` /
`causal_conv1d_fwd_verify` checks — per-token scratch matches the recurrent oracle exactly, and
`scratch[last]==final_state` max|Δ|=0, across fp32/fp16/bf16).

The old snapshot + `_gdn_readvance` (restore pre-verify state, re-run the accepted tokens through a
2nd eager forward) has been removed. `GDNStateCache.snapshot/restore` remain as a general-purpose API
but are no longer on the spec path.

### MTP self-speculation (the appended heads we currently discard) — DESIGNED
GLM-4.7-Flash (`num_nextn_predict_layers=1`) and Qwen3.5/3.6 (`mtp_num_hidden_layers=1`) ship one
MTP head, currently skipped (`weight.py::_is_beyond_decoder` for GLM `layers.47.*`;
`_QWEN35_SKIP_PREFIXES` `"mtp."` for Qwen). Structure (both): `head_out = layer( fc( concat[
norm_e(embed(next_tok)), norm_h(last_hidden) ] ) )` → head. GLM ships its own `embed_tokens` +
untied `shared_head.head` + a full **MLA+MoE** layer at `layers.47`; Qwen reuses the target's embed
+ tied lm_head and ships a single **standard full-attention** `mtp.layers.0` (NOT GDN — so the MTP
draft forward never touches GDN state; only the verify over the 32-layer backbone does, handled
above). Implementation: (1) load the head (exempt those tensors, add module classes reusing
`GLMDecoderLayer` / `Qwen3_5Attn`+MLP); (2) ~~expose `last_hidden` from the target forward~~ **DONE**
— `forward(return_hidden=True)` / `forward_verify(return_hidden=True)` return `last_hidden` (and the
scheduler carries the per-uid seed row into the next `ProposeContext`); see "Target hidden-state
exposure" above; (3) `MTPProposer` runs the head autoregressively K times with its own 1-layer draft
KV, `on_accept` truncates that KV. Validate GLM (no GDN) first, then Qwen3.5.

### DFlash / EAGLE3 (extension targets) — SCAFFOLDED via the abstraction
Both consume `fc(concat of N captured target layers)` → trunk → head, from a **separate checkpoint**
with its **own draft KV pool**. DFlash (`DFlashDraftModel`, local ckpts for Laguna/Qwen3.5/3.6):
N=5–8 captured layers, 5–6-layer trunk, **block-parallel** drafting (denoise `block_size`
mask-tokens in one pass), linear verify; z-lab ckpts borrow the target embed+head, Laguna ships its
own + `d2t/t2d` (compressed draft vocab). EAGLE3: 3 captured layers, 1-layer trunk, autoregressive,
tree verify. To add them: a `DraftModelProposer` (separate `ModelRunner` for the draft ckpt) + the
aux-hidden capture hook (`capture_layer_ids`) + `bind_target`. The abstraction's `capture_layer_ids`
/ `needs_last_hidden` / `bind_target` seams exist for exactly this; EAGLE3's tree verify additionally
needs the `topk>1` custom-mask kernels (see §5 later phases).

---

## 5. Invariants / gotchas

- **Greedy-only correctness** today: assert all reqs in a spec batch are greedy; non-greedy reqs
  fall back to plain decode until sampling acceptance lands.
- **Sync loop only**: spec-decode forces the synchronous spec loop; do not mix with overlap.
- **page_size**: MHA forces 1 (per-token rollback); MLA keeps 16 (the `mla_hip` block size). The
  KV rollback is page-size-aware (frees whole pages beyond the kept run) — see §3 / `_spec_decode_step`.
- **MHA and MLA both supported.** MLA verify uses the absorbed `mla_verify` kernel; only GLM
  (`glm4_moe_lite`) is wired today (the sole MLA model). A new MLA model needs the same
  `max_seqlen_q`-keyed dispatch in its attention layer.
- **LM head row-count**: the TP>1 fast path must gate on logit-row-count, not request-count, or a
  verify batch (K+1 rows, 1 req) collapses to a single row (`embedding.py`).
- The verify forward must **not** be flagged `cold_prefill` (it has a cache prefix) — guaranteed by
  `cached_len > 0`, which always holds for a decoding req.
