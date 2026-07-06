# ZAYA serving on RDNA4 — north-star + implementability triage

Status: design (2026-07-04). Written by applying the CAM_DESIGN methodology (step back → research
the whole decomposition → design the maximal version → triage to a cheapest-falsifier-first roadmap)
to the ZAYA-serving throughput work, INSTEAD of layering the next individual change.

Context: the effort has been a pile of individual levers — spec-decode variants (two-forward →
fused-TiDAR → DFlash), quant variants (RFP458 → RXF → W4A8 → W8A16), and per-kernel work
(GDN/CCA/attn/MoE). Each was measured in isolation. This doc steps back to the ONE decomposition all
of them live inside, and asks which lever actually has the ceiling.

---

## 0. The one commitment everything hangs off

**Serving throughput = (tokens committed per forward) ÷ (forward wall time), and on RDNA4 a decode
forward's wall time is `dispatch_overhead + bandwidth_bound_weight_loads`, NOT compute.** Two — and
only two — orthogonal levers move the ratio:

- **DISPATCH lever** — shrink the per-step host overhead (kernel count / launch / graph coverage).
  Measured: the 0.22 stack gains **+80–107% at low batch from cudagraph** (`[[cudagraph-benefit-stack-dependent]]`);
  clean 35B decode is ~12 ms/token but eager ZAYA-8B AR is **42 ms/token** while its fp8 BW floor is
  ~14 ms → **~2/3 of every eager step is recoverable dispatch overhead.**
- **BANDWIDTH lever** — commit more tokens per weight-load. That is exactly what spec-decode buys
  (amortize the BW-bound weight read across `accept+1` tokens/forward). Fused-TiDAR is this lever;
  its cost blocker (the MoE dequant flood) is now fixed (W8A16, §"cost pivot"), leaving acceptance.

They **multiply**: a graph-captured serve at 2× dispatch AND a trained spec-decode at accept≈2 is
~4× the eager-no-spec baseline. Everything else (MoE kernel, quant format, AR backend) is a rounding
error on the wall (MoE ≤10%; `[[served-decode-is-idle-bound-not-moe]]`).

**The mistake this reframe corrects:** the session optimized the BW lever's MoE *cost* (real, needed)
and its *acceptance* (model-limited, slow) while the DISPATCH lever — bigger ceiling, proven on this
stack, and simply *turned off* for the CCA path (`--graph 0`) — went untouched.

## 1. The decode decomposition (measured, not assumed)

Per-step wall of an eager decode forward, in falling order of what it costs and whether we've moved it:

| Bucket | Share of wall | Lever | State |
|---|---|---|---|
| Host **dispatch / launch** (eager) | ~⅔ of the eager step (42→~14 ms floor) | cudagraph capture | **AR-decode capture is DONE (commit 3699388) but turned OFF in the serve (`--graph 0`)** ← just flip it |
| BW-bound **weight reads** | the ~14 ms floor | spec-decode (amortize) | fused-TiDAR built; accept-capped |
| **TP all-reduce** (TP≥2) | ~30% of *busy* at conc1, exposed | avoid it (TP=1 / DP / fewer syncs) | custom-AR is a **dead end** on RDNA4 (no XGMI; `[[allreduce-backend-fixed-pynccl-gfx1201]]`) |
| dense **attn/proj GEMVs** | next biggest busy bucket | GEMV kernel (v11 K-gate) | +72% win already banked on 27B down_proj (`[[27b-decode-dispatch-and-tp-bubble]]`) |
| **MoE experts** | ≤10% of wall | W8A16 / native fp8 | cost fixed this session; **not the lever** |

## 2. Non-negotiables (RDNA4 failure modes that kill naive plans)

1. **16 GB/card is the binding budget.** Every acceptance/precision upgrade that needs a bigger
   resident set OOMs (DFlash drafter-TP1, TP=1 target, bf16 target — `[[dflash-laguna-int4-acceptance-wall]]`).
   Any lever must fit two 16 GB cards (or one).
2. **No XGMI → TP all-reduce is a fixed PYNCCL tax.** Custom AR is architecturally impossible on
   consumer RDNA4 (no coherent fabric; every fine-grained x-GPU sync primitive fails —
   `[[allreduce-backend-fixed-pynccl-gfx1201]]`). The only way to cut the ~30% AR tax is to **not
   emit it** (TP=1 if it fits, or DP where AR isn't on the hot path).
3. **CCA/GDN recurrent decode capture is IMPLEMENTED, just disabled in the serve config.** Commit
   3699388 threads per-seq recurrent-state slots through static buffers (`gdn_capture`/`cca_capture`:
   `prepare_for_capture` inside the capture loop → `torch.cuda.graph` → `prepare_for_replay`); the
   serve runs `--graph 0` so it's off. So AR-decode capture is NOT the open question — the SPEEDUP is.
   The genuine gap is **CCA spec-verify capture**: regular decode captures for GDN/CCA/MLA/dense, and
   **MLA spec-verify IS captured** (commit 4986cea, `capture_verify_graphs` graph.py:243–301, precomputed
   `q_seq_idx`/`q_kbound`), but MHA/GDN/CCA spec-verify is still disabled (graph.py:259 + `engine.py:616
   if not is_mla:`). ⇒ the fused-forward capture (v2) is a **PORT of the working MLA verify-capture to
   CCA**, not a from-scratch research problem.
4. **LDS ≤ 64 KB and no direct VMEM→LDS** (`[[rdna4-no-direct-vmem-to-lds]]`) — bounds any resident-
   state megakernel; keep recurrent state fp16/fp32 (not fp8) for fidelity.
5. **Measure with the profiler OFF.** `with_stack` inflates launch-heavy decode ~6× — the "84% idle"
   scare was mostly artifact (`[[served-decode-is-idle-bound-not-moe]]`). Clean streaming ITL only.

## 3. Maximal version → implementability triage

Rank by ceiling (how much throughput it can move) vs cost/risk.

| Lever | Ceiling | Cost/Risk | Tier |
|---|---|---|---|
| **cudagraph-capture the CCA/GDN decode** (dispatch lever) | **decisive (~2× at low batch)** | med-high (recurrent state + dynamic shapes) | **v0** |
| Fused-TiDAR spec-decode, W8A16 cost fixed (BW lever) | high (~accept+1) | built; accept-capped | **v0/v1** |
| Acceptance training (round-2, pos0-protect + reasoning corpus) | high (multiplies the BW lever) | cloud burst, days | v1 |
| v11-style GEMV K-gate audit across all decode GEMVs | medium (+72% seen on one) | low | v1 |
| TP=1 (drop the AR tax) if the 8B fits one 16 GB card at fp8 | medium (~30% busy) | low-med (memory knife-edge) | v1 |
| Graph-capture the FUSED forward too (both levers compound) | high (4× compound) | high (custom-mask + capture) | v2 |
| MoE kernel micro-opt, quant-format churn | ≤10% of wall | any | **do not fund** |

### v0 RESULT (2026-07-04) — GREEN. The dispatch lever is real and large.
`tools/run_cca_graph_v0.sh`, ZAYA-8B AR (no spec), W8A16 default: `--graph 0` **23.1 tok/s (43 ms/tok)**
→ `--graph 8` **36.7 tok/s (27 ms/tok) = 1.59×** (triple-confirmed 1.57/1.61/1.59). CCA graph capture
is clean (sizes [1,2,4,8], 0.34 GiB). **All 5 custom HIP kernels verified engaged AND survive capture**
(one-time `[hip-engage]` log, `_hip_engage.py`): `attn_decode.flash_decode_paged`, `attn_hip.flash_prefill`,
`zaya_cca.cca_decode_qk`, `tail_hip`, `moe_w8a16` — identical set eager vs captured, no Triton fallback
(the `TRITON_unified_FALLBACK` tripwire never fired). ⇒ turn `--graph` on in the serve config (it's a
free ~1.6× on AR); then v1/v2 compound. NOTE the AR forward now processes >1 query token under RSA, so
`attn_prefill_paged` (not `attn_decode`) can carry the attention when max_seqlen_q>1 — both are custom HIP.

### v0 method — MEASURE the dispatch lever (mechanism already implemented; just disabled)
CCA/GDN AR-decode capture is DONE (commit 3699388); the serve just runs `--graph 0`. So the question
is not *whether* it captures but *how much it recovers*.
- Experiment: boot ZAYA-8B AR (no spec) at `--graph 0` vs `--graph N`; clean median ITL each. Toward
  ~14–20 ms (from 42 eager) → the dispatch lever is real and the whole serve inherits it; ALSO verify
  every custom HIP decode kernel still fires under capture (`[hip-engage]`: attn_decode / cca_decode /
  tail_hip / moe_w8a16) — a graph-unsafe kernel would fall back or fail capture.
- Cost: ~1 card, hours (mind the cold graph compile; hold `--max-num-seqs` constant). `tools/run_cca_graph_v0.sh`.

### v1 — compound the two orthogonal levers on a graph-captured base
- Land acceptance training (round-2) so the BW lever pays (accept≈1–2 → ~1.4–2× AR *on top of* v0's
  graph win). Audit the v11 GEMV K-gate across every decode GEMV (free latency). Test TP=1-fits to
  drop the AR tax.

### v2 — graph-capture the fused forward itself (both levers in one step)
The crown jewel: a graph-captured *fused* forward = dispatch-free AND accept+1 tokens/forward = the
~4× compound. This is a **PORT of the working MLA spec-verify capture (commit 4986cea,
`capture_verify_graphs` graph.py:243–301) to the CCA path** (extend the `if not is_mla:` gate at
engine.py:616), NOT a research problem — MLA proves recurrent/verify capture works; CCA needs the same
static-buffer treatment for its conv/prev_hs state + the fused custom-mask shapes. Gated on v0 showing
the dispatch win is worth the port.

### De-risking order (cheapest-falsifier-first)
1. **cudagraph the CCA AR decode** (v0) — biggest untouched lever, one experiment, decisive either way.
2. **Acceptance training + GEMV audit + TP=1-fits** (v1) — compound on the captured base.
3. **Graph-capture the fused forward** (v2) — the 4× once v0 says CCA graphs are possible.

## 4. What this de-prioritizes (stop layering these)
- MoE kernel micro-optimization and quant-format churn — ≤10% of wall; W8A16 already closed the real
  cost. Do NOT keep grinding here.
- Custom all-reduce / AR-backend swaps — architecturally dead on RDNA4 (§2.2).
- More spec-decode *variants* (DFlash INT4, etc.) before the dispatch lever is characterized — pick
  the acceptance-training path on the fused kernel we already validated, not a new drafter.

## 5. Key references
Internal (measured): `[[served-decode-is-idle-bound-not-moe]]` · `[[async-scheduling-tested-no-win-keep-serial]]`
· `[[cudagraph-benefit-stack-dependent]]` · `[[27b-decode-dispatch-and-tp-bubble]]` ·
`[[35b-decode-wall-is-gemm2-weight-bw]]` · `[[allreduce-backend-fixed-pynccl-gfx1201]]` ·
`[[dflash-laguna-int4-acceptance-wall]]` · this session's W8A16 + fused-TiDAR results
(`docs/TRAINING_PLAN_ACCEPTANCE.md`, `docs/CONTINUE_FUSED_TIDAR.md`).
External: spec-decode (Leviathan 2211.17192, Medusa 2401.10774, EAGLE 2401.15077), block-diffusion
TiDAR (2511.08923), CUDA-graph decode (the vLLM/SGLang graph-capture literature).
</content>
