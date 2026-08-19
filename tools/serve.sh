#!/usr/bin/env bash
# serve.sh — turn a handful of CHOICES into a minisgl launch line.
#
# WHY THIS EXISTS
#   docker-compose.yml carried eight near-identical serve services (serve, qwen, glm, qwen35b-mtp,
#   qwen35b-dflash, laguna, laguna-dflash, laguna-radix). They ran the SAME command and differed only
#   in a model id, a spec algorithm, a draft path and an attention backend — so every new model or
#   spec mode meant another copy-pasted 40-line block, and 60 MINISGL_* variables between them. Adding
#   a knob meant editing eight places, and the combinations that were not pre-baked (laguna+mtp,
#   glm+dflash, anything with DP or EP) simply did not exist.
#
#   The variation is a small TABLE, not eight services. This file is that table.
#
# THE CHOICES (all optional; every one has a default)
#   MODEL   alias below, or any HF id / local path      (default qwen35b-awq)
#   SPEC    none|mtp|dflash|dspark|eagle3|ngram|tidar   (default: the model's own default)
#           dspark = the dflash engine path with a DSpark draft (Markov + confidence heads,
#           detected from the checkpoint's tensors); canonical for the qwen38-27b* arms.
#   DRAFT   draft checkpoint for dflash/eagle3          (default: per model; required if none)
#   SPEC_K  draft length                                (default: per model+algorithm, tuned)
#   TP      tensor-parallel size, 1 or 2                (default 2)
#   DP      data-parallel size, 1 or 2                  (default 1)
#   EP      1 to enable expert parallelism              (default 0)
#   CTX     context length in tokens                    (default: the checkpoint's own)
#   CONC    concurrent requests                         (default 4)
#
# LESS-USED, still here rather than in eight copies:
#   ATTN (auto|hip|triton), MEM_RATIO, GRAPH_BS, PAGE_SIZE, CACHE_TYPE, PORT, EXTRA_ARGS,
#   MAX_PREFILL_LENGTH (chunked-prefill chunk, default 2048 — see the note at its assignment).
#   EXTRA_ARGS is appended verbatim and wins — the escape hatch for anything not modelled here.
#
# Print the composed line without running it:  DRY_RUN=1 tools/serve.sh
set -uo pipefail

MODEL="${MODEL:-qwen35b-awq}"
SPEC="${SPEC:-}"
TP="${TP:-2}"; DP="${DP:-1}"; EP="${EP:-0}"
CONC="${CONC:-4}"
PORT="${PORT:-1919}"

# --- the table -----------------------------------------------------------------------------------
# Per model: the checkpoint, the attention backend it wants, its default spec algorithm + draft
# length, and (for dflash) the draft checkpoint. `attn=hip` is the canonical served backend; `auto`
# survives only where a model has not been re-validated on it.
dflash_draft=""; eagle3_draft=""; attn="hip"; spec_default="none"; swa_hybrid=""; tool_format=""
# Per-model default presence penalty (server-side, request-unset only; explicit client values —
# including 0.0 — win). Empty = no default, penalty path skipped.
presence_default=""
# Per-model torch allocator config (exported just before exec). Empty = torch default.
alloc_conf=""
k_mtp=4; k_dflash=15; k_eagle3=4; k_tidar=4; mem_default="0.80"
# Minimum TP a model's WEIGHTS require. A general knob, not a special case: some checkpoints simply
# do not fit on one 16 GB card, and the control panel exposes TP as a free dropdown — so picking one
# of them at TP=1 otherwise dies in a CUDA OOM minutes into weight loading, with nothing in the
# error pointing at the actual cause.
min_tp=1
# Each arm matches the ALIAS *or* the checkpoint id/path, because the two ways of choosing a model
# produce different strings: typing `MODEL=laguna` gives the alias, while the control panel's model
# dropdown is populated from the HF cache and hands over the full id. Matching only the alias meant
# picking a model from the dropdown silently fell through to the catch-all and lost its draft
# checkpoint, attention backend and tuned defaults.
case "$MODEL" in
  qwen35b-awq|cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit)
                  model_id="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit";        spec_default="mtp"; k_mtp=4
                  dflash_draft="z-lab/Qwen3.6-35B-A3B-DFlash"; k_dflash=15
                  # SPEC=dflash on this pair could not boot AT ALL on 16 GB cards until these.
                  # 35B weights are ~11.4 GB/card, so a 737 MB bf16 drafter REPLICATED on every rank
                  # plus the verify graphs do not fit beside a plain-decode-sized KV pool. The two
                  # failures bracket it: at 0.80 the KV pool starves after reserving drafter+graphs
                  # ("num_pages > 1" assert), at 0.93 the drafter itself OOMs (13.54 GiB allocated,
                  # 0 free) — no ratio alone works, the FOOTPRINT has to come down. fp8 weight-only
                  # quant halves the drafter (lossless: the target verifies every draft), and the
                  # graph buckets have to shrink with it. Measured booting and serving at
                  # 0.86 + fp8 + GRAPH_BS<=4. MTP needs none of this (no separate draft model).
                  if [[ "${SPEC:-$spec_default}" == "dflash" ]]; then
                    mem_default_spec="0.86"
                    : "${MINISGL_DFLASH_QUANT:=fp8}"; export MINISGL_DFLASH_QUANT
                    # Cap ADMISSION, not just capture. GRAPH_BS now follows CONC, so capping only the
                    # graph would admit batches above the captured max and run them FULLY EAGER —
                    # the exact failure the capture-coverage rule exists to prevent. Cap CONC and let
                    # GRAPH_BS follow it, so admission and coverage stay equal. A CAP, not a
                    # default: CONC is already assigned above this case block, so `${CONC:=4}` would
                    # be a silent no-op.
                    if [ "$CONC" -gt 4 ]; then CONC=4; fi
                  fi ;;
  qwen35b-mxfp4|pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4)
                  model_id="pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4"; spec_default="mtp"; k_mtp=2
                  dflash_draft="z-lab/Qwen3.6-35B-A3B-DFlash"; k_dflash=15 ;;
  # GLM's MTP head is a measured NET LOSS on this box, so its default is EAGLE3 (K=6 from the
  # spec-len sweep). MTP remains selectable — it is just not the default.
  glm|QuantTrio/GLM-4.7-Flash-AWQ)
                  model_id="QuantTrio/GLM-4.7-Flash-AWQ";              spec_default="eagle3"; k_mtp=2
                  eagle3_draft="thoughtworks/GLM-4.7-Flash-Eagle3"; k_eagle3=6 ;;
  laguna|poolside/Laguna-XS-2.1-NVFP4)
                  model_id="poolside/Laguna-XS-2.1-NVFP4";             spec_default="none"
                  dflash_draft="poolside/Laguna-XS-2.1-DFlash-NVFP4"; k_dflash=16; mem_default="0.85"
                  mem_default_spec="0.93"
                  swa_hybrid=1 ;;
  # Muse-Glimmer: dense SWA hybrid (39 sliding @2048 + 13 full, the full ones NoPE), NVFP4,
  # text-only — the vision tower is skipped by the loader.
  # TP=2 is not a preference: text-only weights are ~19.6 GB (14.2 GB of NVFP4 layers plus an UNTIED
  # bf16 embed and lm_head at 2.69 GB each, both ignore-listed by the quantizer), so it does not fit
  # on one 16 GB card at all. serve.sh cannot enforce TP, so it is stated here and in
  # docs/MUSE_GLIMMER_PORT.md rather than left to OOM at load.
  # tool_format=atem pins its native <atem:invoke> XML. Auto-derivation reaches the same answer from
  # the template, but pinning keeps a FORCED tool call off the JSON fallback.
  muse|muse-glimmer|RedHatAI/Muse-Glimmer-30B-NVFP4)
                  model_id="RedHatAI/Muse-Glimmer-30B-NVFP4";          spec_default="none"
                  tool_format="atem"; min_tp=2
                  # 0.85 mirrors Laguna, the other NVFP4 SWA hybrid: ~9.8 GB/card of weights leaves
                  # room for a real KV pool. Validated booting + serving at TP=2 (83% VRAM).
                  mem_default="0.85"
                  # DFlash block-diffusion drafter, block_size 16 -> at most 15 drafts per step
                  # (the block is [anchor, 15 masks]), captured from target layers 1/13/25/37/49.
                  dflash_draft="meta-models/Muse-Glimmer-30B-assistant"; k_dflash=15
                  # This drafter is 2.556B / 5.11 GB bf16 — SEVEN TIMES the 737 MB one that already
                  # forced qwen35b onto fp8, and `_PlainLinear` replicates it on EVERY rank (by
                  # design: identical argmax per rank means drafts stay in sync with no collective).
                  # bf16 would leave ~1.1 GB/card for KV+graphs, i.e. it cannot boot. fp8 halves it
                  # to ~2.56 GB. Lossless in the sense that matters: the target verifies every draft
                  # token, so weight-only quant can cost ACCEPTANCE, never correctness.
                  if [[ "${SPEC:-$spec_default}" == "dflash" ]]; then
                    # 0.96, NOT the 0.93 this line used to carry. 0.93 was Laguna's validated spec
                    # ratio, copied across with the 0.85 above it; it was never measured on Muse and
                    # it does not boot. This target is far heavier: the load delta is 12.58 GiB/card
                    # (measured 2026-08-13, TP=2), and the four reserves below it — 0.02 snapshot +
                    # 0.12 SWA ring (4 slots) + 1.77 drafter + 0.07 graphs = 1.98 GiB — are ALL
                    # subtracted from `memory_ratio * free`, so 0.93 leaves 14.56 - 12.58 - 1.98 =
                    # 0.00 GiB and dies in engine.py's `num_pages > 1` assert. It misses by single
                    # -digit MiB, which is why it reads as a plausible number that simply crashes.
                    # 0.96 leaves 0.47 GiB = 151,200 KV tokens, and boots: graph capture at
                    # avail_mem 3.06 GiB, verify graphs at free 1.78 GiB, VRAM 90%/89%, generation
                    # verified. Do not lower this without re-measuring the load delta.
                    mem_default_spec="0.96"
                    # nvfp4, NOT fp8. Measured: fp8 reserves 3.08 GiB and the KV pool cannot be
                    # sized at all (num_pages assert), even at CONC=2 / ratio 0.95 — so fp8 is not a
                    # viable step here, it simply does not boot. 4-bit weights + an fp16 per-16 group
                    # scale bring the drafter to ~1.61 GiB (down_proj held at fp8, mirroring Meta's
                    # own GGUF, which is Q4_K everywhere except ffn_down at Q6_K).
                    : "${MINISGL_DFLASH_QUANT:=nvfp4}"; export MINISGL_DFLASH_QUANT
                    # BIDIRECTIONAL block mask, and this MUST be stated here because the derived
                    # default is the opposite. The per-layer rule (vllm qwen3_dflash.py /
                    # sglang models/dflash.py: `sliding_attention` -> causal) is right for the z-lab
                    # Qwen drafters it was written for and for Laguna, but WRONG for this one, and
                    # nothing in the checkpoint distinguishes them — Laguna and Muse are both
                    # all-`sliding_attention`, and neither declares `dflash_config.causal` or
                    # `sliding_window_non_causal`. So it is a MEASURED per-checkpoint fact, and it
                    # belongs in the table rather than as a code branch or a global default.
                    #
                    # Measured 2026-08-14, TP=2, long code/math probe, card sampling
                    # (temp 1.0 / top_p 0.95 / top_k 64), fixtures in
                    # /home/pat/fixtures/minisgl-dflash-matrix/museab-*:
                    #   bidirectional + window 2048 (captured)  39.40 tok/s  accept-len 3.01
                    #   baseline: bidirectional + UNBOUNDED (eager) 37.50 tok/s  accept-len 2.70
                    #   causal + window 2048 (captured)         36.06 tok/s  accept-len 2.56
                    # The model card agrees with the measurement: the drafter "predicts entire
                    # blocks of 16 tokens in a single forward pass" — block diffusion is
                    # bidirectional WITHIN the block by construction.
                    #
                    # Without this line the drafter runs the causal mask and reads as a ~5% tok/s
                    # and ~16% accept-len REGRESSION against today's serve, with nothing to point at.
                    : "${MINISGL_DFLASH_CAUSAL:=0}"; export MINISGL_DFLASH_CAUSAL
                  fi
                  swa_hybrid=1 ;;
  # Qwen3.6-27B + the z-lab DFlash drafter. The drafter is the TIED-VOCAB z-lab dialect (no own
  # embed/lm_head/d2t — it borrows the target's), non-causal and UNWINDOWED, so it attends its whole
  # prefix bidirectionally and takes the EAGER per-uid propose path rather than the captured ring
  # (spec/dflash.py: that is a checkpoint property, not a switch). Its config states no block_size,
  # so block = SPEC_K + 1 and k_dflash=15 gives the block of 16 the other DFlash pairs use.
  #
  # This arm carried NO spec defaults, on the claim that the 27B INT4 target costs "~6.8 GiB/card at
  # TP=2" and so a bf16 drafter "fits with room". That arithmetic was wrong: the checkpoint is
  # 19.1 GiB of safetensors, i.e. >=9.5 GiB/card BEFORE the replicated embed/lm_head and the vision
  # tower — so `MODEL=qwen27b SPEC=dflash` at the global 0.80 default dies in engine.py's
  # `num_pages > 1` assert, having reserved 0.17 (GDN state) + 1.67 (drafter) + 0.04 (graphs) +
  # 0.04 (snapshot store) GiB with nothing left to page. The pair HAS a validated operating point —
  # docs/measurements/DFLASH_GDN_RINGGATE_RETAKE.md §3 measured it at fp8 + CONC=2 + MEM_RATIO=0.90
  # — but that lived only in the doc, so every launch that did not retype the magic numbers failed.
  # Encode them here, like qwen35b-awq (0.86) and laguna (0.93) already do.
  qwen27b|cyankiwi/Qwen3.6-27B-AWQ-INT4)
                  model_id="cyankiwi/Qwen3.6-27B-AWQ-INT4";            spec_default="none"
                  dflash_draft="z-lab/Qwen3.6-27B-DFlash"; k_dflash=15
                  if [[ "${SPEC:-$spec_default}" == "dflash" ]]; then
                    mem_default_spec="0.90"
                    # fp8 is not optional here either: it halves the 3.07 GiB bf16 drafter to a
                    # 1.67 GiB reserve. Bf16 would add ~0.9 GiB, which is more than 0.90 leaves.
                    : "${MINISGL_DFLASH_QUANT:=fp8}"; export MINISGL_DFLASH_QUANT
                    # A CAP, not a default (CONC is already assigned above the case block). 2 is the
                    # only concurrency this pair was measured booting at; GRAPH_BS follows CONC, so
                    # capping admission keeps capture coverage equal to what is admitted.
                    if [ "$CONC" -gt 2 ]; then CONC=2; fi
                  fi ;;
  qwen27b-nvfp4|cyankiwi/Qwen3.6-27B-AWQ-BF16-NVFP4)
                  model_id="cyankiwi/Qwen3.6-27B-AWQ-BF16-NVFP4";      spec_default="none"
                  dflash_draft="z-lab/Qwen3.6-27B-DFlash"; k_dflash=15 ;;
  # Qwen3.8-27B, unsloth's MIXED-PRECISION quant: NVFP4 (group-16 e2m1) for the bulk MLP, fp8 W8A8
  # for attention / GDN in_proj / lm_head / the last 8 MLP layers. Same backbone as the 3.6-27B above
  # (64L, hidden 5120, head_dim 256, 24q/4kv, GDN hybrid at full_attention_interval=4), so it rides
  # the existing Qwen3_5 model; what it needed was per-MODULE quant resolution (quant/config.py
  # `for_module`) — one config-wide scheme cannot describe it.
  #
  # The MIXED NVFP4+fp8 build. NOT the default: qwen38-27b (sakamakismile, all-NVFP4) is 1.65
  # GiB/card lighter for the same probe score, so this is kept only as the higher-precision
  # alternative — its GDN/attention stay fp8, and it drafts BETTER under MTP (0.616 vs 0.551
  # accept), so prefer it when spec throughput matters more than KV pool.
  # Settings follow the qwen27b defaults above (attn=hip, TP=2, conc 4, page 16, chunk 2048) —
  # same backbone — EXCEPT mem_default, which 0.80 cannot satisfy. MEASURED at boot, TP=2, 2026-08-15:
  #     resident weights 11.62 GiB/card  (vs the INT4 27B's ~6.8: NVFP4+fp8 on a bigger effective
  #                                       footprint, plus a bf16 lm_head and the 0.85 GB MTP head)
  #     + ~2.37 GiB/card consumed OUTSIDE torch (torch's own reserved-minus-allocated is just
  #       0.19 GiB, so it is not slack the loader or an empty_cache can recover). It is NOT a fixed
  #       per-process cost: qwen27b on the same box/image pays only 0.06 GiB. It is also NOT spec —
  #       measured at 2.39 GiB with SPEC=none. It tracks this checkpoint's KERNEL SET, which pulls in
  #       BOTH the fp8 W8A8 and the NVFP4/e2m1 paths where the INT4 27B only uses AWQ int4; the
  #       suspect is HIP code objects / kernel workspaces. Unreduced — if someone shrinks it, this
  #       ratio should come back down and the KV pool grows with it.
  #     => the free-memory delta the engine bills as `model` is 14.18 of 15.67 GiB free.
  # At 0.80 the KV pool sizes NEGATIVE (-2.18 GiB) and boot dies in _determine_num_pages. 0.95 leaves
  # 0.39 GiB => 25,296 KV tokens, which is the practical context ceiling here (NOT the checkpoint's
  # 262144) and is why conc stays at 4. Raise TP or drop conc before raising this further.
  #
  # min_tp=2 is a GUARD, not a default change (TP already defaults to 2): 22.6 GB of weights plus the
  # MTP head cannot fit one 16 GB card at any memory ratio, and the panel exposes TP as a free dropdown.
  #
  # SPEC=mtp: the checkpoint ships a full bf16 MTP head as a SEPARATE model_mtp.safetensors, listed
  # in the index under canonical `mtp.*` keys, so the normal loader picks it up with no sidecar path.
  # Same Qwen3.8-27B, quantized ALL-NVFP4 rather than unsloth's NVFP4+fp8 mix: the GDN in_proj /
  # out_proj and attention q/k/v/o are 4-bit here where unsloth keeps them fp8, and lm_head ships
  # plain bf16 (no dequant needed — and bf16 is what the M-invariant LM-head GEMV wants anyway).
  # 20.59 GB on disk vs 23.44, i.e. ~9.8 GiB/card vs 11.62, which is a much larger KV pool.
  # The MTP head is a separate bf16 file (model-mtp-bf16.safetensors), canonical `mtp.*` keys.
  # ACCURACY CAVEAT: 4-bit GDN input projections are exactly what unsloth declined to do, and the
  # int4-GDN Qwen3.6-AWQ is the one that degenerated in the quality probe. Treat as UNPROVEN until
  # A/B'd against qwen38-27b on the same prompts.
  # Qwen3.8-27B quantized INT4 (compressed-tensors pack-quantized, group-32, asymmetric) by the same
  # publisher as the 3.6-27B INT4. Same backbone; lm_head ships bf16 so nothing is dequantized.
  # Its ignore list carries the `...linear_attn` CONTAINER while shipping in_proj_qkv quantized, so
  # it depends on the structural quant oracle (QuantConfig.ckpt_quantized) — under the old substring
  # matching it died in the GDN concat with KeyError in_proj_qkvz.weight.
  # NOT THE DEFAULT, despite being the fastest — see the corruption note on the qwen38-27b
  # arm below. THROUGHPUT, single-stream 400-token decode, TP=2, 2026-08-15 (median of 3-4):
  #   Qwen3.6-27B INT4        31.9 tok/s
  #   Qwen3.8 INT4 (this)     31.8 tok/s   <- default
  #   Qwen3.8 INT4 + MTP      31.3 tok/s
  #   Qwen3.8 all-NVFP4       24.4 tok/s   (-23%: the e2m1 decode path costs more than
  #                                         int4 W4A8 on this card; repeatable to 0.1s)
  # So Qwen3.8 is NOT inherently slower than 3.6 — the all-NVFP4 build is, and it had been
  # the default because it was chosen on footprint + quality without measuring tok/s.
  # spec_default=none: MTP is a WASH here (31.3 vs 31.8) despite ~0.55 accept / 3.3 tok-step,
  # i.e. the per-step cost eats the accept-length win — the same result the repo already
  # recorded for MTP under sampling. Still selectable with SPEC=mtp; it works, it just does
  # not pay. NOTE: spec is also auto-disabled whenever batch>1, so any concurrent traffic
  # turns it off regardless.
  qwen38-27b-int4|cyankiwi/Qwen3.8-27B-AWQ-INT4)
                  model_id="cyankiwi/Qwen3.8-27B-AWQ-INT4";            spec_default="none"; k_mtp=4
                  # MEASURED TP=2 2026-08-15: resident 9.47 GiB/card, the LIGHTEST of the three
                  # Qwen3.8 builds. 0.92, NOT the 0.97 the other two carry: because this model is so
                  # light the pool grows to fill whatever the ratio allows, and at 0.97 it took
                  # 341,936 tokens and left 42 MiB — graph capture then OOM'd. The `graph` reserve
                  # (0.03 GiB) badly under-counts what capture actually needs, so a LIGHTER model
                  # needs a LOWER ratio, not a higher one. 0.92 => 290,576 tokens, captured with
                  # 0.71 GiB spare. Validated: 20/20 sampled probe, 0 degeneration, 5.4k prefill,
                  # MTP 0.550 accept / 3.29 tok-step (the best per-step of the three).
                  mem_default="0.92"; alloc_conf="expandable_segments:True"; min_tp=2 
                  # DSpark drafter: the DFlash backbone plus a rank-256 Markov (bigram) logit bias
                  # that decodes the block SEMI-AUTOREGRESSIVELY — each position is biased by the
                  # token chosen at the previous one. That is the whole point: DFlash scores a block
                  # in one forward, so its positions are conditionally independent and acceptance
                  # decays toward the back of the block; one-step dependency restores it.
                  # config block_size=7 and this path drafts B-1, hence k=6.
                  # SPEC=dspark is the canonical way to pick it (an alias of the dflash engine
                  # path — the proposer detects the Markov + confidence heads from the checkpoint's
                  # own tensors); the confidence head then truncates each block where predicted
                  # survival crosses MINISGL_DSPARK_CONF_TAU (adaptive draft length).
                  # Trained against Qwen3.8-27B-FP8 while every target here is 4-bit, and the draft
                  # consumes TARGET hidden states (layers 4/16/28/40/52) — so expect acceptance
                  # BELOW its published 3.35. The 2.7 GB bf16 drafter is REPLICATED per rank, which
                  # is what the lower spec ratio pays for.
                  # MEASURED 2026-08-19 (coding prompts, SAMPLED seeds 101/202/303, M=1, 1200-tok,
                  # TP=2, graphs on, ledger-verified markov+confidence engaged): NET-NEGATIVE at
                  # M=1 on the INT4 arm — plain 38.85 tok/s vs dspark k=3/block-4 34.81 (0.93
                  # drafts/step), k=6 33.80 (0.95), k=6+tau0.5 ~30, MTP 34.25 (1.08). The FP8-
                  # trained drafter accepts ~1 draft/step against 4-bit target hidden, and the
                  # ~5 GB/step propose weight-stream eats the win. So spec_default stays none;
                  # SPEC=dspark remains selectable and correct (12-probe degen pass clean).
                  # MINISGL_DFLASH_QUANT=fp8 on this drafter is WORSE (27.0 tok/s) — never use.
                  # tau=0 (head computed, never cutting) is fastest at M=1; the default tau=0.5
                  # only pays where verify width costs, i.e. batched spec (MINISGL_SPEC_MAX_BS>1).
                  dflash_draft="RadixArk/Qwen3.8-27B-DSpark"; k_dflash=6
                  if [[ "${SPEC:-$spec_default}" =~ ^(dflash|dspark)$ ]]; then mem_default_spec="0.88"; fi ;;
  # DEFAULT ON CORRECTNESS, NOT SPEED. The INT4 arm is ~23% faster (31.8 vs 24.4 tok/s) and was
  # briefly the default for that reason, until a Hermes agent session surfaced token-level
  # CORRUPTION on it: foreign script fused mid-word ("neutral<cyrillic>", "<hangul>"), repeated
  # fragments ("st collidingding"), a spurious injected token, a stray "</".
  # Scanning ~120 archived Hermes sessions for that exact signature (non-Latin script FUSED to a
  # Latin letter, which legitimate foreign text never produces):
  #     locally-served cyankiwi *-AWQ-INT4 :  6 corrupted / 19 sessions  (~32%)
  #     everything else                    :  0 corrupted / ~100 sessions
  # Zero on 60+ HOSTED-model sessions through the same client (so not Hermes), and zero on local
  # NVFP4 (Laguna, Muse) and GGUF (so not local serving generally). Recurring 08-03 -> 08-16, i.e.
  # it predates this table entry. NOT explained by the missing kv_cache_scheme those checkpoints
  # share — Muse and Qwen3.6-35B have the same gap and stay clean. Root cause still OPEN, and it
  # did NOT reproduce from plain requests (short, 15k-token prompt, 3.5k-token generation,
  # streamed and not): it appears to need agent-shaped context and to be stochastic.
  # Until that is understood, the default takes the slower quant family with no such history.
  # fp8-KV: this checkpoint ships NO kv_cache_scheme, so without a sidecar the cache stores k/v
  # at e4m3 with 1.0 scales (the engine warns). CALIBRATED per-head scales were generated and
  # installed at <snapshot>/kv_scales.safetensors, which fp8_scales auto-discovers — the boot
  # log then says "installed PER-HEAD scales from sidecar" instead of the uncalibrated warning.
  # It earns its keep here: measured V amax ranges 6.844 -> 133 ACROSS LAYERS and the per-head
  # spread is up to 2.36x, none of which a single 1.0 scale can represent.
  #
  # THE SIDECAR IS NOT IN THIS REPO and lives in the HF snapshot dir, so a re-download SILENTLY
  # reverts to uncalibrated (a warning in the log, not an error). Durable copy + report:
  #   /home/pat/fixtures/minisgl-kv-calib/sidecars/qwen38-27b-nvfp4_tp2.{safetensors,report.json}
  # Restore:  cp <that>.safetensors <snapshot>/kv_scales.safetensors
  # Regenerate (under a 2-card lease, in the lean image, with MINISGL_KV_FP8=0 so the amax is
  # measured through a bf16 cache — see the calibrator docstring):
  #   python tools/kv_fp8_calibrate.py --model sakamakismile/Qwen3.8-27B-MTP-NVFP4 --tp 2 \
  #     --text /home/pat/fixtures/minisgl-kv-calib/kv_calib_v1.txt --out <sidecar> --report <json>
  # Costs NOTHING in context: pool is unchanged at 247,264 tokens with the sidecar installed.
  # Does NOT measurably change MTP acceptance (0.563 median calibrated vs 0.551 uncalibrated,
  # inside a 0.490-0.599 spread) — the draft head reads the same KV as the target, so cleaner
  # KV moves both sides together. It is an OUTPUT-QUALITY fix, not a spec-acceptance one.
  qwen38-27b|sakamakismile/Qwen3.8-27B-MTP-NVFP4)
                  model_id="sakamakismile/Qwen3.8-27B-MTP-NVFP4";      spec_default="none"; k_mtp=4
                  # MEASURED 2026-08-16: long THINKING (>1024 reasoning tokens, newly reachable now
                  # the think budget honours the template's reasoning_effort) hits the card's
                  # documented "endless repetition" at the card's own thinking sampling
                  # (t=1.0/p=0.95/k=20): 1/4 probe runs looped (1540-char repeat), plus a live
                  # agent session (4000-char `**ai**ai**` loop; prose after it degraded — dropped
                  # sentence-final periods). With presence_penalty=1.0 (card remedy range 0..2;
                  # its non-thinking preset ships 1.5): 0/4 loops, longer coherent outputs.
                  # REVERTED same day: presence=1.0 replaced the loop with a WORSE failure —
                  # presence penalty is one-shot per token IDENTITY over the whole output, so a
                  # long thinking turn saturates the common-token space and the model degenerates
                  # into letter-by-letter spelling ("T\nh\ne\nm\na\ni\n n", live session
                  # 78cd9d018610 at ~5k chars). The 0/4-loops validation was BLIND: its metrics
                  # (repeat-run + missing-period regex) cannot see letter-spelling, and the arm-B
                  # text was never read. The card's own presets encode the constraint — presence
                  # 1.5 for SHORT non-thinking answers, 0.0 for thinking — penalties of this form
                  # do not mix with long generations. Loop root cause still open (top-k sampler
                  # kernel under test; fp8-KV A/B pending); do NOT re-apply an unbounded-window
                  # penalty here.
                  presence_default=""
                  # MEASURED TP=2 2026-08-15: resident 9.97 GiB/card (vs unsloth's 11.62) -> 0.97
                  # leaves 3.27 GiB = 214,016 KV tokens, 4.2x the unsloth serve. Validated: 20/20 on
                  # the sampled quality probe with ZERO degeneration, a 5.4k-token prefill, and MTP
                  # engaged. The feared 4-bit-GDN quality loss did NOT appear.
                  # TRADE: MTP acceptance is consistently LOWER than unsloth's — 0.551 vs 0.616
                  # (same prompt), 2.72 vs 3.03 tok/step. A more heavily quantized target agrees
                  # with its own draft head less often, so some of the memory win is paid back in
                  # spec throughput. Single-sample measurements; re-measure before relying on it.
                  mem_default="0.97"; alloc_conf="expandable_segments:True"; min_tp=2 
                  # DSpark drafter: the DFlash backbone plus a rank-256 Markov (bigram) logit bias
                  # that decodes the block SEMI-AUTOREGRESSIVELY — each position is biased by the
                  # token chosen at the previous one. That is the whole point: DFlash scores a block
                  # in one forward, so its positions are conditionally independent and acceptance
                  # decays toward the back of the block; one-step dependency restores it.
                  # config block_size=7 and this path drafts B-1, hence k=6.
                  # SPEC=dspark is the canonical way to pick it (an alias of the dflash engine
                  # path — the proposer detects the Markov + confidence heads from the checkpoint's
                  # own tensors); the confidence head then truncates each block where predicted
                  # survival crosses MINISGL_DSPARK_CONF_TAU (adaptive draft length).
                  # Trained against Qwen3.8-27B-FP8 while every target here is 4-bit, and the draft
                  # consumes TARGET hidden states (layers 4/16/28/40/52) — so expect acceptance
                  # BELOW its published 3.35. The 2.7 GB bf16 drafter is REPLICATED per rank, which
                  # is what the lower spec ratio pays for.
                  # MEASURED 2026-08-19 (coding prompts, SAMPLED seeds 101/202/303, M=1, 1200-tok,
                  # TP=2, graphs on, ledger-verified markov+confidence engaged): NET-NEGATIVE at
                  # M=1 on the INT4 arm — plain 38.85 tok/s vs dspark k=3/block-4 34.81 (0.93
                  # drafts/step), k=6 33.80 (0.95), k=6+tau0.5 ~30, MTP 34.25 (1.08). The FP8-
                  # trained drafter accepts ~1 draft/step against 4-bit target hidden, and the
                  # ~5 GB/step propose weight-stream eats the win. So spec_default stays none;
                  # SPEC=dspark remains selectable and correct (12-probe degen pass clean).
                  # MINISGL_DFLASH_QUANT=fp8 on this drafter is WORSE (27.0 tok/s) — never use.
                  # tau=0 (head computed, never cutting) is fastest at M=1; the default tau=0.5
                  # only pays where verify width costs, i.e. batched spec (MINISGL_SPEC_MAX_BS>1).
                  dflash_draft="RadixArk/Qwen3.8-27B-DSpark"; k_dflash=6
                  if [[ "${SPEC:-$spec_default}" =~ ^(dflash|dspark)$ ]]; then mem_default_spec="0.88"; fi ;;
  qwen38-27b-mixed|unsloth/Qwen3.8-27B-NVFP4)
                  # No dflash_draft: the z-lab 27B drafter is the TIED-VOCAB dialect (it borrows the
                  # target's embed/lm_head), so pairing it across a model generation is only valid if
                  # the vocabularies match — unverified here. SPEC=dflash therefore requires DRAFT=.
                  model_id="unsloth/Qwen3.8-27B-NVFP4";                spec_default="none"; k_mtp=4
                  # Same base model + card as qwen38-27b above — see that entry: the presence
                  # default was applied and REVERTED same day (long-generation starvation).
                  presence_default=""
                  mem_default="0.97"; alloc_conf="expandable_segments:True"; min_tp=2 
                  # DSpark drafter: the DFlash backbone plus a rank-256 Markov (bigram) logit bias
                  # that decodes the block SEMI-AUTOREGRESSIVELY — each position is biased by the
                  # token chosen at the previous one. That is the whole point: DFlash scores a block
                  # in one forward, so its positions are conditionally independent and acceptance
                  # decays toward the back of the block; one-step dependency restores it.
                  # config block_size=7 and this path drafts B-1, hence k=6.
                  # SPEC=dspark is the canonical way to pick it (an alias of the dflash engine
                  # path — the proposer detects the Markov + confidence heads from the checkpoint's
                  # own tensors); the confidence head then truncates each block where predicted
                  # survival crosses MINISGL_DSPARK_CONF_TAU (adaptive draft length).
                  # Trained against Qwen3.8-27B-FP8 while every target here is 4-bit, and the draft
                  # consumes TARGET hidden states (layers 4/16/28/40/52) — so expect acceptance
                  # BELOW its published 3.35. The 2.7 GB bf16 drafter is REPLICATED per rank, which
                  # is what the lower spec ratio pays for.
                  # MEASURED 2026-08-19 (coding prompts, SAMPLED seeds 101/202/303, M=1, 1200-tok,
                  # TP=2, graphs on, ledger-verified markov+confidence engaged): NET-NEGATIVE at
                  # M=1 on the INT4 arm — plain 38.85 tok/s vs dspark k=3/block-4 34.81 (0.93
                  # drafts/step), k=6 33.80 (0.95), k=6+tau0.5 ~30, MTP 34.25 (1.08). The FP8-
                  # trained drafter accepts ~1 draft/step against 4-bit target hidden, and the
                  # ~5 GB/step propose weight-stream eats the win. So spec_default stays none;
                  # SPEC=dspark remains selectable and correct (12-probe degen pass clean).
                  # MINISGL_DFLASH_QUANT=fp8 on this drafter is WORSE (27.0 tok/s) — never use.
                  # tau=0 (head computed, never cutting) is fastest at M=1; the default tau=0.5
                  # only pays where verify width costs, i.e. batched spec (MINISGL_SPEC_MAX_BS>1).
                  dflash_draft="RadixArk/Qwen3.8-27B-DSpark"; k_dflash=6
                  if [[ "${SPEC:-$spec_default}" =~ ^(dflash|dspark)$ ]]; then mem_default_spec="0.88"; fi ;;
  zaya|*/ZAYA1-8B-fp8|ZAYA1-8B-fp8)
                  model_id="${ZAYA_MODEL:-/models/ZAYA1-8B-fp8}";      spec_default="none"
                  tool_format="zaya_xml"
                  dflash_draft="/drafts/ZAYA1-8B-DFlash-CCA-5L-minv-ep4"; k_dflash=4 ;;
  *)              model_id="$MODEL" ;;     # any other HF id or local path, straight through
esac
[[ -z "$SPEC" ]] && SPEC="$spec_default"
# Refuse a TP the weights cannot fit in, rather than OOM'ing several minutes into the load. Says
# what to do, because the panel's TP dropdown is where this gets chosen wrongly.
if [ "$TP" -lt "$min_tp" ]; then
  echo "[serve] ERROR: $model_id needs TP>=$min_tp (its weights do not fit on $TP card(s) at 16 GB);" \
       "got TP=$TP. Set TP=$min_tp." >&2
  exit 2
fi

ATTN="${ATTN:-$attn}"
# Laguna's 0.85 default leaves too little KV pool once a DFlash drafter AND the verify graphs are
# also resident: `MODEL=laguna SPEC=dflash` at TP=2 with graph capture dies in engine.py's
# `num_pages <= 1` check before serving a single token. Measured: 0.85 fails, 0.93 boots. The
# per-model default is sized for PLAIN decode, so raise it when a draft model has to fit beside it.
if [[ -n "${mem_default_spec:-}" && "$SPEC" != "none" && -n "$SPEC" ]]; then
  mem_default="$mem_default_spec"
fi
MEM_RATIO="${MEM_RATIO:-$mem_default}"
# Graph capture must COVER the concurrency you serve — a request above the captured batch size falls
# back to eager and the served config is no longer the measured one. But covering it is enough: the
# decode batch can never exceed --max-running-requests (= CONC), so any captured size ABOVE CONC can
# never be selected. It is pure waste, and the waste is VRAM at exactly the moment VRAM is tightest
# (capture runs after the KV pool is sized). The old `floor of 8` captured bs=2/4/8 for a CONC=1
# serve and bs=8 for CONC=4 — graphs that could not be reached.
#
# Spec decode does NOT need the larger sizes either, which is the non-obvious part: the verify
# capture derives its own list and ALREADY clamps it (scheduler.py: `verify_bs = [b for b in
# graph_bs_list if b <= config.max_running_req]`), and the propose capture filters the same list
# against its own row cap. So all three capture families follow CONC; only plain decode was reading
# the inflated value.
GRAPH_BS="${GRAPH_BS:-$CONC}"
PAGE_SIZE="${PAGE_SIZE:-16}"
CACHE_TYPE="${CACHE_TYPE:-radix}"
# Chunked-prefill chunk size. 2048, NOT the engine's 8192 default, because the engine default is
# unsafe on wide-activation models: gemma-4-26B-A4B (hidden 2816, 2*inter 1408, top_k 8, vocab 262144)
# reproducibly killed the scheduler worker mid-forward at 8192 — a ~7k-token prompt arrives as ONE
# chunk and the forward cannot fit in the ~2.85 GiB left after the KV pool. MEASURED 2026-08-07
# (MINISGL_PREFILL_MEM_PROBE=1, TP=2): at a 2048 cap the peak is ~186 MiB per chunk and FLAT in
# context, and `reserved` STOPS ratcheting (13476 -> 13646 -> 13646) because uniform block sizes let
# the caching allocator reuse segments; at 8192 the same prompt OOM'd on an 18 MiB allocation with
# devfree at 0. Note the scheduler's own activation guard cannot prevent this — it only shrinks once
# free memory is ALREADY low, so it never trips on the first oversized chunk.
# Qwen3.6-35B ran fine at 8192 and is capped here only for uniformity; raise per-launch with
# MAX_PREFILL_LENGTH=8192 if a model wants bigger chunks and has the headroom.
MAX_PREFILL_LENGTH="${MAX_PREFILL_LENGTH:-2048}"

# --- spec decode ---------------------------------------------------------------------------------
spec_args=()
# A draft checkpoint is REQUIRED by dflash and eagle3, and meaningless for the others: mtp reads the
# MTP head out of the target checkpoint, tidar reads tidar_config.json from the target, and ngram has
# no model at all. DRAFT= overrides whatever the table resolved.
need_draft=""; resolved_draft=""
case "$SPEC" in
  dflash|dspark) need_draft=1; resolved_draft="${DRAFT:-$dflash_draft}" ;;
  eagle3)        need_draft=1; resolved_draft="${DRAFT:-$eagle3_draft}" ;;
esac
if [[ -n "$need_draft" && -z "$resolved_draft" ]]; then
  echo "serve.sh: SPEC=$SPEC needs a draft checkpoint and MODEL=$MODEL has no default for it." >&2
  echo "serve.sh: set DRAFT=<hf-id or /drafts/... path>, or pick a different SPEC." >&2
  echo "serve.sh: models with a built-in draft: qwen35b-awq qwen35b-mxfp4 (dflash),"  >&2
  echo "serve.sh:                               glm (eagle3), laguna zaya (dflash)."  >&2
  exit 2
fi
case "$SPEC" in
  none|"") : ;;
  mtp)     spec_args=(--spec-algorithm mtp    --spec-num-draft "${SPEC_K:-$k_mtp}") ;;
  ngram)   spec_args=(--spec-algorithm ngram  --spec-num-draft "${SPEC_K:-4}") ;;
  tidar)   spec_args=(--spec-algorithm tidar  --spec-num-draft "${SPEC_K:-$k_tidar}") ;;
  eagle3)  spec_args=(--spec-algorithm eagle3 --spec-draft-model-path "$resolved_draft"
                      --spec-num-draft "${SPEC_K:-$k_eagle3}") ;;
  # dspark IS the dflash engine algorithm — the DFlashProposer detects the DSpark Markov +
  # confidence heads from the draft checkpoint's own tensors, so the engine flag stays `dflash`.
  # The separate SPEC name exists so the launch line SAYS which drafter family was asked for and
  # so the qwen38 arms' tuned defaults key on it.
  dflash|dspark)  spec_args=(--spec-algorithm dflash --spec-draft-model-path "$resolved_draft"
                      --spec-num-draft "${SPEC_K:-$k_dflash}") ;;
  *) echo "serve.sh: unknown SPEC='$SPEC' (none|mtp|dflash|dspark|eagle3|ngram|tidar)" >&2; exit 2 ;;
esac

# --- sampled speculative verify ------------------------------------------------------------------
# ON by default whenever spec-decode is active. The engine defaults MINISGL_SPEC_SAMPLED off, and
# with it off spec-decode only runs for GREEDY requests — every temperature>0 request silently falls
# back to plain decode. Real traffic is sampled, so leaving this off means the spec machinery is
# carried but does nothing for the requests that actually arrive. It also makes the served config
# match how spec is measured: greedy verify inflates acceptance at LATE draft positions and
# over-recommends K, so a greedy-only serve is both slower and measured wrong.
if [[ "$SPEC" != "none" && -n "$SPEC" ]]; then
  export MINISGL_SPEC_SAMPLED="${MINISGL_SPEC_SAMPLED:-1}"
fi
# Per-model engine env that used to live in the deleted per-model services.
[[ -n "$tool_format" ]] && export MINISGL_TOOL_FORMAT="${MINISGL_TOOL_FORMAT:-$tool_format}"
[[ -n "$presence_default" ]] && \
  export MINISGL_DEFAULT_PRESENCE_PENALTY="${MINISGL_DEFAULT_PRESENCE_PENALTY:-$presence_default}"

# --- SWA-hybrid prefix caching ------------------------------------------------------------------
# A sliding-window model downgrades `--cache-type radix` to `naive` unless MINISGL_SWA_RADIX is on
# (scheduler.py feature-flags it, default off), which is why an SWA model silently logs
#   "SWA-hybrid model: forcing prefix cache 'naive' (was 'radix'); SWA-radix disabled"
# and loses prefix reuse. MINISGL_SPEC_MHA_PAGED=1 is the second half: SWA-radix only HITS on
# page-16-aligned snapshot boundaries — measured on the TP=2 serve with an identical 1216-token
# shared prefix, ps=1 gave 0 hits (a SILENT miss) and ps=16 gave 2 hits with warm TTFT dropping.
# Both were set per-service on the old laguna services; they belong to the MODEL, so they live here.
# An explicit value from the caller/panel always wins, and both are inert on non-SWA models.
if [[ -n "$swa_hybrid" ]]; then
  export MINISGL_SWA_RADIX="${MINISGL_SWA_RADIX:-1}"
  export MINISGL_SPEC_MHA_PAGED="${MINISGL_SPEC_MHA_PAGED:-1}"
  # Both are REQUIRED here, and a 0 in either fails SILENTLY — the cache is downgraded, or it is
  # "enabled" and never hits. Say so loudly rather than let the serve look healthy.
  # NOTE the compose lean-env anchor SETS MINISGL_SPEC_MHA_PAGED, so `${VAR:-1}` above cannot
  # default it: an inherited value always wins. Its compose default is therefore 1, not 0.
  [[ "$MINISGL_SWA_RADIX" == "0" ]] && echo \
    "[serve] WARNING: MINISGL_SWA_RADIX=0 on an SWA-hybrid model — prefix cache falls back to naive." >&2
  [[ "$MINISGL_SPEC_MHA_PAGED" == "0" ]] && echo \
    "[serve] WARNING: MINISGL_SPEC_MHA_PAGED=0 on an SWA-hybrid model — SWA-radix will NEVER hit." >&2
fi

# --- parallelism ---------------------------------------------------------------------------------
# TP and DP are counts, not device ids — the GPU lease decides WHICH cards are visible. EP is a flag
# that only means anything on a MoE model; passing it elsewhere is harmless but pointless.
par_args=(--tp "$TP")
[[ "$DP" -gt 1 ]] && par_args+=(--dp-size "$DP")
[[ "$EP" == "1" ]] && par_args+=(--enable-ep)
# pynccl is disabled for every multi-rank config on this box (custom_ar carries the collectives).
[[ "$TP" -gt 1 || "$DP" -gt 1 ]] && par_args+=(--disable-pynccl)

# --- context -------------------------------------------------------------------------------------
# Unset => the checkpoint's own max_position_embeddings. Set => override it, which is how you trade
# context length against KV pool depth at a fixed memory ratio.
ctx_args=()
[[ -n "${CTX:-}" ]] && ctx_args=(--max-seq-len-override "$CTX")

cmd=(python -m minisgl
  --model "$model_id"
  --host 0.0.0.0 --port "$PORT"
  --cache-type "$CACHE_TYPE"
  --attention-backend "$ATTN"
  --page-size "$PAGE_SIZE"
  "${par_args[@]}"
  --cuda-graph-max-bs "$GRAPH_BS"
  --max-running-requests "$CONC"
  --memory-ratio "$MEM_RATIO"
  --max-prefill-length "$MAX_PREFILL_LENGTH"
  "${ctx_args[@]}"
  "${spec_args[@]}"
)
# EXTRA_ARGS last so it can override anything above.
[[ -n "${EXTRA_ARGS:-}" ]] && read -r -a _extra <<< "$EXTRA_ARGS" && cmd+=("${_extra[@]}")

printf '[serve] model=%s spec=%s%s tp=%s dp=%s ep=%s ctx=%s conc=%s attn=%s mem=%s graph_bs=%s\n' \
  "$model_id" "$SPEC" "${SPEC_K:+ k=$SPEC_K}" "$TP" "$DP" "$EP" "${CTX:-checkpoint}" "$CONC" \
  "$ATTN" "$MEM_RATIO" "$GRAPH_BS" >&2
[[ -n "$swa_hybrid" ]] && printf '[serve] SWA-hybrid: MINISGL_SWA_RADIX=%s MINISGL_SPEC_MHA_PAGED=%s\n' \
  "$MINISGL_SWA_RADIX" "$MINISGL_SPEC_MHA_PAGED" >&2
[[ "$SPEC" != "none" && -n "$SPEC" ]] && printf '[serve] spec sampled=%s\n' "$MINISGL_SPEC_SAMPLED" >&2
[[ -n "${MINISGL_DEFAULT_PRESENCE_PENALTY:-}" ]] && \
  printf '[serve] default presence_penalty=%s (request-unset only)\n' "$MINISGL_DEFAULT_PRESENCE_PENALTY" >&2
# The DFlash drafter's block mask and weight format, ON THE BANNER. Both change acceptance
# materially and neither appears on the python command line, so without this the only way to tell
# which mask a run used is to read the engine's ring log — and a table entry that silently stopped
# applying reads as a model regression rather than a launch difference. `causal=` is empty when the
# table says nothing, i.e. the per-layer rule derived from `layer_types` is in force.
[[ "$SPEC" =~ ^(dflash|dspark)$ ]] && printf '[serve] %s: quant=%s causal=%s (empty causal = derived from layer_types)\n' \
  "$SPEC" "${MINISGL_DFLASH_QUANT:-<derived>}" "${MINISGL_DFLASH_CAUSAL:-<derived>}" >&2
# DSpark's adaptive draft length on the banner: tau changes accept-len and tok/s materially and
# never appears on the python command line.
[[ "$SPEC" == "dspark" ]] && printf '[serve] dspark: conf=%s tau=%s markov=%s\n' \
  "${MINISGL_DSPARK_CONF:-1}" "${MINISGL_DSPARK_CONF_TAU:-0.5}" "${MINISGL_DSPARK_MARKOV:-1}" >&2
printf '[serve] %s\n' "${cmd[*]}" >&2
# Per-model torch allocator config, exported for the python we exec below (a fresh process, so the
# allocator reads it at ITS import). ON THE BANNER because it changes the KV pool size materially and
# never appears on the command line — a table entry that silently stopped applying would read as a
# model regression. An explicit PYTORCH_HIP_ALLOC_CONF from the environment always wins.
if [[ -n "$alloc_conf" ]]; then
  export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-$alloc_conf}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-$alloc_conf}"
  printf '[serve] alloc_conf=%s\n' "$PYTORCH_HIP_ALLOC_CONF" >&2
fi
[[ -n "${DRY_RUN:-}" ]] && exit 0
exec "${cmd[@]}"
