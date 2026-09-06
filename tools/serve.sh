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
# WEIGHT OFFLOAD (weights/plan.py). Both empty by default, which puts NO flag on the line and is the
# shipped behaviour for every model in the table below.
#   WOFF_DEVICE_GB  VRAM per rank (GiB) the MoE expert tier may occupy. Every layer above it is
#                   pinned in host RAM and streamed over PCIe. ~200k KV tokens per GiB surrendered.
#   WOFF_HOST_GB    per-rank pinned host arena clamp (GiB). Empty = the measured P3b default.
#   WOFF_CHUNK_MIB  pinned-arena chunk size (MiB). Empty = the 2048 default. Worth 25% of the host
#                   tier when a layer's rows are just over half a chunk — see the block below.
#
#   NOT OPTIONAL ONCE OFFLOAD ENGAGES, which is why it is a table column and not just an EXTRA_ARG.
#   With no --weight-offload-device-gb the engine DERIVES the tier as `total_memory * memory_ratio`
#   — the whole KV budget — and since the tier is billed inside `model` in _determine_num_pages, a
#   non-empty plan under it can never leave a KV pool. `bake.UnconfiguredDeviceTierError` refuses
#   that boot from integers before a page is pinned, so a model that needs offload and has no
#   WOFF_DEVICE_GB does not serve slowly, it does not serve at all.
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
                  if [[ "${SPEC:-$spec_default}" =~ ^(dflash|dspark)$ ]]; then
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
                  dflash_draft="z-lab/Qwen3.6-35B-A3B-DFlash"; k_dflash=15
                  # Same 737 MB bf16 drafter beside a same-size 4-bit 35B target as the AWQ twin
                  # above, so the same footprint arithmetic applies: at the global 0.80 default a
                  # bf16 drafter does not boot (see that arm's bracketing). Mirrors the AWQ twin's
                  # measured point; not independently re-measured on this quant.
                  if [[ "${SPEC:-$spec_default}" =~ ^(dflash|dspark)$ ]]; then
                    mem_default_spec="0.86"
                    : "${MINISGL_DFLASH_QUANT:=fp8}"; export MINISGL_DFLASH_QUANT
                    # A CAP, not a default (CONC is already assigned above the case block): GRAPH_BS
                    # follows CONC, so capping admission keeps capture coverage equal to admission.
                    if [ "$CONC" -gt 4 ]; then CONC=4; fi
                  fi ;;
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
                  if [[ "${SPEC:-$spec_default}" =~ ^(dflash|dspark)$ ]]; then
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
                  if [[ "${SPEC:-$spec_default}" =~ ^(dflash|dspark)$ ]]; then
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
                  if [[ "${SPEC:-$spec_default}" =~ ^(dflash|dspark)$ ]]; then
                    # 0.84 + FULL_CAP=2048 is the DEGEN-VALIDATED point (2026-08-19, 12/12 clean,
                    # 20.7k spec steps, zero restarts, MINISGL_SPEC_MAX_BS=4). Uncapped, this
                    # all-full_attention drafter accumulates aux to 8206 rows = 420 MB PER REQUEST
                    # (torch.cat per step, scheduler.py _spec_decode_step) — measured killing the
                    # serve at 0.88 AND at 0.84 under 4 concurrent 12k-token generations (OOM at
                    # 0 bytes free). The cap bounds aux at ~105 MB/uid and shrinks the propose ring
                    # 504->126 MB; drafting conditions on the newest 2048 committed positions.
                    mem_default_spec="0.84"
                    : "${MINISGL_DFLASH_FULL_CAP:=2048}"; export MINISGL_DFLASH_FULL_CAP
                    # MEASURED 2026-08-25 (same protocol as the 08-19 row: sampled seeds
                    # 101/202/303, M=1, 1200 tok): MAX_BS=1 -> tau=0 serves 36.71 tok/s median vs
                    # 27.78 with MAX_BS=4 -> tau=0.5 — the batched default is the slowest M=1
                    # combination, so the M=1-validated point is the default. MAX_BS=4 (+ tau=0.5)
                    # remains the degen-validated BATCHED override for concurrent spec traffic.
                    : "${MINISGL_SPEC_MAX_BS:=1}"; export MINISGL_SPEC_MAX_BS
                    # tau=0.5 only pays where verify width costs (MAX_BS>1); at bs=1 tau=0 wins
                    # (36.71 vs ~30 tok/s), so a bs=1 gate defaults tau to 0.
                    if [ "$MINISGL_SPEC_MAX_BS" = "1" ]; then
                      : "${MINISGL_DSPARK_CONF_TAU:=0}"; export MINISGL_DSPARK_CONF_TAU
                    fi
                  fi ;;
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
                  if [[ "${SPEC:-$spec_default}" =~ ^(dflash|dspark)$ ]]; then
                    # 0.84 + FULL_CAP=2048 is the DEGEN-VALIDATED point (2026-08-19, 12/12 clean,
                    # 20.7k spec steps, zero restarts, MINISGL_SPEC_MAX_BS=4). Uncapped, this
                    # all-full_attention drafter accumulates aux to 8206 rows = 420 MB PER REQUEST
                    # (torch.cat per step, scheduler.py _spec_decode_step) — measured killing the
                    # serve at 0.88 AND at 0.84 under 4 concurrent 12k-token generations (OOM at
                    # 0 bytes free). The cap bounds aux at ~105 MB/uid and shrinks the propose ring
                    # 504->126 MB; drafting conditions on the newest 2048 committed positions.
                    mem_default_spec="0.84"
                    : "${MINISGL_DFLASH_FULL_CAP:=2048}"; export MINISGL_DFLASH_FULL_CAP
                    # MEASURED 2026-08-25 (same protocol as the 08-19 row: sampled seeds
                    # 101/202/303, M=1, 1200 tok): MAX_BS=1 -> tau=0 serves 36.71 tok/s median vs
                    # 27.78 with MAX_BS=4 -> tau=0.5 — the batched default is the slowest M=1
                    # combination, so the M=1-validated point is the default. MAX_BS=4 (+ tau=0.5)
                    # remains the degen-validated BATCHED override for concurrent spec traffic.
                    : "${MINISGL_SPEC_MAX_BS:=1}"; export MINISGL_SPEC_MAX_BS
                    # tau=0.5 only pays where verify width costs (MAX_BS>1); at bs=1 tau=0 wins
                    # (36.71 vs ~30 tok/s), so a bs=1 gate defaults tau to 0.
                    if [ "$MINISGL_SPEC_MAX_BS" = "1" ]; then
                      : "${MINISGL_DSPARK_CONF_TAU:=0}"; export MINISGL_DSPARK_CONF_TAU
                    fi
                  fi ;;
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
                  if [[ "${SPEC:-$spec_default}" =~ ^(dflash|dspark)$ ]]; then
                    # 0.84 + FULL_CAP=2048 is the DEGEN-VALIDATED point (2026-08-19, 12/12 clean,
                    # 20.7k spec steps, zero restarts, MINISGL_SPEC_MAX_BS=4). Uncapped, this
                    # all-full_attention drafter accumulates aux to 8206 rows = 420 MB PER REQUEST
                    # (torch.cat per step, scheduler.py _spec_decode_step) — measured killing the
                    # serve at 0.88 AND at 0.84 under 4 concurrent 12k-token generations (OOM at
                    # 0 bytes free). The cap bounds aux at ~105 MB/uid and shrinks the propose ring
                    # 504->126 MB; drafting conditions on the newest 2048 committed positions.
                    mem_default_spec="0.84"
                    : "${MINISGL_DFLASH_FULL_CAP:=2048}"; export MINISGL_DFLASH_FULL_CAP
                    # MEASURED 2026-08-25 (same protocol as the 08-19 row: sampled seeds
                    # 101/202/303, M=1, 1200 tok): MAX_BS=1 -> tau=0 serves 36.71 tok/s median vs
                    # 27.78 with MAX_BS=4 -> tau=0.5 — the batched default is the slowest M=1
                    # combination, so the M=1-validated point is the default. MAX_BS=4 (+ tau=0.5)
                    # remains the degen-validated BATCHED override for concurrent spec traffic.
                    : "${MINISGL_SPEC_MAX_BS:=1}"; export MINISGL_SPEC_MAX_BS
                    # tau=0.5 only pays where verify width costs (MAX_BS>1); at bs=1 tau=0 wins
                    # (36.71 vs ~30 tok/s), so a bs=1 gate defaults tau to 0.
                    if [ "$MINISGL_SPEC_MAX_BS" = "1" ]; then
                      : "${MINISGL_DSPARK_CONF_TAU:=0}"; export MINISGL_DSPARK_CONF_TAU
                    fi
                  fi ;;
  zaya|*/ZAYA1-8B-fp8|ZAYA1-8B-fp8)
                  model_id="${ZAYA_MODEL:-/models/ZAYA1-8B-fp8}";      spec_default="none"
                  tool_format="zaya_xml"
                  dflash_draft="/drafts/ZAYA1-8B-DFlash-CCA-5L-minv-ep4"; k_dflash=4 ;;
  # Qwen4-Exp (`qwen4_exp`), NVFP4, 48 layers (36 GDN + 12 full-attn), 512 experts/layer, a PLE
  # n-gram block on decoder index 1, and a 4x-wide (10240) hyper-connection residual. The ONLY model
  # in this table that does not fit its expert tier in VRAM at any TP this box has: 37 of its 48
  # layers are pinned in HOST RAM and streamed over PCIe every step. So every term below is
  # load-bearing and none of them has a safe default — a launch that drops one either does not boot
  # or boots as a different, slower serve.
  #
  # `[ENDGAME-2026-09-04]` THE MEASURED OPERATING POINT. Replaces the earlier `[CAPTURE-2026-09-04]`
  # block, which quoted 79.86 ms/step from a wall-derived instrument that does not reproduce across
  # boots (41 / 59 / 80 ms for the same config, by its own doc). Write-up + every raw artifact:
  # docs/measurements/QWEN4EXP_ENDGAME.md.
  #
  #   MEASURED, sampled at the checkpoint's own settings (temp 1.0 / top_k 20 / top_p 0.95 — never
  #   greedy), 200 tokens in-process, both ranks agreeing to 0.3%, graphs LIVE
  #   (cuda_graph_bs_captured [1,2], 126 replays, 0 harness failures):
  #        12.93 / 12.96 tok/s   raw: docs/measurements/WEIGHT_OFFLOAD_2026-09-02/r2_cpu0.json
  #   identical config, graphs OFF:
  #        12.59 / 12.61 tok/s   raw: .../r3_cpu0t2.json
  #   device decode step, GPU-event instrumented, min-of-3 interleaved, two independent boots:
  #        62.36 / 62.57 ms/step raw: .../attrib/decode_attrib_run{1,2}.json
  #   Tier split 11 device / 37 host / 0 cpu = 8.1 GiB device + 27.10 GiB pinned host PER RANK
  #   (54.20 GiB node-wide). TP=2, card 0 (RX 9070 XT) + card 1 (RX 9070), NVFP4 experts, page 16.
  #
  # Capture is worth 2.80-3.78 ms/step (x1.024 on the client wall) and is kept because it is free
  # and a merge requirement, NOT as a throughput lever: 41.7 ms of the 62.4 ms step is rank 1
  # reading host experts over PCIe (568.32 MB/token/rank; card 1 achieves 13.53-13.62 GB/s = 94% of
  # its measured 14.48 GB/s Gen4 x8 ceiling, so the read itself has nothing left to give), and rank 0
  # then burns 20.6 ms/step idling in the MoE all-reduce waiting for it.
  #
  # THE CPU EXPERT TIER IS OFF ON PURPOSE (`--cpu-moe-layers` unset). Measured same-boot-procedure
  # A/B, graphs off on BOTH legs because the tier cannot be captured at all: 36 CPU layers =
  # 11.19 tok/s vs 12.61 with the tier off. It buys 52.7 GiB of pinned host RAM and costs 11%
  # decode. Turn it on only for CAPACITY. See ENDGAME §5.
  qwen4exp|qwen4-exp|q4e)
                  # A local checkpoint, not an HF id: the expert stacks ship as per-layer shard
                  # files and the n-gram table is a separate 49 GiB sidecar. The measured runs
                  # mounted /home/pat/.cache/hf-q4e at /model and /home/pat/.cache/hf-ple at /ple.
                  model_id="${Q4E_MODEL:-/model}";                     spec_default="none"
                  # min_tp=2 is a HARD requirement, not a preference: at TP=1 the per-rank expert
                  # rows double to 1.465 GiB/layer and the host arena needed is ~54 GiB in ONE
                  # process, which this box cannot pin. TP=2 is the only configuration the 48-layer
                  # serve has ever booted in.
                  min_tp=2
                  # attn=hip is REQUIRED FOR CAPTURE, and it is not a kernel change. `rdna4`'s
                  # init_capture_graph/prepare_for_capture/prepare_for_replay all raise
                  # NotImplementedError ("rdna4 cudagraph capture lands in Phase 4"); the
                  # capture-capable implementation is its SUBCLASS `hip`, which dispatches decode to
                  # the same attn_decode.flash_decode_paged op and prefill to the same
                  # attn_hip.flash_prefill. Every qwen4_exp run before 2026-09-04 booted rdna4,
                  # which is the only reason capture had never been attempted here.
                  attn="hip"
                  # `[ENDGAME-2026-09-04]` 0.90, and this is now the MEASURED value: every 48-layer
                  # boot that has ever run — the two capture A/Bs, the two attribution boots and all
                  # four CPU-tier legs — ran at 0.90 and sized a healthy pool (2,850-2,862 KV pages
                  # @ 196,608 B). The previous line said 0.96 and claimed "at 0.90 the pool is ~0";
                  # that was written against a ~14 GiB device tier and is FALSE at the 8.1 GiB tier
                  # this arm actually ships (3.4 GiB less model on the card). 0.96 remains UNMEASURED
                  # with graphs live and is left as an override, not a default: raising it buys KV
                  # pages the operating point has never needed at CONC=2 and spends the headroom
                  # capture allocates its pool out of. MEM_RATIO=0.96 to try it.
                  mem_default="0.90"
                  # A CAP, not a default (CONC is already assigned above this case block, so
                  # `${CONC:=2}` would be a silent no-op). GRAPH_BS follows CONC, so this is also
                  # the capture coverage: buckets [1,2] are what was captured and measured, and a
                  # batch above the captured max runs FULLY EAGER. 2 is the only concurrency this
                  # operating point has been booted at; raising it raises coverage but is unmeasured
                  # and eats VRAM at exactly the moment VRAM is tightest.
                  if [ "$CONC" -gt 2 ]; then CONC=2; fi
                  # `[QSA-2026-09-06]` AND WITH QSA LIVE THE CAP IS 1, NOT 2 — a MEASURED refusal,
                  # not caution. Two prompts submitted together share a leading token run, the second
                  # resumes its prefill at the matched length, and that length is a TOKEN count, not
                  # a page count. `QSAPlan.build` then raises:
                  #     QSA prefill chunk is not group-aligned: cached_len=2 is not a multiple of
                  #     compress_ratio=4
                  # because the r members of the group straddling that boundary were never stored in
                  # this request's raw-key ring (the prefix was REUSED, not recomputed), so the
                  # compression step has nothing to average. Reproduced twice on deliberately
                  # different configurations — 48 layers TP=2 (cached_len=2) and a 4-layer subset
                  # TP=1 on one card (cached_len=9) — so it is not a TP, depth or scale effect. It
                  # RAISES rather than serving wrong attention, which is the guard working; but a
                  # default that provably fails the second concurrent request is not a default, and
                  # every real serve has shared prefixes (a system prompt, a chat template), so this
                  # is not a corner case.
                  # THE FIX IS NOT IN QSA: round the prefix match DOWN to a multiple of r before the
                  # request resumes (costs at most 3 tokens of reuse and leaves the DSV4 identity
                  # `physical_slot // r` untouched). Raise this back to 2 in the same commit that
                  # lands it. CONC=2 with MINISGL_QSA=0 is unaffected (and re-arms the 2048
                  # refusal). See docs/measurements/QSA_INDEXER.md 4c.
                  if [ "${MINISGL_QSA:-1}" != "0" ] && [ "$CONC" -gt 1 ]; then CONC=1; fi
                  # THE OFFLOAD TIER. `[E4M3-2026-09-05]` 8.1 GiB/rank now buys **12** of 48 layers
                  # device-side, not 11, and the other 36 are host-pinned at 24.12 GiB/rank
                  # (48.23 GiB across the node — MEASURED: 18 chunks, payload 24.00 GiB, 120 MiB
                  # abandoned, fill 99.5%, torch_fallbacks 0, min_device_free 1.04 GiB, 14.55 tok/s).
                  # The NVFP4 two-level scale moved both numbers: an e4m3 block scale is 1 B per 16
                  # weights where the fp16 fold was 2, so a layer's expert rows fell 0.7324 ->
                  # 0.6680 GiB/rank. Same 8.1 GiB budget, one MORE layer on the card, and 5.97 GiB
                  # LESS pinned host RAM than the 54.20 this arm used to need.
                  # * 8.1 still not 9.0, and the old reason has not expired — it has been re-priced.
                  #   Stage B's peak is the tier plus one layer in flight; at 12 e4m3 layers that is
                  #   a MEASURED 38.10 GiB allocated leaving min_device_free 1.04 GiB, i.e. healthier
                  #   than the 0.66 GiB the 11-layer fp16-fold point left. 13 layers (device-gb 9.0)
                  #   is 8.57 GiB of tier against a 14.14 GiB budget and leaves the KV pool at
                  #   ~0.16 GiB — UNMEASURED, and on the wrong side of a load peak. Size the device
                  #   tier against the LOAD peak, not the resting footprint.
                  # * chunk 1372 MiB, not 750 and not the 2048 default. THE CHUNK IS A FEASIBILITY
                  #   TERM AND e4m3 MOVED IT. The arena packs next-fit over ROWS and a region may
                  #   never straddle a chunk, but the row set is no longer {400,200,100,50} = 750: it
                  #   is {400, 200, 50, 25, 5, 1.25} MiB, and torch serves the last two out of ONE
                  #   shared 20 MiB kLargeBuffer segment (weights/chunk_plan.py::torch_charged_rows,
                  #   whose model is checked against a live allocator-callback trace). 1372 = 2x686
                  #   packs the CHARGED rows into 18 chunks with 120 MiB abandoned; the old 750
                  #   wastes 2.417 GiB/rank on the new row set. The curve is NOT monotone in chunk
                  #   size — sweep it with tools/offload/plan_chunk_sweep.py, never extrapolate.
                  # * host_gb 25, not 28: the clamp must sit above the 24.12 GiB the plan reserves.
                  #   28 also admits it; 25 is the point that was actually booted and measured.
                  #
                  # `[QSA-2026-09-06]` THE TIER IS NOW 7.4, NOT 8.1 — 11 device layers, not 12 — and
                  # it is a REACH decision, not a memory-safety one. The sparse-attention path
                  # (§QSA below) removed the 2048-token refusal, and the moment requests get long
                  # the binding constraint stops being the weights and becomes the KV POOL:
                  # `Engine.__init__` sets max_seq_len = min(checkpoint 262144, num_pages*page_size).
                  # MEASURED, same boot procedure, same day, both at memory-ratio 0.90 with graphs
                  # off (docs/measurements/QSA_INDEXER.md §5):
                  #
                  #     device-gb 8.1 -> model 13.45 GiB -> pool 0.51 GiB -> 2,810 pages ->  44,960 tok
                  #     device-gb 7.4 -> model 12.79 GiB -> pool 1.20 GiB -> 6,569 pages -> 105,104 tok
                  #
                  # i.e. ONE MoE layer off the card (0.668 GiB/rank at the e4m3 two-level scale) buys
                  # 3,759 KV pages = 60,144 tokens of context, at 12,288 B/token/rank. The 12-layer
                  # point also OOM'd on its first 4k prefill at the old 2048 chunk (92 MB requested,
                  # 138 MB free — docs/measurements/QSA_2026-09-06/r1.log); 8.1 WITH the 1024 chunk is UNTESTED
                  # and would trade those 60,144 tokens back for one device layer, so it is left as
                  # an override (WOFF_DEVICE_GB=8.1), not a default.
                  # * host_gb 26, not 25: the 37th host layer pushes the plan's reserve from 24.12 to
                  #   24.62 GiB/rank and the clamp must sit above it. 26 is the point that booted.
                  woff_device_gb="7.4"; woff_host_gb="26"; woff_chunk_mib="1372"
                  # ---- QSA SPARSE ATTENTION (`[QSA-2026-09-06]`) --------------------------------
                  # The 2048-token refusal is GONE: the sparse selection exists
                  # (python/minisgl/attention/qsa + the qsa_index HIP package), so this arm serves
                  # LONG CONTEXT — 105,104 tokens, MEASURED coherent with retrieval at 4k and above.
                  # Two more terms had to move and neither is a preference:
                  #
                  # 1. GRAPH_BS=0 — FORCED, not chosen. `QSARuntime.prepare` RAISES inside a
                  #    cudagraph capture: the selection allocates its per-forward
                  #    logits/blocks/tokens workspace and does a host `.max()` to size the block
                  #    window, neither of which a capture admits (plan T5.1). A QSA build therefore
                  #    decodes EAGER. That cost was already priced BEFORE QSA and is small on this
                  #    model — captured 14.55 vs eager 13.93 wall tok/s
                  #    (QWEN4EXP_L48_E4M3_HCFUSE_2026-09-05.json) — because 37 of 48 layers read
                  #    their experts over PCIe and a PCIe-bound decode has little launch overhead to
                  #    remove. GRAPH_BS>0 is only legal together with MINISGL_QSA=0, which re-arms
                  #    the 2048 refusal; it does not "turn off" long context, it REFUSES it.
                  # 2. MAX_PREFILL_LENGTH=1024, not 2048 — a FEASIBILITY term. The selection's
                  #    stage-4b workspace is full CHUNK width (`[chunk, 2051]` int32 tokens and
                  #    slots plus an int64 `flat` gather index), so the prefill activation peak
                  #    scales with the chunk. The structural fix is to row-tile 4b the way the
                  #    sparse attention already is (QSA_INDEXER.md §8.1) and it would let this go
                  #    back to 2048.
                  #
                  # --page-size 16 (PAGE_SIZE's default, below) is now a HARD REQUIREMENT rather
                  # than a convention: a compressed index key is addressed by
                  # `physical_kv_slot // indexer_compress_ratio` with no allocator of its own, and
                  # that identity holds only when a group of r=4 cannot straddle a page.
                  # `QSAProfile.require_page_size` RAISES on anything else rather than falling back
                  # to a second, untested addressing scheme.
                  : "${GRAPH_BS:=0}"; : "${MAX_PREFILL_LENGTH:=1024}"
                  # FLOOR_GIB is a BOX property, not a model property, and it is the one value here
                  # that must NOT be carried to another machine. `[QSA-2026-09-06]` the arena is now
                  # 24.62 x 2 = 49.24 GiB (the 37th host layer; it was 24.12 x 2 = 48.23 at the
                  # 12-device-layer tier), and the rank processes read ~2-6 GiB LESS `MemAvailable` than the host
                  # does because both engines are already up: a host reading 60.7 GiB presented as
                  # 54.60 GiB inside the ranks, so 48.23 + 9 = 57.23 was refused and 48.23 + 6 =
                  # 54.23 booted. 9 is kept as the DEFAULT because it is the value with margin; the
                  # 2026-09-05 run that produced every number in this arm ran at
                  # MINISGL_WEIGHT_ARENA_FLOOR_GIB=6 and is recorded as such.
                  #
                  # WHAT THE FLOOR IS ACTUALLY PROTECTING AGAINST IS SMALLER THAN IT LOOKS ON THIS
                  # BOX, and the reason is worth writing down because it is invisible in every tool:
                  # /home is ZFS and the ARC is NOT counted in `MemAvailable`. It sat at 13.8 GiB
                  # while the gate believed there were only 6 GiB spare. Under the pinning the
                  # kernel's shrinker gave 9.2 GiB of it straight back (13.8 -> 4.5 GiB, watched
                  # live), MemAvailable never fell below 11.0 GiB during Stage B, and SwapFree moved
                  # <1 GiB. So a refusal at floor 9 here is the gate being blind to the ARC, not the
                  # box being full — check `/proc/spl/kstat/zfs/arcstats` before believing it.
                  : "${MINISGL_WEIGHT_ARENA_FLOOR_GIB:=9}"; export MINISGL_WEIGHT_ARENA_FLOOR_GIB
                  # THE PLE N-GRAM SIDECAR. Qwen4ExpPLE reads its embeddings out of a separate
                  # ~49 GiB shard set that is NOT part of the model directory, and the head metadata
                  # (ngram_heads_offsets / ngram_heads_vocab_sizes) lives in a bf16 shard that is
                  # NOT in the plefp8 set, so the two are named separately. Composed here rather
                  # than left to the operator because the failure without them is a KeyError three
                  # minutes into a ten-minute boot.
                  Q4E_PLE_DIR="${Q4E_PLE_DIR:-/ple}"
                  if [[ -z "${MINISGL_PLE_FILES:-}" ]]; then
                    MINISGL_PLE_FILES="$(ls "$Q4E_PLE_DIR"/model-plefp8-*.safetensors 2>/dev/null | paste -sd:)"
                  fi
                  : "${MINISGL_PLE_META_FILES:=$Q4E_PLE_DIR/model-bf16-00010.safetensors}"
                  # Refuse from integers rather than boot a model whose n-gram block cannot be fed.
                  if [[ -z "$MINISGL_PLE_FILES" ]]; then
                    echo "[serve] ERROR: qwen4_exp needs the PLE n-gram shards, and none matched" \
                         "$Q4E_PLE_DIR/model-plefp8-*.safetensors. Mount the sidecar (the measured" \
                         "runs used -v /home/pat/.cache/hf-ple:/ple:ro) or set Q4E_PLE_DIR /" \
                         "MINISGL_PLE_FILES explicitly." >&2
                    exit 2
                  fi
                  # `[SHIP-2026-09-05]` The META shard is checked TOO, and it was not before. The
                  # block above guarded only the plefp8 list, so a HALF-mounted sidecar — the ten
                  # plefp8 shards present, `model-bf16-00010.safetensors` absent — sailed past this
                  # gate and died in `ple/source.py::PLEEmbeddingSource.__init__` with the KeyError
                  # about `ngram_heads_offsets`, which is exactly the "three minutes into a
                  # ten-minute boot" failure the comment above says this block exists to prevent.
                  # It is a separate file in a separate shard SET, so it goes missing independently.
                  # Only the DERIVED default is checked: an operator who names the file explicitly
                  # in MINISGL_PLE_META_FILES may point at a multi-path list or a future layout, and
                  # this gate must not out-guess them.
                  if [[ "$MINISGL_PLE_META_FILES" == "$Q4E_PLE_DIR/model-bf16-00010.safetensors" \
                        && ! -f "$MINISGL_PLE_META_FILES" ]]; then
                    echo "[serve] ERROR: qwen4_exp found $(awk -F: '{print NF}' <<< "$MINISGL_PLE_FILES") PLE shards but" \
                         "not the n-gram HEAD METADATA at $MINISGL_PLE_META_FILES. It lives in a" \
                         "bf16 shard that is NOT part of the plefp8 set, and without it the row ids" \
                         "cannot be formed at all — the boot would reach a KeyError minutes from" \
                         "now. Mount the full sidecar or set MINISGL_PLE_META_FILES explicitly." >&2
                    exit 2
                  fi
                  export MINISGL_PLE_FILES MINISGL_PLE_META_FILES ;;
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
# Compose forwards MINISGL_SPEC_MHA_PAGED as the EMPTY STRING when unset on the host, and the
# engine's gate is `!= "0"`, so "" would silently opt every spec serve into paged MHA. Empty =
# unset: the engine's shipped default (page_size->1) applies, and the SWA block below opts in.
[[ -z "${MINISGL_SPEC_MHA_PAGED:-}" ]] && unset MINISGL_SPEC_MHA_PAGED
if [[ -n "$swa_hybrid" ]]; then
  export MINISGL_SWA_RADIX="${MINISGL_SWA_RADIX:-1}"
  export MINISGL_SPEC_MHA_PAGED="${MINISGL_SPEC_MHA_PAGED:-1}"
  # Both are REQUIRED here, and a 0 in either fails SILENTLY — the cache is downgraded, or it is
  # "enabled" and never hits. Say so loudly rather than let the serve look healthy.
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

# --- weight offload ------------------------------------------------------------------------------
# Per-model default from the table (`woff_device_gb`), overridable by the environment. Empty stays
# empty: no flag on the line, byte-identical to every serve that has ever run. Non-empty is the ONE
# lever that decides how much of the MoE expert tier stays in VRAM — see the header. Both values are
# GiB (2**30), matching plan.GIB_PER_UNIT and every figure the engine logs back.
WOFF_DEVICE_GB="${WOFF_DEVICE_GB:-${woff_device_gb:-}}"
WOFF_HOST_GB="${WOFF_HOST_GB:-${woff_host_gb:-}}"
# Arena chunk size, in MiB. MEASURED 2026-09-03, card 0, qwen4_exp (1.465 GiB of experts per layer):
# a layer's rows may never straddle a chunk, so a 2 GiB chunk holds ONE layer and abandons 0.535 GiB
# — 2.00 GiB of pinned RAM per host layer (24-layer offload serve: 32.23 GiB carved, 44.00 GiB
# pinned, 73% efficient). A 3 GiB chunk holds TWO (2.93 GiB) — 1.50 GiB per host layer, a 25% cut
# (32-layer offload serve: 43.95 GiB carved, 45.00 GiB pinned, 97.7%), and that 45.00 GiB is the
# largest arena ever pinned on this box WITH an engine loaded (the previous record, which
# `host_capacity`'s advisory still quotes, is 34.00 GiB with none). Nothing above 2 GiB had ever been
# pinned here before that run, which is why this is a knob with its measurement attached rather than
# a new default. Raw: docs/measurements/WEIGHT_OFFLOAD_2026-09-02/stage_b_serve/.
#
# THE RIGHT CHUNK IS A FUNCTION OF THE PER-RANK LAYER SIZE, SO IT MOVES WITH TP. MEASURED
# 2026-09-04, cards 0+1, qwen4_exp at TP=2: a layer's expert rows halve to 0.7324 GiB, so the 3 GiB
# chunk that was optimal at TP=1 holds FOUR of them (2.93 GiB, 0.07 abandoned) — still a good fill,
# but the RESERVATION is quantised in 3 GiB steps, and at 30 host layers that rounds 21.97 GiB of
# payload up to 24.00 GiB. A 768 MiB chunk holds exactly ONE 0.7324 GiB layer and reserves
# 22.50 GiB for the same payload: 97.7% fill, 1.50 GiB/rank (3.00 GiB across the node) recovered,
# which on this box is the difference between booting and a HostArenaCapacityError.
#
# THE "ROUND UP TO A 256 MiB GRANULE" RULE WAS WRONG, and it cost 0.633 GiB/rank. The arena packs
# next-fit over ROWS, not over layers, and this checkpoint's rows are 400/200/100/50 MiB — exactly
# 750 MiB per layer per rank at TP=2, with nothing left over. So a chunk that is an EXACT MULTIPLE
# of the row set wastes ZERO, while any rounding up wastes the remainder of every chunk. Measured
# 2026-09-04 with `tools/offload/plan_chunk_sweep.py` (48 layers, 37 host layers, TP=2), pinned per
# rank: 750 MiB -> 27.100 GiB (0 waste), 752 -> 27.172, 768 -> 27.750 (+0.650), 1536/3072 -> 28.500
# (+1.400), 1024 -> 37.000 (+9.900 — one layer per chunk, the worst case). The curve is NOT monotone
# in chunk size, so it must be swept, not extrapolated. THE RULE: chunk = k x (sum of one layer's
# per-rank row bytes), smallest k that the box's `MemAvailable` allows.
#
#   | tp | per-rank layer | chunk       | per chunk | fill | measured serve                          |
#   |----|----------------|-------------|-----------|------|-----------------------------------------|
#   | 1  | 1.465 GiB      | 3072 MiB    | 2 layers  | 97.7%| 32 layers, 45.00 GiB pinned             |
#   | 2  | 0.7324 GiB     |  768 MiB    | 1 layer   | 97.7%| 40 layers, 22.50 GiB/rank, 14.5 tok/s   |
#   | 2  | 0.7324 GiB     |  750 MiB    | 1 layer   | 100% | 48 layers, 27.10 GiB/rank, 11.85 tok/s  |
#
# THE 48-LAYER ROW IS THE FULL MODEL AND IT IS COHERENT (Rayleigh-scattering answer at the
# checkpoint's own sampler, temp 1.0 / top_k 20 / top_p 0.95), so it is the operating point to serve
# qwen4_exp from. Its full configuration, all four terms load-bearing:
#
#   tp=2, WOFF_CHUNK_MIB=750, --weight-offload-device-gb 8.1 (=11/48 layers device, 8.06 GiB/rank),
#   --weight-offload-gb 28, --memory-ratio 0.90, MINISGL_WEIGHT_ARENA_FLOOR_GIB=9
#
# `[ENDGAME-2026-09-04]` THE 11.85 IN THAT TABLE IS AN **EAGER** WALL FIGURE (attention_backend=rdna4,
# --cuda-graph-max-bs 0, 115 tokens including prefill) and it is superseded. The measured serve at
# this operating point is 12.93 / 12.96 tok/s with graphs live and 12.59 / 12.61 without, 200
# sampled tokens, both ranks; the device decode step is 62.36 / 62.57 ms across two boots. The
# earlier "81.87 -> 79.86 ms/step" pair is RETIRED as an absolute (that instrument spread 41/59/80
# ms across three boots of one config); the ~2.8-3.8 ms/step capture DELTA it measured stands. See
# the `qwen4exp` arm above and docs/measurements/QWEN4EXP_ENDGAME.md. That arm is the launch line;
# these numbers stay here because the chunk arithmetic is what they measure.
#
# * device-gb 8.1 not 9.0: 12 device layers OOMs in STAGE B, not at rest. Stage B's peak is the tier
#   plus ONE layer in flight (staged checkpoint tensors + post_load's repack, ~1.46 GiB), and at 12
#   layers that is 14.96 GiB allocated with 224 MiB free — it dies asking for a 400 MiB row. At 11
#   the same run leaves min_device_free 0.66 GiB. Size the device tier against the LOAD peak.
# * `[ENDGAME-2026-09-04]` memory-ratio 0.90, not 0.96. The old line here said "at 0.90 the pool is
#   ~0"; that was computed for a ~14 GiB device tier and is FALSE at the 8.1 GiB tier this arm
#   ships — every 48-layer boot on record ran at 0.90 and got 2,850-2,862 KV pages. 0.96 is
#   UNMEASURED with graphs live and is an override (MEM_RATIO=0.96), not the default.
# * FLOOR_GIB=9 is BELOW the 12 GiB default and is the one term that is a box property rather than a
#   model property: 27.10 GiB/rank x 2 = 54.20 GiB, and the rank processes see ~62-65 GiB of
#   `MemAvailable` (2-3 GiB less than the host reads, because both engines are already up), so
#   54.20 + 12 = 66.20 does not fit and 54.20 + 9 = 63.20 does. Free ~3 GiB on this box and the
#   default floor works unchanged. Do not carry the 9 to a box with more RAM.
#
# Raw: docs/measurements/WEIGHT_OFFLOAD_2026-09-02/offload_serve_tp2/L40_tp2.json (40-layer row),
# L48_tp2_chunk750_dev11.json (48-layer row), chunk_sweep_tp2_L48.json (the sweep).
WOFF_CHUNK_MIB="${WOFF_CHUNK_MIB:-${woff_chunk_mib:-}}"
[[ -n "$WOFF_CHUNK_MIB" ]] && export MINISGL_WEIGHT_ARENA_CHUNK_MIB="$WOFF_CHUNK_MIB"
woff_args=()
[[ -n "$WOFF_DEVICE_GB" ]] && woff_args+=(--weight-offload-device-gb "$WOFF_DEVICE_GB")
[[ -n "$WOFF_HOST_GB" ]] && woff_args+=(--weight-offload-gb "$WOFF_HOST_GB")

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
  "${woff_args[@]}"
)
# EXTRA_ARGS last so it can override anything above.
[[ -n "${EXTRA_ARGS:-}" ]] && read -r -a _extra <<< "$EXTRA_ARGS" && cmd+=("${_extra[@]}")

printf '[serve] model=%s spec=%s%s tp=%s dp=%s ep=%s ctx=%s conc=%s attn=%s mem=%s graph_bs=%s\n' \
  "$model_id" "$SPEC" "${SPEC_K:+ k=$SPEC_K}" "$TP" "$DP" "$EP" "${CTX:-checkpoint}" "$CONC" \
  "$ATTN" "$MEM_RATIO" "$GRAPH_BS" >&2
[[ -n "$swa_hybrid" ]] && printf '[serve] SWA-hybrid: MINISGL_SWA_RADIX=%s MINISGL_SPEC_MHA_PAGED=%s\n' \
  "$MINISGL_SWA_RADIX" "$MINISGL_SPEC_MHA_PAGED" >&2
# The offload tier ON THE BANNER, because it is invisible in every other observable: a serve with
# the flag and a serve without it differ only in a `weight offload:` line buried in the engine log,
# and the one with it holds several GiB less KV pool. Diff the LAUNCH LINE first.
[[ -n "$WOFF_DEVICE_GB" ]] && printf '[serve] weight offload: device tier=%s GiB/rank host arena=%s chunk=%s MiB floor=%s GiB\n' \
  "$WOFF_DEVICE_GB" "${WOFF_HOST_GB:-<default>}" "${WOFF_CHUNK_MIB:-<default 2048>}" \
  "${MINISGL_WEIGHT_ARENA_FLOOR_GIB:-<default 12>}" >&2
# The PLE n-gram sidecar ON THE BANNER, same reasoning as the offload tier: it is a separate ~49 GiB
# shard set that appears nowhere on the python command line, and a serve that silently picked up a
# different (or partial) shard list produces plausible text from the wrong embeddings.
[[ -n "${MINISGL_PLE_FILES:-}" ]] && printf '[serve] PLE n-gram: %s shard(s) meta=%s\n' \
  "$(awk -F: '{print NF}' <<< "$MINISGL_PLE_FILES")" "${MINISGL_PLE_META_FILES:-<unset>}" >&2
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
