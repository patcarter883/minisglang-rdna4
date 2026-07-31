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
#   SPEC    none | mtp | dflash | eagle3 | ngram        (default: the model's own default)
#   SPEC_K  draft length                                (default: per model+algorithm, tuned)
#   TP      tensor-parallel size, 1 or 2                (default 2)
#   DP      data-parallel size, 1 or 2                  (default 1)
#   EP      1 to enable expert parallelism              (default 0)
#   CTX     context length in tokens                    (default: the checkpoint's own)
#   CONC    concurrent requests                         (default 4)
#
# LESS-USED, still here rather than in eight copies:
#   ATTN (auto|hip|triton), MEM_RATIO, GRAPH_BS, PAGE_SIZE, CACHE_TYPE, PORT, EXTRA_ARGS.
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
draft=""; attn="hip"; spec_default="none"; k_mtp=4; k_dflash=15; mem_default="0.80"
case "$MODEL" in
  qwen35b-awq)    model_id="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit";        spec_default="mtp"; k_mtp=4  ;;
  qwen35b-mxfp4)  model_id="pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4"; spec_default="mtp"; k_mtp=2
                  draft="z-lab/Qwen3.6-35B-A3B-DFlash"; k_dflash=15 ;;
  glm)            model_id="QuantTrio/GLM-4.7-Flash-AWQ";              spec_default="mtp"; k_mtp=2  ;;
  laguna)         model_id="poolside/Laguna-XS-2.1-NVFP4";             spec_default="none"
                  draft="poolside/Laguna-XS-2.1-DFlash-NVFP4"; k_dflash=16; mem_default="0.85" ;;
  zaya)           model_id="${ZAYA_MODEL:-/models/ZAYA1-8B-fp8}";      spec_default="none"
                  draft="/drafts/ZAYA1-8B-DFlash-CCA-5L-minv-ep4"; k_dflash=4 ;;
  *)              model_id="$MODEL" ;;     # any HF id or local path, straight through
esac
[[ -z "$SPEC" ]] && SPEC="$spec_default"

ATTN="${ATTN:-$attn}"
MEM_RATIO="${MEM_RATIO:-$mem_default}"
# Graph capture must cover the concurrency you serve, or requests above the captured batch size fall
# back to eager and the served config is not the measured one. Default to CONC, floor of 8.
GRAPH_BS="${GRAPH_BS:-$(( CONC > 8 ? CONC : 8 ))}"
PAGE_SIZE="${PAGE_SIZE:-16}"
CACHE_TYPE="${CACHE_TYPE:-radix}"

# --- spec decode ---------------------------------------------------------------------------------
spec_args=()
case "$SPEC" in
  none|"") : ;;
  mtp)     spec_args=(--spec-algorithm mtp    --spec-num-draft "${SPEC_K:-$k_mtp}") ;;
  eagle3)  spec_args=(--spec-algorithm eagle3 --spec-num-draft "${SPEC_K:-4}") ;;
  ngram)   spec_args=(--spec-algorithm ngram  --spec-num-draft "${SPEC_K:-4}") ;;
  dflash)
    [[ -n "$draft" ]] || { echo "serve.sh: MODEL=$MODEL has no DFlash draft checkpoint; set DRAFT=<path>" >&2; exit 2; }
    spec_args=(--spec-algorithm dflash --spec-draft-model-path "${DRAFT:-$draft}"
               --spec-num-draft "${SPEC_K:-$k_dflash}") ;;
  *) echo "serve.sh: unknown SPEC='$SPEC' (none|mtp|dflash|eagle3|ngram)" >&2; exit 2 ;;
esac

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
  "${ctx_args[@]}"
  "${spec_args[@]}"
)
# EXTRA_ARGS last so it can override anything above.
[[ -n "${EXTRA_ARGS:-}" ]] && read -r -a _extra <<< "$EXTRA_ARGS" && cmd+=("${_extra[@]}")

printf '[serve] model=%s spec=%s%s tp=%s dp=%s ep=%s ctx=%s conc=%s attn=%s mem=%s graph_bs=%s\n' \
  "$model_id" "$SPEC" "${SPEC_K:+ k=$SPEC_K}" "$TP" "$DP" "$EP" "${CTX:-checkpoint}" "$CONC" \
  "$ATTN" "$MEM_RATIO" "$GRAPH_BS" >&2
printf '[serve] %s\n' "${cmd[*]}" >&2
[[ -n "${DRY_RUN:-}" ]] && exit 0
exec "${cmd[@]}"
