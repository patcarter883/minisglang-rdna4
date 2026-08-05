#!/usr/bin/env bash
# Drive every leg of the dense-tile/act-quant SERVE A/B, SERIALLY. Each leg takes its own `gpu-lease
# -n 2` and releases it before the next starts: two 35B/GLM TP=2 serves at once would contend for
# board power, PSU headroom and thermals, and neither number would be valid.
#
# Two models, deliberately:
#   QWEN  cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit  — the PRODUCTION config. Its checkpoint leaves every
#         dense linear in bf16 (modules_to_not_convert covers linear_attn/self_attn/shared_expert),
#         so the W4A8 dense GEMM the tile work optimizes is NEVER dispatched. This leg measures what
#         production actually gets.
#   GLM   QuantTrio/GLM-4.7-Flash-AWQ        — quantizes the SHARED EXPERT (self_attn excluded), so
#         its gate_up/down ARE W4A8 dense linears at M = prompt length. This is the leg that can see
#         the dense tile at all, and it carries the shape (glm.gate_up tp2) of the one known
#         isolated regression (0.8141x at M=512).
#
# KV pool is PINNED with --num-pages in both legs of each pair (see _ab_tile_inner.sh) — auto-sizing
# gave the two images different pools and would have compared admission policy, not GEMM cost.
set -uo pipefail
cd "$(dirname "$0")/.."
PRE_IMG=minisgl-rdna4:pre-tile          # engine 6dc09973 + kernels 36d1ac4
POST_IMG=minisgl-rdna4:post-tile        # engine 0a3c017d + kernels 2ade7c1
PRE_REPO=/home/pat/code/minisgl-rdna4-abpretile
POST_REPO=/home/pat/code/minisgl-rdna4-abposttile
S=/tmp/claude-1000/-home-pat-code-minisgl-rdna4/8b0825a0-a432-4698-8474-5a204a4e63ed/scratchpad

leg() {  # $1 tag  $2 image  $3 repo  $4.. extra env
  local tag="$1" img="$2" repo="$3"; shift 3
  echo "########## LEG $tag ($img) ##########"
  ( cd "$repo" && gpu-lease -n 2 -- env LEG="$tag" IMAGE="$img" REPO="$repo" REPS=3 "$@" \
      bash tools/ab_tile_serve_run.sh ) 2>&1 | tee "$S/leg_${tag}.log"
  echo "########## LEG $tag done ##########"
}

# page_size is 1 for the HIP attention backend and FORCED to 16 for MLA (GLM), so the same 16384-token
# pool is 16384 pages for Qwen and 1024 for GLM. Pinning the TOKEN count, not the page count, is what
# makes the two models' legs comparable to their own baselines.
leg QWEN_BEFORE  "$PRE_IMG"  "$PRE_REPO"  NUM_PAGES=16384
leg QWEN_AFTER   "$POST_IMG" "$POST_REPO" NUM_PAGES=16384
leg GLM_BEFORE   "$PRE_IMG"  "$PRE_REPO"  NUM_PAGES=1024 ATTN=auto MODEL=QuantTrio/GLM-4.7-Flash-AWQ
leg GLM_AFTER    "$POST_IMG" "$POST_REPO" NUM_PAGES=1024 ATTN=auto MODEL=QuantTrio/GLM-4.7-Flash-AWQ
echo "ALL LEGS COMPLETE"
