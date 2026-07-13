#!/usr/bin/env bash
# ============================================================================================
# HIP-kernel performance matrix — answers PR #13 "RDNA4 Native HIP Kernels" perf-numbers ask.
# Runs the serving benchmark (prefill / decode / mixed  ×  concurrency M{1,2,4,8,16}) under CUDA
# graph capture (--attn hip) for a model set chosen so EVERY canonical HIP kernel is exercised,
# at TP=1 (where the model fits one 16 GB card) and TP=2.
#
#   Run unattended tonight (each row leases the cards itself; sequential — 2 cards):
#     nohup bash tools/kernel_perf_matrix.sh > tools/kperf_$(date +%m%d_%H%M).driver.log 2>&1 &
#
# Results: one log per (model,TP) in tools/kperf_results/, plus a parsed summary table at the end.
# Each row is self-contained (boot serve -> bench prefill/decode/mixed -> teardown), fail-continue.
# ============================================================================================
set -uo pipefail
cd /home/pat/code/minisgl-rdna4
OUT=tools/kperf_results; mkdir -p "$OUT"
GRAPH=16                      # graph-captured decode (production path; eager is not a result)
BENCH_M=1,2,4,8,16            # concurrency sweep -> M=1 decode, M=16 mixed/batched

# rows: "LABEL|MODEL|TP|EXTRA_ENV"  — TP=1 only listed where the model fits a single 16 GB card.
# EXTRA_ENV passes model-specific knobs to run_bench_window.sh (e.g. mem-ratio, EP for ZAYA).
ROWS=(
  # ===== SINGLE-CARD FIRST: cached local ZAYA1-8B-fp8 (CCA/zaya_cca + w8a8_fp8_wmma + tail + attn_*) =====
  # The only offline-cached servable model; exercises this session's register-resident attn_prefill_paged
  # (O+Q registers) + attn_hip + tail/rms_norm + w8a8 MoE. TP=1 single card, then TP=2 (EP-over-TP CCA).
  "zaya1-8b-fp8|/models/ZAYA1-8B-fp8|1|MEMRATIO=0.85"
  "zaya1-8b-fp8|/models/ZAYA1-8B-fp8|2|EP=1 MEMRATIO=0.85"
  # ===== SPEC-DECODE ROWS REMOVED (MTP + DFlash) =====
  # Established this session: spec-decode perf on the 27B/35B GDN+MoE is poor — GDN caps accept-len ~1.5-2
  # so MTP/DFlash break even or lose vs no-spec, AND the configs are VRAM-tight on 16 GB (DFlash-EP can't
  # fit the draft model + K=15 verify graph + a usable KV pool; 27B-MTP OOM-crashed GPU0 at runtime). Not
  # worth benchmarking here. The base rows below still exercise every canonical kernel. Re-add spec rows
  # only on >16 GB cards / if a draft-head proposer lands. Was: qwen3.6-{35b-mxfp4,27b-awq}-{mtp,dflash-ep}.
  # ===== base kernel-coverage matrix (no spec-decode) =====
  # --- GDN linear-attn + MHA paged attn + tail (small, fits TP=1 AND TP=2) ---
  "qwen3.5-4b-bf16|Qwen/Qwen3.5-4B|1|"
  "qwen3.5-4b-bf16|Qwen/Qwen3.5-4B|2|"
  # NOTE: the small w4a8 (int4) rows are REMOVED — the only cached Qwen3.5-4B AWQ checkpoints
  # (QuantTrio/Qwen3.5-4B-AWQ, cyankiwi/Qwen3.5-4B-AWQ-BF16-INT4) are Qwen3.5-VL multimodal builds
  # (Qwen3_5ForConditionalGeneration; weights under model.language_model.layers.*, plus a visual tower),
  # which minisgl's text loader can't map (KeyError model.layers.0.mlp.gate_up_proj). Serving the
  # Qwen3.5-VL text tower is a separate bring-up. w4a8_fp8_wmma is still exercised by the 27B/35B AWQ rows.
  # --- pure dense MHA (no GDN) — isolates attn_hip prefill/decode ---
  "qwen3-4b-dense|Qwen/Qwen3-4B-Instruct-2507|1|"
  "qwen3-4b-dense|Qwen/Qwen3-4B-Instruct-2507|2|"
  # --- GDN + MHA + w4a8 + MoE (35B: TP=2 only, >16 GB) ---
  "qwen3.6-35b-awq|cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit|2|MEMRATIO=0.85"
  # --- same, MXFP4 e2m1 decode path on the w4a8 kernel ---
  "qwen3.6-35b-mxfp4|pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4|2|MEMRATIO=0.85"
  # --- 27B GDN-hybrid W4A8 (int4): TP=2; TP=1 attempted (~14 GB, tight) ---
  # 27B TP=1 REMOVED: the 27B INT4 weights are ~15.3 GiB and don't fit a single 16 GB card (OOM mid-load,
  # 68 MiB free) — not a memory_ratio issue, the model is too big for one card. TP=2 (sharded) below fits.
  "qwen3.6-27b-awq|cyankiwi/Qwen3.6-27B-AWQ-INT4|2|"
  # --- MLA + MoE + w4a8 (GLM: TP=2 only) ---
  "glm-4.7-flash-awq|QuantTrio/GLM-4.7-Flash-AWQ|2|MEMRATIO=0.85"
  # --- CCA + w8a8_fp8_wmma + MoE (ZAYA 8B): TP=1 fits; TP=2 is DP2+EP (validated config) ---
  # DISABLED: pat883/zaya1-tidar-megatron is not present offline (its HF-hub cache dir is empty; the
  # sibling zaya1-tidar-opd* repos are single-shard/incomplete). Re-enable after re-downloading the
  # megatron checkpoint. CCA (zaya_cca) kernel coverage is deferred with it.
  # "zaya1-8b-fp8|pat883/zaya1-tidar-megatron|1|"
  # "zaya1-8b-fp8|pat883/zaya1-tidar-megatron|2|DP=2 EP=1"
  # --- STANDARD w8a8 (fp8xfp8), NOT ZAYA's CCA ---------------------------------------------------
  # (a) cached DENSE fp8 SMOKE: minisgl wires fp8-W8A8 on the MoE-EXPERT path only, so a dense fp8
  #     model may NOT serve (dense fp8 linear unwired). If it boots -> clean small standard-arch
  #     w8a8 number; if it errors on a dense fp8 linear -> that is itself a finding for the PR.
  "qwen3.5-4b-fp8-dense|RedHatAI/Qwen3.5-4B-FP8-dynamic|1|"
  "qwen3.5-4b-fp8-dense|RedHatAI/Qwen3.5-4B-FP8-dynamic|2|"
  # (b) representative STANDARD fp8-MoE (mainstream arch + fp8 experts = the real w8a8 case).
  #     NOT cached — download first (pick one; confirm it exists + serves), then uncomment:
  #   huggingface-cli download RedHatAI/Qwen3-30B-A3B-FP8   # standard Qwen3-MoE, fp8 experts, TP=2
  # "qwen3-30b-a3b-fp8|RedHatAI/Qwen3-30B-A3B-FP8|2|MEMRATIO=0.85"
)

echo "[kperf] $(date -u +%FT%TZ) starting matrix: ${#ROWS[@]} rows, GRAPH=$GRAPH, M=$BENCH_M"
for row in "${ROWS[@]}"; do
  IFS='|' read -r LABEL MODEL TP EXTRA <<<"$row"
  log="$OUT/${LABEL}_tp${TP}.log"
  echo "[kperf] === $(date -u +%T) $LABEL TP=$TP  ($MODEL) $EXTRA ==="
  # DP2+EP for ZAYA TP=2: run_bench_window forwards --data-parallel-size/--enable-ep via env if set.
  # Pass ONLY MODEL/TP/BENCH_M here; GRAPH/MEMRATIO/MAXRUN/SPEC/... come from $EXTRA (or default in
  # run_bench_window/_bench_inner). Passing GRAPH here too would DUPLICATE it in the env when a row's
  # $EXTRA also sets GRAPH, and env keeps the FIRST → the row override silently lost. So EXTRA is the
  # sole source of those knobs. GRAPH default is 16 (matches the matrix's graph-capture intent).
  gpu-lease -n "$TP" -- env MODEL="$MODEL" TP="$TP" BENCH_M="$BENCH_M" $EXTRA \
      bash tools/run_bench_window.sh > "$log" 2>&1
  rc=$?
  echo "[kperf]     exit=$rc -> $log $([ $rc -ne 0 ] && echo '(FAILED — continuing)')"
done

echo "[kperf] === parsing summary ==="
python3 - "$OUT" <<'PY'
import sys,glob,re,os
d=sys.argv[1]
print(f"\n{'model':22} {'TP':>2} {'prefill tok/s':>14} {'decode M=1':>11} {'mixed M=16':>11}")
print("-"*66)
for f in sorted(glob.glob(os.path.join(d,'*.log'))):
    txt=open(f,errors='ignore').read()
    name=os.path.basename(f)[:-4]
    def grab(pat):
        m=re.findall(pat,txt); return m[-1] if m else "-"
    # serve_matrix_bench prints lines like: "prefill  M=1 ... <N> tok/s", "decode M=1 ... <N> tok/s", "mixed M=16 ... <N> tok/s"
    pre=grab(r"prefill.*?M=1.*?([\d.]+)\s*tok/s")
    dec=grab(r"decode.*?M=1.*?([\d.]+)\s*tok/s")
    mix=grab(r"mixed.*?M=16.*?([\d.]+)\s*tok/s")
    print(f"{name:22} {'':>2} {pre:>14} {dec:>11} {mix:>11}")
print("\n(If a cell is '-' the harness line format differs — read the per-row log directly.)")
PY
echo "[kperf] DONE $(date -u +%FT%TZ). Logs in $OUT/"
