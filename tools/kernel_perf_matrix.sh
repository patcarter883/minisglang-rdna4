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
  # ===== MTP vs DFlash spec-decode comparison (27B + 35B, TP=2, graph-captured) — run FIRST =====
  # Same target, two proposers: MTP (in-model head, K=4, uniform-K -> reserved verify graph, tolerates
  # MEM=0.85/GRAPH=8) vs DFlash (external z-lab drafter, K=15, memory-razor-thin -> GRAPH=4/MEM=0.80).
  # 35B = MXFP4 GDN+MoE; 27B = AWQ-INT4 GDN-hybrid. Compare decode/mixed tok/s across concurrency.
  "qwen3.6-35b-mxfp4-mtp|pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4|2|SPEC=mtp SPEC_K=4 GRAPH=8 MEMRATIO=0.85 MAXRUN=6"
  "qwen3.6-35b-mxfp4-dflash|pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4|2|SPEC=dflash SPEC_K=15 DFLASH_MODEL=z-lab/Qwen3.6-35B-A3B-DFlash GRAPH=4 MEMRATIO=0.80 MAXRUN=8"
  "qwen3.6-27b-awq-mtp|cyankiwi/Qwen3.6-27B-AWQ-INT4|2|SPEC=mtp SPEC_K=4 GRAPH=8 MEMRATIO=0.85 MAXRUN=6"
  "qwen3.6-27b-awq-dflash|cyankiwi/Qwen3.6-27B-AWQ-INT4|2|SPEC=dflash SPEC_K=15 DFLASH_MODEL=z-lab/Qwen3.6-27B-DFlash GRAPH=4 MEMRATIO=0.80 MAXRUN=8"
  # ===== base kernel-coverage matrix (no spec-decode) =====
  # --- GDN linear-attn + MHA paged attn + tail (small, fits TP=1 AND TP=2) ---
  "qwen3.5-4b-bf16|Qwen/Qwen3.5-4B|1|"
  "qwen3.5-4b-bf16|Qwen/Qwen3.5-4B|2|"
  # + w4a8_fp8_wmma (int4xfp8) on the same GDN+MHA arch
  "qwen3.5-4b-awq|QuantTrio/Qwen3.5-4B-AWQ|1|"
  "qwen3.5-4b-awq|QuantTrio/Qwen3.5-4B-AWQ|2|"
  # --- pure dense MHA (no GDN) — isolates attn_hip prefill/decode ---
  "qwen3-4b-dense|Qwen/Qwen3-4B-Instruct-2507|1|"
  "qwen3-4b-dense|Qwen/Qwen3-4B-Instruct-2507|2|"
  # --- GDN + MHA + w4a8 + MoE (35B: TP=2 only, >16 GB) ---
  "qwen3.6-35b-awq|cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit|2|MEMRATIO=0.85"
  # --- same, MXFP4 e2m1 decode path on the w4a8 kernel ---
  "qwen3.6-35b-mxfp4|pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4|2|MEMRATIO=0.85"
  # --- 27B GDN-hybrid W4A8 (int4): TP=2; TP=1 attempted (~14 GB, tight) ---
  "qwen3.6-27b-awq|cyankiwi/Qwen3.6-27B-AWQ-INT4|1|MEMRATIO=0.90"
  "qwen3.6-27b-awq|cyankiwi/Qwen3.6-27B-AWQ-INT4|2|"
  # --- MLA + MoE + w4a8 (GLM: TP=2 only) ---
  "glm-4.7-flash-awq|QuantTrio/GLM-4.7-Flash-AWQ|2|MEMRATIO=0.85"
  # --- CCA + w8a8_fp8_wmma + MoE (ZAYA 8B): TP=1 fits; TP=2 is DP2+EP (validated config) ---
  "zaya1-8b-fp8|pat883/zaya1-tidar-megatron|1|"
  "zaya1-8b-fp8|pat883/zaya1-tidar-megatron|2|DP=2 EP=1"
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
