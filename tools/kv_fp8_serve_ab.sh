#!/usr/bin/env bash
# RULE-4 serve validation + 3-leg A/B for the fp8 KV cache, under graph capture at the served TP.
#
# The three legs are the three states the KV cache can actually be in on this box:
#   bf16         MINISGL_KV_FP8=0                       — the reference (no quantized KV at all)
#   fp8_uncal    MINISGL_KV_FP8=1, no sidecar           — TODAY'S COMPOSE DEFAULT: fp8 store with
#                                                         whatever scales resolve, which for a
#                                                         checkpoint with no kv_cache_scheme is the
#                                                         identity 1.0 (a WARNING at boot)
#   fp8_sidecar  MINISGL_KV_FP8=1 + MINISGL_KV_FP8_SCALES=<per-head sidecar from kv_fp8_calibrate.py>
#
# Each leg ASSERTS ITS OWN PROVENANCE from the boot log (which scale source installed, which cache
# dtype) before it is allowed to report a number — an A/B whose legs silently ran the same
# configuration is the classic way to measure new-vs-itself.
#
# What each leg records:
#   * `Allocating N tokens for KV cache` — the VRAM half of the question. fp8 halves KV bytes, so
#     the honest comparison is not "same context, is fp8 as accurate" but "at EQUAL VRAM, does
#     fp8-plus-more-context beat bf16-with-less".
#   * tools/serve_acceptance.py — 29 checks including the mid-context fact retrieval at ~7.7k tokens
#     that is the one check fp8-KV was previously measured to lose.
#   * decode tok/s at M=1,4 via tools/serve_matrix_bench.py.
#
# Run from the worktree root under a 2-card lease:
#   gpu-lease -n 2 -- bash tools/kv_fp8_serve_ab.sh
set -uo pipefail

IMAGE="${MINISGL_IMAGE:-minisgl-rdna4:comb}"
SIDECAR="${SIDECAR:-/engine/kv_scales_qwen35b_tp2.safetensors}"
OUT="${OUT:-/home/pat/fixtures/minisgl-kv-calib/serve_ab}"
PORT="${PORT:-1919}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen35b-awq}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
CONC="${CONC:-4}"
ACC_REPS="${ACC_REPS:-1}"
M_LIST="${M_LIST:-1,4}"
LEGS="${LEGS:-bf16 fp8_uncal fp8_sidecar}"
# ONE compose project across legs, reusing this worktree's network: the box's docker address pool is
# heavily subnetted by other agents' abandoned networks, so a per-leg network can fail to allocate.
PROJECT="${PROJECT:-minisgl-kvfp8-serveab}"
mkdir -p "$OUT"

boot() {  # $1 leg
  local leg="$1"
  local -a env=(MINISGL_IMAGE="$IMAGE" MODEL="$MODEL_ALIAS" SPEC=none TP=2 CONC="$CONC" ATTN=hip
                MINISGL_HOST_PORT="$PORT" LEASE_NAME="kvfp8-$leg" COMPOSE_PROJECT_NAME="$PROJECT")
  case "$leg" in
    bf16)        env+=(MINISGL_KV_FP8=0 MINISGL_KV_FP8_SCALES=) ;;
    fp8_uncal)   env+=(MINISGL_KV_FP8=1 MINISGL_KV_FP8_SCALES=) ;;
    fp8_sidecar) env+=(MINISGL_KV_FP8=1 MINISGL_KV_FP8_SCALES="$SIDECAR") ;;
    *) echo "unknown leg $leg"; return 2 ;;
  esac
  echo "== boot leg=$leg =="
  env "${env[@]}" docker compose --profile serve up -d || return 1
  local t0=$SECONDS
  while (( SECONDS - t0 < BOOT_TIMEOUT )); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "   ready after $((SECONDS-t0))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -q "kvfp8-$leg-serve" || { echo "   container died"; return 1; }
    sleep 5
  done
  echo "   TIMEOUT after ${BOOT_TIMEOUT}s"; return 1
}

teardown() { docker rm -f "kvfp8-$1-serve" >/dev/null 2>&1; }  # by container: `down` would delete the shared network

provenance() {  # $1 leg -> assert the leg is the configuration it claims to be
  local leg="$1" log="$OUT/$leg.boot.log"
  docker logs "kvfp8-$leg-serve" > "$log" 2>&1
  grep -E "fp8-KV|Allocating .* tokens for KV|Free memory after|KV cache dtype" "$log" | sort -u > "$OUT/$leg.prov.txt"
  echo "-- provenance ($leg) --"; cat "$OUT/$leg.prov.txt"
  case "$leg" in
    bf16)
      if grep -q "fp8-KV .* pool: installed" "$OUT/$leg.prov.txt"; then
        echo "PROVENANCE MISMATCH: bf16 leg installed fp8 scales"; return 1; fi ;;
    fp8_uncal)
      grep -q "fp8-KV is ON but NO calibrated scales" "$OUT/$leg.prov.txt" || {
        # not necessarily a failure: a checkpoint carrying kv_cache_scheme legitimately resolves
        # per-tensor scales here. Say WHICH it was rather than pretending the leg is uncalibrated.
        grep -q "fp8-KV .* installed" "$OUT/$leg.prov.txt" || {
          echo "PROVENANCE MISMATCH: fp8_uncal leg shows no fp8 scale resolution at all"; return 1; }
        echo "   NOTE: this leg resolved CHECKPOINT scales, not the identity — reported as such"; } ;;
    fp8_sidecar)
      # PER-HEAD (MHA / SWA-ring pools) or PER-LAYER latent (MLA) — both mean "the sidecar
      # installed". The granularity is a property of the cache, not of whether the leg is
      # configured the way it claims.
      grep -qE "installed (PER-HEAD scales|PER-LAYER latent scales) from sidecar" "$OUT/$leg.prov.txt" || {
        echo "PROVENANCE MISMATCH: fp8_sidecar leg did NOT install the sidecar"; return 1; } ;;
  esac
  echo "   provenance OK"
}

run_leg() {
  local leg="$1"
  boot "$leg" || { docker logs "kvfp8-$leg-serve" 2>&1 | tail -60; teardown "$leg"; return 1; }
  provenance "$leg" || { teardown "$leg"; return 1; }

  # ACC_REPS>1: the serve is NOT bit-reproducible past ~32 tokens (fused-MoE atomics), and several
  # acceptance checks are keyword assertions on free text, so a single run cannot tell a real
  # regression from sampling noise. Repeat and report the spread.
  echo "-- acceptance ($leg), $ACC_REPS rep(s) --"
  : > "$OUT/$leg.acceptance.txt"
  for rep in $(seq 1 "$ACC_REPS"); do
    echo "[rep $rep]" >> "$OUT/$leg.acceptance.txt"
    python3 tools/serve_acceptance.py --base "http://127.0.0.1:$PORT" --conc "$CONC" \
      >> "$OUT/$leg.acceptance.txt" 2>&1
  done
  grep -E "RESULT:|FAILED:" "$OUT/$leg.acceptance.txt"

  if [[ "${CTX_PROBE:-0}" != "0" ]]; then
    # The equal-VRAM question: fp8 doubles the pool, so sweep context until the leg stops being able
    # to serve it at all. A bf16 leg does not answer a 32k request worse — it does not answer it.
    echo "-- context capacity ($leg) --"
    python3 tools/kv_ctx_capacity_probe.py --base "http://127.0.0.1:$PORT" --label "$leg" \
      --lengths "${CTX_LENGTHS:-4000,8000,16000,24000,32000,48000}" \
      --out "$OUT/$leg.ctxprobe.json" 2>&1 | tee "$OUT/$leg.ctxprobe.txt"
  fi

  echo "-- decode bench ($leg) --"
  python3 tools/serve_matrix_bench.py --url "http://127.0.0.1:$PORT" --label "$leg" \
    --m "$M_LIST" --workloads decode --decode-tokens 256 > "$OUT/$leg.bench.txt" 2>&1
  grep -E "^\s*[0-9]+\s+[0-9]" "$OUT/$leg.bench.txt" || tail -5 "$OUT/$leg.bench.txt"
  teardown "$leg"
}

for leg in $LEGS; do run_leg "$leg" || echo "LEG $leg FAILED"; done

echo
echo "=================== SUMMARY ==================="
for leg in $LEGS; do
  echo "--- $leg ---"
  grep -E "Allocating .* tokens for KV|fp8-KV" "$OUT/$leg.prov.txt" 2>/dev/null | head -4
  grep -E "RESULT:|FAILED:" "$OUT/$leg.acceptance.txt" 2>/dev/null
  grep -E "^\s*[0-9]+\s+[0-9]" "$OUT/$leg.bench.txt" 2>/dev/null
done
