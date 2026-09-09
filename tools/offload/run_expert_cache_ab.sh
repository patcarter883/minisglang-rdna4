#!/usr/bin/env bash
# A/B the per-expert VRAM residency cache against the SHIPPED all-host arm.
#
# THE ARMS DIFFER IN ONE FLAG. Both run the shipped qwen4exp operating point — all 48 MoE layers in
# system RAM (`woff_device_gb=0.5`, which is how serve.sh spells "no device layer"), TP=2, CONC=2 —
# because all-MoE-in-RAM is a FIXED CONDITION of this project, not a tuning choice: the external
# target (llama.cpp, 23.3 tok/s) was measured that way and a device tier makes our numbers
# non-comparable. The cache does not violate it: the authoritative expert bytes still live in host
# RAM and the slab is a COPY of the recently-routed ones, which is what the reference implementation
# does too.
#
#   base   : shipped, no cache                      -> h = 0 for MoE experts
#   cache  : shipped + --expert-cache-gb <GiB>      -> oracle predicts h = 0.864 at 6.7
#
# The slab is allocated during Stage A, BEFORE the KV pool is sized, so the pool absorbs the cost
# automatically and the trade shows up as context, not as an OOM. That is the trade to read in the
# results: tok/s against KV tokens.
#
#   ARM=base  bash tools/offload/run_expert_cache_ab.sh
#   ARM=cache CACHE_GB=6.7 bash tools/offload/run_expert_cache_ab.sh
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARM="${ARM:?set ARM=base|cache}"
CACHE_GB="${CACHE_GB:-6.7}"
LABEL="${LABEL:-xcache-$ARM$([[ $ARM == cache ]] && echo "-${CACHE_GB}g")}"
OUT="${OUT:-$REPO/tools/offload/xcache}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-68}"

export REPO LABEL READY_TIMEOUT
export PORT="${PORT:-1919}"
export REPS="${REPS:-3}"
export DECODE_TOKENS="${DECODE_TOKENS:-128}"
export DECODE_M="${DECODE_M:-1,2}"

mkdir -p "$OUT"
export LOG="$OUT/$LABEL.serve.log"
rm -f "$LOG.probe"
export JSON="$OUT/$LABEL.bench.json"

avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if (( avail < MIN_AVAIL_GIB )); then
  echo "[xcache] REFUSING: MemAvailable ${avail} GiB < ${MIN_AVAIL_GIB} GiB — the host-arena gate" \
       "runs before pinning and is blind to the ZFS ARC, so booting now means dying 20 min in." >&2
  exit 2
fi

# THE ONE FLAG. `base` passes nothing, so it is the shipped launch line byte for byte rather than
# "the cache arm with the cache set to zero" — an emulated baseline is how a -50% claim died on this
# project before (docs: ab-baseline-must-be-old-code-not-emulated).
case "$ARM" in
  base)  extra="" ;;
  cache) extra="--expert-cache-gb $CACHE_GB" ;;
  *)     echo "[xcache] ARM must be base|cache, got '$ARM'" >&2; exit 2 ;;
esac
export EXTRA_ARGS="$extra"

export MODEL=qwen4exp SPEC=none TP="${TP:-2}" CONC=2 GRAPH_BS=0
export MEM_RATIO="${MEM_RATIO:-0.85}"
export MINISGL_IMAGE="${MINISGL_IMAGE:-minisgl-rdna4:cache-mgr7}"
export MINISGL_HOST_PORT="$PORT"
export MINISGL_HOSTPROF=50
export Q4E_MODEL_DIR="${Q4E_MODEL_DIR:-/home/pat/ai/hf/q4e}"

echo "[xcache] ARM=$ARM EXTRA_ARGS='$EXTRA_ARGS' image=$MINISGL_IMAGE tp=$TP mem_ratio=$MEM_RATIO"
echo "[xcache] MemAvailable=${avail}GiB out=$OUT label=$LABEL"

cd "$REPO" || exit 1
timeout "$((READY_TIMEOUT + 2400))" \
  gpu-lease -n "$TP" -- bash "$REPO/tools/offload/_cpu_tier_leg.sh" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[xcache] leg $LABEL exit=$rc  (76 = the GPU wedged, not a bug in the command — re-run once)"
# The arm's own evidence, pulled from the engine log: the cache must SAY it built and observed.
echo "[xcache] --- expert-cache lines (absent on the base arm, required on the cache arm) ---"
grep -F '[expert-cache]' "$LOG.container" 2>/dev/null | head -6 || echo "  (none)"
grep -F '[route-trace]' "$LOG.container" 2>/dev/null | head -3 || true
echo "[xcache] log=$LOG container=$LOG.container json=$JSON"
