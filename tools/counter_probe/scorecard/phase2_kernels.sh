#!/usr/bin/env bash
# phase2_kernels.sh — IN-CONTAINER. Counters for the regime claims that CLOSED OFF optimisation work.
#
# Claim 1  "Dense GEMV is at its floor; the launch-count/fusion lever is dead"
#          -> bf16 GEMV at the serve shapes. If it is stall-dominated rather than bandwidth-
#             saturated, "at its floor" does not follow from the bytes argument that produced it.
# Claim 2  "MoE decode is reduction-floor bound"
#          -> gemm1 and gemm2 at decode M = 1, 5, 6, 30. The claim rests on a fixed ~17us floor plus
#             a per-output-column cost; the counters that test it are the WAVE count and the stall
#             split, plus GL2C_HIT/MISS which decides the cache-resident-vs-HBM question the claim's
#             own revision history flip-flopped on twice.
# Claim 3  "Decode GEMV is at 81% of HBM"
#          -> memory counters on the dense GEMV. The percentage is recomputed from scratch: the
#             roofline denominator was later corrected to 706.6 GB/s (not 644/674), and the bytes
#             come from GL2C_EA_RDREQ_* which is the actual off-L2 traffic rather than an assumed
#             working-set size.
# Claim 6  "Occupancy is THE lever"
#          -> OccupancyPercent / MeanOccupancyPerCU on everything above, read against the stall split.
#
# BAND DISCIPLINE: everything here is the DECODE band (M <= 32). Several past claims silently mixed
# decode and prefill; nothing in this phase touches M >= 64.
set -uo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SELF/pmc_lib.sh"
OUT="${OUT:-/out}/phase2"; mkdir -p "$OUT"

echo "=== phase2: counter DEFINITIONS (units decide the interpretation) ==="
# WAVE_DEP_WAIT came back as ~95 and WAVE_ISSUE_WAIT as ~2.6 while SQ_WAIT_ANY came back in the
# hundreds of millions. Those cannot both be cycle counts, and whether the small ones are PERCENTAGES
# or raw cycles inverts what they say. Dump what the SDK itself declares rather than guessing.
rocprofv3 --list-avail 2>/dev/null > "$OUT/../list_avail_full.txt" || true
grep -iE -A3 "WAVE_DEP_WAIT|WAVE_ISSUE_WAIT|SQ_WAIT_ANY|SQ_WAIT_INST_ANY|SQ_WAVE_CYCLES|VALUBusy|MemUnitBusy|OccupancyPercent|ValuPipeIssueUtil|FetchSize|L0CacheHit" \
  "$OUT/../list_avail_full.txt" 2>/dev/null | head -120 || echo "(no descriptions available)"

echo "=== phase2: build ==="
hipcc -O3 --offload-arch=gfx1201 -I/kern/fp8_wmma/fp8_wmma_rocm \
      -o /tmp/kp /probe/scorecard/kernel_probes.hip 2>&1 | grep -iE "error" | head -30
[ -x /tmp/kp ] || { echo "BUILD FAILED"; exit 1; }

echo "=== phase2: smoke — every mode must launch cleanly before any counter is trusted ==="
/tmp/kp dense 1 4096 4096 128   || { echo "SMOKE dense FAILED"; exit 1; }
/tmp/kp bf16  1 2048 256        || { echo "SMOKE bf16 FAILED";  exit 1; }
/tmp/kp moe1  1 32 8 512 2048 128 16 || { echo "SMOKE moe1 FAILED"; exit 1; }
/tmp/kp moe2  1 32 8 512 2048 128 16 || { echo "SMOKE moe2 FAILED"; exit 1; }

# Counter groups. Kept SMALL and thematic; pmc_lib re-runs singly if the SQ block cannot schedule a
# group, because a silently dropped counter reads as a zero and would invert a verdict.
G_WAVE="SQ_WAVES SQ_BUSY_CYCLES"
G_STALL_ANY="SQ_WAIT_ANY"
G_STALL_DEP="WAVE_DEP_WAIT"
G_STALL_ISS="WAVE_ISSUE_WAIT"
G_INST="SQ_INSTS_VALU SQ_INSTS_SALU"
G_CYC="SQ_INST_CYCLES_VALU SQ_INST_CYCLES_VMEM"
G_MEM="GL2C_EA_RDREQ GL2C_EA_RDREQ_32B GL2C_EA_RDREQ_64B"
G_MEM2="GL2C_HIT GL2C_MISS"
G_FETCH="FetchSize"
G_TCP="TCP_REQ TCP_REQ_MISS"
G_OCC="OccupancyPercent"
G_OCC2="MeanOccupancyPerCU"
G_BUSY="MemUnitBusy"
G_VALU="VALUBusy"

sweep() {   # sweep <tag> <args...>
  local tag="$1"; shift
  echo ""
  echo "--- sweep $tag :: /tmp/kp $* ---"
  local i=0
  for grp in "$G_WAVE" "$G_STALL_ANY" "$G_STALL_DEP" "$G_STALL_ISS" "$G_INST" "$G_CYC" \
             "$G_MEM" "$G_MEM2" "$G_FETCH" "$G_TCP" "$G_OCC" "$G_OCC2" "$G_BUSY" "$G_VALU"; do
    i=$((i+1))
    pmc_run "$OUT/$tag" "g$i" "$grp" -- /tmp/kp "$@"
  done
}

# ---- Claim 3 + Claim 1's regime control: dense decode GEMV ----------------------------------
# 4096^2 is the shape the int4 COLS=2 tuning note quotes, so a regime call here is directly
# comparable to the number that motivated the claim. Weight sets: fp8 16 MB, int4 8 MB -- BOTH fit
# the 64 MB Infinity Cache, so this shape measures the CACHE-RESIDENT regime and is labelled as such.
sweep dense_4096      dense 1 4096 4096 128
# 16384^2: int4 weights 128 MB, fp8 256 MB -- ABOVE the cache, so this is the true HBM-STREAMING
# regime. The 81%-of-HBM claim was measured in this regime; the 54% figure in the cache-resident one.
# Measuring only one of them is how the two numbers came to contradict each other.
sweep dense_16384     dense 1 16384 16384 128

# ---- Claim 1: dense bf16 GEMV at real serve shapes ------------------------------------------
sweep bf16_down       bf16 1 2048 256      # shared.down  — BYLANE, nw=8, cols=1, grid=8 blocks
sweep bf16_gate       bf16 1 1 2048        # shared.gate  — N=1, the "0% of peak" launch-floor shape
sweep bf16_qkvz       bf16 1 6144 2048     # GDN in_proj_qkvz — K>1024, nw=16, cols=1
sweep bf16_lmhead     bf16 1 32768 2048    # LM head — the shape claim 7 says is the serve lever

# ---- Claim 2: MoE decode gemm1/gemm2 across the decode M band -------------------------------
# Qwen3.6-35B-A3B TP=2: E=32, top_k=8, hidden K=2048, inter=512/rank, group 128, block_m=16.
# gemm1: N=2*inter=1024, K=2048.   gemm2: N=2048, K=inter=512.
for m in 1 5 6 30; do
  sweep "moe1_M$m" moe1 $m 32 8 512 2048 128 16
  sweep "moe2_M$m" moe2 $m 32 8 512 2048 128 16
done

echo "=== phase2 done ==="
