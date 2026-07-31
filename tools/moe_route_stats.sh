#!/usr/bin/env bash
# MEASURE: how many DISTINCT experts does one spec-verify batch actually touch?
#
# CONTINUANCE §2 finding 3 concluded that MoE verify cost is INHERENT expert divergence — that the
# qlen=K+1 draft rows route to near-disjoint expert sets, so the grouped GEMM must stream ~8*qlen
# expert slabs instead of 8. Nobody measured it. If the real overlap is high, the 14x MoE cost at
# M=17 is a kernel/alignment bug and that conclusion flips.
#
# The probe (quant/kernels.py:_route_stats, armed by MINISGL_MOE_ROUTE_STATS) records, per MoE call:
#   pairs        = M * top_k                      -- (token, expert) assignments
#   distinct     = |union of expert ids|          -- what an ideal kernel must stream
#   blocks       = ntp / block_m                  -- what moe_align ACTUALLY launches
#   ideal_blocks = sum_e ceil(rows_e / block_m)   -- the best alignment could do
#   union_curve  = distinct experts after 1..M rows
# blocks/distinct ~ 1 means alignment is already amortizing all the overlap there is.
#
# Runs EAGER (GRAPH_BS=0) on purpose: routing is a deterministic function of the hidden states, so it
# is identical captured or not, and the probe's .item() is illegal mid-capture. This is a routing
# statistic, NOT a timing run — do not read tok/s off it.
#
# MUST be invoked UNDER the shared arbiter, which this script does NOT acquire itself:
#     gpu-lease -n 2 -- bash tools/moe_route_stats.sh
set -uo pipefail

WT=/home/pat/code/minisgl-rdna4-specod
OUT=${OUT:-$WT/tools/moe_route_stats_results.txt}
: > "$OUT"

down() { ( cd "$WT" && docker compose --profile serve down >/dev/null 2>&1 ); }

drive() {
  # Generate enough decode steps to fill the sample cap, then let the probe dump.
  python3 - <<'PY'
import json, urllib.request
BASE="http://localhost:1919"; MODEL="poolside/Laguna-XS-2.1-NVFP4"
PROMPT=("Write a detailed technical explanation of how a B-tree index works, including "
        "insertion, node splitting, and range scans.")
for _ in range(3):
    b={"model":MODEL,"messages":[{"role":"user","content":PROMPT}],"max_tokens":384,
       "temperature":0.0,"stream":False}
    r=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(b).encode(),
                             headers={"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(r,timeout=900).read())
    print("  generated", d.get("usage",{}).get("completion_tokens",0), "tokens")
PY
}

leg() {
  local name=$1 tag=$2; shift 2
  echo "=== $name ===" | tee -a "$OUT"
  down
  rm -f "$WT"/tools/route_stats_"$tag".rank*.json
  ( cd "$WT" && env MINISGL_MOE_ROUTE_STATS="/engine/tools/route_stats_$tag" \
      MINISGL_MOE_ROUTE_STATS_N=1200 "$@" docker compose --profile serve up -d >/dev/null 2>&1 )
  local ok=0
  for _ in $(seq 1 150); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && { ok=1; break; }
    sleep 2
  done
  [ "$ok" = 1 ] || { echo "  FAILED to become ready" | tee -a "$OUT"; \
    ( cd "$WT" && docker compose --profile serve logs --tail 25 2>&1 | tail -25 ) | tee -a "$OUT"; return 1; }
  drive 2>&1 | tee -a "$OUT"
  ls -la "$WT"/tools/route_stats_"$tag".rank*.json 2>&1 | tee -a "$OUT"
  echo | tee -a "$OUT"
}

export MODEL=laguna TP=2 CONC=1 MEM_RATIO=0.96 GRAPH_BS=0
export MINISGL_SPEC_MHA_PAGED=1 MINISGL_SWA_RADIX=1

leg "plain (qlen 1)"      plain SPEC=none
leg "dflash K=7 (qlen 8)" k7    SPEC=dflash SPEC_K=7
leg "dflash K=15 (qlen 16)" k15 SPEC=dflash SPEC_K=15

down
echo "raw -> $WT/tools/route_stats_*.rank*.json ; log -> $OUT"
