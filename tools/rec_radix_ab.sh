#!/usr/bin/env bash
# Is the recurrent-radix snapshot store earning its VRAM? Three legs, one variable.
#
#   off      --no-gdn-radix                       — no store at all, no reservation
#   ladder0  MINISGL_GDN_RADIX_SNAP_LADDER=0      — store on, END-of-sequence snapshots only
#   ladder4  (default)                            — store on, 4 interior resume points per sequence
#
# The store costs VRAM out of the same budget the KV pool is sized from, so each leg records BOTH
# sides of the trade: the KV pool it was left with (cost) and the TTFT on prefix-sharing traffic
# (benefit), plus the hit count the engine logs.
#
#   gpu-lease -n 2 -- bash tools/rec_radix_ab.sh
set -uo pipefail

IMAGE="${MINISGL_IMAGE:-minisgl-rdna4:lean}"
OUT="${OUT:-/home/pat/fixtures/minisgl-kv-calib/rec_radix_ab}"
PORT="${PORT:-1919}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen35b-awq}"
CONC="${CONC:-4}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
LEGS="${LEGS:-off ladder0 ladder4}"
PROJECT="${PROJECT:-minisgl-recradix-ab}"
mkdir -p "$OUT"

boot() {  # $1 leg
  local leg="$1"; local -a env=()
  case "$leg" in
    off)     env=(EXTRA_ARGS=--no-gdn-radix) ;;
    ladder0) env=(MINISGL_GDN_RADIX_SNAP_LADDER=0) ;;
    ladder4) env=(MINISGL_GDN_RADIX_SNAP_LADDER=4) ;;
    # The proposed trade: a smaller prefill chunk so short prompts have interior boundaries at all
    # (below max_extend_tokens the ladder has nothing to attach to), a shallower ladder so the cap's
    # FLOOR drops, and a budget low enough to actually bank that — the budget is what sets the
    # reservation, the ladder only floors it.
    tuned)   env=(MINISGL_GDN_RADIX_SNAP_LADDER=2 MINISGL_GDN_RADIX_SNAP_BUDGET_GIB=0.20
                  "EXTRA_ARGS=--max-prefill-length 2048") ;;
    *) echo "unknown leg $leg"; return 2 ;;
  esac
  echo "== boot leg=$leg (${env[*]}) =="
  env MINISGL_IMAGE="$IMAGE" MODEL="$MODEL_ALIAS" SPEC=none TP=2 CONC="$CONC" ATTN=hip \
      MINISGL_HOST_PORT="$PORT" LEASE_NAME="rr-$leg" COMPOSE_PROJECT_NAME="$PROJECT" "${env[@]}" \
      docker compose --profile serve up -d >/dev/null || return 1
  local t0=$SECONDS
  while (( SECONDS - t0 < BOOT_TIMEOUT )); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "   ready after $((SECONDS-t0))s"; return 0; }
    docker logs "rr-$leg-serve" 2>&1 | grep -q "AssertionError\|Traceback" && { echo "   CRASHED"; return 1; }
    sleep 5
  done
  echo "   TIMEOUT"; return 1
}

run_leg() {
  local leg="$1"
  boot "$leg" || { docker logs "rr-$leg-serve" 2>&1 | tail -30; docker rm -f "rr-$leg-serve" >/dev/null 2>&1; return 1; }
  docker logs "rr-$leg-serve" > "$OUT/$leg.boot.log" 2>&1

  # provenance + cost side
  local store pool
  store=$(grep -oE "recurrent-radix snapshot store: cap=[0-9]+ x [0-9.]+ MiB = [0-9.]+ GiB" "$OUT/$leg.boot.log" | head -1)
  pool=$(grep -oE "Allocating [0-9]+ tokens for KV cache" "$OUT/$leg.boot.log" | head -1)
  echo "   store: ${store:-<none: disabled>}"
  echo "   $pool"
  if [[ "$leg" == "off" && -n "$store" ]]; then
    echo "PROVENANCE MISMATCH: leg 'off' still sized a snapshot store"; docker rm -f "rr-$leg-serve" >/dev/null; return 1
  fi
  if [[ "$leg" != "off" && -z "$store" ]]; then
    echo "PROVENANCE MISMATCH: leg '$leg' has no snapshot store"; docker rm -f "rr-$leg-serve" >/dev/null; return 1
  fi

  python3 tools/rec_radix_traffic.py --base "http://127.0.0.1:$PORT" --label "$leg" \
    --out "$OUT/$leg.traffic.json" --prefix-repeat "${PREFIX_REPEAT:-6}" \
    --agent-reqs "${AGENT_REQS:-8}" 2>&1 | tail -6

  # benefit side, as the engine itself counts it
  docker logs "rr-$leg-serve" 2>&1 | grep -c "recurrent-radix HIT" > "$OUT/$leg.hits.txt"
  echo "   recurrent-radix HITs: $(cat "$OUT/$leg.hits.txt")"
  docker logs "rr-$leg-serve" 2>&1 | grep -oE "restored recurrent state at cached_len=[0-9]+" \
    | grep -oE "[0-9]+" | paste -sd+ | bc 2>/dev/null > "$OUT/$leg.skipped.txt"
  echo "   tokens skipped by restore: $(cat "$OUT/$leg.skipped.txt" 2>/dev/null || echo 0)"
  docker rm -f "rr-$leg-serve" >/dev/null 2>&1
}

for leg in $LEGS; do run_leg "$leg" || echo "LEG $leg FAILED"; done

echo
echo "=================== SUMMARY ==================="
printf '%-9s %-14s %-22s %-10s %-10s %s\n' leg pool_tokens store agent_ttft chat_ttft hits
for leg in $LEGS; do
  pool=$(grep -oE "Allocating [0-9]+ tokens" "$OUT/$leg.boot.log" 2>/dev/null | head -1 | grep -oE "[0-9]+")
  store=$(grep -oE "= [0-9.]+ GiB \(ladder depth [0-9]+" "$OUT/$leg.boot.log" 2>/dev/null | head -1 | tr -d '(')
  aw=$(python3 -c "import json;print(json.load(open('$OUT/$leg.traffic.json'))['summary']['agent_warm']['mean'])" 2>/dev/null)
  ch=$(python3 -c "import json;print(json.load(open('$OUT/$leg.traffic.json'))['summary']['chat']['mean'])" 2>/dev/null)
  printf '%-9s %-14s %-22s %-10s %-10s %s\n' "$leg" "${pool:-?}" "${store:-none}" "${aw:-?}" "${ch:-?}" "$(cat "$OUT/$leg.hits.txt" 2>/dev/null || echo ?)"
done
