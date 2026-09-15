#!/usr/bin/env bash
# Verdict on the prefill expert-staging change (d4de2b5). READ-ONLY: sends no requests, starts
# nothing, changes nothing. Run after the serve is up; re-run after a prompt for the number.
#
# The four things that decide whether the change worked, in the order they become observable. Two of
# the three failure modes are silent, which is why each is checked separately rather than inferred
# from the last one:
#   1. the KV pool actually PAID for the slab          (`stage=` on the KV sizing line)
#   2. the slab was taken, not declined                (ENABLED vs DECLINED)
#   3. a prefill launch was SERVED from it             (engaged ledger)
#   4. prefill throughput moved                        (vs the 4.0 tok/s measured 2026-09-15)
#
# The log is captured ONCE to a file and grepped from there, deliberately. The first cut of this
# script used `if docker logs … | grep -q …`, and under `pipefail` a `grep -q` that matches exits
# early, SIGPIPEs `docker logs`, and the pipeline reports 141 — so every check inverted and item 1
# reported "no KV sizing line" about a line that was present twice.
set -uo pipefail
C=${CONTAINER:-lease-minisgl-serve-serve}
P=${PORT:-1919}
BASE=4.0   # tok/s: minisgl_prefill_seconds_total 558.6 s / 2243 tokens, pre-change

L=$(mktemp) ; trap 'rm -f "$L"' EXIT
docker logs "$C" >"$L" 2>&1 || { echo "cannot read logs for $C"; exit 2; }
has() { grep -qF -- "$1" "$L"; }
hr() { printf '%s\n' "────────────────────────────────────────────────────────"; }

hr; echo "1. DID THE KV POOL PAY FOR THE SLAB?"
line=$(grep -m1 "KV sizing:" "$L")
if [[ -z "$line" ]]; then
  echo "   (no KV sizing line yet — still booting?)"
else
  tr ';' '\n' <<<"$line" | sed 's/^ */   /'
  grep -qE "stage=[0-9]" <<<"$line" \
    && echo "   -> the pool was sized around the slab." \
    || echo "   -> NO stage= term: this serve predates the change (stale mount?)."
fi

hr; echo "2. WAS THE SLAB TAKEN?"
if has "MoE prefill expert staging ENABLED"; then
  grep -m1 -o "MoE prefill expert staging ENABLED.*" "$L" | cut -c1-160 | sed 's/^/   /'
elif has "MoE prefill staging DECLINED"; then
  grep -m1 -o "MoE prefill staging DECLINED.*" "$L" | cut -c1-200 | sed 's/^/   /'
  echo "   => INERT this boot. Lower the device tier or raise --memory-ratio."
elif has "MoE prefill staging: not sized"; then
  grep -m1 -o "MoE prefill staging: not sized.*" "$L" | cut -c1-200 | sed 's/^/   /'
else
  echo "   (nothing yet — still booting, or weight offload is off on this arm)"
fi

hr; echo "3. WAS A PREFILL SERVED FROM THE SLAB?"
if has "weight_offload.moe_stage[prefill]"; then
  echo "   YES — weight_offload.moe_stage[prefill] is in the engaged ledger."
else
  echo "   not yet. Needs one prompt whose chunk sweeps the experts"
  echo "   (num_tokens x top_k >= num_experts; ~64 tokens at top_k 8 / 512 experts)."
  if has "weight_offload.moe_resolve[host]"; then
    echo "   NOTE: moe_resolve[host] IS present, so the seam is live but staging never fired."
    echo "         If a real prompt has run, the GATE is wrong — not the slab."
  fi
fi

hr; echo "4. THE NUMBER"
M=$(curl -s --max-time 6 "http://localhost:$P/metrics" 2>/dev/null)
S=$(awk '/^minisgl_prefill_seconds_total/{print $2; exit}' <<<"$M")
T=$(awk '/^minisgl_prefill_computed_tokens_total/{print $2; exit}' <<<"$M")
if [[ -z "${S:-}" || -z "${T:-}" ]]; then
  echo "   metrics unavailable (serve not up yet)."
elif awk "BEGIN{exit !($S <= 0 || $T <= 0)}"; then
  echo "   no prefill recorded yet (${T:-0} tokens, ${S:-0}s) — send one prompt."
else
  awk -v s="$S" -v t="$T" -v b="$BASE" 'BEGIN{
    r = t/s;
    printf "   prefill: %.0f tokens in %.1f s = %.2f tok/s\n", t, s, r;
    printf "   baseline (pre-change, 2026-09-15): %.2f tok/s  =>  %.2fx\n", b, r/b;
    if (r < b*1.5)
      print "   VERDICT: NOT FIXED. Either the in-place host reads were not the dominant\n            cost, or staging did not engage. Settle 2 and 3 before re-theorising.";
    else if (r < b*10)
      print "   VERDICT: improved but far under the link. Suspect a per-layer sync or copy\n            granularity problem, not the mechanism itself.";
    else
      print "   VERDICT: the sweep is no longer read in place. Record the operating point\n            in the serve.sh table.";
  }'
fi
hr
