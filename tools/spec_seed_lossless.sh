#!/usr/bin/env bash
# LOSSLESSNESS GATE for the prompt-prefill draft seed — with the two CONTROLS that decide whether a
# mismatch is the seed's fault or a property the build already had.
#
# The claim under test: seeding changes only what the DRAFTER conditions on; every draft is still
# verified against the target, so at temperature 0 the emitted text should be byte-identical across
#   BASELINE (no seed)  ==  CANDIDATE (seed)  ==  PLAIN (--spec-algorithm none)
#
# CONTROL 1 (base vs plain): does BASELINE spec already diverge from plain? If yes, spec-vs-plain
#   byte-identity is not a property of this build and the seed cannot be blamed for breaking it.
# CONTROL 2 (plain CODE1 vs plain CODE2): is the engine byte-deterministic AT ALL on this config,
#   with spec entirely off? The two requests are the same prompt in one boot at temperature 0, so a
#   difference here means the gate is unmeasurable, not that anything is "corrupting the verify path".
#
# Runs on the HOST (no GPU) after the three legs have written their per-request dumps.
set -uo pipefail
BASE=/home/pat/code/minisgl-rdna4-seedbase/tools
CAND=/home/pat/code/minisgl-rdna4-specod/tools
BLEG="${BLEG:-BASE_dbg}"; CLEG="${CLEG:-CAND_dbg}"; PLEG="${PLEG:-PLAIN}"
m(){ [ -f "$1" ] && md5sum "$1" | cut -d' ' -f1 || echo "(absent)"; }
echo "=== per-request md5 (reasoning_content + content) ==="
printf '%-6s %-34s %-34s %-34s %s\n' TAG BASELINE CANDIDATE PLAIN VERDICT
for tag in warm CODE1 CODE2 SHORT; do
  b=$(m "$BASE/spec_seed.$BLEG.$tag.txt"); c=$(m "$CAND/spec_seed.$CLEG.$tag.txt")
  p=$(m "$CAND/spec_seed.$PLEG.$tag.txt")
  v=""
  [ "$b" = "$c" ] && v="base==cand" || v="base!=cand"
  [ "$b" = "$p" ] && v="$v  base==plain" || v="$v  base!=plain"
  [ "$c" = "$p" ] && v="$v  cand==plain" || v="$v  cand!=plain"
  printf '%-6s %-34s %-34s %-34s %s\n' "$tag" "$b" "$c" "$p" "$v"
done
echo
echo "=== CONTROL 2: is the engine byte-deterministic with spec OFF? ==="
p1=$(m "$CAND/spec_seed.$PLEG.CODE1.txt"); p2=$(m "$CAND/spec_seed.$PLEG.CODE2.txt")
if [ "$p1" = "$p2" ]; then echo "  plain CODE1 == plain CODE2 -> engine IS byte-deterministic; the gate has teeth."
else echo "  plain CODE1 ($p1) != plain CODE2 ($p2)"
     echo "  -> the SAME prompt, SAME boot, temperature 0, spec OFF already produces different text."
     echo "     Byte-identity is NOT a property of this config (fp8 KV + SWA-radix prefix-cache hit),"
     echo "     so a base-vs-cand text difference is NOT evidence that seeding corrupts the verify path."
fi
echo
echo "=== divergence offsets (base vs cand) ==="
for tag in warm CODE1 CODE2 SHORT; do
  b="$BASE/spec_seed.$BLEG.$tag.txt"; c="$CAND/spec_seed.$CLEG.$tag.txt"
  [ -f "$b" ] && [ -f "$c" ] || continue
  printf '  %-6s ' "$tag"
  if cmp -s "$b" "$c"; then echo "identical"; else cmp "$b" "$c" 2>&1 | sed 's#.*differ#differ#'; fi
done
