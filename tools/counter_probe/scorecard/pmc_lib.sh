#!/usr/bin/env bash
# pmc_lib.sh — in-container helpers for running a rocprofv3 counter sweep.
#
# Sourced by the phase scripts. Assumes it is already running inside ONE profile_standard window
# (see perf_level_guard.sh) and inside the ROCm 7.14 image.
#
# The one non-obvious thing here is `pmc_run`'s FALLBACK. The SQ block has a limited number of
# physical counter slots, and rocprofv3 does not always fail loudly when a requested set does not
# fit -- it can drop counters. So every group is first tried whole, and the result is CHECKED for
# the presence of every requested counter; if any are missing, the group is re-run ONE COUNTER AT A
# TIME. Single-counter passes are always schedulable, and because the kernel work is deterministic
# the values remain comparable across passes. Correctness beats the wall-clock saving.

pmc_have_counter() {  # <csvdir> <counter>  -> 0 if at least one row for that counter exists
  local d="$1" c="$2"
  find "$d" -name "*counter_collection.csv" -print0 2>/dev/null \
    | xargs -0 -r grep -l "\"$c\"" >/dev/null 2>&1
}

# pmc_run <outdir> <tag> "<space separated counters>" -- <command...>
pmc_run() {
  local out="$1" tag="$2" counters="$3"; shift 3
  [ "${1:-}" = "--" ] && shift
  local dir="$out/$tag"
  rm -rf "$dir"; mkdir -p "$dir"

  echo "  [pmc] $tag :: $counters"
  timeout -s KILL "${PMC_TIMEOUT:-300}" rocprofv3 --pmc $counters -f csv -d "$dir" -- "$@" \
      >"$dir/run.log" 2>&1
  local rc=$?

  local missing=""
  for c in $counters; do
    pmc_have_counter "$dir" "$c" || missing="$missing $c"
  done

  if [ $rc -ne 0 ] || [ -n "$missing" ]; then
    echo "    -> grouped pass rc=$rc missing:${missing:- none}; re-running SINGLY"
    local i=0
    for c in $counters; do
      i=$((i+1))
      local sd="$dir/single_$i"
      rm -rf "$sd"; mkdir -p "$sd"
      timeout -s KILL "${PMC_TIMEOUT:-300}" rocprofv3 --pmc "$c" -f csv -d "$sd" -- "$@" \
          >"$sd/run.log" 2>&1
      local src=$?
      if pmc_have_counter "$sd" "$c"; then echo "      OK   $c"
      else echo "      FAIL $c (rc=$src)"; fi
    done
  fi
  return 0
}

# pmc_time <label> <command...> — plain timing run, NO counters. Used for the auto-perf-level pass.
pmc_time() {
  local label="$1"; shift
  echo "  [time] $label"
  "$@" 2>&1 | sed 's/^/    /'
}
