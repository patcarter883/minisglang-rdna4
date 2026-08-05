#!/usr/bin/env bash
# perf_level_guard.sh — open ONE profile_standard window and run an arbitrary command inside it.
#
# This is `perf_level_counters.sh`'s proven guard logic, factored out so a whole counter SWEEP can run
# inside a SINGLE set/restore cycle. Setting and restoring the perf level per kernel is both slow and
# risky (every extra restore path is another chance to leave a card pinned); the sweep wants one window.
#
# The three things this preserves verbatim from the proven script, because each one was a real bug:
#   1. Cards are resolved by PCI SLOT, not index. rocm-smi's GPU index and /sys/class/drm's cardN do
#      NOT correspond (ROCm GPU0 = card1, GPU1 = card2, and the Ryzen iGPU is card3). An index match
#      silently reconfigures the iGPU, which is never a compute target.
#   2. Every set is READ BACK and verified; a silent sysfs rejection would otherwise look like
#      "counters still zero" and get blamed on the counters.
#   3. Restore runs from a trap on EXIT/INT/TERM and is itself verified. A card left on
#      profile_standard runs every LATER timing job at a fixed non-boost clock — silent, plausible,
#      wrong numbers, with nothing anywhere reporting an error.
#
# HAZARD (unchanged): `power_dpm_force_performance_level` is GLOBAL, PER-CARD and PERSISTENT. It is not
# scoped to this process. Hold an EXCLUSIVE `gpu-lease -n 2` around the whole window, or you silently
# corrupt another agent's timings with no error anywhere.
#
# INTERPRETATION (unchanged): profile_standard PINS clocks to a fixed NON-BOOST state. ABSOLUTE TIMINGS
# taken inside this window are NOT comparable to auto-mode numbers and must never be mixed into an
# existing timing surface. Counter RATIOS are the point and are unaffected. It pins clocks LOWER, not
# higher, so there is no thermal/power risk.
#
#   gpu-lease -n 2 -- bash tools/counter_probe/perf_level_guard.sh <command...>
set -uo pipefail

[ "$#" -ge 1 ] || { echo "usage: $0 <command...>" >&2; exit 2; }

mapfile -t CARDS < <(
  for c in /sys/class/drm/card*/device; do
    [ -f "$c/power_dpm_force_performance_level" ] || continue
    slot=$(sed -n 's/^PCI_SLOT_NAME=//p' "$c/uevent" 2>/dev/null)
    case "$slot" in 0000:03:00.0|0000:07:00.0) echo "$c" ;; esac
  done
)
[ "${#CARDS[@]}" -eq 2 ] || { echo "expected 2 gfx1201 cards by PCI slot, found ${#CARDS[@]}" >&2; exit 1; }

declare -A ORIG
for c in "${CARDS[@]}"; do ORIG[$c]=$(cat "$c/power_dpm_force_performance_level"); done

# ---------------------------------------------------------------------------------------------
# DETACHED RESTORE SUPERVISOR — the trap below is NOT sufficient on its own.
#
# MEASURED FAILURE (2026-08-05): profile_standard makes an otherwise IDLE card report util=100% at
# ~79 W with 0% memory activity. That is bit-for-bit the gpu-lease wedged-card signature, so the
# lease watchdog confirms a "wedge" after ~100 s and SIGKILLs the process TREE. SIGKILL cannot be
# trapped, so the restore below never ran and BOTH CARDS WERE LEFT PINNED with the lease already
# released -- the precise disaster the header warns about, reached without a single kernel launched.
#
# So the restore cannot live only inside this process. This supervisor is `setsid`-detached into its
# own process group, which a tree-kill of the guard does not reach; it polls for the guard to
# disappear and restores unconditionally. It is idempotent with the trap: whichever runs first wins,
# and the second is a no-op because it writes the same value.
#
# (Pass GPU_LEASE_WEDGE_WATCH=0 on the lease to stop the false wedge in the first place. Belt AND
# braces: that env var prevents the kill, this supervisor survives one.)
# ---------------------------------------------------------------------------------------------
GUARD_PID=$$
SUPER_STAMP="/tmp/perf_level_guard.$GUARD_PID.restore"
{
  printf '%s\n' "${CARDS[@]}" > "$SUPER_STAMP.cards"
  for c in "${CARDS[@]}"; do printf '%s\t%s\n' "$c" "${ORIG[$c]}"; done > "$SUPER_STAMP.orig"
} 2>/dev/null

setsid nohup bash -c '
  guard_pid="$1"; stamp="$2"
  # Outlive the guard by polling; 4 h ceiling so a leaked supervisor cannot live forever.
  for _ in $(seq 1 14400); do
    kill -0 "$guard_pid" 2>/dev/null || break
    sleep 1
  done
  while IFS=$'"'"'\t'"'"' read -r c lvl; do
    [ -n "$c" ] || continue
    now=$(cat "$c/power_dpm_force_performance_level" 2>/dev/null)
    [ "$now" = "$lvl" ] && continue
    echo "$lvl" | sudo -n tee "$c/power_dpm_force_performance_level" >/dev/null 2>&1
    echo "[perf-level supervisor] restored $c -> $(cat "$c/power_dpm_force_performance_level" 2>/dev/null)"
  done < "$stamp.orig"
  rm -f "$stamp.cards" "$stamp.orig"
' _ "$GUARD_PID" "$SUPER_STAMP" >"$SUPER_STAMP.log" 2>&1 &
echo "[guard] detached restore supervisor armed (log: $SUPER_STAMP.log)"

restore() {
  local rc=$? bad=0
  echo "--- restoring perf level ---"
  for c in "${CARDS[@]}"; do
    echo "${ORIG[$c]}" | sudo -n tee "$c/power_dpm_force_performance_level" >/dev/null 2>&1
    local now; now=$(cat "$c/power_dpm_force_performance_level" 2>/dev/null)
    if [ "$now" = "${ORIG[$c]}" ]; then echo "  OK  $(basename "$(dirname "$c")") -> $now"
    else echo "  *** RESTORE FAILED *** $(basename "$(dirname "$c")") is '$now', wanted '${ORIG[$c]}'"; bad=1; fi
  done
  [ $bad -eq 0 ] || echo "!!! A CARD IS LEFT PINNED. Every later timing job on it is wrong. Fix by hand."
  exit $rc
}
trap restore EXIT INT TERM

echo "=== perf level: original ==="
for c in "${CARDS[@]}"; do echo "  $(basename "$(dirname "$c")") = ${ORIG[$c]}"; done

echo "=== perf level: setting profile_standard ==="
for c in "${CARDS[@]}"; do
  echo profile_standard | sudo -n tee "$c/power_dpm_force_performance_level" >/dev/null 2>&1
  now=$(cat "$c/power_dpm_force_performance_level")
  echo "  $(basename "$(dirname "$c")") -> $now"
  [ "$now" = "profile_standard" ] || { echo "SET FAILED (sudo? kernel rejected?) — aborting" >&2; exit 3; }
done

echo "=== running inside the profile_standard window ==="
"$@"
rc=$?
echo "=== command exited rc=$rc (restore runs next, via trap) ==="
exit $rc
