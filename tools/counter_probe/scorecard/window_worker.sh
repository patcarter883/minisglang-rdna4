#!/usr/bin/env bash
# window_worker.sh — hold ONE profile_standard window open and execute phase scripts as they arrive.
#
#   gpu-lease -n 2 -- bash tools/counter_probe/scorecard/window_worker.sh <queuedir>
#
# WHY A QUEUE INSTEAD OF A FIXED PHASE LIST. The repo rule is that ALL counter work in a session goes
# under ONE exclusive lease and ONE profile_standard window: every extra set/restore cycle is another
# chance to leave a card pinned at a non-boost clock, which silently corrupts every later timing job
# on the box. But the phases are not all authorable up front -- later ones depend on what the earlier
# ones show. A queue reconciles the two: the window opens once, and phases are dropped in as they are
# written.
#
# STARVATION IS THE COST, so it is bounded explicitly. Both cards are held for as long as this runs,
# and other agents block behind it. Two independent limits end the window automatically:
#   IDLE_TIMEOUT  — no new job for this long => the operator has stalled, release the cards.
#   MAX_WALL      — hard cap on the whole window regardless of activity.
# Neither is a nicety; without them a forgotten worker holds the entire box indefinitely.
#
# PROTOCOL. Drop an executable job as "<queuedir>/NN_name.job" (a shell script run on the HOST, with
# the lease env intact). Completion is signalled by "<queuedir>/NN_name.done" containing the exit
# code. Write "<queuedir>/STOP" to end the window immediately.
set -uo pipefail
Q="${1:?usage: window_worker.sh <queuedir>}"
mkdir -p "$Q"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-900}"   # 15 min with nothing to do -> release
MAX_WALL="${MAX_WALL:-7200}"          # 2 h hard cap

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$(dirname "$SELF")"

if [ "${GUARD_OPEN:-0}" != "1" ]; then
  echo "[worker] opening the profile_standard window (one per session)"
  exec env GUARD_OPEN=1 bash "$PROBE/perf_level_guard.sh" bash "$SELF/window_worker.sh" "$Q"
fi

echo "[worker] window OPEN. queue=$Q idle_timeout=${IDLE_TIMEOUT}s max_wall=${MAX_WALL}s"
echo "[worker] lease: HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
: > "$Q/WINDOW_OPEN"

start=$(date +%s); last=$start
while :; do
  now=$(date +%s)
  if [ -e "$Q/STOP" ]; then echo "[worker] STOP requested"; break; fi
  if [ $((now - start)) -ge "$MAX_WALL" ]; then echo "[worker] MAX_WALL reached — releasing"; break; fi
  if [ $((now - last)) -ge "$IDLE_TIMEOUT" ]; then echo "[worker] IDLE_TIMEOUT reached — releasing"; break; fi

  job=$(ls "$Q"/*.job 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then sleep 5; continue; fi

  base="${job%.job}"
  echo ""
  echo "[worker] ===== running $(basename "$job") ====="
  bash "$job" 2>&1 | tee "$base.log"
  rc=${PIPESTATUS[0]}
  echo "$rc" > "$base.done"
  mv "$job" "$base.job.ran"
  echo "[worker] ===== $(basename "$base") exited rc=$rc ====="
  last=$(date +%s)
done

rm -f "$Q/WINDOW_OPEN"
echo "[worker] window closing (restore runs via the guard's trap)"
