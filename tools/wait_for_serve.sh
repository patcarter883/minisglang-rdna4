#!/usr/bin/env bash
# Wait for a minisgl serve to become READY, or fail FAST with the reason.
#
# WHY THIS EXISTS. The obvious readiness loop —
#     until curl -s $URL/v1/models | grep -q '"id"'; do docker ps | grep -q serve || break; sleep 10; done
# — only notices a container that EXITED. It does not notice the common failures:
#
#   * the engine asserts (KV pool too small after reserving drafter + graphs) and the CONTAINER STAYS
#     UP, so the loop spins to its outer timeout. Observed 2026-09-10 burning a card lease on a
#     Qwen3.8-27B + DSpark boot that had already failed 3 minutes in with
#     "AssertionError: Not enough memory for KV cache after reserving recurrent state / draft model".
#   * one TP rank dies while rank0 keeps the container alive.
#   * a HIP OOM mid weight-load.
#   * a boot that simply stalls — no new log lines, no readiness, forever.
#
# So: poll readiness, but ALSO scan the log for fatal signatures and watch for a stalled log, and
# print the lines that explain the failure instead of a bare timeout.
#
#   tools/wait_for_serve.sh [-c CONTAINER] [-u URL] [-t TIMEOUT_S] [-s STALL_S]
#
# Exit codes:  0 ready | 1 fatal signature | 2 container gone | 3 stalled | 4 deadline
set -uo pipefail

CONTAINER=""; URL="http://localhost:1919"; TIMEOUT=900; STALL=240
while getopts "c:u:t:s:" o; do case $o in
  c) CONTAINER=$OPTARG ;; u) URL=$OPTARG ;; t) TIMEOUT=$OPTARG ;; s) STALL=$OPTARG ;;
esac; done

# Fatal signatures. Deliberately narrow: each is a boot that will NEVER become ready, so matching one
# is a decision to stop waiting. Warnings and recoverable retries must not appear here.
FATAL='AssertionError|Traceback \(most recent call last\)|torch\.OutOfMemoryError|out of memory|HIP error|hipError|RuntimeError:|No HIP GPUs are available|Killed|exitcode: [1-9]|NCCL.*timeout|libc10\.so'

[[ -n "$CONTAINER" ]] || CONTAINER=$(docker ps --format '{{.Names}}' | grep -E 'serve$' | head -1)
if [[ -z "$CONTAINER" ]]; then echo "wait_for_serve: no serve container found" >&2; exit 2; fi
echo "wait_for_serve: $CONTAINER -> $URL (timeout ${TIMEOUT}s, stall ${STALL}s)" >&2

t0=$SECONDS; last_lines=0; last_change=$SECONDS
while :; do
  if curl -sf --max-time 5 "$URL/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "wait_for_serve: READY after $((SECONDS - t0))s" >&2; exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "wait_for_serve: container GONE after $((SECONDS - t0))s. Last lines:" >&2
    docker logs --tail 15 "$CONTAINER" 2>&1 | cut -c1-200 >&2; exit 2
  fi
  # A fatal line means this boot is over even though the container is still up.
  hit=$(docker logs "$CONTAINER" 2>&1 | grep -E "$FATAL" | tail -4 | cut -c1-200)
  if [[ -n "$hit" ]]; then
    echo "wait_for_serve: FATAL after $((SECONDS - t0))s -- not waiting for a boot that already failed:" >&2
    printf '  %s\n' "$hit" >&2; exit 1
  fi
  # A boot that stops logging and never serves is stalled; waiting the full deadline teaches nothing.
  n=$(docker logs "$CONTAINER" 2>&1 | wc -l)
  if [[ "$n" -ne "$last_lines" ]]; then last_lines=$n; last_change=$SECONDS
  elif (( SECONDS - last_change > STALL )); then
    echo "wait_for_serve: STALLED -- no log output for $((SECONDS - last_change))s and not ready. Last lines:" >&2
    docker logs --tail 10 "$CONTAINER" 2>&1 | cut -c1-200 >&2; exit 3
  fi
  if (( SECONDS - t0 > TIMEOUT )); then
    echo "wait_for_serve: DEADLINE ${TIMEOUT}s. Last lines:" >&2
    docker logs --tail 10 "$CONTAINER" 2>&1 | cut -c1-200 >&2; exit 4
  fi
  sleep 5
done
