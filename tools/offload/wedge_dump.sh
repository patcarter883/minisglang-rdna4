#!/usr/bin/env bash
# Dump a Python stack for EVERY process of a wedged minisgl serve, classified by role.
#
# WHY THIS EXISTS AS A SCRIPT. A serve that answers `/health`, accepts requests, and tokenizes
# NONE of them is not diagnosable from the log — there is no error to grep. The only evidence is
# where each process is blocked, and that evidence is gone the moment the container is torn down.
# So the readiness path captures it automatically instead of relying on someone being at the
# keyboard during the ~17 minutes it takes to reach the failure again.
#
# ROLES, because "attach to the hung process" needs to know which one that is. In a TP=2 serve:
#   frontend   `python -m minisgl`     — the API/uvicorn process that binds /health and /v1/*
#   tokenizer  a smaller child         — tokenize/detokenize, off the request path's hot loop
#   rank       `spawn_main`, ~30 GiB   — the schedulers; under a CPU tier these SPIN by design,
#                                        so finding them busy is expected and proves nothing
# The interesting stack is almost never the rank: a rank spinning in its sense-reversing barrier is
# a SYMPTOM of work never arriving, not the reason it never arrived.
#
# `dump`, not `record`: we want the instantaneous block, not a profile of a process doing nothing.
# sudo strips PATH, so py-spy is named absolutely.
set -uo pipefail
OUT="${1:-/tmp/wedge_dump.txt}"
PYSPY="${PYSPY:-/home/pat/.local/bin/py-spy}"
CID="${CID:-}"

[[ -x "$PYSPY" ]] || { echo "py-spy not found at $PYSPY" >&2; exit 2; }

# Host PIDs of the container's processes. The container has its own PID namespace but shares the
# host's, so py-spy on the host attaches by HOST pid — which is why this does not need to exec
# into the container (and could not: py-spy is not installed in the image).
if [[ -n "$CID" ]]; then
  pids=$(docker top "$CID" -eo pid= 2>/dev/null | tr -d ' ')
else
  pids=$(pgrep -f "minisgl|spawn_main" 2>/dev/null)
fi

{
  echo "==================== wedge dump $(date -u +%FT%TZ) ===================="
  echo "--- memory ---"
  grep -E "MemAvailable|SwapFree" /proc/meminfo
  echo "--- processes ---"
  for p in $pids; do
    [[ -r "/proc/$p/cmdline" ]] || continue
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-120)
    rss=$(awk '/^VmRSS/{print int($2/1024)}' "/proc/$p/status" 2>/dev/null)
    swp=$(awk '/^VmSwap/{print int($2/1024)}' "/proc/$p/status" 2>/dev/null)
    [[ -z "$rss" ]] && continue
    # Classify by the two things that actually separate them: cmdline shape and resident size.
    role="other"
    case "$cmd" in
      *spawn_main*) role="rank" ;;
      *"-m minisgl"*|*minisgl*) role=$([[ "${rss:-0}" -gt 5000 ]] && echo rank || echo frontend/tokenizer) ;;
    esac
    # CPU AFFINITY, per process AND per thread. This is a decisive check, not decoration: the CPU
    # tier pins its pool threads to specific physical cores, and an affinity mask is INHERITED
    # across fork/exec. A frontend or tokenizer that came out of a pinned parent would be confined
    # to the same one or two cores the spinners own — which starves the request path without ever
    # producing an error, exactly the shape of failure being chased. A full mask here REFUTES that
    # hypothesis and sends the hunt elsewhere.
    aff=$(taskset -pc "$p" 2>/dev/null | sed 's/.*: //')
    echo "pid=$p role=$role rss=${rss}M swap=${swp:-0}M affinity=${aff:-?} cmd=$cmd"
    for t in /proc/$p/task/*; do
      tid=$(basename "$t")
      [[ "$tid" == "$p" ]] && continue
      ta=$(taskset -pc "$tid" 2>/dev/null | sed 's/.*: //')
      tn=$(cat "$t/comm" 2>/dev/null)
      # Only report threads that are NOT on the full core set — those are the pinned ones.
      [[ -n "$ta" && "$ta" != "0-15" && "$ta" != "0-7" ]] && echo "    tid=$tid name=$tn affinity=$ta"
    done
  done
  echo
  for p in $pids; do
    [[ -r "/proc/$p/cmdline" ]] || continue
    rss=$(awk '/^VmRSS/{print int($2/1024)}' "/proc/$p/status" 2>/dev/null)
    [[ -z "$rss" ]] && continue
    echo "-------------------- py-spy dump pid=$p rss=${rss}M --------------------"
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-160; echo
    timeout 60 sudo -n "$PYSPY" dump --pid "$p" 2>&1 | sed 's/^/    /'
    echo
  done
} >>"$OUT" 2>&1
echo "[wedge] stacks -> $OUT"
