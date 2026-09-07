#!/usr/bin/env bash
# ONE leg of the CPU-expert-tier serve measurement, run INSIDE a `gpu-lease` shell.
# Not called directly — `run_cpu_tier_serve.sh` sets the fixed condition and leases the cards.
#
# It is a separate FILE rather than a `bash -c '...'` string on purpose: the readiness check has to
# parse JSON, and a python one-liner nested inside a single-quoted `bash -c` inside a shell script
# gets its quotes eaten silently — producing a readiness loop that never passes on a healthy serve.
set -uo pipefail

REPO="${REPO:?}"
LABEL="${LABEL:?}"
LOG="${LOG:?}"
JSON="${JSON:?}"
PORT="${PORT:-1919}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"
REPS="${REPS:-3}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
DECODE_M="${DECODE_M:-1,2}"

cd "$REPO" || exit 1

# INSIDE the lease shell the arbiter has already rewritten COMPOSE_PROJECT_NAME and LEASE_NAME, so
# the container is asked for, never composed from variables that no longer mean what they did
# outside. Deriving it wrong makes readiness report CRASHED on a healthy serve and makes the cleanup
# trap leave an ORPHAN holding both cards.
# `dc` is the ONE spelling of compose used by up, ps, logs and down, so the cleanup trap and the
# readiness loop can never address a different project than the one that was started.
DC=(docker compose --profile serve)
# THE ENGINE LOG IS STREAMED, NOT COLLECTED AT THE END. A `docker compose logs` in the cleanup trap
# is a MEASURED defect in this repo: when the trap runs under SIGTERM the container is already going
# away and the command returns 0 BYTES, so the one artifact that explains the run is empty exactly
# when the run failed. `docker logs -f` from the moment the container exists keeps everything,
# including whatever the engine printed in its last second.
TAILPID=""
cleanup() {
  [[ -n "${DROPPID:-}" ]] && kill "$DROPPID" 2>/dev/null
  [[ -n "$TAILPID" ]] && kill "$TAILPID" 2>/dev/null
  # Best-effort top-up for anything the follower had not flushed; never the only source.
  "${DC[@]}" logs --no-color >>"$LOG.container" 2>/dev/null
  "${DC[@]}" down -t 30 >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

up_err=$("${DC[@]}" up -d 2>&1) || true
if grep -q "address pools have been fully subnetted" <<<"$up_err"; then
  # BOX STATE, not a fault in this service: docker has no free subnet left because stale compose
  # projects still hold theirs, and gpu-lease makes every leased run a NEW project. Retry on the
  # default bridge, which allocates nothing. See compose.default-bridge.yml.
  echo "[leg] docker has no free subnet; retrying on the default bridge"
  DC=(docker compose -f docker-compose.yml -f tools/offload/compose.default-bridge.yml
      --profile serve)
  up_err=$("${DC[@]}" up -d 2>&1) || true
fi
CID=$("${DC[@]}" ps -q serve)
[[ -z "$CID" ]] && { echo "[leg] compose started no serve container:"; echo "$up_err"; exit 1; }
echo "[leg] project=${COMPOSE_PROJECT_NAME:-<unset>} lease=${LEASE_NAME:-<unset>} container=${CID:0:12}"
: >"$LOG.container"
docker logs -f --timestamps "$CID" >>"$LOG.container" 2>&1 &
TAILPID=$!

# THE LOADER'S OWN PAGE CACHE IS WHAT EVICTS THE TIER IT JUST PACKED.
#
# Boot streams ~107 GiB of checkpoint off disk while the process holds ~64 GiB of host-resident
# weights. The pinned arena cannot be evicted, so the kernel takes the only large evictable thing:
# the CPU tier's PAGEABLE expert weights — and on this box the highest-priority swap device is a
# FILE ON DISK, not zram. Measured on the previous run: MemAvailable hit 0, 24.6 GiB went to swap,
# 18 GiB of it the two rank processes, and the first generation then had to fault it all back.
#
# THAT REASONING WAS WRONG AND THIS IS OFF BY DEFAULT. `ckpt_read.py` loads through
# `safetensors.safe_open(..., framework="pt")`, which MMAPS the shard and hands out zero-copy
# tensors — so the checkpoint is NOT "read once and done", its pages are the live backing store for
# tensors still being consumed. Dropping page cache mid-load forces them to be re-read at ZFS's mmap
# rate (~550 MiB/s, one ARC lookup per 4 KiB). MEASURED: container CPU went from ~200% to 1005%
# while this ran, and fell straight back to ~200% when it was killed. Left in, off, with the reason,
# because "just drop the caches" is an obvious-looking idea that will be had again.
if [[ "${DROP_CACHES_DURING_LOAD:-0}" == "1" ]] && sudo -n true 2>/dev/null; then
  ( while :; do sleep 20; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || exit 0; done ) &
  DROPPID=$!
  echo "[leg] dropping the loader's page cache every 20s during the load (pid $DROPPID)"
else
  DROPPID=""
  echo "[leg] NOT dropping page cache during load (no passwordless sudo, or disabled)"
fi

FATAL='died unexpectedly|OutOfMemoryError|HIP failure|CUDA calloc|CpuTierError|UnconfiguredDeviceTierError|HSA_STATUS_ERROR'
deadline=$(( $(date +%s) + READY_TIMEOUT ))
ready=0
while (( $(date +%s) < deadline )); do
  st=$(docker inspect --format '{{.State.Status}}' "$CID" 2>/dev/null || echo gone)
  rc=$(docker inspect --format '{{.RestartCount}}' "$CID" 2>/dev/null || echo 0)
  if [[ "$st" != "running" || "$rc" != "0" ]]; then
    echo "[leg] container status=$st RestartCount=$rc -> DEAD"
    docker logs "$CID" 2>&1 | tail -40
    break
  fi
  if docker logs "$CID" 2>&1 | grep -qE "$FATAL"; then
    echo "[leg] fatal signature in the engine log:"
    docker logs "$CID" 2>&1 | grep -nE "$FATAL" | tail -6
    break
  fi
  # READINESS = AN ACTUAL GENERATION RETURNING CONTENT. /health lies: the API binds it before the
  # engine is warm and keeps answering 200 after the worker dies, container Up, RestartCount 0.
  # Sampled, never greedy — a temp-0 probe is not what this serve is measured at.
  # 300 s, NOT 60. The first generation on this configuration can have to fault the CPU tier's
  # pageable expert weights back in from swap before it can answer — 17.6 GiB node-wide, and on
  # this box the highest-priority swap device is a FILE ON DISK, not zram. A 60 s budget timed out
  # on a serve that was answering fine, and the run was then killed by the outer timeout: the
  # harness reported a failure the engine had not committed.
  if python3 "$REPO/tools/offload/_serve_probe.py" --url "http://127.0.0.1:$PORT" --timeout 300 \
       >>"$LOG.probe" 2>&1; then
    echo "[leg] READY (a generation returned non-empty content)"
    [[ -n "$DROPPID" ]] && kill "$DROPPID" 2>/dev/null && DROPPID=""
    ready=1
    break
  fi
  sleep 15
done
if (( ready == 0 )); then
  # A serve that answers /health, accepts requests and tokenizes NONE of them leaves no error to
  # grep. The only evidence is where each process is blocked, and the teardown below destroys it.
  echo "[leg] NOT READY within ${READY_TIMEOUT}s — capturing stacks before teardown"
  CID="$CID" bash "$REPO/tools/offload/wedge_dump.sh" "$LOG.wedge" || true
  echo "[leg] --- metrics at the wedge (which counters moved, and which did not) ---"
  curl -s -m 10 "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
    | grep -E "^minisgl_(requests_total|prompt_tokens|generation_tokens|requests_inflight|waiting_requests|running_requests|kv_pool_used)" \
    | tee -a "$LOG.wedge"
  exit 3
fi

python3 "$REPO/tools/ab_tile_serve_bench.py" --url "http://127.0.0.1:$PORT" \
  --label "$LABEL" --reps "$REPS" --decode-tokens "$DECODE_TOKENS" \
  --decode-m "$DECODE_M" --prefill-words 360 --out "$JSON"
rc=$?
echo "[leg] bench rc=$rc"

# The two instruments this leg is quoted from, pulled while the container is still up.
echo "[leg] --- memory at bench time (a tok/s taken while swapping is a different measurement) ---"
grep -E "MemAvailable|SwapFree" /proc/meminfo
for pid in $(pgrep -f "minisgl-DP0-TP" 2>/dev/null); do
  echo "  rank pid=$pid $(awk '/^VmRSS|^VmSwap/{printf "%s%s ", $1, $2}' /proc/$pid/status 2>/dev/null)"
done
echo "[leg] --- hostprof (loop-stage host partition, MINISGL_HOSTPROF) ---"
docker logs "$CID" 2>&1 | grep -F '[hostprof]' | tail -8
echo "[leg] --- cpu-moe counters (both sides of the seam) ---"
docker logs "$CID" 2>&1 | grep -F '[cpu-moe]' | tail -4
echo "[leg] --- weight-offload plan ---"
docker logs "$CID" 2>&1 | grep -iE 'weight offload|cpu\(compute\)|host\(arena\)|KV pool|layers device' | tail -12
exit $rc
