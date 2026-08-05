#!/usr/bin/env bash
# avail_check.sh — STEP 0: what does each ROCm stack ADVERTISE for gfx1201, and does it claim PMC
# support? Runs NO workload and programs NO counters, so it is the lowest-risk probe available.
#
# It answers one question the rest of this directory cannot: our counter NAME list was taken from the
# host 7.2.4 `counter_defs.yaml`, so an all-zero result could in principle mean "7.14 renamed these and
# we asked for dead aliases" rather than "the events are dead". Diffing the advertised set across
# stacks settles that. It is also the check nobody ran before the "counters wedge gfx1201" rule was
# generalised from a single non-minimal observation (torch loaded, serve image, real workload).
#
#   gpu-lease -n 2 -- bash tools/counter_probe/avail_check.sh            # 7.14 container
#   ROCM_HOST=1 gpu-lease -n 2 -- bash tools/counter_probe/avail_check.sh  # host 7.2.4
set -uo pipefail
OUT="${OUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results}"
mkdir -p "$OUT"

LADDER='
set -uo pipefail
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
for d in /opt/rocprof-deps /opt/venv/lib/python3.12/site-packages/torch/lib; do
  [ -f "$d/libdw.so.1" ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$d"
done
V=$(rocprofv3 --version 2>&1 | awk "/rocm_version/{print \$2; exit}")
echo "### rocm=$V rocprofiler-sdk=$(rocprofv3 --version 2>&1 | awk "/version:/{print \$2; exit}")"

echo "=== pmc-check (no workload, no counters programmed) ==="
timeout -s KILL 120 rocprofv3-avail pmc-check 2>&1 | tail -20

echo "=== advertised counter NAMES for gfx1201 ==="
timeout -s KILL 120 rocprofv3-avail list 2>&1 \
  | grep -oE "Counter_Name[[:space:]]*:[[:space:]]*[A-Za-z0-9_]+" \
  | awk -F: "{gsub(/ |\t/,\"\",\$2); print \$2}" | sort -u > /tmp/names.txt
echo "count=$(wc -l < /tmp/names.txt)"
echo "--- are the ones the utilisation breakdown needs advertised? ---"
for c in SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INST_CYCLES_VALU \
         SQ_ACTIVE_INST_VALU SQ_THREAD_CYCLES_VALU GRBM_GUI_ACTIVE VALUBusy MemUnitBusy \
         OccupancyPercent FetchSize; do
  grep -qx "$c" /tmp/names.txt && echo "  ADVERTISED  $c" || echo "  absent      $c"
done
cp /tmp/names.txt /out/avail_names_${V}.txt 2>/dev/null
'

if [ "${ROCM_HOST:-0}" = "1" ]; then
  echo "==== TARGET: HOST /opt/rocm ===="
  OUT_DIR="$OUT" bash -c "${LADDER//\/out\//$OUT/}"
  exit $?
fi

IMG="${IMG:-rocm/dev-ubuntu-24.04:7.14.0-full}"
echo "==== TARGET: $IMG ===="
docker run --rm \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb -v "$OUT":/out \
  --entrypoint bash "$IMG" -lc "$LADDER"
