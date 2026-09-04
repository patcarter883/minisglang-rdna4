#!/usr/bin/env bash
# Paired fp32-vs-VNNI per-core measurement on a DDR-resident table.
#
# PAIRED and INTERLEAVED on purpose: the box is shared with a graph-capture workflow whose load
# drifts over minutes, so running all of one policy then all of the other would confound policy
# with box state.  Each rep runs both policies back to back and records the load either side.
#
# usage: run_percore_ab.sh <tag> <reps> [policies...]
set -u
cd "$(dirname "$0")"
TAG=${1:-run}; REPS=${2:-3}; shift 2 || true
POLICIES=("$@"); [ ${#POLICIES[@]} -eq 0 ] && POLICIES=(e4m3 vnni)
OUT=${OUT:-/tmp/claude-1000/-home-pat-code-minisgl-rdna4/36039161-cc69-402f-a0b6-b5813c739e8b/scratchpad}
PLANS="--plan fixture/L0/plan.txt --plan fixture/L1/plan.txt --plan fixture/L2/plan.txt --plan fixture/L3/plan.txt"

load() {  # busy hw-threads over a 2 s window, plus loadavg and swap traffic
  awk '/^cpu /{i=$5+$6; t=0; for(j=2;j<=11;j++)t+=$j; print i, t}' /proc/stat > /tmp/.st1
  local s1 s2; s1=$(awk '{print $1" "$2}' /proc/stat | head -0); sleep 2
  awk '/^cpu /{i=$5+$6; t=0; for(j=2;j<=11;j++)t+=$j; print i, t}' /proc/stat > /tmp/.st2
  local i1 t1 i2 t2; read i1 t1 < /tmp/.st1; read i2 t2 < /tmp/.st2
  local busy; busy=$(awk -v i1=$i1 -v t1=$t1 -v i2=$i2 -v t2=$t2 'BEGIN{printf "%.2f", 16*(1-(i2-i1)/(t2-t1))}')
  local la; la=$(cut -d' ' -f1-3 /proc/loadavg)
  local sw; sw=$(vmstat 1 2 | tail -1 | awk '{print "si="$7" so="$8}')
  echo "busy_hwthreads=${busy}/16 loadavg=${la} ${sw}"
}

for r in $(seq 1 "$REPS"); do
  for p in "${POLICIES[@]}"; do
    echo "### rep=$r policy=$p"
    echo "# load_before: $(load)"
    ./cpu_moe_layer_vnni $PLANS --mode bench --policy "$p" \
        --threads 16 --iters "${ITERS:-300}" --warm 30 2>&1
    echo "# load_after:  $(load)"
  done
done | tee "$OUT/percore_ab_${TAG}.txt"
