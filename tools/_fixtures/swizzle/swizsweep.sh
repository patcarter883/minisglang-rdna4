#!/usr/bin/env bash
# swiz=0 vs swiz=1 across the dense fixture at prefill M. Tile fixed at 256x128 (what the harness
# compiles), so this derives the RULE's shape dependence, not its tile dependence.
set -uo pipefail
C=${LEASE_ROCR_DEVICES%%,*}
run() { docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --ipc host --shm-size 16gb \
  -e ROCR_VISIBLE_DEVICES=$C -e HIP_VISIBLE_DEVICES=0 -e W4A8_SWIZ=$1 -v /sp:/sp \
  --entrypoint /sp/w4a8_swz rdna4-rocm7.14:latest "${@:2}" 2>&1 | sed -n 's/.* \([0-9.]*\) TFLOPS.*/\1/p'; }
# name K N g   (the fixture's dense linears, plus the Muse NVFP4 column)
while read -r name K N g; do
  [ -z "$name" ] && continue
  for M in 64 256 1024 2048; do
    b1=0; b0=0
    for i in 1 2; do
      a=$(run 1 tuned $M $N $K 5 $g); b=$(run 0 tuned $M $N $K 5 $g)
      awk -v x="$a" -v y="$b1" 'BEGIN{exit !(x>y)}' && b1=$a
      awk -v x="$b" -v y="$b0" 'BEGIN{exit !(x>y)}' && b0=$b
    done
    printf '%s,%s,%s,%s,%s,%s,%s\n' "$name" "$K" "$N" "$g" "$M" "${b1:-0}" "${b0:-0}"
  done
done
