#!/usr/bin/env bash
# Both served A/Bs in ONE lease: the fused arm (bs=5/6) and the scatter arm (bs=1).
#   gpu-lease -n 2 --timeout 10800 -- bash tools/moe_g2_split_serve_all.sh
# SPEC=none throughout, so bs=n gives M=n and the served points line up with the swept surface.
set -uo pipefail
WT="${WT:-/home/pat/code/minisgl-rdna4-g2sk}"
for arm in fused scatter; do
  echo ""
  echo "################################ ARM=$arm ################################"
  ARM="$arm" OUT="$WT/tools/_fixtures/moe_g2_split_serve_ab_$arm.txt" \
    bash "$WT/tools/moe_g2_split_serve_ab.sh"
  echo "arm=$arm rc=$?"
done
