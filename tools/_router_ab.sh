#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; TOKS="${TOKS:-256}"; OUT="$REPO/tools/_router_ab_results.txt"
for p in /home/pat/code/rdna4-hip-kernels-mmax/fp8_wmma/torch-ext/fp8_wmma \
         /home/pat/code/rdna4-hip-kernels-build/moe/torch-ext/moe_hip; do
  [ -f "$p/_ops.py" ] || { echo "FATAL: $p not built"; exit 1; }
done
wait_ready(){ for _ in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
run(){ local v="$1" label="$2"
  echo "=== $label (MINISGL_ROUTER_FUSED=$v) ===" | tee -a "$OUT"
  LEASE_NAME=rab MINISGL_IMAGE=minisgl-rdna4:lean \
    MINISGL_GDN_PROJ_GEMV=1 MINISGL_MINV_DECODE_GEMV=1 VLLM_W4A8_MOE_G2FUSE_BYLANE=1 \
    MINISGL_ROUTER_FUSED="$v" MINISGL_EXTRA_ARGS="--no-gdn-radix" \
    gpu-lease -n 2 --detach --name rab -- docker compose -p lease-rab --profile serve up -d >/dev/null 2>&1
  if ! wait_ready; then echo "  BOOT FAILED — container left up:" | tee -a "$OUT"
    docker compose -p lease-rab --profile serve logs --tail 15 2>&1 | sed 's/^/    /' | tee -a "$OUT"; return 1; fi
  python3 "$REPO/tools/_hostloop_driver.py" "$TOKS" "$REPO/tools/_router_out_$label.txt" 2>&1 | grep tok/s | sed 's/^/  bs=1 /' | tee -a "$OUT"
  docker compose -p lease-rab --profile serve down >/dev/null 2>&1; sleep 5; }
: > "$OUT"
run 0 torch_chain
run 1 fused
echo "=== done ===" | tee -a "$OUT"
