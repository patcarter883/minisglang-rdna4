#!/usr/bin/env bash
# Does the ROCm 7.14 / clang 23 codegen regression show up in a REAL SERVE?
#
# The static tables say `gdn_decode_kernel` loses a third of its occupancy under clang 23
# (VGPR 112-114 -> 169-179, 12 -> 8 waves/SIMD) and that `flash_prefill_paged_kernel` goes
# 165 -> 197 VGPR (9 -> 7 waves). Neither is evidence of a slower SERVE: occupancy is a symptom,
# and the prefill kernel got 32% SHORTER at the same time, which is the ILP-for-occupancy trade
# that sometimes wins. So: serve the same model on both images and measure.
#
# WHAT IS HELD FIXED. Same engine worktree, same model, same config, same client, same card. The
# ONLY difference is the image and the kernel packages, and BOTH kernel legs are built from the same
# kernels commit — one by each image's own compiler. A leg that fell back to the image's baked
# /opt/kernels would be comparing two different SOURCE commits and would not be a compiler A/B at
# all, so the provenance assertion below is not optional decoration.
#
#   bash tools/toolchain_serve_ab.sh [--model M] [--conc N] [--reps R]
#
# Reads: sections 4 (prefill tok/s -> attn_prefill_paged) and 5 (decode tok/s vs context depth ->
# GDN decode) of tools/serve_perf.py are the two that should move if anything does.
set -uo pipefail

WT=/home/pat/code/minisgl-rdna4-gdnab           # ISOLATED worktree; compose mounts . as /engine
MODEL=qwen35b-awq                               # the production serve: Qwen3.6-35B-A3B-AWQ, GDN MoE
TP=2
CONC=6
REPS=1
PORT=1919
OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --conc)  CONC="$2";  shift 2 ;;
    --tp)    TP="$2";    shift 2 ;;
    --reps)  REPS="$2";  shift 2 ;;
    --out)   OUT="$2";   shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
OUT="${OUT:-$WT/_toolchain_serve_ab.txt}"
: > "$OUT"
say() { echo "$@" | tee -a "$OUT"; }

say "# toolchain serve A/B — ROCm 7.2.1/clang22 vs ROCm 7.14/clang23"
say "# model=$MODEL tp=$TP conc=$CONC reps=$REPS  worktree=$WT"
say "# kernels: both legs built from rdna4-hip-kernels 86fed3a, each by its own image's compiler"
say ""

wait_ready() {  # the serve answers /health once the model is resident and graphs are captured
  local proj="$1" t=0
  while [ $t -lt 900 ]; do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    docker ps --format '{{.Names}}' | grep -q "^${proj#lease-}-serve$" || { sleep 3; t=$((t+3)); continue; }
    sleep 5; t=$((t+5))
  done
  return 1
}

FAIL=0
for leg in 72 714; do
  case $leg in
    72)  IMG=minisgl-rdna4:lean    ; DESC="ROCm 7.2.1 / clang 22" ;;
    714) IMG=minisgl-rdna4:lean714 ; DESC="ROCm 7.14  / clang 23" ;;
  esac
  PROJ="lease-tcab$leg"
  say "################################################################"
  say "########## LEG $leg — $DESC ($IMG)"
  say "################################################################"

  gpu-lease -n $TP --detach --name "tcab$leg" -- bash -c "
    cd $WT && env \
      MINISGL_IMAGE='$IMG' \
      MINISGL_PYTHONPATH='/engine/_kern$leg:/opt/kernels:/engine/python:/engine' \
      PYTHONDONTWRITEBYTECODE=1 \
      MINISGL_HOST_PORT=$PORT \
      MODEL='$MODEL' TP=$TP CONC=$CONC SPEC=none MEM_RATIO=0.80 GRAPH_BS=$CONC \
      LEASE_NAME='tcab$leg' \
      docker compose -p '$PROJ' --profile serve up -d" >/dev/null 2>&1

  if ! wait_ready "$PROJ"; then
    say "  BOOT FAILED — logs:"
    docker logs "${PROJ#lease-}-serve" 2>&1 | tail -30 | sed 's/^/    /' | tee -a "$OUT"
    ( cd "$WT" && MINISGL_IMAGE="$IMG" docker compose -p "$PROJ" --profile serve down ) >/dev/null 2>&1
    FAIL=1; continue
  fi

  # ---- PROVENANCE. Which .so is actually loaded, and was it built by THIS leg's compiler? ----
  # `import gdn_hip; gdn_hip.__file__` is the only answer that cannot be assumed: /opt/kernels comes
  # SECOND on the path here, so a typo in MINISGL_PYTHONPATH silently serves the image's baked
  # kernels and the whole comparison collapses into "image A vs image A".
  PROV=$(docker exec "${PROJ#lease-}-serve" python -c "
import gdn_hip, attn_prefill_paged, fp8_wmma, moe_hip, hashlib, pathlib
for m in (gdn_hip, attn_prefill_paged, fp8_wmma, moe_hip):
    p = pathlib.Path(m.__file__).parent
    so = sorted(p.glob('*.so'))
    h = hashlib.sha256(so[0].read_bytes()).hexdigest()[:12] if so else 'NO-SO'
    print(f'{m.__name__}={p} sha={h}')
" 2>/dev/null)
  say "  provenance:"; echo "$PROV" | sed 's/^/    /' | tee -a "$OUT"
  if ! grep -q "_kern$leg" <<<"$PROV"; then
    say "  PROVENANCE FAIL: kernels did not come from /engine/_kern$leg — A/B void"
    FAIL=1
  fi

  for r in $(seq 1 "$REPS"); do
    say ""
    say "  ---- serve_perf rep $r ----"
    python3 "$WT/tools/serve_perf.py" --base "http://localhost:$PORT" --conc "$CONC" \
      --container "${PROJ#lease-}-serve" 2>&1 | sed 's/^/    /' | tee -a "$OUT"
  done

  ( cd "$WT" && MINISGL_IMAGE="$IMG" docker compose -p "$PROJ" --profile serve down ) >/dev/null 2>&1
  # The lease is bound to the container (--detach), so it frees when compose brings it down.
  say ""
done

say ""
say "########## READ THIS AS ##########"
say "section 5 (decode tok/s vs context depth) is the GDN decode path — the kernel whose occupancy"
say "fell 12 -> 8. Section 4 (prefill tok/s) is attn_prefill_paged. Section 1/3 also carry GDN decode."
say "If 4 and 5 are both flat, the static regression does not reach the serve."
[ "$FAIL" = 0 ] || say "!! one or more legs failed or lost provenance — see above"
exit "$FAIL"
