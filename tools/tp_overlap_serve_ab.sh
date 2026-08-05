#!/usr/bin/env bash
# tp_overlap_serve_ab.sh — served A/B for the TP comms/compute overlap seam (layers/tp_overlap.py).
#
# The two arms differ in ONE thing: MINISGL_TP_OVERLAP. Same image, same worktree, same custom_ar .so,
# same model, same prompts, same seeds.
#
#   OFF = MINISGL_TP_OVERLAP=0 — every all_reduce runs inline on the compute stream (the old behaviour)
#   ON  = MINISGL_TP_OVERLAP=1 — the dense-branch all_reduce rides the side stream under the MoE GEMM
#
# PROVENANCE IS ASSERTED, NOT ASSUMED. Setting an env var proves nothing about what the GPU did, so
# each arm is checked against the engine's own log line, which the seam emits from the first collective
# that ACTUALLY moves to the side stream. ON must print it; OFF must not. If that check fails the arms
# were identical and the A/B measured nothing, so it aborts rather than reporting a number.
#
# Three measurements, because they answer different questions:
#
#   PREFILL — a ~3200-token prompt at max_tokens=1. TTFT here IS the prefill latency (nothing else has
#             happened yet), so prefill tok/s = prompt_tokens / TTFT. This is the workload the seam
#             targets: chunked prefill is always eager, so it is where overlap is possible at all.
#   AR GUARD — 256-token greedy completions. bs=1 decode is CAPTURED, where the seam degrades to the
#             inline collective by design, so this must come out FLAT. It is a guard, not a target: a
#             change here means the capture-transparency claim is wrong.
#   TEXT    — the prefill arm runs greedy, and the first sampled token is a hard identity gate.
#
# Usage:  gpu-lease -n 2 --timeout 5400 -- bash tools/tp_overlap_serve_ab.sh
set -uo pipefail

ENGINE="${ENGINE:-$(cd "$(dirname "$0")/.." && pwd)}"
KERNELS="${KERNELS:-/home/pat/code/_k-tpoverlap}"
IMAGE="${MINISGL_IMAGE:-minisgl-rdna4:gemma4}"
MODEL="${MODEL:-cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4}"
OUT="${OUT:-$ENGINE/_tp_overlap_ab}"
PORT=1919
mkdir -p "$OUT"

run_arm() {  # $1 = arm name, $2 = MINISGL_TP_OVERLAP, $3 = MINISGL_CAR_MAX_MIB ("" = default)
  local arm="$1" ov="$2" carmib="${3:-}" name="tpov-$1"
  echo "== arm $arm (MINISGL_TP_OVERLAP=$ov) ============================================"
  # STALE RESULTS ARE THE ENEMY. If this arm dies, the verdict must have nothing to read rather than
  # last run's file -- that mistake compared two different engines and reported it as a +36% win.
  rm -f "$OUT/$arm.json" "$OUT/$arm.serve.log" "$OUT/$arm.bench.log" "$OUT/$arm.boot.log"
  docker rm -f "$name" >/dev/null 2>&1
  docker run -d --name "$name" \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
    -e HF_HUB_OFFLINE=1 -e MINISGL_TP_OVERLAP="$ov" -e MINISGL_CAR_MAX_MIB="$carmib" \
    -e PYTHONPATH=/engine/python:/engine:/opt/kernels \
    -e MODEL="$MODEL" -e TP=2 -e SPEC=none -e PORT=$PORT -e CONC=1 \
    -p $PORT:$PORT \
    -v "$ENGINE":/engine \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    --entrypoint bash "$IMAGE" -lc 'bash /engine/tools/serve.sh' >/dev/null

  # Readiness: ask the server, do not trust a sleep. /v1/models only answers once the engine is up.
  local ready=0 i=0
  for i in $(seq 1 150); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then
      echo "!! $arm container DIED after ${i}x4s — last 30 log lines:"
      docker logs "$name" 2>&1 | tail -30 | tee "$OUT/$arm.boot.log"
      docker rm -f "$name" >/dev/null 2>&1; return 1
    fi
    # A crash loop and a slow boot look the same from outside; at 60s, go and read the log.
    if [ "$i" = 15 ]; then
      echo "-- $arm not ready at 60s; recent log:"; docker logs --tail 8 "$name" 2>&1 | sed 's/^/   /'
    fi
    sleep 4
  done
  if [ "$ready" != 1 ]; then
    echo "!! $arm never became ready (600s)"; docker logs "$name" 2>&1 | tail -40 > "$OUT/$arm.boot.log"
    docker rm -f "$name" >/dev/null 2>&1; return 1
  fi

  python3 "$ENGINE/tools/tp_overlap_client.py" --base "http://127.0.0.1:$PORT" \
      --out "$OUT/$arm.json" 2>&1 | tee "$OUT/$arm.bench.log"

  docker logs "$name" > "$OUT/$arm.serve.log" 2>&1
  docker rm -f "$name" >/dev/null 2>&1
  sleep 5
}

# Free the cards immediately on any failure — this box has two, and other agents are waiting on them.
trap 'docker rm -f "tpov-${ARM_A:-off}" "tpov-${ARM_B:-on}" >/dev/null 2>&1' EXIT
ARM_A="${ARM_A:-off}"; ARM_A_OV="${ARM_A_OV:-0}"; ARM_A_CAR="${ARM_A_CAR:-8}"
ARM_B="${ARM_B:-on}";  ARM_B_OV="${ARM_B_OV:-1}"; ARM_B_CAR="${ARM_B_CAR:-8}"
run_arm "$ARM_A" "$ARM_A_OV" "$ARM_A_CAR" || { echo "!! arm $ARM_A failed — aborting before it can be compared against stale data"; exit 1; }
run_arm "$ARM_B" "$ARM_B_OV" "$ARM_B_CAR" || { echo "!! arm $ARM_B failed — aborting"; exit 1; }

# ---- provenance gate ------------------------------------------------------------------------------
echo
hit_off=$(grep -c "comms/compute overlap ENGAGED" "$OUT/$ARM_A.serve.log" 2>/dev/null; true)
hit_on=$(grep -c "comms/compute overlap ENGAGED" "$OUT/$ARM_B.serve.log" 2>/dev/null; true)
echo "== provenance: 'overlap ENGAGED' log lines — off=$hit_off  on=$hit_on"
grep -m1 "comms/compute overlap ENGAGED" "$OUT/$ARM_B.serve.log" 2>/dev/null || true
echo "== provenance: custom_ar slot size per arm"
for a in "$ARM_A" "$ARM_B"; do printf '   %-8s ' "$a"; grep -m1 -o "slot [0-9.]* MiB" "$OUT/$a.serve.log" || echo "(RCCL only)"; done
if [ "$ARM_A_OV" = 0 ] && { [ "$hit_off" != "0" ] || [ "$hit_on" = "0" ]; }; then
  echo "!! ABORT: the two arms did not actually differ (off must be 0, on must be >0)."
  exit 1
fi

python3 - "$OUT/$ARM_A.json" "$OUT/$ARM_B.json" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
print("\n================================ VERDICT ================================")
print(f"{'metric':<34s} {'ARM A':>12s} {'ARM B':>12s} {'delta':>10s}")
for k, label, better in (
    ("prefill_ttft_s",   "prefill TTFT (s), 3200-tok prompt", "lower"),
    ("prefill_tok_s",    "prefill tok/s",                     "higher"),
    ("ar_tok_s",         "AR guard tok/s (captured decode)",  "flat"),
):
    x, y = a.get(k), b.get(k)
    if x is None or y is None:
        print(f"{label:<34s} {'--':>12s} {'--':>12s}"); continue
    d = (y / x - 1) * 100
    print(f"{label:<34s} {x:12.2f} {y:12.2f} {d:+9.1f}%   ({better})")
same = a.get("prefill_text") == b.get("prefill_text")
print(f"\nprefill first-token TEXT identical: {same}   <- the losslessness gate")
if not same:
    print(f"  off: {a.get('prefill_text')!r}\n  on : {b.get('prefill_text')!r}")
PY
