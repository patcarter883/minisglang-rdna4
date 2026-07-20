#!/usr/bin/env bash
# Runs INSIDE the lean serving container (launched by run_seed_ab.sh under the 2-card lease).
# A/B for the MTP prompt-prefill draft-KV seed: for SEED in 0 (baseline) and 1 (seeded), boot the
# 35B MTP spec-decode serve under --graph, probe REAL-token throughput + coherence + accept-len +
# decode power, then tear down. Also a no-spec base run for the losslessness anchor.
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate

MODEL="${MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"
TP="${TP:-2}"; PORT="${PORT:-1919}"
MEMRATIO="${MEMRATIO:-0.80}"; MAXRUN="${MAXRUN:-4}"; GRAPH="${GRAPH:-4}"
ATTN="${ATTN:-hip}"; SPEC_K="${SPEC_K:-4}"; CONC="${CONC:-4}"
RESULTS=/engine/tools/tp2_results; mkdir -p "$RESULTS"

python -c "import gdn_hip, moe_hip, tail_hip, mla_hip; print('[setup] hip pkgs import OK')" \
  || { echo '[setup] hip pkg import FAILED'; exit 1; }

SRV=""
launch() {  # $1 = tag, $2 = extra env assignments (space sep), $3 = spec_args
  local tag="$1" envs="$2" specargs="$3" log="$RESULTS/seed_$1.server.log"
  echo "[launch:$tag] envs='$envs' spec='$specargs' -> $log"
  local pynccl=""; [ "$TP" -gt 1 ] && pynccl="--disable-pynccl"
  setsid env MINISGL_MOE_SCATTER=0 MINISGL_SPEC_DEBUG=1 MINISGL_SPEC_TIMING=1 $envs python -m minisgl \
    --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" --host 0.0.0.0 --graph "$GRAPH" \
    --attention-backend "$ATTN" $pynccl --memory-ratio "$MEMRATIO" --max-running-requests "$MAXRUN" \
    $specargs \
    > "$log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 400); do
    if python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null; then
      echo "[launch:$tag] ready"; return 0
    fi
    kill -0 "$SRV" 2>/dev/null || { echo "[launch:$tag] server DIED:"; tail -50 "$log"; return 1; }
    if grep -qE '^Process minisgl-|^Traceback \(most recent call last\)|torch\.OutOfMemoryError|RuntimeError: No HIP GPUs|^[A-Za-z_.]*Error: |CUDA error:' "$log" 2>/dev/null; then
      echo "[launch:$tag] worker CRASHED:"; tail -60 "$log"; return 1
    fi
    sleep 3
  done
  echo "[launch:$tag] not ready in time:"; tail -50 "$log"; return 1
}
stop() {
  [ -n "$SRV" ] || return 0
  kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""
  for _ in $(seq 1 20); do
    python -c "import socket,sys;s=socket.socket();r=s.connect_ex(('127.0.0.1',$PORT));s.close();sys.exit(0 if r!=0 else 1)" 2>/dev/null && break
    sleep 1
  done
  sleep 2
}
trap stop EXIT

SPEC_ARGS="--spec-algorithm mtp --spec-num-draft ${SPEC_K} --reasoning-parser auto"

run_one() {  # $1 = tag, $2 = envs, $3 = specargs, $4 = probe-extra
  local tag="$1"
  launch "$tag" "$2" "$3" || { echo "[FAIL] $tag did not boot"; return 1; }
  grep -iE 'captur|cuda.?graph|prefill draft-KV seed|BUFFERED propose' "$RESULTS/seed_$tag.server.log" | head -6 || true
  python /engine/tools/seed_probe.py --url "http://127.0.0.1:$PORT" --tag "$tag" \
    --out "$RESULTS/seed_$tag.probe.json" --conc "$CONC" --decode-tokens "${DECODE_TOKENS:-256}" $4 || true
  echo "===== [$tag] server accept stats (last [spec]) ====="
  grep -E '\[spec\] step=|\[spec\] mean accept|\[spec-timing\]' "$RESULTS/seed_$tag.server.log" | tail -8 || true
  stop
}

MODE="${MODE:-all}"
echo "######## MTP prompt-prefill draft-KV seed  MODE=$MODE ($MODEL, TP=$TP, graph=$GRAPH) ########"
case "$MODE" in
  base_nospec) run_one base_nospec "" "--reasoning-parser auto" "--skip-power" ;;
  seed_off)    run_one seed_off "" "$SPEC_ARGS" "" ;;
  seed_on)     run_one seed_on "MINISGL_SPEC_PREFILL_SEED=1" "$SPEC_ARGS" "" ;;
  all)
    run_one base_nospec "" "--reasoning-parser auto" "--skip-power"
    run_one seed_off "" "$SPEC_ARGS" ""
    run_one seed_on "MINISGL_SPEC_PREFILL_SEED=1" "$SPEC_ARGS" "" ;;
  focus)
    # A/A determinism floor + conc=1 tok/s leverage (power already measured in the `all` run).
    run_one seed_off "" "$SPEC_ARGS" "--aa --skip-power"
    run_one seed_on "MINISGL_SPEC_PREFILL_SEED=1" "$SPEC_ARGS" "--aa --skip-power" ;;
  *) echo "unknown MODE=$MODE"; exit 1 ;;
esac
echo "[done] artifacts in $RESULTS/ (seed_*.probe.json, seed_*.server.log)"
