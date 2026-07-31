#!/usr/bin/env bash
# FIX GATE 4 — are the review fixes BEHAVIOUR-NEUTRAL on the shipped paths?
#
# The fixes touch the propose-capture stats API, the verify-width ladder cap (now derived from the
# kernels' own M thresholds, and int4-aware), the pad/truncate decision (moved into one pure
# function), and DDTree verify capture. Three of those four sit directly on the serving path, so the
# claim that needs evidence is: on the SHIPPED configurations, the emitted greedy text is BYTE-
# IDENTICAL to 753f08d2 (the head the review examined).
#
# Legs: Laguna+DFlash (e2m1 -> ceiling 16, ladder [3,7,15]) and Qwen3.6-35B-AWQ+MTP (int4 -> ceiling
# 8, K=4 so ladder [2,3,4] UNCHANGED — the int4 cap is a no-op on everything this repo ships, and
# that is exactly what this leg has to demonstrate rather than assert).
#
# TWO KNOWN NOISE SOURCES ARE FORCED OFF on every leg, because a determinism gate with a nonzero
# noise floor proves nothing: MINISGL_KV_FP8=0 (fp8 KV made MTP non-reproducible against ITSELF)
# and MINISGL_MOE_G2FUSE=0 (quant/kernels.py says in place that the fused MoE gemm2's atomic
# reduction order VARIES run to run — observed live: Laguna+DDTree repeated the same greedy request
# in one boot and differed at 256 tokens). Each request is still issued TWICE per boot to show the
# floor rather than assume it.
#
#   gpu-lease -n 2 -- bash tools/fixgate_neutral.sh
set -uo pipefail

NEW_WT=${NEW_WT:-/home/pat/code/minisgl-rdna4-propose}
OLD_WT=${OLD_WT:-/home/pat/code/minisgl-rdna4-prefix753}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
OUT=${OUT:-$NEW_WT/tools/fixgate_neutral.txt}
export TP=2 CONC=1 MINISGL_KV_FP8=0 MINISGL_MOE_G2FUSE=0
: > "$OUT"

down() { ( cd "$1" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }
alldown() { down "$NEW_WT"; down "$OLD_WT"; }
trap alldown EXIT INT TERM

wait_ready() { for _ in $(seq 1 300); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && return 0
    sleep 2; done; return 1; }

drive() { python3 - <<'PY'
import hashlib, json, urllib.request
BASE = "http://localhost:1919"
PROMPT = "Write a Python function that reverses a singly linked list in place, then explain it."
try:
    M = json.loads(urllib.request.urlopen(f"{BASE}/v1/models", timeout=30).read())["data"][0]["id"]
except Exception as e:                                                       # noqa: BLE001
    print(f"  NO SERVER: {type(e).__name__}: {e}"); raise SystemExit
for MT in (32, 64, 128, 256):
    md5s = []
    for _ in (1, 2):
        b = {"model": M, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": MT,
             "temperature": 0.0, "seed": 1234, "stream": False}
        r = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(b).encode(),
                                   headers={"Content-Type": "application/json"})
        try:
            d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
        except Exception as e:                                               # noqa: BLE001
            md5s.append(f"ERR:{type(e).__name__}"); continue
        m = d["choices"][0]["message"]
        t = (m.get("reasoning_content") or "") + "\0" + (m.get("content") or "")
        md5s.append(hashlib.md5(t.encode()).hexdigest())
    print(f"  MT={MT:4d}  md5={md5s[0]}  repeat={'SAME' if md5s[0] == md5s[-1] else md5s[-1]}")
PY
}

leg() {  # leg <tag> <worktree> <MODEL> <SPEC> [extra env...]
  local tag=$1 wt=$2 model=$3 spec=$4; shift 4
  echo "=== $tag  model=$model spec=$spec ===" | tee -a "$OUT"
  alldown
  ( cd "$wt" && env MINISGL_IMAGE="$IMAGE" MODEL="$model" SPEC="$spec" "$@" \
      docker compose --profile serve up -d >/dev/null 2>&1 )
  if wait_ready; then
    local c; c=$( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -q serve )
    echo -n "  PROVENANCE width.py/scheduler.py md5: " | tee -a "$OUT"
    docker exec "$c" sh -c \
      "md5sum /engine/python/minisgl/spec/width.py /engine/python/minisgl/scheduler/scheduler.py | cut -d' ' -f1 | tr '\n' ' '" \
      2>/dev/null | tee -a "$OUT"; echo "" | tee -a "$OUT"
    drive 2>&1 | tee -a "$OUT"
  else
    echo "  NEVER BECAME READY" | tee -a "$OUT"
  fi
  ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
    | grep -aE "ADAPTIVE verify width|PROPOSE graphs CAPTURED|M thresholds have DRIFTED" \
    | sed 's/^[^ ]* *| *//' | tail -3 | tee -a "$OUT"
  down "$wt"; echo "" | tee -a "$OUT"
}

leg "753f08d2 laguna" "$OLD_WT" laguna     dflash MEM_RATIO=0.93 GRAPH_BS=8
leg "FIXED    laguna" "$NEW_WT" laguna     dflash MEM_RATIO=0.93 GRAPH_BS=8
leg "753f08d2 qwen35b" "$OLD_WT" qwen35b-awq mtp
leg "FIXED    qwen35b" "$NEW_WT" qwen35b-awq mtp
echo "== FIX GATE 4 COMPLETE ==" | tee -a "$OUT"
