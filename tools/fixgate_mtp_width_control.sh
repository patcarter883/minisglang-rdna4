#!/usr/bin/env bash
# FIX GATE 3 — the MTP losslessness objection, answered with the only control that can settle it.
#
# THE OBJECTION (review, BLOCKING): Qwen3.6-35B-A3B-AWQ + MTP emits different greedy text under the
# adaptive width than under the pre-change fixed width, at max_tokens 128 and 256, with a zero
# in-boot noise floor. Isolated to the ADAPTIVE WIDTH (pinning a single captured width makes the new
# tree byte-identical to the pre-change run), and the widths in play (qlen 3/4/5) are all below M=16,
# so the documented M<=16 cliff does not explain it. The reviewer's decisive point: "unlike DFlash
# there is NO fixed pre-change width that reproduces the new default".
#
# WHAT THIS RUN TESTS, and why it is the right test. If a WIDTH CHANGE moves the greedy text on the
# UNTOUCHED engine, then no width-VARYING scheme can ever reproduce a single fixed width, and
# "no fixed pre-change width reproduces the new default" is a restatement of that property, not
# evidence of a defect in the controller. So: take the untouched PARENT of every commit in this
# phase (d276137c — no capture.py, no width.py, propose eager, fixed width) and run it at
# --spec-num-draft 4, 3 and 2, one greedy prompt, each request issued TWICE, with BOTH known
# nondeterminism sources forced off (MINISGL_KV_FP8=0, MINISGL_MOE_G2FUSE=0 — the fused MoE gemm2's
# atomic reduction order varies run to run, by its own docstring).
#
#   * If K=4 / K=3 / K=2 give the SAME text -> the engine IS width-invariant and the adaptive width
#     really did introduce a difference. That would be a genuine defect.
#   * If they give DIFFERENT text, with the in-boot repeats identical (zero noise floor), then
#     width -> text is a pre-existing property of this engine's MoE verify forward, and the adaptive
#     width inherits it.
#
# In exact arithmetic the emitted sequence is the TARGET's own greedy continuation regardless of
# width (verify_greedy emits target argmax tokens; drafts only decide how many arrive per step), so
# any width-driven text change is a rounding effect, not a logical one. This run measures whether
# that rounding effect predates the change.
#
#   gpu-lease -n 2 -- bash tools/fixgate_mtp_width_control.sh
set -uo pipefail

NEW_WT=${NEW_WT:-/home/pat/code/minisgl-rdna4-propose}
PARENT_WT=${PARENT_WT:-/home/pat/code/minisgl-rdna4-vparent}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
OUT=${OUT:-$NEW_WT/tools/fixgate_mtp_width_control.txt}
export MODEL="${MODEL:-qwen35b-awq}" TP=2 CONC=1
export MINISGL_KV_FP8=0 MINISGL_MOE_G2FUSE=0     # fp8 KV made MTP non-reproducible against ITSELF; a gate needs 0 noise
: > "$OUT"

down() { ( cd "$1" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }
alldown() { down "$NEW_WT"; down "$PARENT_WT"; }
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

leg() {  # leg <tag> <worktree> <SPEC_K or ''>
  local tag=$1 wt=$2 k=$3
  echo "=== $tag  (SPEC_K=${k:-default}) ===" | tee -a "$OUT"
  alldown
  ( cd "$wt" && env MINISGL_IMAGE="$IMAGE" SPEC=mtp SPEC_K="$k" \
      docker compose --profile serve up -d >/dev/null 2>&1 )
  if wait_ready; then
    local c; c=$( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -q serve )
    echo -n "  PROVENANCE spec/width.py present? " | tee -a "$OUT"
    docker exec "$c" sh -c \
      "test -f /engine/python/minisgl/spec/width.py && md5sum /engine/python/minisgl/spec/width.py | cut -d' ' -f1 || echo ABSENT" \
      2>/dev/null | tee -a "$OUT"
    echo -n "  PROVENANCE env KV_FP8/SPEC_K: " | tee -a "$OUT"
    docker exec "$c" sh -c "tr '\0' '\n' < /proc/1/environ | grep -E 'KV_FP8|SPEC_K=' | tr '\n' ' '" \
      2>/dev/null | tee -a "$OUT"; echo "" | tee -a "$OUT"
    drive 2>&1 | tee -a "$OUT"
  else
    echo "  NEVER BECAME READY" | tee -a "$OUT"
  fi
  ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
    | grep -aE "ADAPTIVE verify width|PROPOSE graphs CAPTURED|verify graphs captured|num_draft" \
    | sed 's/^[^ ]* *| *//' | tail -4 | tee -a "$OUT"
  down "$wt"; echo "" | tee -a "$OUT"
}

# THE CONTROL: three fixed widths on the UNTOUCHED PARENT. Nothing from this phase is in that tree.
leg "PARENT K=4"  "$PARENT_WT" 4
leg "PARENT K=3"  "$PARENT_WT" 3
leg "PARENT K=2"  "$PARENT_WT" 2
# ...and the shipped tree, for the record (ladder + engagement).
leg "NEW default" "$NEW_WT"    ""
echo "== FIX GATE 3 COMPLETE ==" | tee -a "$OUT"
