#!/usr/bin/env bash
# Plugin/backend isolation matrix, re-run now that the GDN HIP path is fixed.
#
# WHY RE-RUN: the original matrix (memory `vllm24-hip-gdn-decomp-and-fused-decode-lever`) found
# every non-GDN change worth ~0 — but that was measured at 46 tok/s where GDN WAS the ceiling, so
# everything else was masked. GDN is now 83.3 (stride-aware paged state + fused conv/decode), and
# stock is 92, so the residual ~9% should now be attributable.
#
# All rows are NO-SPEC at one config (32768 ctx, MNS=8, fp8 KV, TP=2) so they compare directly to the
# recorded all-stock 92. Metric is server-side DECODE tok/s (excludes TTFT/HTTP), plus e2e.
#
# Boots are ~130 s thanks to the persistent attn-autotune + torch.compile caches (see compose).
# Row 6 (all-stock GDN) may still pay a one-off Triton compile — it is last for that reason.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO="$PWD"
OUT="${OUT:-$REPO/tools/vhip_patches/plugin_matrix.tsv}"
HIP="gdn_hip,tail_hip_register,w4a8_fp8_wmma_register"

printf 'row\tplugins\tattn_backend\textra\tdecode_tok_s\te2e_tok_s\tboot_s\n' > "$OUT"

run_row() {
  local name="$1" plugins="$2" backend="$3" extra="$4"
  echo "=== $name | plugins=[$plugins] attn=$backend extra=[$extra] ==="
  docker compose -p lease-vhip-qwen down >/dev/null 2>&1; sleep 4

  local t0=$(date +%s)
  ( export VHIP_MAXLEN=32768 VLLM_PLUGINS="$plugins"
    [ -n "$backend" ] && export VLLM_ATTENTION_BACKEND="$backend"
    # shellcheck disable=SC2086
    [ -n "$extra" ] && export $extra
    tools/vhip_launch.sh qwen >/dev/null 2>&1 )

  local ready=0
  for _ in $(seq 1 240); do
    curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && { ready=1; break; }
    docker ps -q -f name=lease-vhip-qwen-vhip-1 | grep -q . || break
    sleep 5
  done
  local boot=$(( $(date +%s) - t0 ))
  if [ "$ready" != 1 ]; then
    echo "  !! FAILED TO BOOT (${boot}s)"
    printf '%s\t%s\t%s\t%s\tBOOT_FAIL\tBOOT_FAIL\t%s\n' "$name" "$plugins" "${backend:-default}" "${extra:-none}" "$boot" >> "$OUT"
    docker logs lease-vhip-qwen-vhip-1 2>&1 | grep -m2 -E "Error|error:|Traceback" | head -2
    return
  fi

  read -r dec e2e < <(python3 - <<'PY'
import json, time, urllib.request
def metrics():
    with urllib.request.urlopen("http://localhost:8000/metrics", timeout=30) as r:
        o = {}
        for ln in r.read().decode().splitlines():
            if ln.startswith("#") or " " not in ln: continue
            k, v = ln.rsplit(" ", 1); o[k.split("{")[0]] = o.get(k.split("{")[0], 0.0) + float(v)
        return o
def req(n):
    b = json.dumps({"model":"x","messages":[{"role":"user","content":
        "Write a detailed technical explanation of how a gated delta-net linear-attention layer "
        "maintains its recurrent state across tokens. Be thorough and precise."}],
        "max_tokens":n,"temperature":0,"stream":False}).encode()
    rq = urllib.request.Request("http://localhost:8000/v1/chat/completions", data=b,
                                headers={"Content-Type":"application/json"})
    t0=time.perf_counter()
    with urllib.request.urlopen(rq, timeout=600) as r: o=json.load(r)
    return time.perf_counter()-t0, o["usage"]["completion_tokens"]
req(16)
ds, es = [], []
for _ in range(3):
    a=metrics(); w,n=req(200); b=metrics()
    d=b["vllm:request_decode_time_seconds_sum"]-a["vllm:request_decode_time_seconds_sum"]
    ds.append((n-1)/d); es.append(n/w)
print(f"{sum(ds)/len(ds):.1f} {sum(es)/len(es):.1f}")
PY
)
  echo "  -> decode ${dec} tok/s | e2e ${e2e} tok/s | boot ${boot}s"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$plugins" "${backend:-default}" "${extra:-none}" "$dec" "$e2e" "$boot" >> "$OUT"
}

run_row "1-hip-baseline"      "$HIP"                              ""               ""
run_row "2-triton-attn"       "$HIP"                              "TRITON_ATTN"    ""
run_row "3-rocm-attn"         "$HIP"                              "ROCM_ATTN"      ""
run_row "4-no-w4a8"           "gdn_hip,tail_hip_register"         ""               "VLLM_ROCM_W4A8_AWQ_DENSE=0"
run_row "5-no-tail"           "gdn_hip,w4a8_fp8_wmma_register"    ""               ""
run_row "6-all-stock"         ""                                  "TRITON_ATTN"    ""

docker compose -p lease-vhip-qwen down >/dev/null 2>&1
echo; echo "=== RESULTS ($OUT) ==="; column -t -s $'\t' "$OUT"
