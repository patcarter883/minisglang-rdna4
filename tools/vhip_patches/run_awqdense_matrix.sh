#!/usr/bin/env bash
# CORRECTED matrix: does routing AWQ DENSE linears to our W4A8 FP8-WMMA kernel close the 83.3 -> 92 gap?
#
# The first matrix (run_plugin_matrix.sh) came back flat at 83.3-83.4 across attention backends,
# w4a8-on/off and tail-on/off. Reason: on ROCm an AWQ checkpoint NEVER consults
# _POSSIBLE_KERNELS[ROCM] (AutoAWQConfig.get_quant_method -> AutoAWQLinearMethod -> pure-Triton
# awq_gemm), so our W4A8 dense kernel was never engaged and "w4a8 off" removed nothing. `awq_dense_hip`
# is the plugin that actually routes it, and it was only wired into the glm/laguna presets.
#
# Profile motivation: at bs=1 decode the #1 kernel was `elementwise_kernel_manual_unroll` at 64.4%
# (231 us/call, weight-sized) paired with the fp16 GEMV wvSplitK at 6.9% — the signature of
# dequant-to-fp16-then-fp16-GEMV, i.e. the stock AWQ path. Routing to an int4-native W4A8 kernel is
# the direct attack on that.
#
# All rows NO-SPEC, 32768 ctx, MNS=8, fp8 KV, TP=2 — directly comparable to the recorded all-stock 92.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OUT="${OUT:-$PWD/tools/vhip_patches/awqdense_matrix.tsv}"
BASE="gdn_hip,tail_hip_register,w4a8_fp8_wmma_register"

printf 'row\tplugins\textra\tdecode_tok_s\te2e_tok_s\tboot_s\n' > "$OUT"

run_row() {
  local name="$1" plugins="$2" extra="$3"
  echo "=== $name | plugins=[$plugins] extra=[$extra] ==="
  docker compose -p lease-vhip-qwen down >/dev/null 2>&1; sleep 4
  local t0=$(date +%s)
  ( export VHIP_MAXLEN=32768 VLLM_PLUGINS="$plugins"
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
    docker logs lease-vhip-qwen-vhip-1 2>&1 | grep -m3 -E "Error|error:|Traceback|RuntimeError" | head -3
    printf '%s\t%s\t%s\tBOOT_FAIL\tBOOT_FAIL\t%s\n' "$name" "$plugins" "${extra:-none}" "$boot" >> "$OUT"
    return
  fi
  # confirm the router actually engaged, so a null result can't be a silent no-op again
  local routed
  routed=$(docker logs lease-vhip-qwen-vhip-1 2>&1 | grep -ciE "awq.?dense|W4A8.*dense|routed" || true)
  echo "  router log hits: $routed"

  read -r dec e2e < <(python3 - <<'PY'
import json, time, urllib.request
def metrics():
    with urllib.request.urlopen("http://localhost:8000/metrics", timeout=30) as r:
        o={}
        for ln in r.read().decode().splitlines():
            if ln.startswith("#") or " " not in ln: continue
            k,v=ln.rsplit(" ",1); o[k.split("{")[0]]=o.get(k.split("{")[0],0.0)+float(v)
        return o
def req(n):
    b=json.dumps({"model":"x","messages":[{"role":"user","content":
      "Write a detailed technical explanation of how a gated delta-net linear-attention layer "
      "maintains its recurrent state across tokens. Be thorough and precise."}],
      "max_tokens":n,"temperature":0,"stream":False}).encode()
    rq=urllib.request.Request("http://localhost:8000/v1/chat/completions",data=b,
                              headers={"Content-Type":"application/json"})
    t0=time.perf_counter()
    with urllib.request.urlopen(rq,timeout=600) as r: o=json.load(r)
    return time.perf_counter()-t0, o["usage"]["completion_tokens"]
req(16)
ds,es=[],[]
for _ in range(3):
    a=metrics(); w,n=req(200); b=metrics()
    d=b["vllm:request_decode_time_seconds_sum"]-a["vllm:request_decode_time_seconds_sum"]
    ds.append((n-1)/d); es.append(n/w)
print(f"{sum(ds)/len(ds):.1f} {sum(es)/len(es):.1f}")
PY
)
  echo "  -> decode ${dec} tok/s | e2e ${e2e} tok/s | boot ${boot}s"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$plugins" "${extra:-none}" "$dec" "$e2e" "$boot" >> "$OUT"
}

# control: what we have been measuring all session (AWQ dense = stock Triton)
run_row "A-stock-awq-dense"   "$BASE"                 ""
# THE FIX: route AWQ dense to our W4A8 FP8-WMMA kernel
run_row "B-w4a8-awq-dense"    "$BASE,awq_dense_hip"   ""
# + bypass the MoE shape oracle (it defers to stock Triton moe_wna16 below M=64, i.e. always at bs=1)
run_row "C-w4a8-dense+moe"    "$BASE,awq_dense_hip"   "VLLM_ROCM_W4A8_FORCE=on"
# TRUE all-stock reference (the recorded 92). Requires the compose `${VLLM_PLUGINS-...}` fix: with
# `:-` an empty value silently fell back to the default list and this row was a fake null.
# May pay a one-off GDN Triton compile — it is last for that reason.
run_row "D-all-stock"         ""                      ""
# stock GDN only, keeping our tail/w4a8 — isolates GDN from the rest of the stack
run_row "E-stock-gdn-only"    "tail_hip_register,w4a8_fp8_wmma_register,awq_dense_hip" ""

docker compose -p lease-vhip-qwen down >/dev/null 2>&1
echo; echo "=== RESULTS ($OUT) ==="; column -t -s $'\t' "$OUT"
