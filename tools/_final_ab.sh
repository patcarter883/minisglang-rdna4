#!/usr/bin/env bash
# Best measured config: dot2 GEMV + BYLANE gemm2 kernels, GDN+minv decode-GEMV routing,
# with/without the overlap loop. SAMPLED (checkpoint generation_config) = the real path.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; TOKS="${TOKS:-256}"; OUT="$REPO/tools/_final_ab_results.txt"
PKG=/home/pat/code/rdna4-hip-kernels-mmax/fp8_wmma/torch-ext/fp8_wmma
[ -f "$PKG/_ops.py" ] || { echo "FATAL: not built"; exit 1; }
wait_ready(){ for _ in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
cleanup(){ docker compose -p lease-fin --profile serve down >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
conc(){ python3 - "$1" <<'PY'
import json,sys,time,urllib.request,threading
n=int(sys.argv[1]); P="Explain step by step how a GPU executes a matmul. Be thorough."
res=[]
def one():
    b=json.dumps({"prompt":P,"max_tokens":128,"ignore_eos":True}).encode()
    r=urllib.request.Request("http://127.0.0.1:1919/generate",data=b,headers={"Content-Type":"application/json"})
    c=0;t0=None
    with urllib.request.urlopen(r,timeout=600) as resp:
        for raw in resp:
            if raw.startswith(b"data: "):
                if t0 is None: t0=time.perf_counter()
                c+=1
    res.append((c,time.perf_counter()-t0))
t=time.perf_counter(); th=[threading.Thread(target=one) for _ in range(n)]
[x.start() for x in th]; [x.join() for x in th]; w=time.perf_counter()-t
print(f"  conc={n}: {sum(c for c,_ in res)/w:.1f} tok/s")
PY
}
run(){ local args="$1" bl="$2" label="$3"
  echo "=== $label ===" | tee -a "$OUT"
  LEASE_NAME=fin MINISGL_IMAGE=minisgl-rdna4:lean FP8_WMMA_PKG="$PKG" \
    MINISGL_GDN_PROJ_GEMV=1 MINISGL_MINV_DECODE_GEMV=1 VLLM_W4A8_MOE_G2FUSE_BYLANE="$bl" \
    MINISGL_EXTRA_ARGS="$args" \
    gpu-lease -n 2 --detach --name fin -- docker compose -p lease-fin --profile serve up -d >/dev/null 2>&1
  if ! wait_ready; then echo "  BOOT FAILED" | tee -a "$OUT"
    docker compose -p lease-fin --profile serve logs --tail 12 2>&1 | sed 's/^/    /' | tee -a "$OUT"; cleanup; sleep 5; return 1; fi
  python3 "$REPO/tools/_hostloop_driver.py" "$TOKS" "$REPO/tools/_fin_$label.txt" 2>&1 | grep tok/s | sed 's/^/  bs=1 /' | tee -a "$OUT"
  conc 4 | tee -a "$OUT"; conc 8 | tee -a "$OUT"
  cleanup; sleep 5; }
: > "$OUT"
run ""               ""  A_gemv_only
run ""               1   B_gemv_bylane
run "--no-gdn-radix" 1   C_bylane_overlap
echo "=== done ===" | tee -a "$OUT"
