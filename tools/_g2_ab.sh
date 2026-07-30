#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; TOKS="${TOKS:-256}"; OUT="$REPO/tools/_g2_ab_results.txt"
wait_ready(){ for _ in $(seq 1 200); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
conc(){ local n="$1"
  python3 - "$n" <<'PY'
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
print(f"  conc={n}: aggregate {sum(c for c,_ in res)/w:.1f} tok/s")
PY
}
run(){ local v="$1" label="$2"
  echo "=== $label (VLLM_W4A8_MOE_G2FUSE_BYLANE='$v') ===" | tee -a "$OUT"
  LEASE_NAME=g2ab MINISGL_IMAGE=minisgl-rdna4:lean VLLM_W4A8_MOE_G2FUSE_BYLANE="$v" \
    gpu-lease -n 2 --detach --name g2ab -- docker compose -p lease-g2ab --profile serve up -d >/dev/null 2>&1
  if ! wait_ready; then echo "  BOOT FAILED" | tee -a "$OUT"
    docker compose -p lease-g2ab --profile serve logs --tail 20 2>&1 | sed 's/^/    /' | tee -a "$OUT"
    docker compose -p lease-g2ab --profile serve down >/dev/null 2>&1; sleep 5; return 1; fi
  python3 "$REPO/tools/_hostloop_driver.py" "$TOKS" "$REPO/tools/_g2_out_$label.txt" 2>&1 | grep "tok/s" | sed 's/^/  bs=1 /' | tee -a "$OUT"
  conc 4 2>&1 | tee -a "$OUT"; conc 8 2>&1 | tee -a "$OUT"
  docker compose -p lease-g2ab --profile serve down >/dev/null 2>&1; sleep 5; }
: > "$OUT"
run ""  klanes
run "1" bylane
echo "=== done ===" | tee -a "$OUT"
