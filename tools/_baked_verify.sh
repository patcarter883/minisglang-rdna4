#!/usr/bin/env bash
# Verify the BAKED image (no shadow mounts) reproduces the shadow-mounted numbers.
# No EXIT trap: on failure the container is LEFT UP so the logs can be read.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; OUT="$REPO/tools/_baked_verify.txt"
wait_ready(){ for _ in $(seq 1 200); do curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0; sleep 5; done; return 1; }
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
: > "$OUT"
echo "=== BAKED minisgl-rdna4:lean, NO shadow mounts ===" | tee -a "$OUT"
LEASE_NAME=bk MINISGL_IMAGE=minisgl-rdna4:lean \
  MINISGL_GDN_PROJ_GEMV=1 MINISGL_MINV_DECODE_GEMV=1 VLLM_W4A8_MOE_G2FUSE_BYLANE=1 \
  MINISGL_EXTRA_ARGS="--no-gdn-radix" \
  gpu-lease -n 2 --detach --name bk -- docker compose -p lease-bk --profile serve up -d >/dev/null 2>&1
if ! wait_ready; then
  echo "BOOT FAILED — container left up, last 25 log lines:" | tee -a "$OUT"
  docker compose -p lease-bk --profile serve logs --tail 25 2>&1 | sed 's/^/  /' | tee -a "$OUT"
  exit 1
fi
docker compose -p lease-bk --profile serve logs 2>&1 | grep -c "gdn_proj" | sed 's/^/  gdn_proj gemv engaged: /' | tee -a "$OUT"
python3 "$REPO/tools/_hostloop_driver.py" 256 "$REPO/tools/_baked_out.txt" 2>&1 | grep tok/s | sed 's/^/  bs=1 SAMPLED /' | tee -a "$OUT"
conc 4 | tee -a "$OUT"; conc 8 | tee -a "$OUT"
docker compose -p lease-bk --profile serve down >/dev/null 2>&1
