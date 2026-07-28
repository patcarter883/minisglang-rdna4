#!/usr/bin/env bash
# Reusable vLLM decode-profiler container wrapper. Run UNDER a lease:
#   gpu-lease -n 2 --name prof -- bash tools/profiling/run_profile.sh
# Config via env (all optional except PROF_MODEL):
#   PROF_MODEL   (req)  HF id, e.g. cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
#   PROF_TP=2  PROF_MAXLEN=24000  PROF_KV=fp8  PROF_DTYPE=float16  PROF_MEM=0.92
#   PROF_MNBT=2048  PROF_MNS=8  PROF_EAGER=0  PROF_PREFILL=300  PROF_DECODE=30
#   PROF_IMAGE=vllm24-hip:combined
#   PROF_OUTDIR=<host dir for traces>   PROF_EXTRA_ENV="-e VLLM_GDN_HIP_RECURRENT_ONLY=1 ..."
# Guarantees: exact ROCm passthrough, forwards the lease's HIP/ROCR devices verbatim, warm triton
# cache, --rm + EXIT/INT/TERM trap so a kill NEVER leaks a container (the orphan bug), and it verifies
# the per-rank traces landed before returning success.
set -uo pipefail

: "${PROF_MODEL:?set PROF_MODEL to the HF model id}"
IMAGE="${PROF_IMAGE:-vllm24-hip:combined}"
OUTDIR="${PROF_OUTDIR:-/tmp/claude-1000/-home-pat-code-minisgl-rdna4/prof}"
NAME="prof_$$_$(date +%s 2>/dev/null || echo x)"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUTDIR"; rm -f "$OUTDIR"/*.json* 2>/dev/null || true

cleanup(){ docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
cleanup

# Forward the PROF_* knobs the inner python reads.
PROF_ENVS=()
for v in PROF_MODEL PROF_TP PROF_MAXLEN PROF_KV PROF_DTYPE PROF_MNBT PROF_MNS PROF_MEM PROF_EAGER PROF_PREFILL PROF_DECODE; do
  [ -n "${!v:-}" ] && PROF_ENVS+=(-e "$v=${!v}")
done

echo "[run_profile] image=$IMAGE model=$PROF_MODEL tp=${PROF_TP:-2} outdir=$OUTDIR"
docker run --rm --name "$NAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}" \
  -e NCCL_P2P_DISABLE=1 -e NCCL_PROTO=Simple \
  -e VLLM_TORCH_PROFILER_DIR=/prof \
  "${PROF_ENVS[@]}" ${PROF_EXTRA_ENV:-} \
  -v "$OUTDIR":/prof \
  -v "$HERE/offline_profile.py":/opt/offline_profile.py:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -v /home/pat/code/vllm-gfx1201/.triton-ab:/root/.triton \
  --entrypoint bash "$IMAGE" -lc \
  'export PYTHONPATH=/opt/kernels:$PYTHONPATH; python3 /opt/offline_profile.py'
rc=$?

echo "[run_profile] container rc=$rc; traces:"
found=$(find "$OUTDIR" -name "*.pt.trace.json*" -printf "  %p (%s bytes)\n" 2>/dev/null)
if [ -z "$found" ]; then
  echo "  NONE — profiling FAILED (check container log above)"; exit 1
fi
echo "$found"
echo "[run_profile] OK"
