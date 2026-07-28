#!/usr/bin/env bash
# vhip launcher — books both cards through the shared arbiter and starts the vllm24-hip serve
# from THIS worktree (source isolation: never the shared /home/pat/code/minisgl-rdna4 tree).
#
#   tools/vhip_launch.sh glm            # GLM-4.7-Flash-AWQ, no spec
#   tools/vhip_launch.sh glm-mtp        # + native glm4_moe_lite_mtp
#   tools/vhip_launch.sh glm-eagle3     # + thoughtworks/GLM-4.7-Flash-Eagle3
#   tools/vhip_launch.sh qwen           # Qwen3.6-35B-A3B-AWQ, no spec
#   tools/vhip_launch.sh qwen-mtp       # + qwen3_next_mtp
#   tools/vhip_launch.sh qwen-dflash    # + z-lab/Qwen3.6-35B-A3B-DFlash
#
# Stop:  docker compose -p lease-<name> down     (releases the lease)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${1:?usage: vhip_launch.sh <preset>}"; shift || true

# Point every source mount at THIS worktree, not the shared tree.
export VHIP_GLUE_SRC="$REPO/tools/profiling/vllm_oot_slotfix.py"
export VHIP_REG_SRC="$REPO/tools/profiling/register_bf16ssm.py"
export VHIP_MLAGLUE_SRC="$REPO/tools/vhip_patches/mla_vllm"
export VHIP_MLAMETA_SRC="$REPO/tools/vhip_patches/mla_vllm-0.1.0.dist-info"
export VHIP_MLASO_SRC="$REPO/tools/vhip_patches/mla/torch-ext/mla_hip/mla_hip_C.cpython-312-x86_64-linux-gnu.so"
export VHIP_MLAINIT_SRC="$REPO/tools/vhip_patches/mla/torch-ext/mla_hip/__init__.py"
export VHIP_SO_SRC="${VHIP_SO_SRC:-$REPO/tools/vhip_patches/gdn/torch-ext/gdn_hip/gdn_hip_C.cpython-312-x86_64-linux-gnu.so}"
export VHIP_AWQDENSE_SRC="$REPO/tools/vhip_patches/awqdense_vllm"
export VHIP_AWQMETA_SRC="$REPO/tools/vhip_patches/awqdense_vllm-0.1.0.dist-info"
# Patched startup attention-autotune: adds a persistent disk cache (and a kill-switch) to
# tcclaviger's ALWAYS-ON tuner, which otherwise re-profiles every boot. See docker-compose.yml.
export VHIP_ATTNTUNE_SRC="$REPO/tools/vhip_patches/attn_autotune.py"
# Patched w4a8_vllm MoE hook: adds the W4A16 (fp16-activation) decode path over the same
# standard-layout int4 weights. See the mount comment in docker-compose.yml.
export VHIP_FP8SO_SRC="${VHIP_FP8SO_SRC:-/home/pat/code/rdna4-hip-kernels/fp8_wmma/torch-ext/fp8_wmma/fp8_wmma_C.cpython-312-x86_64-linux-gnu.so}"
export VHIP_FP8INIT_SRC="${VHIP_FP8INIT_SRC:-/home/pat/code/rdna4-hip-kernels/fp8_wmma/torch-ext/fp8_wmma/__init__.py}"
export VHIP_MOEEXP_SRC="$REPO/tools/vhip_patches/moe_experts.py"
export VHIP_RDNA4_SRC="$REPO/tools/vhip_patches/rdna4_vllm"
export VHIP_RDNA4META_SRC="$REPO/tools/vhip_patches/rdna4_vllm-0.1.0.dist-info"

GLM=QuantTrio/GLM-4.7-Flash-AWQ
QWEN=cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
LAGUNA=poolside/Laguna-XS-2.1-NVFP4

# Spec decode costs KV budget twice over: the drafter's own weights come out of the same per-card
# 16 GB, and every in-flight request reserves K extra draft slots. So a preset's no-spec max-model-len
# does NOT fit once spec is on (Qwen MTP: vLLM computed a 36704 ceiling against the 60000 default and
# refused to boot). Each spec preset therefore lowers the default context; VHIP_MAXLEN still wins.
case "$PRESET" in
  *-mtp)     SPEC_MAXLEN=32768 ;;
  *-eagle3)  SPEC_MAXLEN=32768 ;;
  *-dflash)  SPEC_MAXLEN=16384 ;;   # K=16 draft slots per request are the widest verify here
  *)         SPEC_MAXLEN="" ;;
esac

case "$PRESET" in
  glm*)
    export MINISGL_MODEL="$GLM"
    # MLA decode runs vLLM's TRITON_MLA; MLA PREFILL runs our mla_hip kernel via the mla_vllm glue.
    # That kernel is bf16-typed, so the serve MUST be bf16 — under --dtype float16 the backend's
    # supported_dtypes check skips it and the selector falls back to the CK FMHA that segfaults.
    export VHIP_KV="${VHIP_KV:-auto}"
    export VHIP_DTYPE="${VHIP_DTYPE:-bfloat16}"
    export VHIP_MAXLEN="${VHIP_MAXLEN:-${SPEC_MAXLEN:-32768}}"
    export VHIP_MEM="${VHIP_MEM:-0.90}"
    export VHIP_MNS="${VHIP_MNS:-8}"
    export VHIP_REASONING_PARSER="${VHIP_REASONING_PARSER:-glm47}"
    export VHIP_TOOL_PARSER="${VHIP_TOOL_PARSER:-glm47}"
    export VHIP_CHAT_TEMPLATE_KWARGS="${VHIP_CHAT_TEMPLATE_KWARGS:-off}"
    # no GDN in GLM — don't load the gdn_hip glue at all; do load the MLA + AWQ-dense glue.
    export VLLM_PLUGINS="${VLLM_PLUGINS:-mla_hip,w4a8_fp8_wmma_register,awq_dense_hip}"
    # MoE expert dispatch stays on the tuned never-regress crossover. MEASURED 2026-07-27: forcing
    # our grouped kernel on for every batch (VLLM_ROCM_W4A8_FORCE=on) gave 53.5 tok/s vs 53.0 on
    # auto — inside noise, so there is no reason to override the measured window.
    export VLLM_ROCM_W4A8_FORCE="${VLLM_ROCM_W4A8_FORCE:-auto}"
    ;;
  qwen*)
    export MINISGL_MODEL="$QWEN"
    export VHIP_MAXLEN="${VHIP_MAXLEN:-${SPEC_MAXLEN:-60000}}"
    export VHIP_MEM="${VHIP_MEM:-0.92}"
    ;;
  laguna*)
    # SWA-hybrid gated-attention MoE; NVFP4 folds into the e2m1 W4A8 kernel. No GDN, no MLA.
    export MINISGL_MODEL="$LAGUNA"
    export VHIP_MAXLEN="${VHIP_MAXLEN:-${SPEC_MAXLEN:-32768}}"
    export VHIP_MEM="${VHIP_MEM:-0.90}"
    export VHIP_MNS="${VHIP_MNS:-${SPEC_MNS:-8}}"
    export VHIP_DTYPE="${VHIP_DTYPE:-bfloat16}"
    export VHIP_CHAT_TEMPLATE_KWARGS="${VHIP_CHAT_TEMPLATE_KWARGS:-off}"
    export VHIP_REASONING_PARSER="${VHIP_REASONING_PARSER:-poolside_v1}"
    export VHIP_TOOL_PARSER="${VHIP_TOOL_PARSER:-off}"
    export VLLM_PLUGINS="${VLLM_PLUGINS:-w4a8_fp8_wmma_register,awq_dense_hip}"
    ;;
  *) echo "unknown preset: $PRESET" >&2; exit 2 ;;
esac

# spec presets (VHIP_SPEC set by the caller always wins; K overrides the draft length)
if [ -z "${VHIP_SPEC:-}" ]; then
  case "$PRESET" in
    glm-mtp)     VHIP_SPEC='{"method":"glm4_moe_lite_mtp","num_speculative_tokens":'"${K:-2}"'}' ;;
    glm-eagle3)  VHIP_SPEC='{"method":"eagle3","model":"thoughtworks/GLM-4.7-Flash-Eagle3","num_speculative_tokens":'"${K:-3}"'}' ;;
    qwen-mtp)    VHIP_SPEC='{"method":"qwen3_next_mtp","num_speculative_tokens":'"${K:-2}"'}' ;;
    # z-lab's Qwen DFlash drafters have MIXED layer_types (5 sliding + 1 full), which vLLM 0.24's
    # DFlash proposer rejects outright (vllm#40898). Kept for when that lands upstream.
    qwen-dflash) VHIP_SPEC='{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","num_speculative_tokens":'"${K:-15}"'}' ;;
    # The supported DFlash pairing here: Laguna target + Laguna DFlash speculator (uniform
    # all-sliding layer_types). K defaults to the drafter's own dflash_config.block_size = 16.
    laguna-dflash) VHIP_SPEC='{"method":"dflash","model":"poolside/Laguna-XS-2.1-DFlash-NVFP4","num_speculative_tokens":'"${K:-16}"'}' ;;
  esac
  export VHIP_SPEC="${VHIP_SPEC:-}"
fi

echo "[launch] preset=$PRESET model=$MINISGL_MODEL spec=${VHIP_SPEC:-none}"
cd "$REPO"
exec gpu-lease -n 2 --detach --name "vhip-$PRESET" -- \
  docker compose --profile vhip up -d --force-recreate
