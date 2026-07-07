"""CAMRuntime — the base-forward bridge for the CAM serving path (the seam WS-C's /cam/* API needs).

WS-C's endpoints need `base_logits(token_ids) -> Tensor` (one frozen-base forward, no CAM staged) for the
write gate's prompt probe and the router-gated decode loop; `CAMMemory` deliberately does NOT run the base
(it operates on logits passed in). minisgl's *served* model lives in the backend scheduler process and is
not directly callable from the FastAPI frontend, so for the MVP the CAM runtime loads its OWN frozen base
(HF Qwen3.5-4B) co-located in the API process and serves /cam/* via the LOGIT-ONLY router path (no residual
tap needed — matches memory-organ's serve_gen). This decouples CAM from the backend.

TWO caveats this MVP carries (documented, not hidden):
  1. RDNA4: HF Qwen3.5-4B's gated-delta-net (GDN) layers use the `fla` path, which HANGS on gfx1201. The
     runtime patches them to the native `gdn_hip` kernels (minisgl ships them) when CAM_NATIVE_GDN=1 — the
     same patch memory-organ applies. Without a GDN-capable base the forward will hang, so we require it.
  2. Resources: a co-located base is a SECOND ~8 GB model alongside minisgl's backend model — fine on a
     2-card box or when CAM runs standalone, tight on one 16 GB card. Sharing the exact backend model
     (ZMQ control-plane into the scheduler) is the future first-class integration.

Enable by setting MINISGL_CAM=1 and MINISGL_CAM_CHECKPOINT=<dir> (a memory-organ export).
"""
import os
import json
import logging

import torch

logger = logging.getLogger(__name__)
_RUNTIME = None


def _patch_native_gdn(model):
    """Patch HF Qwen3.5 GDN layers to native gdn_hip (RDNA4-safe). No-op if unavailable or not requested."""
    if os.environ.get("CAM_NATIVE_GDN", "1") != "1":
        return model
    try:
        # minisgl ships the native Triton-free gdn_hip patcher (fla HANGS on gfx1201 as the tap opens).
        from minisgl.gdn.hf_patch import patch_qwen3_5_gdn
        n = patch_qwen3_5_gdn(model)
        logger.info("CAM: patched %d Qwen3.5 GDN layer(s) to native gdn_hip.", n)
        return model
    except Exception as e:  # noqa
        logger.warning("CAM: native-GDN patch unavailable (%s) — HF GDN will run its default fla path "
                       "(HANGS on RDNA4). Run on CUDA/CPU or fix the gdn_hip import.", e)
        return model


class CAMRuntime:
    """Co-located frozen base + tokenizer + CAMMemory. Provides the base_logits seam for /cam/*."""

    def __init__(self, checkpoint_dir: str, base_model_id: str = None, device: str = None):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        from .memory import CAMMemory

        meta = json.load(open(os.path.join(checkpoint_dir, "meta.json")))
        base_model_id = base_model_id or meta.get("base_model") or "Qwen/Qwen3.5-4B"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base, last = None, None
        for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):   # mirror memory-organ load_frozen_base
            try:
                base = loader.from_pretrained(base_model_id, dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True).to(self.device).eval()
                break
            except Exception as e:  # noqa
                last = e
        if base is None:
            raise last
        for p in base.parameters():
            p.requires_grad_(False)
        base.config.use_cache = False
        base = _patch_native_gdn(base)
        self.base = base
        self.base_embed = base.get_input_embeddings()
        self.memory = CAMMemory(checkpoint_dir, self.base_embed, base.get_output_embeddings().weight)
        logger.info("CAMRuntime ready: base=%s device=%s cam_enabled=%s tap_layer=%s",
                    base_model_id, self.device, self.memory.enabled, getattr(self.memory, "tap_layer", None))

    @torch.no_grad()
    def base_logits(self, token_ids) -> torch.Tensor:
        """One frozen-base forward on token_ids (no CAM staged) -> last-position logits [vocab]."""
        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        return self.base(inputs_embeds=self.base_embed(ids)).logits[0, -1]

    def encode(self, text: str):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids))


def get_cam_runtime():
    """Lazy singleton. Returns None (so /cam/* replies 503) when CAM is not configured or fails to load."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    ckpt = os.environ.get("MINISGL_CAM_CHECKPOINT")
    if not ckpt or not os.path.isdir(ckpt):
        logger.warning("CAM: MINISGL_CAM_CHECKPOINT unset/missing (%s) — /cam/* disabled.", ckpt)
        return None
    try:
        _RUNTIME = CAMRuntime(ckpt)
    except Exception as e:  # noqa
        logger.error("CAM: runtime load failed — /cam/* disabled. (%s)", e)
        return None
    return _RUNTIME
