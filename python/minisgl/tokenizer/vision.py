"""Image preprocessing for vision-capable checkpoints (Gemma-4 / DiffusionGemma), on the tokenizer worker.

The frontend hands over raw image FILES (bytes) plus a prompt whose chat template rendered one
`IMAGE_SENTINEL` per image. This module:

  1. decodes each image and runs the checkpoint's OWN image processor (`Gemma4ImageProcessor` via
     AutoImageProcessor) — the aspect-preserving resize to the soft-token budget, the patchify and the
     position grid are therefore the reference's, not a re-implementation of them;
  2. keeps only the REAL patches and ships them as uint8. The processor resizes a uint8 image (so the
     result is uint8) and then rescales by 1/255 in fp32, so `round(pixel * 255)` recovers the exact
     bytes — the GPU's `tail_hip.vision_patch_in` redoes the rescale in the same fp32 op order. That is
     a quarter of the fp32 bytes and none of the 2520-row padding the processor adds;
  3. expands each placeholder token to `<boi> + n x pad + <eoi>`, where n is the image's soft-token
     count and `pad` is a content-hash id ABOVE the vocabulary. The radix cache keys on token ids, so
     two different images can never share a cached prefix, while the same image re-sent (a chat
     history) still hits. The model never embeds a pad id: the engine overwrites those rows with the
     vision tower's output before the embedding lookup sees them.
"""

from __future__ import annotations

import hashlib
import io
from typing import List, Tuple

import numpy as np

# Rendered by the frontend in place of each image part; swapped for the model's own image token text
# after the chat template runs, so the frontend never needs to know which model it is serving.
IMAGE_SENTINEL = "<|minisgl_image|>"

_PAD_SPAN = 1 << 30  # pad ids are vocab_size + hash % 2^30: above the vocab, inside int32


class VisionPreprocessor:
    def __init__(self, model_path: str, tokenizer) -> None:
        from minisgl.utils.hf import cached_load_hf_config

        cfg = cached_load_hf_config(model_path)
        self.supported = getattr(cfg, "vision_config", None) is not None and \
            getattr(cfg, "image_token_id", None) is not None
        if not self.supported:
            return
        text_cfg = getattr(cfg, "text_config", cfg)
        self.vocab_size = int(text_cfg.vocab_size)
        self.image_token_id = int(cfg.image_token_id)
        self.boi_token_id = int(cfg.boi_token_id)
        self.eoi_token_id = int(cfg.eoi_token_id)
        self.image_token_text = tokenizer.convert_ids_to_tokens(self.image_token_id)
        from transformers import AutoImageProcessor

        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.patch = int(self.processor.patch_size)
        self.pool = int(self.processor.pooling_kernel_size)

    def render(self, prompt: str) -> str:
        """Swap the frontend's sentinel for this model's image token text (before encoding)."""
        return prompt.replace(IMAGE_SENTINEL, self.image_token_text)

    def process_image(self, data: bytes) -> Tuple[np.ndarray, int, int]:
        """Raw image file -> (uint8 patches [pw*ph, patch*patch*3], pw, ph)."""
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
        out = self.processor(images=[img], return_tensors="pt")
        n_soft = int(out["num_soft_tokens_per_image"][0])
        n = n_soft * self.pool * self.pool
        pv = out["pixel_values"][0, :n].numpy()
        pos = out["image_position_ids"][0, :n].numpy()
        u8 = np.rint(pv * 255.0)
        # The recovery is exact or the reference did not rescale a uint8 image; refuse to guess.
        if np.abs(u8 * np.float32(1 / 255) - pv).max() > 1e-6:
            raise ValueError("image processor output is not a rescaled uint8 image")
        pw, ph = int(pos[:, 0].max()) + 1, int(pos[:, 1].max()) + 1
        assert pw * ph == n, (pw, ph, n)
        return np.ascontiguousarray(u8.astype(np.uint8)), pw, ph

    def expand(self, ids: List[int], images: List[bytes]):
        """ids with one image token per image -> (expanded ids, [MMImage])."""
        from minisgl.message import MMImage

        where = [i for i, t in enumerate(ids) if t == self.image_token_id]
        if len(where) != len(images):
            raise ValueError(f"the prompt has {len(where)} image placeholders but {len(images)} images")
        out: List[int] = []
        items = []
        prev = 0
        for pos, data in zip(where, images):
            px, pw, ph = self.process_image(data)
            n_soft = (pw // self.pool) * (ph // self.pool)
            digest = hashlib.sha256(px.tobytes() + pw.to_bytes(4, "little") + ph.to_bytes(4, "little"))
            h = int.from_bytes(digest.digest()[:8], "little")
            pad = self.vocab_size + (h % _PAD_SPAN)
            out.extend(ids[prev:pos])
            out.append(self.boi_token_id)
            items.append(MMImage(pixels=px.tobytes(), pw=pw, ph=ph, offset=len(out), length=n_soft,
                                 content_hash=h))
            out.extend([pad] * n_soft)
            out.append(self.eoi_token_id)
            prev = pos + 1
        out.extend(ids[prev:])
        return out, items
