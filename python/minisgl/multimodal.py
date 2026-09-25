"""Per-request image state for vision-capable models (Gemma-4 / DiffusionGemma).

One `ReqVision` is created when a request with images is admitted and is shared by every prefill chunk
of that request (the scheduler builds a fresh `Req` per chunk, so the state cannot live on the Req).
It carries the images' soft-token SPANS in the prompt and caches each image's tower output from the
chunk that first needs it until the chunk that finishes it — an image is encoded exactly once however
the prompt is chunked, and a prefix-cache hit that covers an image skips its encode entirely.

Two invariants the rest of the engine relies on:

  * A prefill chunk never ENDS strictly inside an image (`clip_chunk_end`). Gemma-4's sliding layers
    let every token of an image attend to the whole image; a chunk that stopped halfway would compute
    the first half without the second. A chunk may START inside one (a prefix-cache hit that ends
    mid-image): the cached half was computed with the whole image visible by the request that cached
    it, and it is the same image because the pad ids are its content hash.
  * An image's rows are merged from `embeds[j]`, never looked up: the token ids at those positions are
    content-hash pads above the vocabulary (tokenizer/vision.py).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


class ReqVision:
    __slots__ = ("images", "spans", "embeds")

    def __init__(self, images) -> None:
        self.images = list(images)
        self.spans: List[Tuple[int, int]] = [(im.offset, im.offset + im.length) for im in self.images]
        self.embeds: Dict[int, torch.Tensor] = {}

    def clip_chunk_end(self, start: int, end: int) -> int:
        """Move a chunk end out of any image it would cut: back to the image's start when that leaves
        a non-empty chunk, otherwise forward to the image's end (a chunk may exceed its token budget by
        at most one image)."""
        for a, b in self.spans:
            if a < end < b:
                return a if a > start else b
        return end

    def pixels(self, j: int, device: torch.device) -> Tuple[torch.Tensor, int, int]:
        im = self.images[j]
        n = im.pw * im.ph
        t = torch.frombuffer(bytearray(im.pixels), dtype=torch.uint8).view(n, -1)
        return t.to(device, non_blocking=False), im.pw, im.ph

    def release_pixels(self, j: int) -> None:
        self.images[j].pixels = b""  # encoded: the host copy is dead weight from here on
