"""Pylate-free GTE-ModernColBERT subject encoder.

`pylate` (the ColBERT library) pins an OLDER `transformers` than the serve needs (5.13), so it can't be
installed alongside the engine. But GTE-ModernColBERT is just a `ModernBertModel` backbone + a 768->128
`Dense` projection — both load through the serve's own transformers. This reimplements the encoding
(backbone -> Dense -> per-token L2-norm -> masked mean-pool -> one subject-key vector) so no extra
dependency is needed. The SAME module fits the whitening (tools/gte_whiten_precompute) and runs in the
serve (CAMMemory._gte_key), so the two are guaranteed consistent.
"""
from __future__ import annotations

import glob
import os
from typing import List

import torch
import torch.nn.functional as F


class GTEEncoder:
    """Mean-pooled whitened-input GTE-ModernColBERT key encoder (CPU by default — short subjects encode in
    ms and this keeps GPU memory for the base model)."""

    def __init__(self, model: str = "lightonai/GTE-ModernColBERT-v1", device: str = "cpu",
                 max_len: int = 299) -> None:
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        d = model
        if not os.path.isdir(d):
            hits = glob.glob(os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*/"))
            hits = hits or glob.glob(
                f"/root/.cache/huggingface/hub/models--{model.replace('/', '--')}/snapshots/*/")
            d = hits[0] if hits else model
        self.max_len = max_len
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(d)
        self.model = AutoModel.from_pretrained(d, dtype=torch.float32).to(device).eval()
        dense = load_file(os.path.join(d, "1_Dense", "model.safetensors"))
        wk = next(k for k in dense if "weight" in k)
        self.Wd = dense[wk].float().to(device)                     # [out=128, in=768]
        self.dim = int(self.Wd.shape[0])

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """[N] texts -> [N, dim] masked-mean-pooled, per-token-L2-normed ColBERT key vectors (CPU float32)."""
        t = self.tok([x or "" for x in texts], padding=True, truncation=True,
                     max_length=self.max_len, return_tensors="pt").to(self.device)
        h = self.model(**t).last_hidden_state                      # [B,T,768]
        proj = F.normalize(h @ self.Wd.t(), dim=-1)                # [B,T,128] per-token L2 (ColBERT)
        m = t["attention_mask"].unsqueeze(-1).float()
        return ((proj * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu()
