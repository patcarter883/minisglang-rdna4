from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import nvtx_annotate

from . import _sampler_hip

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    # Structured output: packed xgrammar token bitmask [bs, ceil(vocab/32)] (constrained rows carry
    # the grammar's allowed set; unconstrained rows are all-ones). Applied to logits before sampling.
    grammar_bitmask: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def _apply_top_k(probs: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    # per-row top-k mask (top_k: [bs] int); keep the k highest probs, zero the rest.
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    ranks = torch.arange(probs.shape[-1], device=probs.device).unsqueeze(0)
    sorted_probs = sorted_probs.masked_fill(ranks >= top_k.unsqueeze(-1), 0.0)
    return torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)


def _apply_top_p(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    # per-row nucleus mask (top_p: [bs] float); keep the smallest prefix whose mass > top_p.
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    sorted_probs = sorted_probs.masked_fill((cumsum - sorted_probs) > top_p.unsqueeze(-1), 0.0)
    return torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    # Fused native HIP sampler (temperature+softmax+top-k+top-p+multinomial in one kernel, no sort)
    # when available; otherwise the torch reference below. Only tensor top_k/top_p route to the op
    # (the int/float scalar forms are unused by the engine's Sampler.prepare).
    if (
        _sampler_hip.available()
        and logits.is_cuda
        and logits.dtype == torch.float32
        and not isinstance(top_k, int)
        and not isinstance(top_p, float)
    ):
        return _sampler_hip.sample(logits, temperatures, top_k, top_p)
    # torch port of the former flashinfer.sampling path (greedy goes through argmax in Sampler).
    probs = torch.softmax(logits / temperatures.unsqueeze(-1).clamp_min(1e-6), dim=-1)
    if top_k is not None:
        probs = _apply_top_k(probs, top_k)
    if top_p is not None:
        probs = _apply_top_p(probs, top_p)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probs, num_samples=1).squeeze(-1).to(torch.int32)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.grammar_bitmask is not None:  # structured output: mask disallowed tokens to -inf
                from .grammar import apply_token_bitmask

                logits = apply_token_bitmask(logits.float(), args.grammar_bitmask)
            if args.temperatures is None:  # greedy sampling
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
