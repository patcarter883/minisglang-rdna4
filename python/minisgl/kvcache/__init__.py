from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry

if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
    if model_config.is_mla:
        # MLA (DeepSeek / GLM-4.x MoE): one compressed latent per token per layer
        # (kv_lora_rank + qk_rope_head_dim), shared across heads.
        from .mla_pool import MLAKVCache

        return MLAKVCache(
            num_layers=model_config.num_kv_layers,
            latent_dim=model_config.kv_lora_rank + model_config.qk_rope_head_dim,
            num_pages=num_pages,
            page_size=page_size,
            device=device,
            dtype=dtype,
        )

    from .mha_pool import MHAKVCache

    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_kv_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


@SUPPORTED_CACHE_MANAGER.register("recurrent_radix")
def create_recurrent_radix_cache(device: torch.device):
    # Radix prefix cache with per-node recurrent-state (GDN/CCA) snapshots — reuses a shared prefix's
    # paged KV AND its linear-attention recurrent state (opt-in via MINISGL_GDN_RADIX; the scheduler
    # selects this only for recurrent-hybrid models, else forces "naive"). See radix_cache.py.
    import os

    from .radix_cache import RadixPrefixCache

    cap = int(os.environ.get("MINISGL_GDN_RADIX_MAX_SNAPSHOTS", "64"))
    return RadixPrefixCache(device=device, recurrent=True, max_rec_snapshots=cap)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
