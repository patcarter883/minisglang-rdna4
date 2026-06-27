from __future__ import annotations

import torch

from .base import BaseKVCachePool


class MLAKVCache(BaseKVCachePool):
    """Paged latent KV cache for multi-head latent attention (DeepSeek / GLM-4.x MoE).

    Unlike MHA (which stores per-head K and V), MLA stores ONE compressed latent vector per
    token per layer — the kv_a output ``[c_KV (kv_lora_rank) ‖ k_rope (qk_rope_head_dim)]`` —
    shared across all heads (MQA over the rope part). The absorbed-decode kernel
    (``mla_hip.mla_decode``) attends q directly over this latent, so the cache cell is
    ``latent_dim = kv_lora_rank + qk_rope_head_dim`` wide and TP-replicated (no per-head split).

    Buffer: ``[num_layers, num_pages, page_size, latent_dim]``. ``store_kv`` writes the new
    tokens' latent (passed as ``k``; ``v`` is unused) at the flat ``out_loc`` slots, exactly like
    MHAKVCache — so the scheduler's page-table / out_loc plumbing is unchanged.
    """

    def __init__(
        self,
        num_layers: int,
        latent_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._latent_buffer = torch.empty(
            (num_layers, num_pages, page_size, latent_dim), device=device, dtype=dtype
        )
        self._num_layers = num_layers
        self._device = device
        self._storage_shape = (num_pages * page_size, latent_dim)

    def latent_cache(self, index: int) -> torch.Tensor:
        # [num_pages, page_size, latent_dim] — the per-layer paged latent the mla_decode kernel reads.
        return self._latent_buffer[index]

    def k_cache(self, index: int) -> torch.Tensor:
        # Alias so generic code that asks for the "k" cache gets the latent.
        return self._latent_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:  # pragma: no cover - MLA has no separate V cache
        raise NotImplementedError("MLA has no separate V cache; V is absorbed from the latent")

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor | None, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        # k = latent [tokens, latent_dim] (kv_a output, c_KV-norm ‖ k_rope). v is unused.
        latent_dim = self._storage_shape[1]
        cache = self._latent_buffer[layer_id].view(self._storage_shape)
        cache[out_loc] = k.view(-1, latent_dim).to(cache.dtype)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._latent_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
