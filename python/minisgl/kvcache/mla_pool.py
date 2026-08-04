from __future__ import annotations

import os

import torch

from .base import BaseKVCachePool
from .mha_pool import kv_amax_to_descale


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

    fp8 (e4m3) LATENT CACHE — one scale PER LAYER, and why that is the right granularity here.
    MHA calibrates per (layer, head) because K and V are separate tensors with a head axis. A
    latent has neither: it is a SINGLE stored tensor per token, read back in both the "K" role
    (folded into the pre-softmax score) and the "V" role (folded linearly into the output) — which
    is exactly why the mla_hip fp8 kernels want ``k_descale == v_descale == cache_descale``. So the
    only granularity MLA admits is one scalar per layer. It does admit that one, though, and until
    it existed the fp8 latent cache was stored with an IMPLICIT scale of 1.0 while
    ``MINISGL_KV_FP8=1`` was the compose default: no range fitting at all, and everything below
    2^-9 flushed to zero.

    The descale lives in a persistent per-layer DEVICE tensor (never reallocated) for the same
    reason the MHA one does: the decode/verify kernels are cuda-graph captured and read the value
    through a pointer, so a fresh tensor per forward would leave the graph replaying a stale
    address, and a host scalar would freeze at capture time.
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

        # ---- fp8 scale table (identity until a calibration installs one) -------------------------
        self.kv_is_fp8 = dtype == torch.float8_e4m3fn
        # [num_layers] fp32, one scalar per layer. A 1-D tensor rather than python floats so
        # `latent_descale[l : l+1]` is a stable-address 1-element view — which is what the mla_hip
        # fp8 ops take (they read `descale[0]` off a device pointer).
        self.latent_descale = torch.ones(num_layers, dtype=torch.float32, device=device)
        self.latent_inv_scale = torch.ones(num_layers, dtype=torch.float32, device=device)

        # ---- offline calibration accumulator (tools/kv_fp8_calibrate.py) -------------------------
        # Same contract as MHAKVCache: NOT gated on kv_is_fp8, because the calibrator deliberately
        # runs against the bf16 cache so the activations it measures carry no fp8 feedback.
        self._calibrating = os.environ.get("MINISGL_KV_FP8_CALIBRATE", "0") != "0"
        if self._calibrating:
            # [num_layers, 1] — the trailing axis is a degenerate "head" axis so the calibrator can
            # handle MHA and MLA amax rows with one piece of code.
            self._k_amax = torch.zeros(num_layers, 1, dtype=torch.float32, device=device)
            # MLA has no separate V. The alias keeps the calibrator's (k, v) pair shape-compatible
            # and makes the emitted sidecar's k_scale == v_scale, which is what the kernels want.
            self._v_amax = self._k_amax

    def latent_cache(self, index: int) -> torch.Tensor:
        # [num_pages, page_size, latent_dim] — the per-layer paged latent the mla_decode kernel reads.
        return self._latent_buffer[index]

    def k_cache(self, index: int) -> torch.Tensor:
        # Alias so generic code that asks for the "k" cache gets the latent.
        return self._latent_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:  # pragma: no cover - MLA has no separate V cache
        raise NotImplementedError("MLA has no separate V cache; V is absorbed from the latent")

    def descale_view(self, layer_id: int) -> torch.Tensor:
        """Layer `layer_id`'s descale as a 1-element device view, for the mla_hip fp8 ops.

        A VIEW of the persistent table, so a boot-time in-place install is visible to a graph
        captured afterwards and the address never moves."""
        return self.latent_descale[layer_id : layer_id + 1]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor | None, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        # k = latent [tokens, latent_dim] (kv_a output, c_KV-norm ‖ k_rope). v is unused.
        latent_dim = self._storage_shape[1]
        cache = self._latent_buffer[layer_id].view(self._storage_shape)
        lat = k.view(-1, latent_dim)

        if self._calibrating:
            # amax over the WHOLE latent (tokens and channels): one stored tensor, one scale.
            self._k_amax[layer_id] = torch.maximum(
                self._k_amax[layer_id], lat.detach().abs().amax().float().reshape(1)
            )

        if self.kv_is_fp8:
            # Scale before the cast, per layer. torch's cast to float8_e4m3fn SATURATES at ±448 (it
            # does not return NaN the way a bare hardware convert does), so an activation above the
            # calibration amax clips — the correct behaviour for a max-calibrated quantizer.
            cache[out_loc] = (lat.float() * self.latent_inv_scale[layer_id]).to(cache.dtype)
            return
        cache[out_loc] = lat.to(cache.dtype)

    def finalize_kv_calibration(self) -> None:
        """Freeze the per-layer descale from the amax accumulated so far. No-op unless calibrating.

        For the offline calibrator only: it mutates the table every already-stored token was
        quantized against, so calling it on a live cache invalidates the whole cache."""
        if not self._calibrating:
            return
        scale = kv_amax_to_descale(self._k_amax).reshape(-1)
        self.latent_descale.copy_(scale)
        self.latent_inv_scale.copy_(1.0 / scale)
        self._calibrating = False
        del self._k_amax, self._v_amax

    def set_fp8_kv_scales(
        self, layer_id: int, k_scale: torch.Tensor, v_scale: torch.Tensor
    ) -> None:
        """Install layer `layer_id`'s latent descale, written IN PLACE (stable address).

        Takes the MHA signature so `fp8_scales.install_kv_fp8_scales` can drive both pool kinds.
        A latent has ONE scale, so the (k, v) pair — and a per-head row, which this cache cannot
        honour — are reduced with MAX rather than silently picking one: a too-large descale wastes
        a little range, a too-small one CLIPS."""
        assert self.kv_is_fp8, "set_fp8_kv_scales on a non-fp8 MLA pool"
        rows = [
            r.detach().to(device=self._device, dtype=torch.float32).reshape(-1)
            for r in (k_scale, v_scale)
        ]
        scale = torch.cat(rows).max()
        # A layer the calibrator never saw legitimately has amax 0 -> "no data", not "scale 0".
        if float(scale) <= 0:
            scale = torch.ones_like(scale)
        self.latent_descale[layer_id] = scale
        self.latent_inv_scale[layer_id] = 1.0 / scale

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._latent_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
