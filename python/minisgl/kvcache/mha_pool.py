from __future__ import annotations

import os

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool

# Fused store_kv (cast + per-tensor 1/scale + scatter) from the canonical tail_hip kernel. Imported
# directly (not via minisgl.layers, to avoid an import cycle) and gated by the same MINISGL_TAIL_HIP
# switch as the other tail ops. Soft: absent .so -> torch scatter fallback below.
_STORE_KV = None
if os.environ.get("MINISGL_TAIL_HIP", "1") != "0":
    try:
        import tail_hip

        _STORE_KV = tail_hip.store_kv
    except Exception:
        _STORE_KV = None

_FP8_MAX = 448.0  # e4m3 (OCP) max representable magnitude


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

        # fp8-KV per-tensor descale (A5). bf16/fp16 KV keeps scale 1.0 (the fused store just casts).
        # For fp8 KV the scale is CALIBRATED once at warmup (see accumulate/finalize below): a static
        # per-tensor scale is required because a single descale must undo every stored token's scale,
        # so it cannot vary per step. k/v_scale are plain python floats (passed scalar to the kernel
        # and read by the attention backend for its descale arg).
        self.kv_is_fp8 = dtype == torch.float8_e4m3fn
        self.k_scale = [1.0] * num_layers
        self.v_scale = [1.0] * num_layers
        # Calibration (A5): accumulate pre-cast |k|/|v| amax per layer, then finalize to a static
        # per-tensor scale = amax / FP8_MAX. OFF by default (MINISGL_KV_FP8_CALIBRATE=1 to enable) so
        # the default fp8-KV path is byte-identical to before (scale 1.0). A correct static scale
        # needs REPRESENTATIVE data + a freeze BEFORE real requests store, so enabling it is meant to
        # be driven by a calibration harness: enable, run a representative prefill, call
        # finalize_kv_calibration(), THEN serve. (Auto-calibrating on engine dummy-warmup data is not
        # representative, hence not the default.)
        self._calibrating = self.kv_is_fp8 and os.environ.get("MINISGL_KV_FP8_CALIBRATE", "0") != "0"
        if self._calibrating:
            self._k_amax = torch.zeros(num_layers, dtype=torch.float32, device=device)
            self._v_amax = torch.zeros(num_layers, dtype=torch.float32, device=device)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        # Persist new K/V into the paged buffer at `out_loc`. Fused native store when available
        # (one kernel: cast to the cache dtype + per-tensor 1/scale + scatter); else the torch
        # scatter+cast fallback. (bf16/fp16 use scale 1.0; fp8 uses the calibrated per-tensor scale.)
        _, kv_heads, head_dim = self._storage_shape
        kv = k.view(-1, kv_heads, head_dim)
        vv = v.view(-1, kv_heads, head_dim)

        if self._calibrating:
            # No device sync: accumulate amax on-device; read once in finalize_kv_calibration().
            self._k_amax[layer_id] = torch.maximum(self._k_amax[layer_id], kv.detach().abs().amax())
            self._v_amax[layer_id] = torch.maximum(self._v_amax[layer_id], vv.detach().abs().amax())

        k_cache = self._k_buffer[layer_id].view(self._storage_shape)
        v_cache = self._v_buffer[layer_id].view(self._storage_shape)
        ks, vs = self.k_scale[layer_id], self.v_scale[layer_id]

        if _STORE_KV is not None and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16:
            _STORE_KV(
                kv.contiguous(), vv.contiguous(), k_cache, v_cache, out_loc.to(torch.int32), ks, vs
            )
            return

        # ---- torch fallback (bf16/fp16 direct; fp8 divides by the calibrated scale before cast) ----
        if self.kv_is_fp8 and (ks != 1.0 or vs != 1.0):
            k_cache[out_loc] = (kv.float() / ks).to(k_cache.dtype)
            v_cache[out_loc] = (vv.float() / vs).to(v_cache.dtype)
        else:
            k_cache[out_loc] = kv.to(k_cache.dtype)
            v_cache[out_loc] = vv.to(v_cache.dtype)

    def finalize_kv_calibration(self) -> None:
        """Freeze the fp8-KV per-tensor descale from the amax accumulated over the warmup forward(s).
        No-op unless KV is fp8 and calibration is on. Call ONCE after the warmup forward, before real
        requests store into the cache (calibration writes are dummy-warmup tokens that get freed)."""
        if not self._calibrating:
            return
        kmax = self._k_amax.cpu().tolist()
        vmax = self._v_amax.cpu().tolist()
        # amax 0 (a layer never stored) -> keep scale 1.0.
        self.k_scale = [max(m / _FP8_MAX, 1e-4) if m > 0 else 1.0 for m in kmax]
        self.v_scale = [max(m / _FP8_MAX, 1e-4) if m > 0 else 1.0 for m in vmax]
        self._calibrating = False
        del self._k_amax, self._v_amax

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
