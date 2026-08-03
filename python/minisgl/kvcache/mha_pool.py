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
# Does the loaded tail_hip carry the PER-HEAD store schema (k_inv_scale/v_inv_scale/stochastic)?
# A pre-per-head .so would accept only the 7 positional args, and silently pairing a per-tensor
# STORE with a per-head READ would dequantize every head but one with the wrong scale — wrong
# numbers, no error. So the schema is probed once here and checked at pool construction, where it
# can fail loudly at boot instead of quietly at token 1.
_STORE_KV_PER_HEAD = False
if os.environ.get("MINISGL_TAIL_HIP", "1") != "0":
    try:
        import tail_hip

        _STORE_KV = tail_hip.store_kv
        try:
            _STORE_KV_PER_HEAD = len(_STORE_KV.default._schema.arguments) >= 10
        except Exception:
            _STORE_KV_PER_HEAD = False
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

        # fp8-KV PER-HEAD descale. bf16/fp16 KV keeps scale 1.0 (the fused store just casts).
        # For fp8 KV the scale is CALIBRATED once at warmup (see accumulate/finalize below): a
        # STATIC scale is required because one descale must undo every stored token's scale, so it
        # cannot vary per step. What it CAN vary over is the KV HEAD: each head owns a disjoint
        # slice of every cache row, so head h's descale only ever has to undo head h's stores.
        # Per-head therefore costs nothing in correctness and recovers the dynamic range a single
        # per-tensor amax throws away whenever one head is much hotter than the rest.
        # Store and read index the SAME [num_layers, num_kv_heads] table.
        self.num_kv_heads = local_kv_heads
        self.kv_is_fp8 = dtype == torch.float8_e4m3fn
        if self.kv_is_fp8 and _STORE_KV is not None and not _STORE_KV_PER_HEAD:
            raise RuntimeError(
                "fp8 KV cache needs the per-head tail_hip.store_kv schema, but the loaded "
                "tail_hip only accepts the old per-tensor one. The engine and the kernels ship "
                "together (Dockerfile builds rdna4-hip-kernels into /opt/kernels); rebuild the "
                "image, or run with MINISGL_TAIL_HIP=0 to take the torch fallback."
            )
        # Per-LAYER summary scalars (max over heads) — informational, and the scalar fallback the
        # fused store op still accepts. The hot path uses the device tensors below.
        self.k_scale = [1.0] * num_layers
        self.v_scale = [1.0] * num_layers
        # Persistent [num_layers, num_kv_heads] descale TENSORS for the attention kernels. DEVICE
        # tensors, not host floats: a captured graph would freeze a host scalar at capture time.
        # They persist for the pool's lifetime (stable address = graph-safe) and are refreshed
        # IN-PLACE by finalize_kv_calibration. Indexing [layer_id] yields a contiguous
        # [num_kv_heads] row; the canonical attn kernels pick per-head vs per-tensor purely from
        # that row's numel (numel == num_kv_heads -> stride 1; numel 1 -> stride 0, the legacy read).
        self.k_descale = torch.ones(num_layers, local_kv_heads, dtype=torch.float32, device=device)
        self.v_descale = torch.ones(num_layers, local_kv_heads, dtype=torch.float32, device=device)
        # Reciprocals for the STORE side, kept as their own persistent tensors so the store kernel
        # neither divides per element nor disagrees with a torch reference by a division ULP.
        self.k_inv_scale = torch.ones(num_layers, local_kv_heads, dtype=torch.float32, device=device)
        self.v_inv_scale = torch.ones(num_layers, local_kv_heads, dtype=torch.float32, device=device)
        # Calibration: accumulate pre-cast |k|/|v| amax per (layer, head), then finalize to a static
        # scale = amax / FP8_MAX. OFF by default (MINISGL_KV_FP8_CALIBRATE=1 to enable) so the
        # default fp8-KV path is byte-identical to before (scale 1.0). A correct static scale needs
        # REPRESENTATIVE data + a freeze BEFORE real requests store, so enabling it is meant to be
        # driven by a calibration harness: enable, run a representative prefill, call
        # finalize_kv_calibration(), THEN serve. (Auto-calibrating on engine dummy-warmup data is
        # not representative, hence not the default.)
        self._calibrating = self.kv_is_fp8 and os.environ.get("MINISGL_KV_FP8_CALIBRATE", "0") != "0"
        if self._calibrating:
            self._k_amax = torch.zeros(num_layers, local_kv_heads, dtype=torch.float32, device=device)
            self._v_amax = torch.zeros(num_layers, local_kv_heads, dtype=torch.float32, device=device)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        # Persist new K/V into the paged buffer at `out_loc`. Fused native store when available
        # (one kernel: cast to the cache dtype + per-head 1/scale + scatter); else the torch
        # scatter+cast fallback. (bf16/fp16 use scale 1.0; fp8 uses the calibrated per-head scale.)
        _, kv_heads, head_dim = self._storage_shape
        kv = k.view(-1, kv_heads, head_dim)
        vv = v.view(-1, kv_heads, head_dim)

        if self._calibrating:
            # No device sync: accumulate amax PER HEAD on-device (reduce over tokens and head_dim,
            # keep the head axis); read once in finalize_kv_calibration().
            self._k_amax[layer_id] = torch.maximum(
                self._k_amax[layer_id], kv.detach().abs().amax(dim=(0, 2)).float()
            )
            self._v_amax[layer_id] = torch.maximum(
                self._v_amax[layer_id], vv.detach().abs().amax(dim=(0, 2)).float()
            )

        k_cache = self._k_buffer[layer_id].view(self._storage_shape)
        v_cache = self._v_buffer[layer_id].view(self._storage_shape)

        if _STORE_KV is not None and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16:
            from minisgl._hip_engage import engaged

            engaged("tail_hip.store_kv")
            # fp8: hand the kernel the persistent PER-HEAD reciprocal rows (stable address =>
            # graph-safe; a host scalar would freeze at capture time).
            #
            # ROUNDING = RNE, i.e. `stochastic=False`. tail_hip.store_kv HAS a stochastic-rounding
            # path (see tail_kernels.hip f32_to_e4m3_sr) and it is correct — its parity test proves
            # it is unbiased and stays inside the rounding bracket. It is off because it was
            # MEASURED WORSE end to end, not because it is unfinished. On real Qwen3-0.6B K/V over
            # real text (tail/local/kv_fp8_error.py, 28 layers): SR cuts the store bias 1.61e-5 ->
            # 8.66e-6 of RMS — a bias already ~1e-5, because a roughly symmetric activation
            # distribution cancels RNE's rounding errors across signs — and pays sqrt(2) more
            # variance for it (storage rel-RMSE 0.0267 -> 0.0382). The attention output that the
            # model actually consumes gets ~45% WORSE (median ratio 1.44-1.48), and SR won on
            # 1 of 84 (layer, context) cells at ctx 1024/4096/8192. There is no crossover up to
            # 32768 real tokens. Do not turn this on without a measurement that beats those.
            if self.kv_is_fp8:
                ki, vi = self.k_inv_scale[layer_id], self.v_inv_scale[layer_id]
            else:
                ki = vi = None
            _STORE_KV(
                kv.contiguous(), vv.contiguous(), k_cache, v_cache, out_loc.to(torch.int32),
                1.0, 1.0, ki, vi, False,
            )
            return

        # ---- torch fallback (bf16/fp16 direct; fp8 scales per head before the cast) --------------
        # NOTE: this path is round-to-nearest-even only — there is no SR without the kernel. It is
        # the no-tail_hip fallback, not the served path.
        if self.kv_is_fp8:
            ki = self.k_inv_scale[layer_id].view(1, -1, 1)
            vi = self.v_inv_scale[layer_id].view(1, -1, 1)
            k_cache[out_loc] = (kv.float() * ki).to(k_cache.dtype)
            v_cache[out_loc] = (vv.float() * vi).to(v_cache.dtype)
        else:
            k_cache[out_loc] = kv.to(k_cache.dtype)
            v_cache[out_loc] = vv.to(v_cache.dtype)

    def finalize_kv_calibration(self) -> None:
        """Freeze the fp8-KV PER-HEAD descale from the amax accumulated over the warmup forward(s).
        No-op unless KV is fp8 and calibration is on. Call ONCE after the warmup forward, before real
        requests store into the cache (calibration writes are dummy-warmup tokens that get freed)."""
        if not self._calibrating:
            return
        # scale[l, h] = amax[l, h] / FP8_MAX, clamped off zero. A (layer, head) that never stored
        # has amax 0 -> keep scale 1.0 so it degrades to the un-calibrated direct cast.
        kscale = (self._k_amax / _FP8_MAX).clamp_(min=1e-4)
        vscale = (self._v_amax / _FP8_MAX).clamp_(min=1e-4)
        kscale = torch.where(self._k_amax > 0, kscale, torch.ones_like(kscale))
        vscale = torch.where(self._v_amax > 0, vscale, torch.ones_like(vscale))
        # In-place into the persistent tensors -> addresses stay stable, so any graph already
        # captured against them replays with the new values instead of a stale baked constant.
        self.k_descale.copy_(kscale)
        self.v_descale.copy_(vscale)
        self.k_inv_scale.copy_(1.0 / kscale)
        self.v_inv_scale.copy_(1.0 / vscale)
        # Per-layer summary scalars (max over heads) for logging / the scalar op fallback.
        self.k_scale = kscale.amax(dim=1).cpu().tolist()
        self.v_scale = vscale.amax(dim=1).cpu().tolist()
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
