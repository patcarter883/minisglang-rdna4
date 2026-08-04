from __future__ import annotations

import os

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool

# Fused store_kv (cast + per-head 1/scale + scatter) from the canonical tail_hip kernel. Imported
# directly (not via minisgl.layers, to avoid an import cycle) and gated by the same MINISGL_TAIL_HIP
# switch as the other tail ops. Soft: absent .so -> torch scatter fallback below.
_STORE_KV = None
# Does the loaded tail_hip carry the PER-HEAD store schema (k_inv_scale/v_inv_scale)?
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
            _STORE_KV_PER_HEAD = len(_STORE_KV.default._schema.arguments) >= 9
        except Exception:
            _STORE_KV_PER_HEAD = False
    except Exception:
        _STORE_KV = None

_FP8_MAX = 448.0  # e4m3 (OCP) max representable magnitude


def kv_amax_to_descale(amax: torch.Tensor) -> torch.Tensor:
    """`amax -> e4m3 DESCALE` — the one place the calibration formula lives.

    `descale = amax / 448`, so the largest observed magnitude lands exactly at e4m3's max and the
    store is `x / descale`. A (layer, head) that never stored has amax 0, which means "no data", not
    "scale 0": it keeps 1.0, degrading to the un-calibrated direct cast rather than dividing by zero.

    Shared by `finalize_kv_calibration` (in-pool, TP=1) and `tools/kv_fp8_calibrate.py`, which at
    TP>1 must gather the per-rank amax into GLOBAL per-head rows on the HOST before converting — a
    pool only ever holds its own head shard. One formula, two callers, so a TP=2 sidecar is
    numerically the table a TP=1 run would have written."""
    scale = (amax / _FP8_MAX).clamp(min=1e-4)
    return torch.where(amax > 0, scale, torch.ones_like(scale))


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
        # WHERE THE SCALES COME FROM. A served engine does NOT calibrate: kvcache/fp8_scales.py
        # resolves them once at boot (sidecar -> checkpoint kv_cache_scheme -> warn) and calls
        # set_fp8_kv_scales() below, between pool construction and graph capture. That ordering is
        # the correctness requirement, not a preference — one descale has to undo every store ever
        # written under it, so mutating the table with a live cache corrupts everything already in
        # it.
        # The accumulate/finalize pair below is the OFFLINE CALIBRATOR (tools/kv_fp8_calibrate.py),
        # which is the only way to get genuinely PER-HEAD scales: per-head amax is a property of the
        # activations, and no checkpoint ships it. It runs in its own process over representative
        # text, freezes at the END of the run, and writes a sidecar the next boot reads. OFF unless
        # MINISGL_KV_FP8_CALIBRATE=1, so the served path never accumulates.
        # NOT gated on kv_is_fp8: the calibrator deliberately runs against the DEFAULT bf16 cache
        # so the activations it measures are the exact ones, with no fp8 feedback loop, and so a
        # first-ever calibration does not have to boot the very fp8 path it is trying to configure.
        self._calibrating = os.environ.get("MINISGL_KV_FP8_CALIBRATE", "0") != "0"
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
            # ROUNDING = round-to-nearest-even. Stochastic rounding on this store was implemented,
            # measured and REMOVED (not flag-gated — see the kernel's "why NOT stochastic rounding"
            # note for the numbers). Short version: SR removes a COHERENT ACCUMULATING bias, and a
            # KV cache is WRITE-ONCE storage — each element is rounded once and read, so there is no
            # accumulation, and sign-symmetric activations already cancel RNE's bias down to 1.6e-5
            # of RMS. SR paid the textbook sqrt(2) variance penalty for that (storage rel-RMSE
            # 0.0267 -> 0.0382) and made the attention output ~45% worse. The SAME technique is an
            # 8x WIN on the GDN recurrent state, which IS a recurrence that re-rounds its own state
            # every step. Recurrence vs write-once storage is the distinction, not the technique.
            if self.kv_is_fp8:
                ki, vi = self.k_inv_scale[layer_id], self.v_inv_scale[layer_id]
            else:
                ki = vi = None
            _STORE_KV(
                kv.contiguous(), vv.contiguous(), k_cache, v_cache, out_loc.to(torch.int32),
                1.0, 1.0, ki, vi,
            )
            return

        # ---- torch fallback (bf16/fp16 direct; fp8 scales per head before the cast) --------------
        # NOTE: not the served path — the fused kernel above is. Kept so a build without tail_hip
        # still stores correctly (same per-head reciprocal, same RNE cast).
        if self.kv_is_fp8:
            ki = self.k_inv_scale[layer_id].view(1, -1, 1)
            vi = self.v_inv_scale[layer_id].view(1, -1, 1)
            k_cache[out_loc] = (kv.float() * ki).to(k_cache.dtype)
            v_cache[out_loc] = (vv.float() * vi).to(v_cache.dtype)
        else:
            k_cache[out_loc] = kv.to(k_cache.dtype)
            v_cache[out_loc] = vv.to(v_cache.dtype)

    def finalize_kv_calibration(self) -> None:
        """Freeze the fp8-KV PER-HEAD descale from the amax accumulated so far. No-op unless
        calibration is on.

        Called by tools/kv_fp8_calibrate.py at the END of an offline calibration run, whose cache
        contents are then discarded. It is NOT for a live serve: it mutates the table that every
        already-stored token was quantized against, so calling it with a live cache invalidates
        every entry in it (measured: max|Δ| 4.85e-01 on the attention output of a captured decode
        graph, which does pick the new value up)."""
        if not self._calibrating:
            return
        # scale[l, h] = amax[l, h] / FP8_MAX, clamped off zero (see kv_amax_to_descale).
        kscale = kv_amax_to_descale(self._k_amax)
        vscale = kv_amax_to_descale(self._v_amax)
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

    def set_fp8_kv_scales(
        self, layer_id: int, k_scale: torch.Tensor, v_scale: torch.Tensor
    ) -> None:
        """Install layer `layer_id`'s fp8-KV descale (already TP-sharded by the caller).

        `k_scale`/`v_scale` are fp32 rows of numel 1 (per-tensor — broadcast to every head) or
        num_kv_heads (per-head). Both the descale and its reciprocal are written IN PLACE so the
        tensor ADDRESSES never change: an attention or store kernel already captured against them
        replays the new values instead of a baked constant. Boot-time only — see the class comment
        on why this must happen before anything the engine will later read has been stored."""
        assert self.kv_is_fp8, "set_fp8_kv_scales on a non-fp8 KV pool"
        for row, dst, inv in (
            (k_scale, self.k_descale, self.k_inv_scale),
            (v_scale, self.v_descale, self.v_inv_scale),
        ):
            r = row.detach().to(device=dst.device, dtype=torch.float32).reshape(-1)
            assert r.numel() in (1, self.num_kv_heads), (
                f"kv scale row numel {r.numel()}, expected 1 or {self.num_kv_heads}"
            )
            # A zero/negative scale would divide by zero on the store side; a (layer, head) that a
            # calibrator never saw legitimately has amax 0, which means "no data", not "scale 0".
            r = torch.where(r > 0, r, torch.ones_like(r))
            dst[layer_id].copy_(r.expand(self.num_kv_heads) if r.numel() == 1 else r)
            inv[layer_id].copy_(1.0 / dst[layer_id])
        self.k_scale[layer_id] = float(self.k_descale[layer_id].max())
        self.v_scale[layer_id] = float(self.v_descale[layer_id].max())

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
