from __future__ import annotations

import torch


class GDNStateCache:
    """Per-sequence recurrent state for GDN (Gated Delta Net) layers.

    Unlike the paged MHA KV cache, GDN state is a FIXED-size slot per running sequence
    (constant memory regardless of context length) and is NOT prefix-cacheable — the
    delta-rule state is updated in place and cannot be rolled back to a prefix. So this
    is a simple slot allocator (free-list), not a radix/paged structure:

      - one slot allocated when a sequence starts (prefill), zero-initialized,
      - the slot's conv_state + ssm_state are read+updated in place across decode steps,
      - the slot is freed when the sequence finishes.

    ★ Slot 0 is RESERVED as the NULL block and is NEVER handed to a real sequence.
    Cache index 0 == ``NULL_BLOCK_ID``: ``causal_conv1d_fn`` / ``causal_conv1d_update``
    treat any sequence whose ``cache_indices[seq] == 0`` as a null/padding block and
    return their output buffer UNWRITTEN (garbage, no error). So the free-list starts at
    slot 1 (real slots in ``[1, num_slots-1]``); slot 0 exists in the buffer but is only
    ever the padding target. (This was the root cause of the vacuous 3b-3 prefill PASS;
    see PORT.md "★ 3c CONSTRAINT".) Size ``num_slots = max_running_req + 2`` (one NULL +
    one per running req), mirroring the engine's "+1 dummy page" habit.

    Buffers are indexed [gdn_layer_id, slot] — gdn_layer_id enumerates ONLY the linear-
    attention layers (the 1-in-4 full-attention layers use the normal MHA KV cache).

    Shapes (per slot), from mamba_utils gated_delta_net_state_shape (TP-divided):
      conv_state: (conv_dim, conv_kernel - 1)   conv_dim = head_k_dim*num_k_heads*2 + head_v_dim*num_v_heads
      ssm_state:  (num_v_heads, head_v_dim, head_k_dim)
    """

    def __init__(
        self,
        num_gdn_layers: int,
        num_slots: int,
        conv_dim: int,
        conv_kernel: int,
        num_v_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        ssm_dtype: torch.dtype | None = None,
    ) -> None:
        self.num_gdn_layers = num_gdn_layers
        self.num_slots = num_slots
        self._device = device
        self._dtype = dtype
        # The ssm_state is the memory hog (num_v_heads*head_v_dim*head_k_dim/slot); it may use a
        # narrower dtype than conv_state to cut recurrent-state HBM (~2x max_running_req). The
        # gdn_hip kernels read/write it at this dtype but compute fp32 in-register. conv_state stays
        # `dtype` (fp32). Defaults to `dtype` (no change) unless an ssm_dtype is given.
        self._ssm_dtype = ssm_dtype or dtype
        # conv state layout follows the causal_conv1d kernel contract (dim-first);
        # orientation is re-verified when the conv1d kernel is wired (Phase 3b).
        self.conv_state = torch.zeros(
            (num_gdn_layers, num_slots, conv_dim, conv_kernel - 1), dtype=dtype, device=device
        )
        self.ssm_state = torch.zeros(
            (num_gdn_layers, num_slots, num_v_heads, head_v_dim, head_k_dim),
            dtype=self._ssm_dtype,
            device=device,
        )
        # LIFO free-list of slot ids. Slot 0 is the reserved NULL block (see class
        # docstring): the range STOPS at 1, so slot 0 is never popped/allocated.
        self.NULL_SLOT = 0
        self._free: list[int] = list(range(num_slots - 1, 0, -1))

    def alloc_many(self, n: int) -> torch.Tensor:
        """Allocate n state slots; returns their ids as an int32 device tensor."""
        if n > len(self._free):
            raise RuntimeError(f"GDNStateCache: need {n} slots, only {len(self._free)} free")
        slots = [self._free.pop() for _ in range(n)]
        return torch.tensor(slots, dtype=torch.int32, device=self._device)

    def free(self, slots: torch.Tensor | list[int]) -> None:
        ids = slots.tolist() if isinstance(slots, torch.Tensor) else list(slots)
        self._free.extend(int(s) for s in ids)

    def reset_slots(self, slots: torch.Tensor) -> None:
        """Zero a fresh sequence's state across all GDN layers (called at prefill)."""
        self.conv_state[:, slots] = 0
        self.ssm_state[:, slots] = 0

    def conv(self, gdn_layer_id: int) -> torch.Tensor:
        """conv_state for one GDN layer: (num_slots, conv_dim, conv_kernel-1)."""
        return self.conv_state[gdn_layer_id]

    def ssm(self, gdn_layer_id: int) -> torch.Tensor:
        """ssm_state for one GDN layer: (num_slots, num_v_heads, head_v_dim, head_k_dim)."""
        return self.ssm_state[gdn_layer_id]

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def ssm_dtype(self) -> torch.dtype:
        return self._ssm_dtype
