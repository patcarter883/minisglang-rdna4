from __future__ import annotations

import torch


class CCAStateCache:
    """Per-sequence recurrent state for ZAYA CCA layers (the conv front-end's two streams).

    A CCA layer keeps TWO fixed-size recurrent buffers per running sequence (constant memory
    regardless of context length), exactly the GDN-state pattern (see `gdn_state.py`):

      - ``conv_states`` : the ``total_padding``-wide causal-conv window over the packed q|k channels.
        Rolled left + appended at every decode step; seeded from / written back to per prefill seg.
      - ``prev_hs``     : the PREVIOUS token's hidden state, consumed by ``val_proj2`` (the CCA value
        is built from val_proj1(current hs) ‖ val_proj2(previous hs)). Read-then-write on decode;
        shift+seed+store-last on prefill.

    Both are fp32 (config `mamba_cache_dtype: float32`) and the conv kernels REQUIRE fp32
    conv_states. Neither is prefix-cacheable (in-place recurrent update), so this is a plain
    free-list slot allocator, not a radix/paged structure.

    ★ Slot 0 is RESERVED as the NULL block and is NEVER handed to a real sequence — the decode
    kernel treats ``slot == 0`` (``is_pad``) as a padding row and leaves its output unwritten. The
    free-list therefore starts at slot 1; ``num_slots = max_running_req + 2`` (one NULL + one dummy),
    mirroring the engine's habit.

    Buffers are indexed ``[cca_layer_id, slot]`` — ``cca_layer_id`` enumerates ONLY the
    attention-bearing (even) layers (the contiguous 0..num_cca_layers-1 id that is ALSO the paged-KV
    layer slice). Per-slot shapes:
      conv_states: (conv_dim=1280, conv_kernel=total_padding=2)
      prev_hs:     (hidden_size=2048,)
    """

    def __init__(
        self,
        num_cca_layers: int,
        num_slots: int,
        conv_dim: int,
        conv_kernel: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_cca_layers = num_cca_layers
        self.num_slots = num_slots
        self._device = device
        self._dtype = dtype
        # conv_states keeps the FULL total_padding window (NOT kernel-1) — the CCA conv front-end is a
        # two-stage causal conv whose recurrent window is total_padding = (K0-1)+(K1-1) columns wide.
        self.conv_states = torch.zeros(
            (num_cca_layers, num_slots, conv_dim, conv_kernel), dtype=dtype, device=device
        )
        self.prev_hs = torch.zeros(
            (num_cca_layers, num_slots, hidden_size), dtype=dtype, device=device
        )
        # LIFO free-list. Slot 0 is the reserved NULL block (range stops at 1).
        self.NULL_SLOT = 0
        self._free: list[int] = list(range(num_slots - 1, 0, -1))

    def alloc_many(self, n: int) -> torch.Tensor:
        """Allocate n state slots; returns their ids as an int32 device tensor."""
        if n > len(self._free):
            raise RuntimeError(f"CCAStateCache: need {n} slots, only {len(self._free)} free")
        slots = [self._free.pop() for _ in range(n)]
        return torch.tensor(slots, dtype=torch.int32, device=self._device)

    def free(self, slots: torch.Tensor | list[int]) -> None:
        ids = slots.tolist() if isinstance(slots, torch.Tensor) else list(slots)
        self._free.extend(int(s) for s in ids)

    def reset_slots(self, slots: torch.Tensor) -> None:
        """Zero a fresh sequence's state across all CCA layers (called at prefill)."""
        self.conv_states[:, slots] = 0
        self.prev_hs[:, slots] = 0

    def snapshot(self, slots: torch.Tensor):
        """Clone conv+prev_hs state for `slots` (across all CCA layers). Opaque handle for `restore`."""
        sl = slots.to(torch.long)
        return (sl, self.conv_states[:, sl].clone(), self.prev_hs[:, sl].clone())

    def restore(self, snapshot) -> None:
        """Restore a `snapshot()` back into the same slots (conv + prev_hs, all CCA layers)."""
        sl, conv, prev = snapshot
        self.conv_states[:, sl] = conv
        self.prev_hs[:, sl] = prev

    def clone_slot(self, slot: int):
        """Slot-agnostic snapshot of ONE slot's conv+prev_hs state (all CCA layers) for radix
        prefix-caching — mirrors GDNStateCache.clone_slot. Returns an opaque handle installable into a
        DIFFERENT slot via `load_slot`."""
        s = int(slot)
        return (self.conv_states[:, s : s + 1].clone(), self.prev_hs[:, s : s + 1].clone())

    def load_slot(self, slot: int, snap) -> None:
        """Install a `clone_slot` snapshot into `slot` (conv + prev_hs, all CCA layers)."""
        s = int(slot)
        conv, prev = snap
        self.conv_states[:, s : s + 1] = conv
        self.prev_hs[:, s : s + 1] = prev

    def install_verify_state(
        self,
        conv_scratch: dict,
        prev_scratch: dict,
        slots: torch.Tensor,
        t_index: torch.Tensor,
    ) -> None:
        """Install the per-token state captured by a spec-decode VERIFY forward into the live slots,
        for every CCA layer at once (mirrors GDNStateCache.install_verify_state). ``conv_scratch`` /
        ``prev_scratch`` map cca_layer_id -> the verify scratch ([Q, N, ...], Q=verify_max_qlen,
        N=num_seqs). ``slots`` (long, [N]) is the CCA slot per sequence (batch order); ``t_index``
        (long, [N]) is the per-seq token index to install (= accepted_count-1, the state AFTER the
        last accepted/confirmed token). Replaces the snapshot + re-advance: the verify capture already
        holds the exact accepted-prefix conv window + prev_hs, so this is a pure gather.

        Vectorized: scratch[t_index[i], i] -> state_cache[layer, slots[i]] for each seq i.
        """
        n = slots.numel()
        seq_ar = torch.arange(n, device=slots.device)
        for lid in range(self.num_cca_layers):
            cs = conv_scratch[lid]  # [Q, N, C, TP]
            ps = prev_scratch[lid]  # [Q, N, hidden]
            self.conv_states[lid, slots] = cs[t_index, seq_ar].to(self.conv_states.dtype)
            self.prev_hs[lid, slots] = ps[t_index, seq_ar].to(self.prev_hs.dtype)

    def conv(self, cca_layer_id: int) -> torch.Tensor:
        """conv_states for one CCA layer: (num_slots, conv_dim, conv_kernel)."""
        return self.conv_states[cca_layer_id]

    def prev(self, cca_layer_id: int) -> torch.Tensor:
        """prev_hs for one CCA layer: (num_slots, hidden_size)."""
        return self.prev_hs[cca_layer_id]

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
