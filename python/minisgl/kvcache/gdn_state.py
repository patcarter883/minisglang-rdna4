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
        # ---- ReplaySSM ring (one per GDN layer) ----------------------------------------------
        # The decode kernel does not read-modify-write ssm_state every step; it decodes against the
        # CHECKPOINT in ssm_state plus a small ring of (k, vr, g) appended since, and folds the ring
        # back only every L steps (or when the Frobenius bound says so). So `ssm_state` alone is NOT
        # the state any more — the true state is fold(ssm_state, ring). Two rules follow, and every
        # method below that touches ssm_state obeys one of them:
        #   READ  ssm_state from outside the decode kernel  -> `flush_ring` FIRST (materialises it).
        #   WRITE ssm_state from outside the decode kernel  -> `reset_ring` AFTER (the ring's entries
        #                                                      and its cached ||S0||_F are now stale).
        # `reset_ring` marks ||S0||_F unknown (-1), which the decode kernel treats as "re-establish on
        # the next step" — that is the self-heal path for a freshly prefilled slot, not a separate
        # invalidation mechanism.
        # Allocated ONCE, here, alongside the state it shadows: the decode op is graph-captured, so
        # every buffer it touches needs a stable device address and there is no per-step allocation.
        # Per-layer `make_replay_ring` rather than one big [layers, ...] tensor so the shapes can
        # never drift from the kernel's own allocator.
        self._ring: list | None = None
        self.ring_len = 0
        if device.type == "cuda":   # the replay kernels are HIP-only; a CPU cache keeps the old path
            try:
                import gdn_hip as _gdn
                if hasattr(_gdn, "gdn_decode_conv_gated_replay"):
                    self.ring_len = int(_gdn.REPLAY_RING_LEN)
                    self._ring = [
                        _gdn.make_replay_ring(num_slots, num_v_heads, head_v_dim, head_k_dim,
                                              self.ring_len, dtype=self._ssm_dtype, device=device)
                        for _ in range(num_gdn_layers)
                    ]
            except Exception:
                # Not ImportError alone: torch.ops.load_library raises OSError when the extension is
                # built but its runtime is not present. Either way the replay op is unreachable, so
                # fall back to the materialised decode rather than failing the cache ctor — and any
                # real breakage in gdn_hip surfaces loudly at the first forward, not here.
                pass

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
        self.reset_ring(slots)

    # ---- ReplaySSM ring lifecycle ---------------------------------------------------------------

    def ring(self, gdn_layer_id: int):
        """The ReplaySSM ring for one GDN layer, or None when the kernel package has no replay op."""
        return None if self._ring is None else self._ring[gdn_layer_id]

    def flush_ring(self, slots: torch.Tensor) -> None:
        """Fold the ring into the checkpoint for `slots`, in every GDN layer, so `ssm_state` IS the
        true state afterwards. Call before ANY direct read of ssm_state (snapshot, radix clone, a
        prefill that continues an existing sequence).

        Cheap when the ring is empty: the kernel's blocks return on their own cursor, so an untouched
        slot costs the dispatch and nothing else."""
        if self._ring is None:
            return
        import gdn_hip as gdn
        sl = slots.to(torch.long)
        for lid in range(self.num_gdn_layers):
            r = self._ring[lid]
            gdn.gdn_replay_flush(self.ssm_state[lid], sl, r["k"], r["vr"], r["g"], r["len"],
                                 r["s0n"])

    def reset_ring(self, slots: torch.Tensor | int | None = None) -> None:
        """Drop the buffered entries and mark ||S0||_F unknown, for `slots` (or all). Call after ANY
        direct write to ssm_state — the ring's entries belong to the state that was just overwritten,
        and the cached ||S0||_F is stale. `-1` (unknown) makes the next decode step re-establish the
        checkpoint norm itself, so a freshly written slot needs no other invalidation."""
        if self._ring is None:
            return
        if slots is None:
            for r in self._ring:
                r["len"].zero_()
                r["s0n"].fill_(-1.0)
            return
        if isinstance(slots, int):
            slots = torch.tensor([slots], dtype=torch.long, device=self._device)
        sl = slots.to(torch.long)
        for r in self._ring:
            r["len"].index_fill_(0, sl, 0)
            r["s0n"].index_fill_(0, sl, -1.0)

    def snapshot(self, slots: torch.Tensor):
        """Clone conv+ssm state for `slots` (across all GDN layers). Returns an opaque handle for
        `restore`.

        NOTE: no longer used by spec-decode. The verify path now installs the exact accepted-prefix
        state directly via the per-token-state verify kernel (`install_verify_state` below), so the
        old snapshot + re-advance is gone. Kept as a general-purpose state-clone API."""
        sl = slots.to(torch.long)
        self.flush_ring(sl)          # a direct ssm_state READ: materialise the ReplaySSM ring first
        return (sl, self.conv_state[:, sl].clone(), self.ssm_state[:, sl].clone())

    def restore(self, snapshot) -> None:
        """Restore a `snapshot()` back into the same slots (conv + ssm, all layers)."""
        sl, conv, ssm = snapshot
        self.conv_state[:, sl] = conv
        self.ssm_state[:, sl] = ssm
        self.reset_ring(sl)          # a direct ssm_state WRITE: the ring's entries are now stale

    def clone_slot(self, slot: int):
        """Slot-agnostic snapshot of ONE slot's conv+ssm state across all GDN layers, for radix
        prefix-caching. Returns an opaque handle (cloned tensors, no source-slot binding) that
        `load_slot` can install into a DIFFERENT slot — a future request that hits the cached prefix
        restores this exact recurrent state instead of re-prefilling the shared prefix. ~17 MB / slot
        for the 35B (all 30 GDN layers, per rank)."""
        s = int(slot)
        self.flush_ring(torch.tensor([s], dtype=torch.long, device=self._device))
        return (self.conv_state[:, s : s + 1].clone(), self.ssm_state[:, s : s + 1].clone())

    def load_slot(self, slot: int, snap) -> None:
        """Install a `clone_slot` snapshot into `slot` (conv + ssm, all GDN layers). The recurrent
        state is decomposition-invariant under the bit-exact recurrent kernel, so continuing a prefill
        from this restored state is byte-identical to prefilling the shared prefix from zero."""
        s = int(slot)
        conv, ssm = snap
        self.conv_state[:, s : s + 1] = conv
        self.ssm_state[:, s : s + 1] = ssm
        self.reset_ring(s)           # a direct ssm_state WRITE: the ring's entries are now stale

    def install_verify_state(
        self,
        conv_scratch: dict,
        ssm_scratch: dict,
        slots: torch.Tensor,
        t_index: torch.Tensor,
    ) -> None:
        """Install the per-token state captured by a spec-decode VERIFY forward into the live slots,
        for every GDN layer at once. ``conv_scratch``/``ssm_scratch`` map gdn_layer_id -> the kernel's
        scratch ([Q, N, ...] — Q=max_qlen, N=num_seqs). ``slots`` (long, [N]) is the GDN slot per
        sequence (batch order); ``t_index`` (long, [N]) is the per-seq token index to install
        (= accepted_count-1, the state AFTER the last accepted/confirmed token). This replaces the
        snapshot + re-advance: the recurrent verify already computed the exact accepted-prefix state,
        we just gather it. Both conv + ssm are installed so the next decode step continues correctly.

        Vectorized gather: scratch[t_index[i], i] -> state_cache[layer, slots[i]] for each seq i.
        """
        n = slots.numel()
        seq_ar = torch.arange(n, device=slots.device)
        for lid in range(self.num_gdn_layers):
            cs = conv_scratch[lid]  # [Q, N, C, W-1]
            ss = ssm_scratch[lid]   # [Q, N, HV, V, K]
            # gather the chosen t per seq: result [N, ...]
            conv_pick = cs[t_index, seq_ar]  # [N, C, W-1]
            ssm_pick = ss[t_index, seq_ar]   # [N, HV, V, K]
            self.conv_state[lid, slots] = conv_pick.to(self.conv_state.dtype)
            self.ssm_state[lid, slots] = ssm_pick.to(self.ssm_state.dtype)
        self.reset_ring(slots)       # a direct ssm_state WRITE: the ring's entries are now stale

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
