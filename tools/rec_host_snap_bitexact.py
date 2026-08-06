#!/usr/bin/env python3
"""Gate: a host-tier recurrent snapshot round-trips BIT-EXACTLY vs the legacy device clone.

Model-free and fast (seconds). It builds a real `GDNStateCache` with the 35B's shapes and compares
the two `clone_slot`/`load_slot` paths byte-for-byte, so it isolates the TRANSPORT from every other
moving part — the model, the radix, the scheduler, the attention kernel.

Two things make this gate actually mean something:

  * ALL-BITS-RANDOM fill, not `randn`. A memcpy has to be exact for NaN, +/-inf, denormals and
    signalling-NaN payloads; `randn` never produces any of them, so it cannot distinguish a byte
    copy from a lossy-but-plausible one. We fill via uint8 and reinterpret.
  * sha256 over RAW BYTES, not allclose. Per `kvcache/state_digest.py:22-25`: "a 1-ULP difference
    is a different digest; 'close' is not a defence for a cache that claims to be lossless."

And a NEGATIVE CONTROL, without which the gate is unfalsified: we disable `flush_ring` and confirm
the comparison FAILS. Otherwise a gate that passes proves nothing about whether it could ever fail
— the same argument `tp_overlap.inline_collectives()` makes at `tp_overlap.py:246-260`.

Run:  gpu-lease -n 1 -- python tools/rec_host_snap_bitexact.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch

from minisgl.kvcache.gdn_state import GDNStateCache
from minisgl.kvcache.host_arena import PinnedFrameArena

# Qwen3.5-MoE 35B at TP=2, per rank: 30 GDN layers of 40 (full_attention_interval=4),
# conv_dim 8192/2, num_v_heads 32/2. One snapshot = 17,203,200 B = 16.406 MiB.
DEFAULTS = dict(
    num_gdn_layers=30,
    num_slots=6,
    conv_dim=4096,
    conv_kernel=4,
    num_v_heads=16,
    head_v_dim=128,
    head_k_dim=128,
)


# Bound at import, BEFORE the negative control can monkeypatch the class attribute. The reference
# fold must keep using the real implementation while `clone_slot`'s call is the one being broken —
# otherwise the control breaks both sides and proves nothing, which is the failure mode this whole
# file exists to avoid.
_REAL_FLUSH = GDNStateCache.flush_ring


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _fill_all_bits_random(t: torch.Tensor, seed: int) -> None:
    """Fill with uniformly random BITS (so NaN/inf/denormal/sNaN all occur), not random floats.

    Built as a CONTIGUOUS source and copied in, because the destination is a strided slice
    (`conv_state[:, s:s+1]` spans two contiguous subspaces) and cannot be reinterpreted in place.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randint(0, 256, (t.numel() * t.element_size(),), dtype=torch.uint8, generator=g)
    t.copy_(raw.view(t.dtype).reshape(t.shape).to(t.device))


def build(dev: torch.device, ssm_dtype: torch.dtype) -> GDNStateCache:
    return GDNStateCache(
        num_gdn_layers=DEFAULTS["num_gdn_layers"],
        num_slots=DEFAULTS["num_slots"],
        conv_dim=DEFAULTS["conv_dim"],
        conv_kernel=DEFAULTS["conv_kernel"],
        num_v_heads=DEFAULTS["num_v_heads"],
        head_v_dim=DEFAULTS["head_v_dim"],
        head_k_dim=DEFAULTS["head_k_dim"],
        dtype=torch.float32,
        ssm_dtype=ssm_dtype,
        device=dev,
    )


def roundtrip(cache: GDNStateCache, src: int, dst: int, use_host: bool, arena) -> None:
    cache._host_arena = arena if use_host else None
    snap = cache.clone_slot(src)
    assert snap is not None, "arena exhausted during the gate — raise MINISGL_REC_SNAP_HOST_GIB"
    cache.load_slot(dst, snap)
    torch.cuda.synchronize()
    del snap


def case_plain(cache: GDNStateCache, arena) -> tuple[str, str]:
    """Device path vs host path, from the same source slot into two different slots."""
    _fill_all_bits_random(cache.conv_state[:, 1:2], 0xC0FFEE)
    _fill_all_bits_random(cache.ssm_state[:, 1:2], 0xBEEF)
    roundtrip(cache, src=1, dst=2, use_host=False, arena=arena)
    roundtrip(cache, src=1, dst=3, use_host=True, arena=arena)
    dev = _sha(cache.conv_state[:, 2]) + _sha(cache.ssm_state[:, 2])
    hst = _sha(cache.conv_state[:, 3]) + _sha(cache.ssm_state[:, 3])
    return dev, hst


def _prime_ring(cache: GDNStateCache, slot: int, seed: int) -> None:
    """Put REAL, non-zero buffered entries in the ring for `slot`.

    Zero-filled entries fold to a no-op, which is how the first version of this case managed to
    pass while proving nothing. The values need only be plausible, not physical: `g` is a log-decay
    so it is kept <= 0, and `s0n` is set to -1 ("unknown"), which is the self-heal path the decode
    kernel already takes for a freshly-installed slot.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    sl = torch.tensor([slot], dtype=torch.long, device=cache._device)
    for r in cache._ring:
        for key in ("k", "vr"):
            t = r[key]
            t.copy_(torch.randn(t.shape, generator=g, dtype=torch.float32).to(t.device).to(t.dtype))
        t = r["g"]
        t.copy_((-torch.rand(t.shape, generator=g, dtype=torch.float32) * 0.1).to(t.device).to(t.dtype))
        r["len"].index_fill_(0, sl, 4)
        r["s0n"].index_fill_(0, sl, -1.0)


def case_ring(cache: GDNStateCache, arena) -> tuple[str, str] | None:
    """Prove `flush_ring` survived the rewrite, by comparing against the TRULY FOLDED state.

    Comparing the device path to the host path cannot prove this: both call `flush_ring` through
    the same line, so breaking it breaks both identically and they still agree. (That is exactly
    what the negative control caught on the first version of this file.) The oracle has to be
    independent — so we fold the ring EXPLICITLY, digest the result, and require `clone_slot` to
    reproduce it. Skip the flush and the snapshot is a checkpoint up to REPLAY_RING_LEN decode
    steps stale: the restore installs a state behind the tokens it claims to represent, which is
    silently wrong text and never a crash.

    Returns (reference, host) digests, or None if this build has no replay op — then there is no
    ring and nothing to get wrong.
    """
    if cache._ring is None:
        return None
    src = 1
    _fill_all_bits_random(cache.conv_state[:, src : src + 1], 0x1234)
    _fill_all_bits_random(cache.ssm_state[:, src : src + 1], 0x5678)
    pristine_conv = cache.conv_state[:, src : src + 1].clone()
    pristine_ssm = cache.ssm_state[:, src : src + 1].clone()

    def reset_state() -> None:
        cache.conv_state[:, src : src + 1] = pristine_conv
        cache.ssm_state[:, src : src + 1] = pristine_ssm

    # Reference: fold the ring explicitly, then read the slot directly. No snapshot involved.
    reset_state()
    _prime_ring(cache, src, seed=99)
    _REAL_FLUSH(cache, torch.tensor([src], dtype=torch.long, device=cache._device))
    torch.cuda.synchronize()
    ref = _sha(cache.conv_state[:, src]) + _sha(cache.ssm_state[:, src])

    # Host path from the SAME starting condition; must land on the same bytes.
    reset_state()
    _prime_ring(cache, src, seed=99)
    roundtrip(cache, src=src, dst=3, use_host=True, arena=arena)
    host = _sha(cache.conv_state[:, 3]) + _sha(cache.ssm_state[:, 3])
    return ref, host


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssm-dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("FAIL: no HIP/CUDA device (run under gpu-lease, inside the ROCm container)")
        return 2

    os.environ.setdefault("MINISGL_REC_SNAP_HOST", "1")
    dev = torch.device("cuda")
    cache = build(dev, torch.bfloat16 if args.ssm_dtype == "bf16" else torch.float32)
    layout = cache.snapshot_layout
    arena = PinnedFrameArena(layout, 4)
    print(f"snapshot frame = {layout.nbytes} B = {layout.nbytes / (1 << 20):.3f} MiB")
    print(f"arena: {arena.stats()}")

    ok = True

    d, h = case_plain(cache, arena)
    print(f"[plain] device={d[:16]}… host={h[:16]}…  {'MATCH' if d == h else 'MISMATCH'}")
    ok &= d == h

    r = case_ring(cache, arena)
    if r is None:
        print("[ring ] SKIPPED — gdn_hip has no replay op in this build, so there is no ring")
    else:
        print(f"[ring ] folded-ref={r[0][:16]}… host={r[1][:16]}…  "
              f"{'MATCH' if r[0] == r[1] else 'MISMATCH'}")
        ok &= r[0] == r[1]

    # ---- NEGATIVE CONTROL ----------------------------------------------------------------
    # Break flush_ring and require the ring case to FAIL. A gate that cannot be made to fail is
    # not evidence. Monkeypatched here rather than added as a test-only env var in gdn_state.py:
    # production code should not carry a switch whose only purpose is to corrupt state.
    if r is not None:
        real_flush = GDNStateCache.flush_ring
        GDNStateCache.flush_ring = lambda self, slots: None  # type: ignore[assignment]
        try:
            broken = case_ring(cache, arena)
        finally:
            GDNStateCache.flush_ring = real_flush  # type: ignore[assignment]
        caught = broken is not None and broken[0] != broken[1]
        print(f"[nctrl] flush_ring disabled -> {'DETECTED (good)' if caught else 'NOT DETECTED'}")
        ok &= caught
        if not caught:
            print("        The gate cannot distinguish a flushed from an unflushed capture, so a "
                  "PASS above proves nothing about ring ordering.")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
