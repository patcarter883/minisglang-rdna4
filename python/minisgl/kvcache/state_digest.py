"""Bit-exact per-layer digests of the KV state a prefill leaves behind. Diagnostic only.

WHY. A prefix-reuse bug on this engine presents as TEXT that differs, which is the least useful
possible signal: by the time it shows up it has been through every layer, the sampler and (for block
diffusion) a 16-step denoising trajectory whose entropy bound SORTS 256 values, so one flipped bit
anywhere rewrites the whole answer. `tools/cca_chunk_bisect.py` set the precedent for the fix —
compare the two paths per LAYER and report the first that differs — and this is the same idea moved
one level down, to the state itself rather than to hidden activations:

    a prefill is CORRECT iff the KV it leaves behind is bit-identical to a cold prefill's.

That is the whole contract of prefix reuse. Everything downstream (decode, a canvas, the sampler) is
a pure function of it, so a matching digest EXONERATES the prefill and points downstream, and a
mismatching one names the layer and the pool. Neither conclusion is available from the text.

TWO POOLS, because a SWA hybrid keeps its state in two places and they fail differently:
  * the MAIN paged pool holds the FULL-attention layers over [0, device_len) — a mismatch here means
    the reused radix PAGES are wrong (wrong content, wrong page, or mutated after insert);
  * the sliding-window RING holds the last min(device_len, W) positions per sliding layer — a
    mismatch here means the window snapshot/restore is wrong, which is the thing SWA-radix adds.

Digests are sha256 over the RAW BYTES, so they are exact rather than tolerant: a 1-ULP difference is
a different digest. That is deliberate — "close" is not a defence for a cache that claims to be
lossless, and the two paths being compared run the same kernels on the same tokens.

SEGMENTED BY ABSOLUTE POSITION, because "layer L differs" is only half an answer. A prefix-reused
prefill has two regions with completely different failure stories — the REUSED span, whose K/V came
out of the cache, and the NEW span the forward just computed — and one digest over both cannot say
which one moved. Splitting at fixed `SEG` boundaries in ABSOLUTE position (not relative to the
window, so the bins line up across two runs with different match lengths) separates them:

    only the NEW span differs   -> the cache handed over correct state and the EXTEND miscomputed;
    the REUSED span differs     -> what was restored is not what a cold prefill produces.

Inert unless MINISGL_STATE_DIGEST=1. It copies the whole reused prefix to host and hashes it, which
costs hundreds of ms per request; it is a bisect instrument, never a serve path.
"""
from __future__ import annotations

import hashlib
import os

import torch


# Segment width in absolute positions. 256 keeps a 3.2k prompt at 13 bins per layer while still
# resolving the boundary between a reused prefix and the tail a chunked extend computes.
SEG = 256


def state_digest_enabled() -> bool:
    """Read in ONE place so the call site and any future reader cannot disagree about the gate."""
    return os.environ.get("MINISGL_STATE_DIGEST") == "1"


def _sha(t: torch.Tensor) -> str:
    """sha256 of a tensor's raw bytes, contiguous-normalised. `.cpu()` first so the digest does not
    depend on the device allocator's stride choices."""
    b = t.detach().to("cpu").contiguous()
    return hashlib.sha256(b.numpy().tobytes()).hexdigest()[:16]


def _segments(lo: int, hi: int):
    """[lo, hi) split on ABSOLUTE multiples of SEG, so two runs that reused different amounts still
    produce the same bins and can be compared bin by bin."""
    start = lo
    while start < hi:
        end = min(((start // SEG) + 1) * SEG, hi)
        yield start, end
        start = end


def main_pool_digest(kv_cache, page_table_row: torch.Tensor, length: int) -> list:
    """(layer, seg_start, seg_end, k_sha, v_sha) for the FULL-attention pool over [0, length).

    `page_table_row` is `engine.page_table[req.table_idx]`, i.e. this sequence's slot for every
    position — so the digest follows the sequence, not the physical pages. That matters: a prefix HIT
    reuses another request's pages, so a physical-page digest would compare different addresses and
    report a difference that is only a different allocation."""
    slots = page_table_row[:length].to(torch.long)
    out = []
    for lid in range(kv_cache.num_layers):
        kc = kv_cache.k_cache(lid).view(-1, *kv_cache.k_cache(lid).shape[2:])
        vc = kv_cache.v_cache(lid).view(-1, *kv_cache.v_cache(lid).shape[2:])
        for a, b in _segments(0, length):
            sl = slots[a:b]
            out.append((lid, a, b, _sha(kc[sl]), _sha(vc[sl])))
    return out


def swa_ring_digest(swa_kv, table_idx: int, boundary: int, window: int, ring_stride: int) -> list:
    """(layer, seg_start, seg_end, k_sha, v_sha) for the ring window [boundary-Wp, boundary), in
    ascending absolute position — the SAME addressing `SWAWindowSnapshotter._window_slots` and
    rdna4.py's gather use, so a mismatch here is a mismatch the attention kernel would actually
    read."""
    Wp = min(boundary, window)
    base = table_idx * ring_stride
    out = []
    for lid in range(swa_kv.num_layers):
        kc, vc = swa_kv.k_cache(lid), swa_kv.v_cache(lid)
        for a, b in _segments(boundary - Wp, boundary):
            pos = torch.arange(a, b, device=swa_kv.device, dtype=torch.long)
            slots = base + (pos % ring_stride)
            out.append((lid, a, b, _sha(kc[slots, 0]), _sha(vc[slots, 0])))
    return out


def log_prefill_state_digest(logger, tag: str, engine, req, swa_snap=None,
                             matched: int = 0) -> None:
    """Emit one `[state-digest]` line per layer for a request whose prefill just completed.

    Printed rather than returned because the two runs being compared are different PROCESSES (a
    prefix hit and its cold reference cannot coexist in one serve — measuring the prompt cold inserts
    it, so the second request is a full hit, not a partial one). The scrape key is
    `tag uid boundary pool layer seg`, and tools/canvas_state_bisect.py diffs two serve logs on it.
    `hit` is recorded on every line so the reader can tell the REUSED span from the NEW one without
    reconstructing the match from the prompt."""
    length = int(req.device_len)
    # `matched` must be passed IN, not read back off req.cache_handle: the caller commits the prefix
    # before digesting, and `cache_req` REPLACES the handle with the freshly inserted one — whose
    # cached_len is the INSERT boundary, not the reuse boundary. Reading it here silently reported
    # align_down(device_len) as the reused span, which would put every newly-computed position on the
    # wrong side of the line and invert the tool's central verdict.
    hit = int(matched or 0)
    try:
        rows = [("main", r) for r in
                main_pool_digest(engine.kv_cache, engine.page_table[req.table_idx], length)]
        # The window geometry comes from the SNAPSHOTTER, not from a second copy of the arithmetic:
        # the whole point of the digest is to read the ring exactly as clone/restore address it, so
        # borrowing W and R from the object that owns them is what keeps the two from drifting.
        if engine.swa_kv_cache is not None and swa_snap is not None:
            rows += [("swa", r) for r in swa_ring_digest(
                engine.swa_kv_cache, req.table_idx, length, swa_snap.W, swa_snap.R)]
        for pool, (lid, a, b, ks, vs) in rows:
            logger.info_rank0(f"[state-digest] {tag} uid={req.uid} boundary={length} hit={hit} "
                              f"pool={pool} layer={lid} seg={a}:{b} k={ks} v={vs}")
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never take the serve down
        logger.warning_rank0(f"[state-digest] {tag} failed: {type(exc).__name__}: {exc}")


__all__ = [
    "log_prefill_state_digest",
    "main_pool_digest",
    "state_digest_enabled",
    "swa_ring_digest",
]
