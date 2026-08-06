"""Measure-only oracle: how much cross-request prefix reuse is the live cache throwing away?

MEASUREMENT, NOT A FEATURE. Nothing here changes what the engine serves — it only counts. It
exists to answer one go/no-go question before anyone writes a host-RAM KV tier: *is there reuse
in the live traffic that the current cache fails to capture, and if so, why?*

The "why" is the whole point, because there are two independent reasons a shared prefix is not
reused, and they have completely different fixes:

  * The KV pages were EVICTED (`radix_cache.evict`). Only a host-RAM KV tier recovers this.
  * The KV pages are still resident, but the recurrent-state snapshot for that boundary was
    dropped, so `match_prefix` caps the match at a shallower node (`radix_cache.py:194-199`,
    `_enforce_rec_cap` at `:272-277`). This needs a BIGGER SNAPSHOT CAP, not a KV tier — and
    once the snapshot store lives in host RAM the cap is nearly free, so this half may resolve
    itself for a fraction of the work.

Conflating the two is exactly how you build a KV tier that turns out to have been the wrong
lever. So the oracle reports them separately:

    prompt        total prompt tokens seen                            (denominator)
    actual        tokens actually reused  (== MatchResult.cached_len)
    kv            tokens whose KV was still resident (uncapped tree walk)
    potential     tokens some EARLIER request already covered

    gap_rec   = kv - actual         KV was there, the snapshot was not   -> raise the cap
    gap_evict = potential - kv      neither was there                    -> host KV tier

DECISION RULE: `gap_evict / prompt` under ~5% on a day of real traffic means a host KV tier has
nothing to buy. `gap_rec` being the dominant term means raise `MINISGL_GDN_RADIX_MAX_SNAPSHOTS`
instead.

`potential` is computed against an LRU of page-prefix hashes of every prompt previously seen —
i.e. an idealised cache with no eviction and no snapshot cap, bounded only by this oracle's own
capacity. It is an UPPER BOUND on what any tier could recover, which is the right thing to
compare against: if the upper bound is small, no implementation can be large.

Default OFF (`MINISGL_GHOST_ORACLE=1` to enable) because it costs one `.tolist()` of the prompt
plus O(pages) hashing per admitted request, on the admission path.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import List, Sequence

# 64-bit FNV-1a over per-page hashes. The hash is CHAINED (each page folds into the running
# value), so a hit at page p means "some earlier prompt shared this exact p-page prefix" — not
# merely "contained this page". That chaining is what makes the count a prefix length rather
# than a bag-of-pages overlap.
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK = (1 << 64) - 1


def ghost_oracle_enabled() -> bool:
    return os.environ.get("MINISGL_GHOST_ORACLE") == "1"


def ghost_oracle_path(model_path: str, dp_rank: int = 0) -> str | None:
    """Where this model's oracle state lives, or None if persistence is off.

    Requires an EXPLICIT `MINISGL_GHOST_ORACLE_DIR` rather than defaulting to somewhere under the
    repo: the state is a measurement fixture accumulated over days, so it must land on a path the
    operator chose and mounted, not in a container layer or a bind-mounted worktree that gets
    removed. No dir set => in-memory only, and the run is still valid, just not cumulative.

    Keyed by MODEL because the hashes are over token ids: two checkpoints with different
    tokenizers produce incomparable page hashes, and pooling their counters would silently average
    two different questions. Switching models therefore parks the old file and resumes it later
    rather than corrupting it. `dp_rank` is in the name because each DP replica sees its own
    disjoint request stream.
    """
    d = os.environ.get("MINISGL_GHOST_ORACLE_DIR")
    if not d:
        return None
    slug = "".join(c if c.isalnum() or c in "-._" else "_" for c in (model_path or "unknown"))
    return os.path.join(d, f"ghost-{slug}-dp{dp_rank}.bin")


class GhostPrefixOracle:
    """Bounded LRU of page-prefix hashes + the four counters above.

    Not thread-safe and does not need to be: the scheduler loop is single-threaded, and this is
    only ever called from `CacheManager.match_req`.
    """

    __slots__ = (
        "page_size",
        "capacity",
        "_seen",
        "prompt_tokens",
        "actual_tokens",
        "kv_tokens",
        "potential_tokens",
        "requests",
    )

    def __init__(self, page_size: int, capacity: int | None = None) -> None:
        self.page_size = max(1, int(page_size))
        # ~40 B/entry in a CPython dict of int->None, so the 262,144 default is ~10-25 MB. Each
        # entry is one PAGE of one prompt, so at page_size=16 this remembers ~4.2M tokens of
        # distinct prefix material — comfortably more than the 141k-token device pool, which is
        # what makes `potential` a genuine upper bound rather than a second cache-size artifact.
        # Injected by the caller rather than read from the environment here: this module is
        # deliberately dependency-free so tests/core/test_ghost_oracle.py can load it by PATH on the
        # bare host (importing the package would pull in torch, which needs the ROCm container).
        # Reaching for the shared env helper would have re-introduced exactly that coupling.
        self.capacity = int(capacity if capacity is not None else 1 << 18)
        self._seen: "OrderedDict[int, None]" = OrderedDict()
        self.prompt_tokens = 0
        self.actual_tokens = 0
        self.kv_tokens = 0
        self.potential_tokens = 0
        self.requests = 0

    # ---------------------------------------------------------------- observation

    def observe(self, input_ids: Sequence[int], actual_len: int, kv_len: int) -> None:
        """Record one admitted request.

        `actual_len` is `MatchResult.cached_len` (what was really reused). `kv_len` is the
        UNCAPPED tree-walk prefix (what was still resident in the radix), which equals
        `actual_len` for a dense cache and is >= it for a recurrent one.
        """
        ps = self.page_size
        n = len(input_ids)
        n_pages = n // ps
        self.requests += 1
        self.prompt_tokens += n
        self.actual_tokens += actual_len
        self.kv_tokens += max(kv_len, actual_len)
        if n_pages == 0:
            return

        ids = list(input_ids)
        seen = self._seen
        h = _FNV_OFFSET
        deepest = 0
        hashes: List[int] = []
        for p in range(n_pages):
            page = tuple(ids[p * ps : (p + 1) * ps])
            h = ((h ^ (hash(page) & _MASK)) * _FNV_PRIME) & _MASK
            hashes.append(h)
            if h in seen:
                # Deepest, not first: an interior page-prefix may have been LRU-dropped while a
                # deeper one survives. The deepest surviving hit is the reusable prefix length.
                deepest = p + 1

        self.potential_tokens += max(deepest * ps, kv_len, actual_len)

        cap = self.capacity
        for hv in hashes:
            if hv in seen:
                seen.move_to_end(hv)
            else:
                seen[hv] = None
                if len(seen) > cap:
                    seen.popitem(last=False)

    # ---------------------------------------------------------------- reporting

    @property
    def gap_rec_tokens(self) -> int:
        """KV still resident but the recurrent snapshot was gone -> raise the snapshot cap."""
        return max(0, self.kv_tokens - self.actual_tokens)

    @property
    def gap_evict_tokens(self) -> int:
        """Neither KV nor snapshot resident -> only a host-RAM KV tier recovers this."""
        return max(0, self.potential_tokens - self.kv_tokens)

    # ---------------------------------------------------------------- persistence

    # The LRU is the part that MUST survive a restart, not just the counters. `potential` is
    # defined against "prompts seen earlier"; if the LRU resets, every prompt after a restart looks
    # novel and `potential` collapses to ~0 until the traffic happens to repeat itself. A day of
    # data punctuated by restarts would then systematically UNDER-report reuse — i.e. bias the
    # answer toward "don't build the tier", which is exactly the wrong way for a go/no-go probe to
    # fail. Persisting the counters alone would keep the bias and hide it behind a big denominator.
    _MAGIC = b"GHOSTOR1"

    def save(self, path: "os.PathLike[str] | str") -> None:
        """Atomically write state. Best-effort: a probe must never take the server down."""
        import array
        import struct
        import tempfile

        p = os.fspath(path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        buf = array.array("Q", self._seen.keys())  # insertion order == LRU order, oldest first
        header = self._MAGIC + struct.pack(
            "<7Q",
            self.page_size,
            self.requests,
            self.prompt_tokens,
            self.actual_tokens,
            self.kv_tokens,
            self.potential_tokens,
            len(buf),
        )
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(header)
                buf.tofile(f)
            os.replace(tmp, p)  # atomic: a torn file after SIGKILL would poison the next boot
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load(self, path: "os.PathLike[str] | str") -> bool:
        """Restore state. Returns True if anything was loaded.

        A page_size change invalidates every hash (pages are hashed at that granularity), so it
        starts fresh rather than silently mixing two page geometries into one number.
        """
        import array
        import struct

        p = os.fspath(path)
        if not os.path.exists(p):
            return False
        with open(p, "rb") as f:
            head = f.read(len(self._MAGIC) + 56)
            if len(head) < len(self._MAGIC) + 56 or head[: len(self._MAGIC)] != self._MAGIC:
                return False
            (ps, reqs, prompt, actual, kv, potential, n) = struct.unpack(
                "<7Q", head[len(self._MAGIC) :]
            )
            if ps != self.page_size:
                return False
            buf = array.array("Q")
            try:
                buf.fromfile(f, n)
            except (EOFError, ValueError):
                # EOFError = fewer items than the header promised; ValueError = a trailing partial
                # item (byte count not a multiple of 8). Both mean a torn write, and both must be a
                # fresh start rather than an exception that takes the boot down with it.
                return False
        self.requests = reqs
        self.prompt_tokens = prompt
        self.actual_tokens = actual
        self.kv_tokens = kv
        self.potential_tokens = potential
        self._seen.clear()
        # Keep the NEWEST `capacity` entries when the capacity shrank between runs: the tail of the
        # file is the most-recently-used end.
        for h in buf[-self.capacity :] if len(buf) > self.capacity else buf:
            self._seen[h] = None
        return True

    def summary(self) -> str:
        p = self.prompt_tokens or 1
        return (
            f"ghost-oracle: reqs={self.requests} prompt={self.prompt_tokens} "
            f"actual={self.actual_tokens} ({self.actual_tokens / p:.1%}) "
            f"potential={self.potential_tokens} ({self.potential_tokens / p:.1%}) | "
            f"gap_rec={self.gap_rec_tokens} ({self.gap_rec_tokens / p:.1%}, raise snapshot cap) "
            f"gap_evict={self.gap_evict_tokens} ({self.gap_evict_tokens / p:.1%}, needs host KV tier) "
            f"| lru={len(self._seen)}/{self.capacity}"
        )
