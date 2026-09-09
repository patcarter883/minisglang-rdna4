"""Recurrent-state snapshots for a model that carries TWO kinds of per-sequence state.

qwen4_exp (Qwen3.8-Flash-Next) is a GDN hybrid that ALSO has PLE n-gram layers, and both keep
per-sequence recurrent state in the SAME slot space (`PLERuntime.prepare`: "`slots` is the
per-sequence PLE/GDN state slot"). The recurrent-radix store was built around a single
`clone_slot`/`load_slot` pair, so before this the engine had one honest option: refuse prefix
caching entirely for PLE models (`resolve_prefix_cache`), because a radix hit would restore the GDN
state at the prefix boundary and leave the PLE state at zero/EOS — silent wrong output.

This composes the two caches behind that same two-method interface, so the scheduler seam, the
`rec_state: Any` handle in `radix_cache`, and every drop/eviction path stay exactly as they were.

THE INVARIANT IS ALL-OR-NOTHING. A snapshot is only usable if EVERY member captured; if any member
declines (the GDN host arena is exhausted, say), the whole snapshot is dropped. A partial restore is
precisely the failure the PLE gate existed to prevent, and "shallower match, more re-prefill" is the
outcome the store already produces at its existing drop sites — never a wrong answer.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple


class CompositeRecurrentState:
    """Fan `clone_slot`/`load_slot` across several per-slot recurrent caches."""

    def __init__(self, members: Sequence[Tuple[str, Any]]) -> None:
        #: (name, cache) in a FIXED order — the handle is positional, so the order that wrote a
        #: snapshot must be the order that reads it. Names are for diagnostics only.
        self._members = tuple(members)
        if not self._members:
            raise ValueError("CompositeRecurrentState needs at least one member cache")

    @property
    def member_names(self) -> Tuple[str, ...]:
        return tuple(n for n, _ in self._members)

    def clone_slot(self, slot: int):
        """Snapshot every member, or return None if any member declines.

        None is the store's existing "no snapshot" signal: `match_prefix` caps the reusable prefix
        to the deepest node that HAS one, so a decline costs re-prefill, not correctness.
        """
        parts = []
        for _name, cache in self._members:
            snap = cache.clone_slot(slot)
            if snap is None:
                return None
            parts.append(snap)
        return tuple(parts)

    def load_slot(self, slot: int, snap) -> None:
        """Restore every member from a composite handle. Refuses a handle of the wrong arity."""
        if snap is None:
            return
        if not isinstance(snap, tuple) or len(snap) != len(self._members):
            raise ValueError(
                f"composite recurrent snapshot has {len(snap) if isinstance(snap, tuple) else '?'} "
                f"part(s), expected {len(self._members)} ({', '.join(self.member_names)}). A handle "
                f"written by a different member set would restore one kind of state and silently "
                f"leave another at its initial value."
            )
        for (_name, cache), part in zip(self._members, snap):
            cache.load_slot(slot, part)

    def flush_ring(self, *args, **kwargs) -> None:
        """Forwarded to any member that has one (GDN's ReplaySSM ring)."""
        for _name, cache in self._members:
            fn = getattr(cache, "flush_ring", None)
            if fn is not None:
                fn(*args, **kwargs)
