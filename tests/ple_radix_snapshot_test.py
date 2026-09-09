"""PLE recurrent state under recurrent-radix prefix caching.

THE DEFECT THIS GUARDS. A PLE model keeps TWO per-sequence states in one slot space: GDN's conv+ssm
and PLE's dilated conv window + n-gram token history. The snapshot store only ever captured the
first, so `resolve_prefix_cache` refused prefix caching outright for PLE models — correct, and it
cost a live serve 100% of its prefix reuse (measured 2026-09-09: 204,618 prompt tokens, 0 hit
tokens, on a serve whose every Hermes turn resends the whole transcript).

Restoring one state and not the other is WRONG TEXT WITH NO ERROR, so these tests check both halves
and the all-or-nothing rule, not just that a snapshot exists.

    python3 -m pytest tests/ple_radix_snapshot_test.py -q -o addopts=""
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from minisgl.kvcache.composite_state import CompositeRecurrentState  # noqa: E402
from minisgl.ple.state import PLEStateCache  # noqa: E402

EOS = 7


def _ple(slots=4):
    return PLEStateCache(num_slots=slots, wide=3, state_len=8, context_len=2,
                         eos_token_id=EOS, dtype=torch.float32, device=torch.device("cpu"))


def test_snapshot_captures_BOTH_halves():
    """conv window AND token history. Capturing only the device half is the original defect in
    miniature — the restored slot would carry the right conv state against a history still at EOS."""
    c = _ple()
    c.conv_state[1].fill_(2.5)
    c.token_history[1] = np.array([11, 22], dtype=np.int64)
    snap = c.clone_slot(1)
    conv, hist = snap
    assert torch.equal(conv, torch.full((3, 8), 2.5))
    assert np.array_equal(hist, np.array([11, 22]))


def test_snapshot_is_slot_agnostic_and_restores_into_a_DIFFERENT_slot():
    """The whole point: a later request lands in whatever slot is free and must be able to resume
    another sequence's prefix state there."""
    c = _ple()
    c.conv_state[0].fill_(1.25)
    c.token_history[0] = np.array([3, 4], dtype=np.int64)
    snap = c.clone_slot(0)
    c.load_slot(2, snap)
    assert torch.equal(c.conv_state[2], c.conv_state[0])
    assert np.array_equal(c.token_history[2], c.token_history[0])


def test_the_snapshot_is_a_COPY_not_a_view():
    """A view would track the source slot as it keeps decoding, so the 'restore' would install
    state from an arbitrarily later point in that sequence."""
    c = _ple()
    c.conv_state[0].fill_(1.0)
    c.token_history[0] = np.array([5, 6], dtype=np.int64)
    snap = c.clone_slot(0)
    c.conv_state[0].fill_(9.0)                      # source keeps advancing
    c.token_history[0] = np.array([99, 98], dtype=np.int64)
    c.load_slot(1, snap)
    assert torch.equal(c.conv_state[1], torch.ones(3, 8)), "conv snapshot aliased the live slot"
    assert np.array_equal(c.token_history[1], np.array([5, 6])), "history snapshot aliased"


def test_composite_is_all_or_nothing():
    """If ANY member declines (GDN's host arena exhausted), the whole snapshot must be dropped.
    A partial handle is exactly the half-restored state the PLE gate existed to prevent."""
    class Declines:
        def clone_slot(self, slot): return None
        def load_slot(self, slot, snap): raise AssertionError("must not be called")

    comp = CompositeRecurrentState((("gdn", Declines()), ("ple", _ple())))
    assert comp.clone_slot(0) is None, "composite returned a partial snapshot"


def test_composite_fans_out_and_round_trips():
    a, b = _ple(), _ple()
    a.conv_state[0].fill_(4.0); a.token_history[0] = np.array([1, 2], dtype=np.int64)
    b.conv_state[0].fill_(8.0); b.token_history[0] = np.array([3, 4], dtype=np.int64)
    comp = CompositeRecurrentState((("a", a), ("b", b)))
    snap = comp.clone_slot(0)
    assert isinstance(snap, tuple) and len(snap) == 2
    comp.load_slot(3, snap)
    assert torch.equal(a.conv_state[3], a.conv_state[0]) and torch.equal(b.conv_state[3], b.conv_state[0])
    assert np.array_equal(a.token_history[3], a.token_history[0])


def test_composite_refuses_a_handle_of_the_WRONG_ARITY():
    """A handle written by a different member set must not silently restore a subset — that is the
    original bug wearing a tuple."""
    comp = CompositeRecurrentState((("a", _ple()), ("b", _ple())))
    with pytest.raises(ValueError, match="expected 2"):
        comp.load_slot(0, (("only-one",),))


def test_load_of_None_is_a_noop():
    """`clone_slot` returning None is the store's existing 'no snapshot' signal and reaches here."""
    c = _ple()
    c.conv_state[0].fill_(3.0)
    c.load_slot(0, None)
    assert torch.equal(c.conv_state[0], torch.full((3, 8), 3.0))
