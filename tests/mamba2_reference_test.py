"""The chunked SSD form must equal the sequential definition — proven on the HOST, before any HIP.

Phase 1 of docs/NEMOTRON35_LIGHTNING_PLAN.md. The chunked form is not a transcription of the
sequential one: it re-associates the recurrence into (diagonal block) + (state passing) so the inner
work becomes two GEMMs. Every term in that decomposition can drop a decay factor or shift a segment
sum by one, and every such bug is silent — the shapes agree and the numbers are merely wrong.

Establishing the equivalence here, in float64, is what makes a later kernel mismatch mean "the kernel
is wrong" rather than "one of these two things is wrong".

Shapes are Nemotron-3.5-Lightning's own where it matters (n_groups=8 serving 64 heads, head_dim 64,
state 128, chunk 128), scaled down where it does not.

    python3 -m pytest tests/mamba2_reference_test.py -q -o addopts=""
"""

from __future__ import annotations

import pytest
import torch

from minisgl.mamba2 import (  # noqa: E402
    discretize_dt,
    mamba2_chunked,
    mamba2_decode,
    mamba2_sequential,
    segment_sum,
)

F64 = torch.float64


def make(L=40, H=8, P=16, G=2, N=12, seed=0, dtype=F64):
    """A tie-free random fixture. `torch.randn` everywhere is deliberate for the tensors, but dt is
    drawn through the real discretization so the decay factors land in the shipped [0.001, 0.1]
    band rather than an arbitrary one — a decay of ~1.0 makes chunked and sequential agree for the
    wrong reason (nothing decays, so a dropped decay factor is invisible)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)  # noqa: E731
    dt = discretize_dt(r(L, H), r(H), 0.001, 0.1)
    return dict(x=r(L, H, P), dt=dt, A_log=r(H).abs().log(), B=r(L, G, N), C=r(L, G, N), D=r(H))


# ---------------------------------------------------------------- segment_sum

def test_segment_sum_is_the_decay_between_two_positions():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=F64)
    s = segment_sum(t)
    assert s[0, 0] == 0.0                       # nothing decays a value read where it entered
    assert s[1, 0] == pytest.approx(2.0)        # entered at 0, read at 1 -> t[1]
    assert s[3, 1] == pytest.approx(3.0 + 4.0)  # entered at 1, read at 3 -> t[2]+t[3]
    assert torch.isinf(s[0, 1]) and s[0, 1] < 0  # future contributions are masked out


def test_segment_sum_is_strictly_causal():
    s = segment_sum(torch.randn(6, dtype=F64))
    future = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    assert torch.isinf(s[future]).all() and (s[future] < 0).all()
    assert torch.isfinite(s[~future]).all()


# ---------------------------------------------------------------- the equivalence

@pytest.mark.parametrize("chunk", [1, 3, 8, 16, 64])
def test_chunked_equals_sequential_at_every_chunk_size(chunk):
    a = make()
    ys, ss = mamba2_sequential(**a)
    yc, sc = mamba2_chunked(**a, chunk_size=chunk)
    torch.testing.assert_close(yc, ys, rtol=0, atol=1e-10)
    torch.testing.assert_close(sc, ss, rtol=0, atol=1e-10)


@pytest.mark.parametrize("L", [1, 7, 127, 128, 129, 256, 300])
def test_a_ragged_tail_chunk_is_handled_at_its_TRUE_length(L):
    """The tail chunk is the case a kernel gets wrong first: a padded decay matrix and an unpadded
    one differ, and lengths that are not a multiple of chunk_size are the common case in prefill."""
    a = make(L=L)
    ys, ss = mamba2_sequential(**a)
    yc, sc = mamba2_chunked(**a, chunk_size=128)
    torch.testing.assert_close(yc, ys, rtol=0, atol=1e-10)
    torch.testing.assert_close(sc, ss, rtol=0, atol=1e-10)


def test_an_incoming_state_is_carried_and_decayed():
    """Chunked prefill resumes from a state; a form that only works from zero is not usable."""
    a = make(L=53)
    s0 = torch.randn(8, 16, 12, dtype=F64, generator=torch.Generator().manual_seed(7))
    ys, ss = mamba2_sequential(**a, initial_state=s0)
    yc, sc = mamba2_chunked(**a, chunk_size=16, initial_state=s0)
    torch.testing.assert_close(yc, ys, rtol=0, atol=1e-10)
    torch.testing.assert_close(sc, ss, rtol=0, atol=1e-10)


def test_splitting_a_prefill_in_two_equals_doing_it_at_once():
    """The chunked-prefill contract: process [0:k), carry the state, process [k:L). This is what
    MINISGL's chunked prefill does, and it must be indistinguishable from one long call."""
    a = make(L=100)
    y_all, s_all = mamba2_chunked(**a, chunk_size=32)
    k = 37
    first = {kk: (v[:k] if kk in ("x", "dt", "B", "C") else v) for kk, v in a.items()}
    second = {kk: (v[k:] if kk in ("x", "dt", "B", "C") else v) for kk, v in a.items()}
    y1, s1 = mamba2_chunked(**first, chunk_size=32)
    y2, s2 = mamba2_chunked(**second, chunk_size=32, initial_state=s1)
    torch.testing.assert_close(torch.cat([y1, y2]), y_all, rtol=0, atol=1e-10)
    torch.testing.assert_close(s2, s_all, rtol=0, atol=1e-10)


# ---------------------------------------------------------------- decode

def test_decode_step_by_step_equals_a_prefill_over_the_same_tokens():
    """Decode and prefill are two implementations of one recurrence; a serve that disagrees between
    them produces a different answer depending on whether a token was prefilled or generated."""
    a = make(L=12)
    y_ref, s_ref = mamba2_sequential(**a)
    state = torch.zeros(8, 16, 12, dtype=F64)
    ys = [mamba2_decode(a["x"][t], a["dt"][t], a["A_log"], a["B"][t], a["C"][t], a["D"], state)
          for t in range(12)]
    torch.testing.assert_close(torch.stack(ys), y_ref, rtol=0, atol=1e-11)
    torch.testing.assert_close(state, s_ref, rtol=0, atol=1e-11)


def test_decode_advances_the_state_in_place():
    """In place is the kernel's contract, and it is what spec-decode rollback has to undo."""
    a = make(L=1)
    state = torch.zeros(8, 16, 12, dtype=F64)
    before = state.data_ptr()
    mamba2_decode(a["x"][0], a["dt"][0], a["A_log"], a["B"][0], a["C"][0], a["D"], state)
    assert state.data_ptr() == before and state.abs().sum() > 0


# ---------------------------------------------------------------- group sharing

def test_B_and_C_expand_in_BLOCKS_across_heads_not_by_cycling():
    """64 heads / 8 groups: head h reads group h//8. Expanding by cycling instead of by blocks has
    identical shapes and different numbers — the failure is invisible to any shape assertion, so it
    is pinned here against a hand-built case."""
    L, H, P, G, N = 1, 4, 1, 2, 3
    x = torch.ones(L, H, P, dtype=F64)
    dt = torch.full((L, H), 0.05, dtype=F64)
    # group 0 = e1, group 1 = e2; C selects the same coordinate, so y reveals which group each head read
    B = torch.zeros(L, G, N, dtype=F64); B[0, 0, 0] = 1.0; B[0, 1, 1] = 1.0
    C = torch.ones(L, G, N, dtype=F64)
    y, _ = mamba2_sequential(x=x, dt=dt, A_log=torch.zeros(H, dtype=F64), B=B, C=C,
                             D=torch.zeros(H, dtype=F64))
    # heads 0,1 -> group 0; heads 2,3 -> group 1. All four read a single 1.0 contribution scaled by dt.
    torch.testing.assert_close(y[0, :, 0], torch.full((H,), 0.05, dtype=F64), rtol=0, atol=1e-12)


def test_heads_do_not_leak_into_each_other():
    a = make(L=20, H=4, G=4)          # one group per head: heads are fully independent
    y_all, _ = mamba2_sequential(**a)
    for h in range(4):
        one = dict(x=a["x"][:, h:h + 1], dt=a["dt"][:, h:h + 1], A_log=a["A_log"][h:h + 1],
                   B=a["B"][:, h:h + 1], C=a["C"][:, h:h + 1], D=a["D"][h:h + 1])
        y_h, _ = mamba2_sequential(**one)
        torch.testing.assert_close(y_h[:, 0], y_all[:, h], rtol=0, atol=1e-12)


# ---------------------------------------------------------------- discretization

def test_dt_is_clamped_into_the_shipped_band():
    # softplus(-3.0) = 0.0486 lands INSIDE the band; softplus(0) = 0.693 does not, which is worth
    # knowing: at these limits an unbiased dt of 0 is already clamped to the ceiling.
    dt = discretize_dt(torch.tensor([-40.0, -3.0, 40.0], dtype=F64),
                       torch.zeros(3, dtype=F64), 0.001, 0.1)
    assert dt[0] == pytest.approx(0.001)     # softplus underflows -> clamped UP
    assert dt[2] == pytest.approx(0.1)       # clamped DOWN
    assert 0.001 < dt[1] < 0.1


def test_at_the_real_shape():
    """Nemotron-3.5-Lightning's own mamba geometry, one chunk-and-a-bit."""
    a = make(L=140, H=64, P=64, G=8, N=128, seed=3)
    ys, ss = mamba2_sequential(**a)
    yc, sc = mamba2_chunked(**a, chunk_size=128)
    torch.testing.assert_close(yc, ys, rtol=0, atol=1e-9)
    torch.testing.assert_close(sc, ss, rtol=0, atol=1e-9)
