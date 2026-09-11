"""PLE state must be spec-safe: a verify pass that rejects drafts must leave EXACTLY the state of a
pass that only ever saw the accepted tokens.

WHY THIS EXISTS. The backbone's PLE block is recurrent in two independent ways — a host n-gram token
history and a device conv window — and a speculative verify runs it over `[confirmed, d0 .. d_{K-1}]`,
most of which is typically rejected. Before this test, both halves were advanced over ALL of those
tokens and neither was ever rolled back, so every rejection left the block reading a lexical context
that never existed. That is silent: the text stays fluent, the target's hidden states are simply
wrong from then on, which (a) makes speculative decoding LOSSY rather than merely slow and (b)
poisons the very hidden states the MTP draft head is seeded from, so it compounds with the rejection
rate. GDN and CCA have had the equivalent accepted-prefix install since spec decode landed.

The invariant asserted here is the definition of losslessness for this block, not a proxy for it:

    verify(n tokens) then roll back to `keep`   ==   a pass that only ever processed `keep` tokens

Both halves are checked, and each has a FALSIFICATION arm reproducing the pre-fix behaviour (advance
over everything). If an arm agreed, this test could not tell the fixed code from the bug.

CPU only, no GPU, no checkpoint, no engine.

Run:  PYTHONPATH=python python3 tests/ple_spec_rollback_test.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "python")

from minisgl.ple.runtime import PLEBatch, PLERuntime  # noqa: E402
from minisgl.ple.state import PLEStateCache  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILED.append(name)


WIDE, STATE_LEN, CTX, EOS = 8, 9, 4, 7
SLOT = 1


def fresh_state() -> PLEStateCache:
    return PLEStateCache(num_slots=4, wide=WIDE, state_len=STATE_LEN, context_len=CTX,
                         eos_token_id=EOS, dtype=torch.float32, device=torch.device("cpu"))


class StubSource:
    """Only `advance` is exercised here; it is what PLEEmbeddingSource does with the history."""

    def advance(self, state, slots, token_lists):
        for slot, toks in zip(slots, token_lists):
            state.push_tokens(int(slot), toks)


def runtime(state) -> PLERuntime:
    rt = PLERuntime.__new__(PLERuntime)          # bypass __init__: no mmap'd table on this path
    rt.source, rt.state = StubSource(), state
    rt._pending = None
    rt.batch = None
    rt.commits = rt.prepares = rt.commit_noops = 0
    return rt


# ---------------------------------------------------------------------------- the pass under test
# 4 staged tokens (1 confirmed + 3 drafts); 2 of them survive (confirmed + 1 accepted draft).
STAGED = np.array([11, 12, 13, 14], dtype=np.int64)
KEEP = 2

print("HISTORY: the n-gram context must advance over the COMMITTED tokens only")
st = fresh_state()
rt = runtime(st)
rt._pending = PLEBatch(embeddings=torch.zeros(0), state_indices=torch.zeros(0, dtype=torch.int64),
                       slots=[SLOT], seq_lens=[len(STAGED)], is_decode=False,
                       tokens=[STAGED], defer_commit=True)
rt.commit_verified({SLOT: KEEP})
got_hist = st.history(SLOT).copy()

ref = fresh_state()
ref.push_tokens(SLOT, STAGED[:KEEP])             # a pass that only ever saw the accepted tokens
check("history == accepted-only pass", np.array_equal(got_hist, ref.history(SLOT)),
      f"{got_hist} vs {ref.history(SLOT)}")

bad = fresh_state()
bad.push_tokens(SLOT, STAGED)                    # PRE-FIX: advanced over the rejected drafts too
check("FALSIFICATION: advancing over all drafts disagrees",
      not np.array_equal(got_hist, bad.history(SLOT)), f"both {got_hist}")

print()
print("CONV WINDOW: the state must be the one after the COMMITTED tokens only")
torch.manual_seed(3)
n = len(STAGED)
chunk = torch.randn(WIDE, n)
st2 = fresh_state()
st2.conv_state[SLOT] = torch.randn(WIDE, STATE_LEN)
window = torch.cat([st2.conv_state[SLOT], chunk], dim=-1)   # what _short_conv builds, (wide, s+n)

full_pass = window[:, -STATE_LEN:].clone()                  # PRE-FIX write: state after ALL n
st2.install_verify_conv(SLOT, window, KEEP)
got_conv = st2.conv_state[SLOT].clone()

ref_conv = torch.cat([torch.zeros(0), window[:, KEEP:KEEP + STATE_LEN]], dim=-1)
check("conv == state after exactly `keep` tokens", torch.equal(got_conv, ref_conv),
      f"max|d|={(got_conv-ref_conv).abs().max().item():.3e}")
check("FALSIFICATION: the full-pass window disagrees", not torch.equal(got_conv, full_pass))

# keep == n is the no-rejection case and MUST be the untouched full-pass state, or a fully-accepted
# step would corrupt the very state it got right.
st3 = fresh_state()
st3.conv_state[SLOT] = torch.randn(WIDE, STATE_LEN)
w3 = torch.cat([st3.conv_state[SLOT], chunk], dim=-1)
st3.install_verify_conv(SLOT, w3, n)
check("keep == n reproduces the full-pass state", torch.equal(st3.conv_state[SLOT], w3[:, -STATE_LEN:]))

print()
print("NULL slot (cudagraph padding) is never advanced")
st4 = fresh_state()
rt4 = runtime(st4)
rt4._pending = PLEBatch(embeddings=torch.zeros(0), state_indices=torch.zeros(0, dtype=torch.int64),
                        slots=[0], seq_lens=[len(STAGED)], is_decode=False,
                        tokens=[STAGED], defer_commit=True)
rt4.commit_verified({})
check("NULL slot history untouched", np.array_equal(st4.history(0), np.full(CTX, EOS, dtype=np.int64)))

print()
print("an un-adjudicated real slot falls back to the full advance (pre-fix behaviour, deliberately)")
st5 = fresh_state()
rt5 = runtime(st5)
rt5._pending = PLEBatch(embeddings=torch.zeros(0), state_indices=torch.zeros(0, dtype=torch.int64),
                        slots=[SLOT], seq_lens=[len(STAGED)], is_decode=False,
                        tokens=[STAGED], defer_commit=True)
rt5.commit_verified({})
ref5 = fresh_state(); ref5.push_tokens(SLOT, STAGED)
check("missing slot -> full advance", np.array_equal(st5.history(SLOT), ref5.history(SLOT)))

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
