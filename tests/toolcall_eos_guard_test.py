"""A turn must not END inside a tool-call block the model opened.

MEASURED DEFECT (2026-09-21). Until 3799717d an xgrammar structural tag constrained the body of a
`<tool_call>` wrapper, and a SIDE EFFECT of that constraint was that the turn-ending token could not
be sampled mid-structure -- checked against the live Flash-Next tokenizer, `<|im_end|>` (248046) was
MASKED inside an open call while `<|endoftext|>` was not. Removing the tag fixed a much worse problem
(it forced a JSON body on a checkpoint whose own template mandates XML: 25% junk arguments, code
bodies capped at 369 chars) but it also removed that side effect. The model then began ending turns
mid-call, which the frontend recovers as prose with finish_reason=length -- twice in eleven turns of
Hermes session e5b8b76e21e8, at 13:26:33 and 13:27:54.

This guard restores ONLY that property: it refuses EOS while a block is open and constrains no
format. Torch-free, so it runs with no GPU.

    python3 tests/toolcall_eos_guard_test.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

_P = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "..", "python", "minisgl", "scheduler", "think_gate.py")
_spec = importlib.util.spec_from_file_location("tg_under_test", _P)
tg = importlib.util.module_from_spec(_spec)
sys.modules["tg_under_test"] = _spec.name and tg     # dataclass needs the module registered
_spec.loader.exec_module(tg)

OPEN, CLOSE, EOS, OTHER_EOS = (248058,), (248059,), 248046, 248044
FAILS: list = []


def check(label, got, want):
    ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'}  {label}: got {got!r}, want {want!r}")
    if not ok:
        FAILS.append(label)


def gate(budget=8192):
    g = tg.ToolCallGate(budget=budget)
    assert g.arm("u", openers=[OPEN], closers=[CLOSE], eos_ids=[EOS, OTHER_EOS])
    return g


def feed(g, toks):
    for t in toks:
        g.commit("u", t)


print("THE GUARD: EOS is refused only while a call block is open")
g = gate()
check("outside a call, EOS allowed", g.suppress_eos("u"), False)
feed(g, OPEN)
check("block open, EOS refused", g.suppress_eos("u"), True)
feed(g, [1, 2, 3])
check("still inside the body", g.suppress_eos("u"), True)
feed(g, CLOSE)
check("closed again, EOS allowed", g.suppress_eos("u"), False)

print("\nBOUNDED: a block that never closes must not hold the turn open forever")
g = gate(budget=4)
feed(g, OPEN)
feed(g, [7] * 3)
check("under budget, still held", g.suppress_eos("u"), True)
feed(g, [7] * 3)
check("past budget, released", g.suppress_eos("u"), False)

print("\nDEADLOCK GUARD: an EOS id INSIDE a delimiter disables suppression entirely")
g2 = tg.ToolCallGate()
g2.arm("v", openers=[(EOS, 11)], closers=[CLOSE], eos_ids=[EOS])
for t in (EOS, 11):
    g2.commit("v", t)
check("would-deadlock gate never suppresses", g2.suppress_eos("v"), False)

print("\nARMING: nothing to gate means nothing armed")
g3 = tg.ToolCallGate()
check("no openers", g3.arm("w", openers=[], closers=[CLOSE]), False)
check("no closers", g3.arm("w", openers=[OPEN], closers=[]), False)
check("disabled", tg.ToolCallGate(enabled=False).arm("w", openers=[OPEN], closers=[CLOSE]), False)
g4 = gate()
check("re-arm is a no-op", g4.arm("u", openers=[OPEN], closers=[CLOSE]), False)

print("\nLIFECYCLE: freeing a uid drops its state")
g5 = gate()
feed(g5, OPEN)
check("armed and open", g5.suppress_eos("u"), True)
g5.free("u")
check("after free", g5.suppress_eos("u"), False)
check("nothing armed", g5.any_armed(), False)

print("\nNESTING IS NOT ASSUMED: a second opener inside a block does not need two closers")
g6 = gate()
feed(g6, OPEN)
feed(g6, OPEN)          # the model repeating the opener must not require two closers to recover
feed(g6, CLOSE)
check("one closer clears it", g6.suppress_eos("u"), False)

print()
if FAILS:
    print(f"FAILED: {len(FAILS)} check(s): {FAILS}")
    raise SystemExit(1)
print("ALL CHECKS PASS")
