"""Re-derive `toolcall_degen_probe`'s calibration by replaying REAL traffic through its detectors.

The probe's detector thresholds are not guesses: they were tuned against the assistant turns Hermes
has already stored, and this script is how that was done and how it is re-checked. It replays every
stored assistant turn for a set of models through `score_turn` and prints (a) whether the six
hand-read junk turns are still caught, (b) how many OTHER turns fire, and (c) the per-model rate
table reproduced in the probe's docstring.

Run it after ANY detector change. A detector edit that silently starts firing on a third of healthy
traffic is the failure this exists to catch — the first cut of `junk_args` did exactly that, by
treating the integer `6` in `{"query": "...", "limit": 6}` as a two-character placeholder.

    python3 tools/toolcall_degen_replay_validate.py

Read-only against ~/.hermes/state.db. No GPU, no serve, no lease.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toolcall_degen_probe import DIAGNOSTIC, PRIMARY, SECONDARY, score_turn  # noqa: E402

DB = os.path.expanduser("~/.hermes/state.db")

# Hand-read from the transcripts on 2026-09-20. cdb27addb762 turns 37027/37077/37079/37082/37085
# (`{"code": "x"}` and `# placeholder\nprint('ok')`) and 30935df66949 turn 34460 (`# placeholder`).
KNOWN_JUNK = {37027, 37077, 37079, 37082, 37085, 34460}
Q4E = "Qwen3.8-Flash-Next"
CONTROLS = ("cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit", "sakamakismile/Qwen3.8-27B-MTP-NVFP4",
            "Qwen3.8-27B", "deepseek/deepseek-v4.1-flash")
KEYS = PRIMARY + SECONDARY + DIAGNOSTIC


def scan(con, model):
    rows = con.execute(
        """select m.id, m.session_id, m.finish_reason, m.content, m.tool_calls,
                  m.reasoning_content, m.reasoning
           from sessions s join messages m on m.session_id = s.id
           where s.model = ? and m.role = 'assistant'
           order by m.session_id, m.timestamp, m.id""", (model,)).fetchall()
    prior, out = {}, []
    for mid, sid, fin, content, tc, rc, rr in rows:
        msg = {"content": content, "reasoning_content": rc, "reasoning": rr,
               "tool_calls": json.loads(tc) if tc else []}
        pt = prior.setdefault(sid, [])
        sc = score_turn(msg, pt, fin or "-")
        pt.append(len(rc or rr or ""))
        out.append((mid, sc))
    return out


def main() -> int:
    if not os.path.exists(DB):
        print(f"no Hermes db at {DB} — nothing to replay")
        return 2
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    q4e = scan(con, Q4E)
    caught = sorted(m for m, sc in q4e if m in KNOWN_JUNK and sc["junk_args"])
    missed = sorted(KNOWN_JUNK - set(caught))
    extra = sorted(m for m, sc in q4e if m not in KNOWN_JUNK and sc["junk_args"])
    print(f"junk_args vs hand-read ground truth: caught {len(caught)}/{len(KNOWN_JUNK)}")
    if missed:
        print(f"  MISSED (detector went blind): {missed}")
    if extra:
        print(f"  EXTRA firings on {len(extra)} of {len(q4e) - len(KNOWN_JUNK)} other turns: {extra}")
    if not missed and not extra:
        print("  clean: every known junk turn caught, no other turn flagged")

    print(f"\n{'model':<42}{'turns':>6}" + "".join(f"{k:>22}" for k in KEYS))
    for model in (Q4E,) + CONTROLS:
        rows = scan(con, model)
        n = max(len(rows), 1)
        cells = ""
        for k in KEYS:
            hits = sum(int(sc[k]) for _, sc in rows)
            cells += f"{hits:>8} {hits / n * 100:>11.1f}%"
        print(f"{model:<42}{len(rows):>6}{cells}")

    print(f"\nPRIMARY={PRIMARY}  SECONDARY={SECONDARY}  DIAGNOSTIC (not defects)={DIAGNOSTIC}")
    return 1 if (missed or extra) else 0


if __name__ == "__main__":
    raise SystemExit(main())
