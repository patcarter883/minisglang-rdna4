"""Regression test for reasoning-delimiter DERIVATION and the derived `thinking_open`.

The bug this locks down: serving a checkpoint whose reasoning markup is not `<think>`/`</think>`
(Gemma-4's `<|channel>thought` … `<channel|>`) returned `content=""` with the entire answer in
`reasoning_content`, on EVERY request. Two defaults conspired — a family-name delimiter table that
fell through to `<think>` for any checkpoint without a row, and a `thinking_open` that defaulted
True. Both are now derived from the checkpoint's own artifacts.

Two halves, run independently:

* PURE — parser semantics, no model files. Always runs.
* ARTIFACT — the derivation cascade against the real cached checkpoints. Skipped per-model when a
  checkpoint is not in the HF cache, so this is runnable anywhere; the models that ARE cached are
  asserted against the legacy table, which is the regression oracle for Qwen3/GLM/Laguna.

Run:  PYTHONPATH=python python3 tests/reasoning_delimiter_derivation_test.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from minisgl.server.reasoning import (  # noqa: E402
    ReasoningParser,
    derive_delimiters,
    derive_end_wildcard,
    get_reasoning_parser,
    resolve_reasoning_parser,
)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------------------------------------
# PURE: parser semantics
# ---------------------------------------------------------------------------------------------
THINK = ReasoningParser("<think>", "</think>")
# Gemma-4's pair. ASYMMETRIC — the opener carries a channel name the closer does not, which no
# `<x>`/`</x>` table row can express.
CHAN = ReasoningParser("<|channel>thought", "<channel|>")

print("PURE 1: prompt_state reads the rendered prompt, never a default")
CASES = [
    # (parser, rendered generation prompt, span_open, reasoning_possible, why)
    (THINK, "<|im_start|>assistant\n<think>\n", True, True, "Qwen3.6/GLM: template opens the span"),
    (THINK, "<|im_start|>assistant\n<think>\n\n</think>\n\n", False, False, "Qwen3 thinking-off"),
    (THINK, "<user>hi</user>\n<assistant><think>", True, True, "Laguna thinking-on"),
    (THINK, "<user>hi</user>\n<assistant></think>", False, False, "Laguna default (pre-closed)"),
    (THINK, "<|im_start|>assistant\n", False, True, "Qwen3-0.6B: opener left to the MODEL"),
    (CHAN, "<|turn>model\n<|channel>thought\n<channel|>", False, False, "Gemma-4 thinking-off"),
    (CHAN, "<|turn>model\n", False, True, "Gemma-4 thinking-on: opener left to the model"),
    (CHAN, "<|turn>model\n<|channel>thought\n", True, True, "Gemma-4 mid-span"),
    # History already closed; the generation prompt then reopens -> the LAST delimiter wins.
    (THINK, "<think>old</think>answer<|im_start|>assistant\n<think>\n", True, True, "multi-turn"),
]
for parser, prompt, want_open, want_poss, why in CASES:
    got = parser.prompt_state(prompt)
    check(f"prompt_state {why}", got == (want_open, want_poss), f"got {got}")

print("PURE 2: parse — a missing close tag must not eat the answer")
check("closed span splits", THINK.parse("reasoning</think>The answer is 4.", thinking_open=True)
      == ("reasoning", "The answer is 4."))
check("span not open, no close tag -> ALL content (the bug)",
      THINK.parse("4", thinking_open=False) == (None, "4"))
check("span open, no close tag -> ALL reasoning (truncated CoT)",
      THINK.parse("still thinking", thinking_open=True) == ("still thinking", ""))
check("model-side opener, no close tag -> reasoning even with thinking_open=False",
      CHAN.parse("<|channel>thought\nhmm", thinking_open=False) == ("hmm", ""))
check("gemma asymmetric pair splits", CHAN.parse(
    "<|channel>thought\nlet me add\n<channel|>4", thinking_open=False) == ("let me add", "4"))
check("gemma thinking-off passthrough", CHAN.parse("4", thinking_open=False) == (None, "4"))
check("explicit close wins over span state",
      CHAN.parse("r<channel|>a", thinking_open=False) == ("r", "a"))

print("PURE 3: streaming splitter mirrors parse")


def stream(parser: ReasoningParser, chunks: list[str], active: bool):
    st = parser.stream_state(active=active)
    r, c = "", ""
    for ch in chunks:
        rd, cd = st.push(ch)
        r += rd or ""
        c += cd or ""
    rt, ct = st.flush()
    return r + (rt or ""), c + (ct or "")


check("prompt-side open, split across chunks",
      stream(THINK, ["reason", "ing</thi", "nk>the ans", "wer"], True) == ("reasoning", "the answer"))
check("span closed -> everything is content",
      stream(THINK, ["the ", "answer"], False) == ("", "the answer"))
check("model-side opener detected mid-stream",
      stream(CHAN, ["<|chan", "nel>thought\nmus", "ing<channel|>42"], False) == ("musing", "42"))
# The live gemma-4 serve emits `<|channel>thought` as its OWN chunk (it is one token) and the
# separator `\n` as the next, so trimming only at the opener left a leading newline on streamed
# reasoning_content that non-streaming `parse` (.strip()) did not have. Chunk boundaries must never
# change the payload.
check("opener alone in a chunk, separator in the next",
      stream(CHAN, ["<|channel>thought", "\n", "musing", "<channel|>", "42"], False)
      == ("musing", "42"))
RAW = "<|channel>thought\nmusing<channel|>42"
_r, _c = CHAN.parse(RAW, thinking_open=False)
check("streamed == non-streamed for the same text",
      stream(CHAN, ["<|channel>thought", "\nmusing", "<channel|>42"], False) == (_r or "", _c or ""))
check("opener probe releases plain text unharmed",
      stream(CHAN, ["<", "b>bold</b> answer"], False) == ("", "<b>bold</b> answer"))
check("probe tail is flushed, not dropped", stream(CHAN, ["<|ch"], False) == ("", "<|ch"))

print("PURE 4: recipient-wildcard closer (channel-routed templates)")
# Muse-Glimmer: reasoning is a separate TURN and the closer names the RECIPIENT, so the concrete
# `to=user` form derived from a history probe cannot match a reply that ends reasoning by routing to
# a TOOL. Matched literally, such a reply has no closer at all and the whole thing — tool call
# included — is filed under reasoning_content.
MUSE = ReasoningParser(
    " to=self<|message|>", "<|eom|><|start|>assistant to=user<|message|>",
    " to=user<|message|>",
    "<|eom|><|start|>assistant to=", "<|message|>",
)
check("wildcard: user-routed closer splits",
      MUSE.parse(" to=self<|message|>musing<|eom|><|start|>assistant to=user<|message|>42",
                 thinking_open=False) == ("musing", "42"))
check("wildcard: TOOL-routed closer splits (the defect)",
      MUSE.parse(" to=self<|message|>musing<|eom|><|start|>assistant to=browser.search"
                 "<|message|><atem:function_calls/>", thinking_open=False)
      == ("musing", "<atem:function_calls/>"))
check("wildcard: no closer at all is still all-reasoning",
      MUSE.parse(" to=self<|message|>still thinking", thinking_open=False)
      == ("still thinking", ""))
check("wildcard: a direct answer keeps its header stripped and stays content",
      MUSE.parse(" to=user<|message|>42", thinking_open=False) == (None, "42"))
check("wildcard: prefix alone is not a closer",
      MUSE.parse(" to=self<|message|>a<|eom|><|start|>assistant to=", thinking_open=False)
      == ("a<|eom|><|start|>assistant to=", ""))
check("wildcard: streamed == non-streamed, tool-routed",
      stream(MUSE, [" to=self<|message|>mus", "ing<|eom|><|start|>assist",
                    "ant to=browser.sea", "rch<|message|>", "<atem:function_calls/>"], False)
      == ("musing", "<atem:function_calls/>"))
check("wildcard: recipient split across chunks is not leaked as content",
      stream(MUSE, [" to=self<|message|>m<|eom|><|start|>assistant to=", "tool", "<|message|>x"],
             False) == ("m", "x"))
# A literal-closer parser must be completely unaffected by the new machinery.
check("literal closer unaffected (rpartition semantics kept)",
      THINK.parse("a</think>b</think>c", thinking_open=True) == ("a</think>b", "c"))

print("PURE 5: explicit --reasoning-parser still wins, `none` still disables")
p, how = resolve_reasoning_parser(None, requested="qwen3")
check("explicit name selects the table row", (p.start_token, p.end_token) == ("<think>", "</think>"))
check("explicit provenance", "--reasoning-parser" in how)
p, how = resolve_reasoning_parser(None, requested="none")
check("`none` disables", p is None)
check("declared name still resolves through the table",
      (lambda t: (t[0].start_token, t[0].end_token))(
          resolve_reasoning_parser(None, requested="auto", declared="poolside_v1"))
      == ("<think>", "</think>"))

# ---------------------------------------------------------------------------------------------
# ARTIFACT: derive from the real cached checkpoints
# ---------------------------------------------------------------------------------------------
# `None` = the legacy table had no row, so the old code silently used <think>/</think>. Anything
# else is the pair the table WOULD have produced — derivation must reproduce it exactly.
ORACLE = {
    "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit": ("<think>", "</think>"),
    "Qwen/Qwen3-0.6B": ("<think>", "</think>"),
    "QuantTrio/GLM-4.7-Flash-AWQ": ("<think>", "</think>"),
    "poolside/Laguna-XS-2.1-NVFP4": ("<think>", "</think>"),
    "Zyphra/ZAYA1-8B-MXFP4-Experts": ("<think>", "</think>"),
    "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4": ("<|channel>thought", "<channel|>"),
    "cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4": ("<|channel>thought", "<channel|>"),
    # Genuinely non-reasoning: the template has no enable_thinking branch, so derivation must find
    # nothing rather than invent a pair.
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8": None,
}

print("ARTIFACT: derivation vs the legacy table (the regression oracle)")
try:
    from minisgl.utils import load_tokenizer
except Exception as e:  # noqa: BLE001
    print(f"  SKIP all — transformers unavailable ({e})")
    load_tokenizer = None

if load_tokenizer is not None:
    for model, expect in ORACLE.items():
        try:
            tok = load_tokenizer(model)
        except Exception as e:  # noqa: BLE001 — not cached on this box
            print(f"  SKIP  {model} (not cached: {type(e).__name__})")
            continue
        got = derive_delimiters(tok)
        pair = (got[0], got[1]) if got else None
        how = got[2] if got else "no reasoning convention"
        check(f"{model} -> {pair}", pair == expect, f"expected {expect}")
        print(f"        via: {how}")
        # A `<think>`-family template does not vary its closer by recipient, so the wildcard
        # derivation must find NOTHING for every model in the oracle — that is what keeps the common
        # path on the literal matcher, byte for byte.
        check(f"{model}: no recipient wildcard", derive_end_wildcard(tok, got[1] if got else "")
              is None)

    # Muse-Glimmer: derivation step 2 finds nothing (no enable_thinking branch), step 3 reads the
    # pair off an assistant turn carrying reasoning_content, and the wildcard probe then brackets the
    # recipient. This is the checkpoint the whole change exists for.
    MUSE_ID = "RedHatAI/Muse-Glimmer-30B-NVFP4"
    try:
        mtok = load_tokenizer(MUSE_ID)
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP  {MUSE_ID} (not cached: {type(e).__name__})")
        mtok = None
    if mtok is not None:
        mp, mhow = resolve_reasoning_parser(mtok)
        check("muse opener", mp.start_token == " to=self<|message|>", repr(mp.start_token))
        check("muse closer", mp.end_token == "<|eom|><|start|>assistant to=user<|message|>",
              repr(mp.end_token))
        check("muse answer-turn header", mp.turn_header == " to=user<|message|>",
              repr(mp.turn_header))
        check("muse recipient wildcard derived",
              (mp.end_prefix, mp.end_suffix) == ("<|eom|><|start|>assistant to=", "<|message|>"),
              f"{(mp.end_prefix, mp.end_suffix)}")
        # The gate bug in one assertion: every delimiter ends in the SAME token, so a gate keyed on
        # the last id opens on the OPENER.
        last = [tuple(mtok.encode(s, add_special_tokens=False))[-1]
                for s in (mp.start_token, mp.end_token, mp.turn_header)]
        check("muse delimiters share a trailing token (why the gate must match sequences)",
              len(set(last)) == 1, f"{last}")
        check("muse closer is multi-token",
              len(mtok.encode(mp.end_token, add_special_tokens=False)) > 1)
        # The header must NOT be a suffix of the opener, or registering it as a gate release trigger
        # would fire at the start of the reasoning turn (see _reasoning_answer_delim's guard).
        check("muse header is not a suffix of the opener",
              not mp.start_token.endswith(mp.turn_header))
        print(f"        via: {mhow}")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
