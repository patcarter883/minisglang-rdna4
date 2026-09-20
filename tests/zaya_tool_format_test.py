"""ZAYA's native tool-call format must be DERIVABLE, not only pinnable.

MEASURED DEFECT (2026-09-10). `MINISGL_TOOL_FORMAT=zaya_xml` was set by exactly one thing: the
serve.sh arm matching `*/ZAYA1-8B-fp8`. The MXFP4 conversion of the same checkpoint took the
catch-all instead, so the running serve had no format pinned and fell back to `_derive_tool_format`,
which knew Gemma-4's and Muse-Glimmer's wrappers and not ZAYA's — returning None, i.e. the JSON
default. A forced tool call was then grammar-constrained to a JSON shape this checkpoint was never
trained to emit, while its template plainly renders `<zyphra_tool_call>`.

    docker exec: python -m pytest tests/zaya_tool_format_test.py -q -o addopts=""
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

import minisgl.server.api_server as api  # noqa: E402

ZAYA_CKPT = "/models/ZAYA1-8B-MXFP4"


class _StubTok:
    def __init__(self, rendered: str):
        self._rendered = rendered

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return self._rendered


@pytest.fixture(autouse=True)
def _reset_derivation_cache():
    """`_resolve_tool_format` memoises; each case must derive afresh."""
    api._DERIVED_TOOL_FORMAT = None
    api._DERIVED_TOOL_FORMAT_SET = False
    yield
    api._DERIVED_TOOL_FORMAT = None
    api._DERIVED_TOOL_FORMAT_SET = False


def _derive_with(monkeypatch, rendered: str):
    monkeypatch.setattr(api, "_frontend_tokenizer", lambda: _StubTok(rendered))
    return api._derive_tool_format()


def test_zyphra_wrapper_derives_zaya_xml(monkeypatch):
    rendered = "<|im_start|>assistant\n<zyphra_tool_call>\n<function=f>\n</function>\n</zyphra_tool_call>\n"
    assert _derive_with(monkeypatch, rendered) == "zaya_xml"


def test_other_families_are_untouched(monkeypatch):
    assert _derive_with(monkeypatch, "<|tool_call>call:f{}") == "gemma_native"
    assert _derive_with(monkeypatch, '<atem:invoke name="f">') == "atem"


def test_an_unrecognised_template_still_keeps_the_json_default(monkeypatch):
    assert _derive_with(monkeypatch, '{"name": "f", "arguments": {}}') is None


def test_the_grammar_targets_the_same_wrapper_the_probe_looks_for():
    """Detection and constraint must agree — a probe that fires on a shape the grammar cannot
    produce would pin a format that then fails to constrain anything."""
    ebnf = api._wrapped_xml_grammar(
        [{"function": {"name": "get_weather", "parameters": {"properties": {"city": {}}}}}],
        None, "<zyphra_tool_call>", "</zyphra_tool_call>")
    assert ebnf is not None
    assert "<zyphra_tool_call>" in ebnf


@pytest.mark.skipif(not os.path.isdir(ZAYA_CKPT),
                    reason=f"{ZAYA_CKPT} not mounted — real-template half NOT covered")
def test_the_real_checkpoint_template_renders_that_wrapper():
    """The stub above is only worth something if the real template actually emits this marker.

    Renders the checkpoint's own jinja on `_TOOL_PROBE_MESSAGES` — the same probe the server uses.
    """
    jinja2 = pytest.importorskip("jinja2")
    src = open(os.path.join(ZAYA_CKPT, "chat_template.jinja")).read()
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    rendered = env.from_string(src).render(
        messages=api._TOOL_PROBE_MESSAGES, add_generation_prompt=False)
    assert "<zyphra_tool_call>" in rendered, rendered[-400:]
