"""`preserve_thinking` defaults OFF on templates that read it — derived, not model-name branched.

WHY. The template's own default carries every prior turn's raw reasoning forward. On
Qwen3.8-Flash-Next a trace that has begun to spiral is then re-read on every later turn, and the
spiral is what gets reinforced. MEASURED 2026-09-21 replaying an already-degenerated conversation,
3 runs per arm: baseline 1/3 produced a tool call (2/3 hit the token cap), min_p=0.05/temp 0.6 1/3,
preserve_thinking=False 3/3 (0/3 cap-outs).

SCOPE IS THE TEMPLATE, NOT A NAME. Qwen3.8-Flash-Next and Qwen3.8-27B read the kwarg; Qwen3.6 (35B
and 27B) and GLM-4.7-Flash ignore it and cannot be affected.

    docker exec: python -m pytest tests/preserve_thinking_default_test.py -q -o addopts=""
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

import minisgl.server.api_server as api  # noqa: E402

MP = "/fake/model"


class _Tok:
    """Renders differently under preserve_thinking=False, like the Qwen3.8 templates."""
    def __init__(self, reads: bool): self.reads = reads
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        if self.reads and kw.get("preserve_thinking") is False:
            return "RENDER-WITHOUT-PRIOR-THINKING"
        return "RENDER-WITH-PRIOR-THINKING"


class _Req:
    def __getattr__(self, n): return None


@pytest.fixture(autouse=True)
def _clean():
    api._template_reads_preserve_thinking.cache_clear()
    os.environ.pop("MINISGL_PRESERVE_THINKING", None)
    yield
    api._template_reads_preserve_thinking.cache_clear()
    os.environ.pop("MINISGL_PRESERVE_THINKING", None)


def _tok(monkeypatch, reads):
    monkeypatch.setattr(api, "_frontend_tokenizer", lambda: _Tok(reads))
    monkeypatch.setattr(api, "_template_level_kwarg", lambda p: None)
    monkeypatch.setattr(api, "_server_default_template_kwargs", lambda: {})


def test_a_template_that_reads_it_gets_preserve_thinking_False(monkeypatch):
    _tok(monkeypatch, reads=True)
    assert api._resolve_chat_template_kwargs(_Req(), MP)["preserve_thinking"] is False


def test_a_template_that_IGNORES_it_is_left_alone(monkeypatch):
    """Qwen3.6 / GLM must not acquire a kwarg their template never reads."""
    _tok(monkeypatch, reads=False)
    kw = api._resolve_chat_template_kwargs(_Req(), MP) or {}
    assert "preserve_thinking" not in kw


def test_an_explicit_client_value_wins(monkeypatch):
    """A caller naming the model's own kwarg is more specific than our default."""
    _tok(monkeypatch, reads=True)
    r = _Req(); r.chat_template_kwargs = {"preserve_thinking": True}
    assert api._resolve_chat_template_kwargs(r, MP)["preserve_thinking"] is True


def test_the_env_escape_hatch_restores_the_template_default(monkeypatch):
    _tok(monkeypatch, reads=True)
    os.environ["MINISGL_PRESERVE_THINKING"] = "1"
    kw = api._resolve_chat_template_kwargs(_Req(), MP) or {}
    assert "preserve_thinking" not in kw


def test_no_model_path_means_no_probe_and_no_default(monkeypatch):
    _tok(monkeypatch, reads=True)
    kw = api._resolve_chat_template_kwargs(_Req(), None) or {}
    assert "preserve_thinking" not in kw
