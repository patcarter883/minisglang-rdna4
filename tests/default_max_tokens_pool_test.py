"""The uncapped-chat default is DERIVED from the KV pool, and can never shrink below the old one.

WHY (measured, Qwen3.8-Flash-Next, 2026-09-21). A flat 8192 had to be both big enough for a
reasoning model's whole span and small enough that `max_running` of them fit a small serve's pool.
It is not: reasoning alone ran 3.8k / 6.4k / 6.5k / 9.2k tokens across an A/B, and 2 of 3 baseline
runs ended at the cap with finish_reason=length and NO tool call — the agent turn dies there. The
same 8192, on a 69k pool at max_running 6, is 49k of reservation before any prompt is counted.

The floor is the property that makes this safe to land everywhere at once: no arm gets a SMALLER
default than it has today, so an arm nobody re-measured cannot regress.

    docker exec: python -m pytest tests/default_max_tokens_pool_test.py -q -o addopts=""
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

import minisgl.server.api_server as api  # noqa: E402


def _state(max_seq_len, max_running):
    return types.SimpleNamespace(max_seq_len=max_seq_len,
                                 config=types.SimpleNamespace(max_running_req=max_running))


@pytest.fixture(autouse=True)
def _clean_env():
    os.environ.pop("MINISGL_DEFAULT_MAX_TOKENS", None)
    yield
    os.environ.pop("MINISGL_DEFAULT_MAX_TOKENS", None)


def _with(monkeypatch, st):
    monkeypatch.setattr(api, "get_global_state", lambda: st)


def test_no_global_state_falls_back_to_the_old_constant(monkeypatch):
    """Early boot and unit tests legitimately have no pool size — that path must be the old value,
    not an exception and not a guess."""
    def boom():
        raise AssertionError("Global state is not initialized")
    monkeypatch.setattr(api, "get_global_state", boom)
    assert api.default_max_tokens(is_text_completion=False) == 8192


def test_pool_size_not_reported_yet_falls_back(monkeypatch):
    """`max_seq_len` arrives in a scheduler message AFTER boot; until then, the floor."""
    _with(monkeypatch, _state(None, 2))
    assert api.default_max_tokens(is_text_completion=False) == 8192


def test_q4e_shape_derives_a_generous_cap(monkeypatch):
    """The live q4e serve: 195248-token pool, max_running 2 -> 195248//2//4."""
    _with(monkeypatch, _state(195248, 2))
    assert api.default_max_tokens(is_text_completion=False) == 24406


def test_a_small_oversubscribed_pool_never_goes_BELOW_the_old_default(monkeypatch):
    """69k pool at max_running 6 derives 2880 — clamped UP to 8192. THE NO-REGRESSION PROPERTY:
    landing this cannot shrink any arm's default, including arms never re-measured."""
    _with(monkeypatch, _state(69120, 6))
    assert 69120 // 6 // 4 == 2880
    assert api.default_max_tokens(is_text_completion=False) == 8192


def test_a_huge_pool_is_capped_so_one_request_cannot_starve_admission(monkeypatch):
    _with(monkeypatch, _state(1_000_000, 1))
    assert api.default_max_tokens(is_text_completion=False) == 32768


def test_an_explicit_env_value_still_wins(monkeypatch):
    _with(monkeypatch, _state(195248, 2))
    os.environ["MINISGL_DEFAULT_MAX_TOKENS"] = "4096"
    assert api.default_max_tokens(is_text_completion=False) == 4096


def test_the_text_completion_lane_is_untouched(monkeypatch):
    """/v1/completions keeps OpenAI's documented 16 — this change is the chat lane only."""
    _with(monkeypatch, _state(195248, 2))
    assert api.default_max_tokens(is_text_completion=True) == 16
