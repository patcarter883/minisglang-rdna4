"""Every reasoning-delimiter kwarg api_server passes must survive to the rollout.

THE DEFECT THIS PINS. `api_server` calls `run_markovian_rsa(..., think_tool_release=...,
think_answer_delim=..., think_close_prefix=..., think_close_suffix=...)`. None of the four were in
the signature, so EVERY rsa request 500'd:

    TypeError: run_markovian_rsa() got an unexpected keyword argument 'think_tool_release'

RSA was unreachable, not degraded — found 2026-09-09 on the first RSA call ever made against ZAYA.
A signature check is the right gate because the failure is a pure interface drift: three layers
(api_server -> run_markovian_rsa -> BackendClient.chat) each grew the reasoning set at a different
time, and nothing forced them to agree.

    python3 -m pytest tests/rsa_think_kwargs_test.py -q -o addopts=""
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

REASONING_KWARGS = {
    "think_close_delim", "think_budget", "think_tool_release",
    "think_answer_delim", "think_close_prefix", "think_close_suffix",
}
ROOT = pathlib.Path(__file__).resolve().parents[1] / "python" / "minisgl"


def _kwonly(path: pathlib.Path, func: str, cls: str | None = None) -> set[str]:
    tree = ast.parse(path.read_text())
    nodes = ast.walk(tree)
    if cls:
        klass = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
        nodes = ast.walk(klass)
    fn = next(n for n in nodes
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func)
    return {a.arg for a in fn.args.kwonlyargs} | {a.arg for a in fn.args.args}


def test_api_server_passes_only_kwargs_run_markovian_rsa_accepts():
    """The exact 500. Anything api_server sends must be in the signature."""
    api = (ROOT / "server" / "api_server.py").read_text()
    tree = ast.parse(api)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "run_markovian_rsa"]
    assert calls, "no run_markovian_rsa call found — has it been renamed?"
    accepted = _kwonly(ROOT / "rsa" / "core.py", "run_markovian_rsa")
    for c in calls:
        passed = {k.arg for k in c.keywords if k.arg}
        missing = passed - accepted
        assert not missing, f"api_server passes {sorted(missing)} which run_markovian_rsa rejects"


def test_run_markovian_rsa_accepts_the_whole_reasoning_set():
    got = _kwonly(ROOT / "rsa" / "core.py", "run_markovian_rsa")
    assert REASONING_KWARGS <= got, f"missing {sorted(REASONING_KWARGS - got)}"


@pytest.mark.parametrize("mod,cls", [
    ("inproc.py", "InProcessBackendClient"),   # the in-engine path, primary for a minisgl serve
    ("core.py", "BackendClient"),              # the HTTP shim
])
def test_the_backend_clients_accept_the_whole_reasoning_set(mod, cls):
    """Forwarding from run_markovian_rsa only moves the TypeError unless the CLIENT accepts them
    too — and the in-process client is the primary path for an in-engine serve."""
    got = _kwonly(ROOT / "rsa" / mod, "complete", cls)
    assert REASONING_KWARGS <= got, f"{mod}:chat missing {sorted(REASONING_KWARGS - got)}"


def test_the_inproc_client_actually_forwards_them_to_SamplingParams():
    """Accepting a kwarg and dropping it is the quieter version of the same bug: the rollout would
    close its think span by a different rule than the /v1/chat/completions lane."""
    src = (ROOT / "rsa" / "inproc.py").read_text()
    for k in REASONING_KWARGS:
        assert f"{k}=" in src, f"{k} accepted but never forwarded into SamplingParams"
