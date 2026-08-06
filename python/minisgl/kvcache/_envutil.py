"""Env parsing that matches this repo's compose convention. Deliberately dependency-free.

Every knob in docker-compose.yml is declared as `FOO: "${FOO:-}"`, so an unset variable arrives in
the container as a variable that IS SET, to the EMPTY STRING. That makes the obvious spelling

    int(os.environ.get("FOO", "200"))

wrong: the default is only used when the key is ABSENT, which under compose it never is, so the
call becomes `int("")` and raises at import time — a boot crash, not a fallback. Use `env_int`.
"""

from __future__ import annotations

import os


def env_int(name: str, default: int) -> int:
    """Integer env knob where empty-or-unset means `default`. Never raises on junk input."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    """Float env knob where empty-or-unset means `default`. Never raises on junk input."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default
