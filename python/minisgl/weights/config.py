"""Resolved settings for the pinned weight arena.

**There is deliberately NO on/off env knob here.** Plan §6.2 and the repo's standing rule forbid
env-gating a feature at merge, and a control leg produced by flipping a knob in the same binary is
exactly the emulated baseline that is not permitted. Whether the arena exists at all is a *derived*
decision (offloadable bytes vs measured VRAM), owned by the plan resolver next to
`resolve_prefix_cache` — not by an environment variable.

What IS here is tuning that a measurement fixture needs to sweep: chunk size, floor, whether the
expensive first-touch and self-test run. Every one of them has a validated default, and the
validated operating point belongs in `tools/serve.sh`'s per-model table, not only in this file.

Env is read through `kvcache/_envutil.env_int` / `env_float` — imported lazily so that the planning,
capacity and layout modules stay importable (and unit-testable) on a machine with no torch and no
ROCm. That indirection also means there is exactly ONE implementation of the compose empty-string
rule: under compose every knob arrives SET-BUT-EMPTY, so `int(os.environ.get("FOO", "200"))` becomes
`int("")` and crashes at import.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunk_plan import ALIGN, CHUNK_GRANULE, DEFAULT_CHUNK_BYTES, GIB, MIB, round_up
from .host_capacity import DEFAULT_FLOOR_BYTES


def _env_int(name: str, default: int) -> int:
    from minisgl.kvcache._envutil import env_int

    return env_int(name, default)


def _env_float(name: str, default: float) -> float:
    from minisgl.kvcache._envutil import env_float

    return env_float(name, default)


@dataclass(frozen=True)
class ArenaSettings:
    chunk_bytes: int = DEFAULT_CHUNK_BYTES
    align: int = ALIGN
    floor_bytes: int = DEFAULT_FLOOR_BYTES
    selftest: bool = True
    first_touch: bool = True

    def describe(self) -> str:
        return (
            f"chunk={self.chunk_bytes // MIB} MiB align={self.align} "
            f"floor={self.floor_bytes / GIB:.1f} GiB selftest={self.selftest} "
            f"first_touch={self.first_touch}"
        )


def resolve_arena_settings() -> ArenaSettings:
    """The one place env is read. Defaults are the measured ones:

    * chunk 2048 MiB — P3b's demonstrated size (34.0 GiB/rank as 17 x 2 GiB, 4.88 GB/s median).
      Nothing larger has ever been pinned on this box.
    * floor 12 GiB — the floor P3b ran under, so its 62 GiB two-rank ceiling is a figure measured
      under this exact policy.
    * self-test ON — Phase 0 produced four independent cases of the driver returning success over
      wrong state; an out-of-band data check is the only defence (plan §5.4 A1.4). Turning it off is
      a measurement convenience, never a production setting.
    """
    chunk_mib = _env_int("MINISGL_WEIGHT_ARENA_CHUNK_MIB", DEFAULT_CHUNK_BYTES // MIB)
    chunk_bytes = max(CHUNK_GRANULE, round_up(chunk_mib * MIB, CHUNK_GRANULE))
    floor_gib = _env_float("MINISGL_WEIGHT_ARENA_FLOOR_GIB", DEFAULT_FLOOR_BYTES / GIB)
    return ArenaSettings(
        chunk_bytes=chunk_bytes,
        align=ALIGN,
        floor_bytes=max(0, int(floor_gib * GIB)),
        selftest=_env_int("MINISGL_WEIGHT_ARENA_SELFTEST", 1) != 0,
        first_touch=_env_int("MINISGL_WEIGHT_ARENA_FIRST_TOUCH", 1) != 0,
    )


def create_pinned_weight_arena(
    device_index: int,
    *,
    rank: int = 0,
    local_ranks: int = 1,
    label: str = "weights",
    settings: ArenaSettings | None = None,
):
    """Engine-facing factory: resolved settings + an unattached arena. Nothing is pinned yet."""
    from .pinned_arena import PinnedWeightArena

    s = settings or resolve_arena_settings()
    return PinnedWeightArena(
        device_index,
        rank=rank,
        local_ranks=local_ranks,
        chunk_bytes=s.chunk_bytes,
        align=s.align,
        floor_bytes=s.floor_bytes,
        label=label,
        # EVERY field of `s` must land somewhere. `selftest` and `first_touch` used to be resolved
        # here and then dropped on the floor, so the two knobs this module exists to plumb were a
        # silent no-op for any caller that did not separately re-read the settings and hand the
        # booleans to `attach()` itself.
        selftest_default=s.selftest,
        first_touch_default=s.first_touch,
    )
