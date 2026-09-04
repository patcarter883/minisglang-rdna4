"""Weight offload — the host tier (M1-A).

`PinnedWeightArena` is the pinned host arena: `hipHostMalloc(...Mapped)` chunks whose device-visible
addresses come from `hipHostGetDevicePointer`, a forward-only bump allocator over them, a capacity
gate against a `MemAvailable` floor, and an out-of-band data self-test.

Import layering — deliberate, and load-bearing for the tests:

    chunk_plan      pure integers                      no torch, no HIP, no /proc
    host_capacity   integers + /proc/meminfo           no torch, no HIP
    hipmem          ctypes; dlopens only on first use  no torch
    pinned_arena    the arena; torch only inside host_tensor()
    torch_pool      torch MemPool plumbing (P5b)       torch required
    config          env resolution                     imports minisgl.kvcache lazily
    granule         the granule descriptor             torch + minisgl.kvcache AT MODULE SCOPE

Nothing above `torch_pool` imports torch at module scope, so the planning and capacity logic is
unit-testable on a machine where `import torch` fails outright.

`stream_tier` is deliberately NOT re-exported here either, and for the same reason as `granule`: it
imports torch and `moe_interpose` at module scope. Import it by path
(`from minisgl.weights.stream_tier import ExpertStreamTier`); `stage_b` and `bake` already do, from
inside the methods that need it, so the torch-free planning tests stay torch-free.

`granule` is therefore deliberately NOT re-exported here: it imports torch and
`minisgl.kvcache.host_arena` (for `FrameComponent`/`FrameLayout`) at module scope, and
`minisgl.layers.moe` / `minisgl.layers.linear` import it eagerly, so pulling it into this
`__init__` would make `import minisgl.weights` — and with it every torch-free planning test — depend
on torch and on the layers package. Import it by path: `from minisgl.weights.granule import ...`.
"""

from __future__ import annotations

from .chunk_plan import (
    ALIGN,
    DEFAULT_CHUNK_BYTES,
    ArenaExhaustedError,
    ArenaLayoutError,
    BumpAllocator,
    ChunkPlan,
    Placement,
    RegionRequest,
    RegionTooLargeError,
    fmt_bytes,
    headroom_chunks,
    plan_regions,
    suggest_chunk_bytes,
    torch_allocation_bytes,
)
from .config import ArenaSettings, create_pinned_weight_arena, resolve_arena_settings
from .host_capacity import (
    DEFAULT_SWAP_TRIPWIRE_ARM_MULTIPLE,
    DEFAULT_SWAP_TRIPWIRE_FRACTION,
    DEFAULT_SWAP_TRIPWIRE_PAGES,
    CapacityVerdict,
    HostArenaCapacityError,
    HostArenaSwapThrashError,
    SwapTripwire,
    cgroup_available_bytes,
    check_capacity,
    evaluate_capacity,
    mem_available_bytes,
    parse_cgroup_available,
)
from .pinned_arena import (
    ArenaChunk,
    ArenaPhase,
    ArenaRegion,
    ArenaSelfTestError,
    ArenaStateError,
    PinnedWeightArena,
    PlanVerification,
    SelfTestResult,
    chunk_fingerprint,
    decode_fingerprint,
    verify_offsets,
)

__all__ = [
    # layout
    "ALIGN",
    "DEFAULT_CHUNK_BYTES",
    "ArenaExhaustedError",
    "ArenaLayoutError",
    "BumpAllocator",
    "ChunkPlan",
    "Placement",
    "RegionRequest",
    "RegionTooLargeError",
    "fmt_bytes",
    "headroom_chunks",
    "plan_regions",
    "suggest_chunk_bytes",
    "torch_allocation_bytes",
    # capacity
    "DEFAULT_SWAP_TRIPWIRE_ARM_MULTIPLE",
    "DEFAULT_SWAP_TRIPWIRE_FRACTION",
    "DEFAULT_SWAP_TRIPWIRE_PAGES",
    "CapacityVerdict",
    "HostArenaCapacityError",
    "HostArenaSwapThrashError",
    "SwapTripwire",
    "cgroup_available_bytes",
    "check_capacity",
    "evaluate_capacity",
    "mem_available_bytes",
    "parse_cgroup_available",
    # arena
    "ArenaChunk",
    "ArenaPhase",
    "ArenaRegion",
    "ArenaSelfTestError",
    "ArenaStateError",
    "PinnedWeightArena",
    "PlanVerification",
    "SelfTestResult",
    "chunk_fingerprint",
    "decode_fingerprint",
    "verify_offsets",
    # config
    "ArenaSettings",
    "create_pinned_weight_arena",
    "resolve_arena_settings",
]
