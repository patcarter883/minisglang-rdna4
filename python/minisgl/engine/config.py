from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, List, Literal, NamedTuple

import torch
from minisgl.distributed import DistributedInfo, DpInfo
from minisgl.utils import cached_load_hf_config, is_rocm

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


# ======================= prefix-cache resolution (single source of truth) =======================
# The snapshot-capable radix ("recurrent_radix") is the ONLY consumer of the snapshot store, and that
# store is a real VRAM reservation taken out of the KV pool at boot (engine._rec_snapshot_store_bytes).
# So exactly two things have to agree: whether the SCHEDULER builds that cache manager, and whether the
# ENGINE reserves for it. They used to be decided independently, and they DIVERGED. The engine keyed
# off `gdn_radix` (default True) plus "per-slot state bytes > 0" — and that per-slot term counts the
# SWA ring pool as well as GDN/CCA recurrent state. So:
#   * a SWA hybrid looked reservable to the engine no matter what the SWA flag said, while the
#     scheduler gated SWA separately and, when it said no, forced 'naive' and stored nothing.
#     MEASURED on gemma-4-26B-A4B TP=2 (16 GB cards): 0.98 GiB — ~100k KV tokens — reserved for a
#     store that could not exist; and
#   * a dense/MHA/MLA model was saved only by per_slot == 0, i.e. by accident rather than by anything
#     that says "this model has no snapshot store".
# `gdn_radix` also has no bearing on the SWA path, and `cache_type == "naive"` (which the scheduler
# honours) was invisible to the engine entirely. Resolve it ONCE, here, and let both sides read the
# answer instead of the engine guessing what the scheduler is about to do.

SnapshotKind = Literal["", "recurrent", "swa"]


class PrefixCachePlan(NamedTuple):
    """What prefix cache a given (model, flags) combination will ACTUALLY run.

    cache_type     — the CacheManager key the scheduler must build ("radix" / "naive" /
                     "recurrent_radix"), with every forced downgrade already applied.
    snapshot_kind  — what will populate the recurrent-radix snapshot store, hence what the engine must
                     reserve VRAM for:
                       ""           nothing — reserve ZERO (dense/MHA/MLA, and any hybrid whose
                                    snapshot path is switched off)
                       "recurrent"  GDN/CCA per-slot conv+ssm / conv+prev_hs clones
                       "swa"        sliding-window ring clones (a SWA hybrid keeps NO recurrent state;
                                    its snapshots are window KV, sized by the ring)
    reason         — human-readable why, for the boot log.
    """

    cache_type: str
    snapshot_kind: SnapshotKind
    reason: str


def swa_radix_enabled() -> bool:
    """SWA-radix prefix caching: now DEFAULT ON for sliding-window hybrids. MINISGL_SWA_RADIX=0 disables.

    It shipped flag-gated OFF (2e5d6d5e) as merge-time conservatism, not because anything was wrong
    with it: the window snapshot/restore was proven byte-identical to a cold prefill (per-layer hidden
    hash over every layer and every decode step, generated token-id sequence, output sha) and cut TTFT
    on a reused long prefix ~11x (2.20 s -> 0.20 s). Default-off meant the feature was, in practice,
    never on — while the KV pool paid for it anyway, because the engine's reservation did not read this
    flag. Its one recorded cost is that, like recurrent radix, it forces the synchronous normal_loop
    (no overlap scheduling), so it trades a little decode throughput for a large TTFT win on
    prefix-sharing traffic. Re-validated on Gemma4, whose sliding and full layers do NOT share a
    head_dim (256/8 sliding vs 512/2 full) and whose ring geometry changed under this feature — see
    the report in this branch.

    Read in ONE place so the engine's reservation and the scheduler's cache-type choice cannot read it
    differently — or, as before, one of them not at all."""
    return os.environ.get("MINISGL_SWA_RADIX", "1") != "0"


def resolve_prefix_cache(config: "EngineConfig") -> PrefixCachePlan:
    """Decide the prefix cache and the snapshot store ONCE, from config alone.

    Config-only on purpose: the engine has to answer this while it is SIZING the KV pool, i.e. before
    gdn_state / cca_state / swa_kv_cache exist, whereas the scheduler answers it afterwards. Keying
    both off the same `ModelConfig` predicates — the very ones that decide whether those caches get
    built — is what turns "will the store be used?" into a single fact instead of two guesses.

    `cache_type` and `gdn_radix` live on SchedulerConfig, not EngineConfig, and the object handed to
    the engine may be either — so read them defensively, with the SchedulerConfig defaults.
    """
    mc = config.model_config
    cache_type = getattr(config, "cache_type", "radix")

    # GDN (Qwen3.5/3.6) and CCA (ZAYA) recurrent state is not prefix-cacheable UNLESS it is
    # snapshotted: a plain radix hit would report cached_len>0 with no state behind it (silent
    # garbage). --gdn-radix (default on) opts into the snapshot-capable radix; --no-gdn-radix forces
    # naive, and then there is nothing to reserve for.
    if mc.is_gdn_hybrid or mc.is_cca_hybrid:
        if cache_type == "naive":
            return PrefixCachePlan("naive", "", "prefix cache explicitly 'naive'")
        if getattr(config, "gdn_radix", True):
            return PrefixCachePlan(
                "recurrent_radix", "recurrent", "recurrent-state hybrid, --gdn-radix on"
            )
        return PrefixCachePlan(
            "naive", "", "GDN/CCA recurrent state is not prefix-cacheable (--no-gdn-radix set)"
        )

    # SWA hybrids (Laguna, Gemma4) keep NO recurrent state; their sliding layers keep a window ring
    # whose boundary is transient, so prefix reuse needs the same snapshot treatment. Default ON now
    # (see swa_radix_enabled) — so for these models the snapshot-store reservation is LEGITIMATE: it
    # is a store that will actually be filled. MINISGL_SWA_RADIX=0 turns both the store and the
    # reservation off together, which is the whole point of resolving it here.
    if mc.is_swa_hybrid:
        if cache_type == "naive":
            return PrefixCachePlan("naive", "", "prefix cache explicitly 'naive'")
        if swa_radix_enabled():
            return PrefixCachePlan("recurrent_radix", "swa", "SWA hybrid, MINISGL_SWA_RADIX=1")
        return PrefixCachePlan("naive", "", "SWA-radix disabled (MINISGL_SWA_RADIX=0)")

    # Dense / MHA / MLA: whatever was asked for, and no snapshot store at all.
    return PrefixCachePlan(cache_type, "", "no recurrent state and no sliding window")


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    # Data-parallel coordinates for this engine replica. Defaults to the inert single-replica DP
    # (dp_rank=0, dp_size=1) so every existing programmatic EngineConfig build is unchanged. The
    # server launcher overrides it per spawned replica. `enable_ep` is reserved for the expert-parallel
    # toggle (shards MoE experts across dp ranks); it stays False / inert in the DP-launcher-only path.
    dp_info: DpInfo = field(default_factory=lambda: DpInfo(0, 1))
    enable_ep: bool = False
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    # PyNCCL is a CUDA-only collective: pynccl.cu pulls in NVIDIA NCCL (nccl227.h / -lnccl, not RCCL)
    # and loads via apache-tvm-ffi. Neither is present on ROCm, so default it OFF there — tp>1 then
    # falls back to torch.distributed backend="nccl" (→ RCCL on ROCm) in engine._init_communication.
    # default_factory (not a plain `= not is_rocm()`) so the arch is probed at construct time, not
    # import time. The CLI path sets its own ROCm-aware default in server/args.py (argparse always
    # supplies use_pynccl, so this dataclass default only governs programmatic EngineConfig builds).
    use_pynccl: bool = field(default_factory=lambda: not is_rocm())
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # --- speculative decoding (off by default; see SPEC_DECODE.md) ------------------------------
    # "none" disables every spec path (byte-for-byte unchanged serve). "ngram" enables the
    # prompt-lookup MVP. These flat fields mirror the argparse dests; spec_config assembles them.
    spec_algorithm: str = "none"
    spec_num_draft: int = 4
    spec_ngram_max: int = 3
    spec_ngram_min: int = 1
    spec_draft_model_path: str | None = None  # EAGLE3/DFlash: separate draft checkpoint path

    @cached_property
    def spec_config(self):
        from minisgl.spec import SpecConfig

        if self.spec_algorithm == "none":
            return None
        return SpecConfig(
            algorithm=self.spec_algorithm,
            num_draft=self.spec_num_draft,
            ngram_max=self.spec_ngram_max,
            ngram_min=self.spec_ngram_min,
            draft_model_path=self.spec_draft_model_path,
        )

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig
        from minisgl.models.weight import checkpoint_tensor_names

        # Pass the checkpoint's actual tensor names so an MTP head is built from the WEIGHTS, not
        # from a config field that may claim one the checkpoint does not ship (see from_hf).
        # Header-only read; no tensor data. Skipped for dummy weights — there is no checkpoint.
        names = None if self.use_dummy_weight else checkpoint_tensor_names(self.model_path)
        return ModelConfig.from_hf(
            self.hf_config, spec_algorithm=self.spec_algorithm, ckpt_tensor_names=names
        )

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def device_index(self) -> int:
        """Physical card slot for this replica's TP rank, within the lease-visible device set.

        The gpu-lease/HIP_VISIBLE_DEVICES exposes the leased cards as cuda:0..N-1; this picks the
        slot for (dp_rank, tp_rank). With dp_size=1 this collapses to tp_info.rank — the historical
        `cuda:{tp_rank}` mapping — so single-replica runs are unchanged. With dp_size>1 (tp_size=1
        for ZAYA) each replica lands on its own card: dp_rank=0 -> cuda:0, dp_rank=1 -> cuda:1.
        """
        return self.dp_info.dp_rank * self.tp_info.size + self.tp_info.rank

    @property
    def distributed_addr(self) -> str:
        # Each DP replica is an INDEPENDENT TP process group (with EP off there is no cross-replica
        # collective), so give each replica its own rendezvous port to avoid init_method collisions
        # when several replicas come up on one host. dp_size=1 keeps the historical 2333.
        return f"tcp://127.0.0.1:{2333 + self.dp_info.dp_rank}"


def snapshot_ladder_depth(config) -> int:
    """Interior-resume snapshots a sequence holds beyond its end boundary, per snapshot kind.

    Lives here, next to `resolve_prefix_cache`, because the ENGINE needs it to size the reservation
    and the SCHEDULER needs it to trim the live ladder — and when those two disagreed about how many
    snapshots exist, the store silently held more than the KV pool was sized around.

    * RECURRENT (GDN/CCA): a sequence genuinely holds `ladder` interior resume points plus its end
      boundary; 4 is the measured working set and shrinking it makes the ladder thrash against itself.
    * SWA: there is NO interior ladder — a window snapshot is taken at the page-aligned prefix
      boundary and nowhere else — so the live set is one per concurrent sequence and the default is 0.
      Depth above that buys only CROSS-REQUEST reuse, which is priced very differently here: a Gemma4
      window snapshot is 100 MiB against 16.4 MiB for the 35B's recurrent state, so inheriting the
      recurrent ladder of 4 cost 1.95 GiB = 204,800 KV-pool tokens, 59% of Gemma4's entire pool, to
      buy depth the recurrent A/B measured as worth nothing on this box (cap 12 and cap 23 gave the
      same hits and the same TTFT).

    MINISGL_GDN_RADIX_SNAP_LADDER overrides either kind, so a prefix-sharing-heavy deployment buys the
    depth back explicitly and pays the KV tokens knowingly."""
    env = os.environ.get("MINISGL_GDN_RADIX_SNAP_LADDER")
    if env:
        return max(0, int(env))
    return 0 if resolve_prefix_cache(config).snapshot_kind == "swa" else 4
