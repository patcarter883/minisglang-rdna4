from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


# --- Data parallel (DP) coordinates --------------------------------------------------------------
# DP replicates the WHOLE model (attention/CCA backbone + experts) across `dp_size` independent
# engine replicas — used for models that cannot tensor-parallelize (e.g. ZAYA's CCA backbone). Each
# replica owns a distinct (dp_rank) and internally still runs `tp_size` TP ranks. With dp_size=1 the
# whole DP layer is inert: _DP_INFO stays DpInfo(0, 1) and every existing single-replica path is
# byte-for-byte unchanged. EP (expert parallel) later builds its collective group over these dp ranks.
@dataclass(frozen=True)
class DpInfo:
    dp_rank: int
    dp_size: int

    def __post_init__(self):
        assert 0 <= self.dp_rank < self.dp_size

    def is_primary(self) -> bool:
        return self.dp_rank == 0


_DP_INFO: DpInfo | None = None
# Expert-parallel toggle (module global so MoELayer.__init__ — built before the engine wires the EP
# context — knows to SIZE itself to the local expert shard). Set by the Engine alongside set_dp_info;
# stays False for every DP-off / DP-only run (experts replicated, full count loaded).
_ENABLE_EP: bool = False
# EP-OVER-TP mode (vllm-style TP+EP): with dp_size==1 and tp_size>1, shard the experts across the TP
# ranks (each rank holds a FULL subset of experts, attention stays TP-sharded) instead of across DP
# replicas. The EP sharding group is then the TP group. False => the historical DP+EP (experts across
# dp replicas, attention replicated). get_ep_size/get_ep_rank abstract the two so MoELayer/weight
# loader are mode-agnostic.
_EP_OVER_TP: bool = False


def set_dp_info(dp_rank: int, dp_size: int, enable_ep: bool = False,
                ep_over_tp: bool = False) -> None:
    global _DP_INFO, _ENABLE_EP, _EP_OVER_TP
    if _DP_INFO is not None:
        raise RuntimeError("DP info has been set")
    _DP_INFO = DpInfo(dp_rank, dp_size)
    _EP_OVER_TP = bool(enable_ep) and bool(ep_over_tp) and dp_size == 1
    _ENABLE_EP = bool(enable_ep) and (dp_size > 1 or _EP_OVER_TP)


def is_ep_enabled() -> bool:
    return _ENABLE_EP


def is_ep_over_tp() -> bool:
    return _EP_OVER_TP


def get_ep_size() -> int:
    """Expert-sharding degree: TP size under EP-over-TP, else DP size. 1 when EP is off."""
    if not _ENABLE_EP:
        return 1
    return get_tp_info().size if _EP_OVER_TP else get_dp_info().dp_size


def get_ep_rank() -> int:
    """This rank's index within the expert-sharding group (tp_rank under EP-over-TP, else dp_rank)."""
    if not _ENABLE_EP:
        return 0
    return get_tp_info().rank if _EP_OVER_TP else get_dp_info().dp_rank


def get_dp_info() -> DpInfo:
    # Default to the inert single-replica DP when nothing was set (dp_size=1 paths never call
    # set_dp_info, so this keeps them no-op without forcing every caller to special-case None).
    if _DP_INFO is None:
        return DpInfo(0, 1)
    return _DP_INFO


def try_get_dp_info() -> DpInfo | None:
    return _DP_INFO


__all__ = [
    "DistributedInfo",
    "set_tp_info",
    "get_tp_info",
    "try_get_tp_info",
    "DpInfo",
    "set_dp_info",
    "get_dp_info",
    "try_get_dp_info",
    "is_ep_enabled",
    "is_ep_over_tp",
    "get_ep_size",
    "get_ep_rank",
]
