from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    # Recurrent-radix prefix cache for GDN/CCA hybrid models: snapshot the linear-attention recurrent
    # state at page-aligned prefix boundaries so prefix hits are reusable (lossless). ON by default;
    # it forces the synchronous (non-overlap) scheduler loop, so `--no-gdn-radix` opts back out. Inert
    # for non-recurrent models and auto-disabled under spec-decode / expert-parallelism (untested combo).
    gdn_radix: bool = True

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        # Per-replica ingress: each DP replica's rank-0 binds its OWN backend addr so the front-end
        # can deliver each UserMsg to exactly ONE replica (per-replica routing). dp_size=1 keeps the
        # historical single address (dp_rank=0 -> ".dp=0" suffix is appended unconditionally; the
        # tokenizer/scheduler always agree on the same formula so the channel still matches).
        return f"ipc:///tmp/minisgl_0{self._unique_suffix}.dp={self.dp_info.dp_rank}"

    @property
    def zmq_detokenizer_addr(self) -> str:
        # Replies are shared across replicas — ONE detokenizer demuxes by globally-unique uid — so
        # this stays a single address (no dp_rank key).
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        # Within-replica TP fan-out (rank0 -> rank1..). Keyed by dp_rank so replicas don't cross-wire.
        return f"ipc:///tmp/minisgl_2{self._unique_suffix}.dp={self.dp_info.dp_rank}"

    def backend_addr_for(self, dp_rank: int) -> str:
        """The backend ingress addr for replica ``dp_rank`` (front-end routing target)."""
        return f"ipc:///tmp/minisgl_0{self._unique_suffix}.dp={dp_rank}"

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
