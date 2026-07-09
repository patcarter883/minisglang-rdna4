"""Prometheus metrics for the minisgl serving path.

The lean serve image (`minisgl-rdna4:lean`) does NOT ship `prometheus_client`, so this module emits
the Prometheus text exposition format by hand (it is trivial: ``# HELP`` / ``# TYPE`` / ``name{labels}
value``). No new dependency is added to the image.

Two families of metrics, split by which process can observe them cheaply:

  * FRONTEND-observed (this file, in the FastAPI/uvicorn process): throughput (generated / prompt
    tokens), request counts, and the TTFT / TPOT / end-to-end latency histograms. All of these are
    derived from the ``UserReply`` stream the frontend already receives, so they cost only a few int
    adds + a bucket bisect per reply — nothing on the decode hot path.

  * BACKEND-observed (`SchedulerMetrics` in the scheduler process): spec-decode acceptance
    (draft / accepted / emitted / steps) and queue / KV / GDN occupancy. These live in the scheduler
    and are threaded to the frontend by piggybacking a ``StatsMsg`` on the existing scheduler ->
    detokenizer -> frontend ZMQ path (no new socket). The frontend keeps the latest snapshot per DP
    replica and sums across replicas at render time.

Everything runs inside uvicorn's single async event loop (listen() writer + /metrics reader), so no
locking is required.
"""
from __future__ import annotations

import time
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

# ---- histogram bucket boundaries (seconds) --------------------------------------------------------
_TTFT_BUCKETS: Sequence[float] = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0,
)
_TPOT_BUCKETS: Sequence[float] = (
    0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 1.0,
)
_E2E_BUCKETS: Sequence[float] = (
    0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0,
)


class _Histogram:
    """A minimal cumulative-bucket histogram (Prometheus semantics)."""

    __slots__ = ("bounds", "counts", "sum")

    def __init__(self, bounds: Sequence[float]) -> None:
        self.bounds = list(bounds)
        self.counts = [0] * (len(self.bounds) + 1)  # last cell == +Inf overflow
        self.sum = 0.0

    def observe(self, value: float) -> None:
        self.counts[bisect_left(self.bounds, value)] += 1
        self.sum += value

    def render(self, name: str, help_text: str, labels: str = "") -> List[str]:
        # `labels` is the inner label text (e.g. 'model_name="foo"'); merged with le= on buckets.
        pre = f"{labels}," if labels else ""
        base = f"{{{labels}}}" if labels else ""
        out = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        cumulative = 0
        for i, ub in enumerate(self.bounds):
            cumulative += self.counts[i]
            out.append(f'{name}_bucket{{{pre}le="{_fmt(ub)}"}} {cumulative}')
        cumulative += self.counts[-1]
        out.append(f'{name}_bucket{{{pre}le="+Inf"}} {cumulative}')
        out.append(f"{name}_sum{base} {_fmt(self.sum)}")
        out.append(f"{name}_count{base} {cumulative}")
        return out


def _fmt(x: float) -> str:
    """Render a float without a trailing exponent surprise; ints stay integral-looking."""
    if x == int(x):
        return str(int(x))
    return repr(x)


def _escape_label(v: str) -> str:
    """Escape a Prometheus label value (backslash, double-quote, newline)."""
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@dataclass
class _ReqMetric:
    start: float
    first_token: float = -1.0
    last_completion: int = 0
    prompt_counted: bool = False


@dataclass
class BackendSnapshot:
    """Latest per-replica scheduler stats (one per DP replica), threaded in via StatsMsg."""

    dp_rank: int = 0
    spec_draft_tokens: int = 0
    spec_accepted_tokens: int = 0
    spec_emitted_tokens: int = 0
    spec_steps: int = 0
    running_requests: int = 0
    waiting_requests: int = 0
    kv_tokens_total: int = 0
    kv_tokens_used: int = 0
    gdn_slots_total: int = 0
    gdn_slots_used: int = 0


class FrontendMetrics:
    """Owns all frontend-observed counters/histograms plus the latest per-replica backend snapshot."""

    def __init__(self, model_path: str = "") -> None:
        self.model_path = model_path
        # `model_name` label on every series so the Grafana dashboard's $model selector populates and
        # two concurrent serves are distinguishable (mirrors vLLM's model_name label).
        self._labels = f'model_name="{_escape_label(model_path)}"'
        # counters
        self.generation_tokens = 0
        self.prompt_tokens = 0
        self.requests_total = 0
        self.requests_success = 0
        self.requests_aborted = 0
        # histograms
        self.ttft = _Histogram(_TTFT_BUCKETS)
        self.tpot = _Histogram(_TPOT_BUCKETS)
        self.e2e = _Histogram(_E2E_BUCKETS)
        # per-uid in-flight tracking (frontend-side latency/throughput derivation)
        self._req: Dict[int, _ReqMetric] = {}
        # latest backend snapshot per DP replica (dp_rank -> BackendSnapshot)
        self._backend: Dict[int, BackendSnapshot] = {}

    # -- lifecycle hooks (called from FrontendManager) --------------------------------------------
    def on_request_start(self, uid: int) -> None:
        self._req[uid] = _ReqMetric(start=time.perf_counter())
        self.requests_total += 1

    def on_reply(self, uid: int, completion_tokens: int, prompt_tokens: int,
                 has_output: bool, finished: bool) -> None:
        m = self._req.get(uid)
        if m is None:
            return
        now = time.perf_counter()
        if not m.prompt_counted and prompt_tokens:
            self.prompt_tokens += prompt_tokens
            m.prompt_counted = True
        if completion_tokens > m.last_completion:
            self.generation_tokens += completion_tokens - m.last_completion
            m.last_completion = completion_tokens
        if m.first_token < 0 and (has_output or finished):
            m.first_token = now
            self.ttft.observe(now - m.start)
        if finished:
            self.e2e.observe(now - m.start)
            # mean per-output-token latency over this request (generation phase only).
            if m.last_completion > 1 and m.first_token >= 0:
                self.tpot.observe((now - m.first_token) / (m.last_completion - 1))
            self.requests_success += 1
            self._req.pop(uid, None)

    def on_abort(self, uid: int) -> None:
        if self._req.pop(uid, None) is not None:
            self.requests_aborted += 1

    def update_backend(self, snap: BackendSnapshot) -> None:
        self._backend[snap.dp_rank] = snap

    # -- aggregation + rendering ------------------------------------------------------------------
    def _backend_totals(self) -> BackendSnapshot:
        total = BackendSnapshot()
        for s in self._backend.values():
            total.spec_draft_tokens += s.spec_draft_tokens
            total.spec_accepted_tokens += s.spec_accepted_tokens
            total.spec_emitted_tokens += s.spec_emitted_tokens
            total.spec_steps += s.spec_steps
            total.running_requests += s.running_requests
            total.waiting_requests += s.waiting_requests
            total.kv_tokens_total += s.kv_tokens_total
            total.kv_tokens_used += s.kv_tokens_used
            total.gdn_slots_total += s.gdn_slots_total
            total.gdn_slots_used += s.gdn_slots_used
        return total

    def render(self) -> str:
        lines: List[str] = []
        lbl = self._labels
        suffix = f"{{{lbl}}}" if lbl else ""

        def counter(name: str, help_text: str, value: float) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name}{suffix} {_fmt(value)}")

        def gauge(name: str, help_text: str, value: float) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{suffix} {_fmt(value)}")

        b = self._backend_totals()

        # ---- throughput -------------------------------------------------------------------------
        counter("minisgl_generation_tokens_total",
                "Total generated (decode) tokens served.", self.generation_tokens)
        counter("minisgl_prompt_tokens_total",
                "Total prompt (prefill) tokens processed.", self.prompt_tokens)

        # ---- request counts ---------------------------------------------------------------------
        counter("minisgl_requests_total", "Total requests received.", self.requests_total)
        counter("minisgl_requests_success_total",
                "Requests that finished successfully.", self.requests_success)
        counter("minisgl_requests_aborted_total",
                "Requests aborted (client disconnect / stop / abort).", self.requests_aborted)
        gauge("minisgl_requests_inflight",
              "Requests currently in flight (frontend view).", len(self._req))

        # ---- latency histograms -----------------------------------------------------------------
        lines += self.ttft.render("minisgl_ttft_seconds", "Time to first token (seconds).", lbl)
        lines += self.tpot.render(
            "minisgl_tpot_seconds", "Mean per-output-token latency per request (seconds).", lbl)
        lines += self.e2e.render(
            "minisgl_request_latency_seconds", "End-to-end request latency (seconds).", lbl)

        # ---- spec-decode acceptance (backend) ---------------------------------------------------
        counter("minisgl_spec_draft_tokens_total",
                "Speculative draft tokens proposed (summed over DP replicas).",
                b.spec_draft_tokens)
        counter("minisgl_spec_accepted_tokens_total",
                "Speculative draft tokens accepted by verify.", b.spec_accepted_tokens)
        counter("minisgl_spec_emitted_tokens_total",
                "Tokens emitted by spec steps (accepted + bonus).", b.spec_emitted_tokens)
        counter("minisgl_spec_steps_total",
                "Speculative verify steps executed.", b.spec_steps)
        # Derived gauges. mean_accept_len = tokens committed per verify step (accepted + 1 bonus);
        # >= 1 always, == 1 means nothing is being accepted. accept_rate = accepted / proposed.
        if b.spec_steps > 0:
            gauge("minisgl_spec_mean_accept_len",
                  "Mean tokens committed per verify step (accepted drafts + 1 bonus).",
                  1.0 + b.spec_accepted_tokens / b.spec_steps)
        if b.spec_draft_tokens > 0:
            gauge("minisgl_spec_accept_rate",
                  "Fraction of proposed draft tokens accepted.",
                  b.spec_accepted_tokens / b.spec_draft_tokens)

        # ---- queue / occupancy (backend) --------------------------------------------------------
        gauge("minisgl_running_requests",
              "Requests in the running (decode) set.", b.running_requests)
        gauge("minisgl_waiting_requests",
              "Requests waiting in the prefill queue.", b.waiting_requests)
        # KV-pool token capacity/usage. Names match the provisioned minisgl-serving Grafana dashboard.
        gauge("minisgl_kv_pool_total_tokens",
              "Total KV-cache token capacity (pages * page_size).", b.kv_tokens_total)
        gauge("minisgl_kv_pool_used_tokens",
              "KV-cache tokens currently allocated.", b.kv_tokens_used)
        # GDN/CCA recurrent-state slots (only meaningful for hybrid models; 0/0 otherwise).
        gauge("minisgl_gdn_state_total_slots",
              "GDN/CCA recurrent-state slot capacity.", b.gdn_slots_total)
        gauge("minisgl_gdn_state_used_slots",
              "GDN/CCA recurrent-state slots in use.", b.gdn_slots_used)

        return "\n".join(lines) + "\n"


__all__ = ["FrontendMetrics", "BackendSnapshot"]
