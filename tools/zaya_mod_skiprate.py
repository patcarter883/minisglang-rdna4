"""Measure ZAYA1 MoD 'skip' expert routing rate at DECODE (go/no-go for the skip optimization).

Loads ZAYA1-8B-fp8 offline (single card, eager) and monkeypatches ZayaMoEBlock.forward to count,
per routing decision, how often top-1 lands on the MOD skip slot, split by prefill vs decode phase.
Reports overall + per-layer decode skip rate.

The skip signal is the router's `keep` mask (0 == the raw top-1 was the skip slot), NOT the expert
id: the fused route clamps the id to a real expert inside the kernel, so `expert_ids == num_experts`
is unreachable by construction.

Run INSIDE the lean image via gpu-lease:
    PYTHONPATH=/opt/kernels:/engine/python:/engine python /engine/tools/zaya_mod_skiprate.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import torch
from minisgl.core import SamplingParams, get_global_ctx
from minisgl.llm import LLM
from minisgl.models import zaya as zaya_mod

MODEL = "/models/ZAYA1-8B-fp8"

# stats[(phase, layer_id)] = [n_skip, n_total]
STATS: dict = defaultdict(lambda: [0, 0])
_LAYER_COUNTER = {"n": 0}


def _instrument() -> None:
    Block = zaya_mod.ZayaMoEBlock
    orig = Block.forward

    def wrapped(self, hidden_states, prev_router_states):
        # assign a stable per-instance layer id lazily
        lid = getattr(self, "_skiprate_lid", None)
        if lid is None:
            lid = _LAYER_COUNTER["n"]
            _LAYER_COUNTER["n"] += 1
            self._skiprate_lid = lid
        # Counterfactual: optionally override the skip-slot balancing bias to see what skip rate MoD
        # WOULD produce if the checkpoint's -1.0 (skip-disabling) bias were relaxed. Off by default.
        _ov = os.environ.get("MOD_SKIP_BIAS_OVERRIDE")
        if _ov is not None:
            with torch.no_grad():
                self.router.balancing_biases[self._num_experts] = float(_ov)
        # ROUTER CONTRACT (fused route, models/zaya.py:299): FOUR values, and the ids are ALREADY
        # CLAMPED to [0, ne-1] by the kernel epilogue. The raw winner `ne` never reaches python any
        # more, so the old `expert_idx == ne` skip test can no longer fire — the skip signal is
        # `keep == 0`, which the router emits (in the model dtype) exactly when the RAW winner was
        # the MOD slot. This tool read the pre-fusion 3-tuple and that dead metric; both are fixed.
        route_prob, expert_ids, keep, rsn = self.router.forward(hidden_states, prev_router_states)
        try:
            phase = "decode" if get_global_ctx().batch.is_decode else "prefill"
        except Exception:
            phase = "unknown"
        assert keep is not None, (
            "skip-rate is a MoD metric and the router only emits `keep` with MOD on "
            "(config.zaya_use_mod); this checkpoint has it off, so there is no skip slot to count."
        )
        n_skip = int((keep == 0).sum().item())
        n_total = int(keep.numel())
        s = STATS[(phase, lid)]
        s[0] += n_skip
        s[1] += n_total
        # Replicate the real forward from the route we already have (avoid a second router call).
        # This is ZayaMoEBlock.forward verbatim, minus the router call.
        experts_out = self.experts.forward(
            hidden_states, topk_weights=route_prob, topk_ids=expert_ids
        )
        prob = route_prob.to(hidden_states.dtype)
        mod_out = hidden_states * prob
        out = keep * experts_out + (1.0 - keep) * mod_out
        return out, rsn

    Block.forward = wrapped
    print("[skiprate] instrumented ZayaMoEBlock.forward", flush=True)


def _report() -> None:
    phases = defaultdict(lambda: [0, 0])
    per_layer_decode = {}
    for (phase, lid), (nsk, ntot) in sorted(STATS.items()):
        phases[phase][0] += nsk
        phases[phase][1] += ntot
        if phase == "decode":
            per_layer_decode[lid] = (nsk, ntot)
    print("\n==================== MoD skip-rate ====================", flush=True)
    for phase, (nsk, ntot) in phases.items():
        rate = (nsk / ntot * 100.0) if ntot else 0.0
        print(f"[{phase:7s}] skip {nsk}/{ntot} = {rate:.1f}%", flush=True)
    if per_layer_decode:
        print("\n-- per-MoE-layer DECODE skip rate --", flush=True)
        for lid in sorted(per_layer_decode):
            nsk, ntot = per_layer_decode[lid]
            rate = (nsk / ntot * 100.0) if ntot else 0.0
            print(f"  layer#{lid:2d}: {nsk:5d}/{ntot:5d} = {rate:5.1f}%", flush=True)
    print("======================================================\n", flush=True)


def main() -> int:
    print(f"[skiprate] torch={torch.__version__} hip={torch.cuda.is_available()}", flush=True)
    _instrument()
    llm = LLM(
        model_path=MODEL,
        dtype=torch.bfloat16,
        attention_backend="hip",
        cuda_graph_max_bs=0,  # eager; skip rate is a routing property, graph-independent
        memory_ratio=0.80,
        max_running_req=4,
    )
    print("[skiprate] engine built; generating", flush=True)
    prompts = [
        "Q: What is the capital of France?\nA:",
        "Q: Write a short story about a robot who learns to paint.\nA:",
        "Q: Explain how a transformer neural network works, step by step.\nA:",
        "Q: List five prime numbers and explain what makes a number prime.\nA:",
        "def fibonacci(n):\n    # returns the nth fibonacci number\n",
        "The three most important discoveries in physics are",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=128)
    outs = llm.generate(prompts, sp)
    for p, o in zip(prompts, outs):
        print(f"\n[PROMPT] {p[:40]!r}...\n[OUT] {o['text'][:120]!r}", flush=True)
    _report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
