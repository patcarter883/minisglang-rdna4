"""Gate for the native fused top-k/top-p sampler (canonical ``sampler_hip`` kernel).

Replaces the two full-vocab ``torch.sort`` passes in ``sample_impl`` (top-k mask + top-p nucleus)
with ONE flashinfer-style rejection-sampling HIP kernel that fuses temperature + softmax + top-k +
top-p + multinomial over fp32 logits.

Default-ON (``MINISGL_FUSED_SAMPLER=1``) but a SOFT fallback: unlike ``layers/_tail_hip`` (which
hard-requires its .so because the user opted default-on), the pure-torch sampler in ``sample.py`` is
an always-correct reference, so a missing/unbuilt ``sampler_hip`` must NOT break serving — it just
routes back to torch. The kernel and the torch path are validated equivalent by
``rdna4-hip-kernels/sampler/tests/sampler_parity.py`` under a GPU lease before this default is
trusted; ``MINISGL_FUSED_SAMPLER=0`` forces the torch reference (A/B / debugging).

Greedy sampling never reaches here — ``Sampler.sample`` branches to ``torch.argmax`` first — and the
grammar bitmask is applied to logits (disallowed -> -inf) before ``sample_impl``, so it composes with
this op unchanged (-inf logit -> prob 0 -> never sampled), exactly like the torch fallback.
"""
from __future__ import annotations

import os

import torch

ENABLED = os.environ.get("MINISGL_FUSED_SAMPLER", "1") != "0"

_OP = None
MAX_ROUNDS = 32
if ENABLED:
    try:
        import sampler_hip

        _OP = sampler_hip.top_k_top_p_sampling_from_logits_
        MAX_ROUNDS = int(sampler_hip.MAX_ROUNDS)
    except Exception:
        _OP = None  # unbuilt / absent -> torch fallback in sample_impl


def available() -> bool:
    return _OP is not None


def sample(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | None,
    top_p: torch.Tensor | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Fused sampler. logits fp32 [bs, vocab], temperatures fp32 [bs], top_k int32 [bs] | None
    (None / k==vocab disabled), top_p fp32 [bs] | None (None / p==1.0 disabled). Returns int32 [bs].

    Temperature is applied INSIDE the kernel (with the same clamp_min(1e-6) as the torch reference),
    so pass pre-temperature logits + the temperatures tensor — do not pre-divide. The caller owns the
    per-round uniform deviates; identical uniforms in => identical tokens out (capturable), but
    sampling runs eager here (it is outside the decode attention graph)."""
    from minisgl._hip_engage import engaged

    bs = logits.shape[0]
    u = torch.rand(MAX_ROUNDS, bs, device=logits.device, dtype=torch.float32, generator=generator)
    out = torch.empty(bs, device=logits.device, dtype=torch.int32)
    engaged("sampler_hip.top_k_top_p_sampling_from_logits_")
    _OP(logits.contiguous(), temperatures, top_k, top_p, u, out)
    return out
