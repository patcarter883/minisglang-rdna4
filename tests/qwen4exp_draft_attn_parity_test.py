"""The MTP draft attention must equal a plain full-sequence causal attention over the same tokens.

WHY THIS EXISTS. `forward_draft_masked` is the one component of the Qwen3.8-Flash-Next MTP head that
was never checked numerically. Its q/k/v/gate/norm/rotary lines were verified by READING them against
`Qwen3_5Attn.forward`; the attention itself is hand-rolled — a GLOBAL ring buffer keyed by slot, a
`write_col` cursor, an additive -inf mask over absolute positions, and a GQA contraction that
deliberately does NOT expand K/V to nq heads. None of that is exercised by the backbone, so none of
it is covered by the serve working.

That matters because the head drafts at p ~= 0.28 per token where Qwen trained and measured it in
exactly this configuration (no PLE table of its own, consuming target hiddens), and other engines
report 0.77-0.89 on this same checkpoint. "Verified by reading" is also what was said about
`pre_fc_norm_hidden` right before measuring it moved acceptance 36%.

THE INVARIANT. Attention has no memory beyond its KV. So stepping the ring ONE TOKEN AT A TIME —
seed a prefix, then call `forward_draft_masked` per token, advancing the cursor exactly as
`MTPProposer.propose_body` does — MUST equal a single dense causal attention over the whole
sequence, computed in one shot. This is self-consistent: it needs no external reference, and it is
sensitive to precisely the things reading cannot check — ring wraparound, an off-by-one in the mask,
a cursor that advances wrongly, a rotary position that drifts from the column it wrote.

The falsification arms matter as much as the invariant: each perturbs the reference the way a
plausible transcription slip would, and each MUST disagree. An arm that agrees means this test
cannot tell the right implementation from that particular wrong one.

CPU only, no GPU, no engine, no serve. Needs the checkpoint for real `mtp.layers.0.self_attn.*`.

Run:  PYTHONPATH=python python3 tests/qwen4exp_draft_attn_parity_test.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MINISGL_TAIL_HIP", "0")   # CPU: keep the torch fallbacks

import torch  # noqa: E402

FAILED: list[str] = []
CKPT = "/model" if os.path.isdir("/model") else os.path.expanduser("~/ai/hf/q4e")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail and not ok else ''}")
    if not ok:
        FAILED.append(name)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / max(b.abs().max().item(), 1e-6)).item()


if not os.path.isdir(CKPT):
    print(f"SKIPPED: {CKPT} not present — this test asserts nothing without the checkpoint.")
    sys.exit(0)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from minisgl.distributed import set_tp_info, try_get_tp_info  # noqa: E402

if try_get_tp_info() is None:
    set_tp_info(0, 1)

torch.set_default_dtype(torch.float32)   # fp32: a 1-ulp bf16 wobble must not mask a real bug

from minisgl.models.config import ModelConfig  # noqa: E402
from minisgl.models.qwen4exp import Qwen4ExpMTPAttn  # noqa: E402
from minisgl.utils.hf import cached_load_hf_config  # noqa: E402
from safetensors import safe_open  # noqa: E402

# NOT AutoConfig: the installed transformers does not register `qwen4_exp`, so it raises. minisgl's
# own loader falls back to a generic PretrainedConfig built from config.json, which is what the
# serve itself uses — so this test builds the config by exactly the path the engine does.
hf = cached_load_hf_config(CKPT)
cfg = ModelConfig.from_hf(hf, spec_algorithm="mtp")
print(f"config: hidden={cfg.hidden_size} heads={cfg.num_qo_heads}/{cfg.num_kv_heads} "
      f"head_dim={cfg.head_dim} rotary={cfg.rotary_config}")

attn = Qwen4ExpMTPAttn(cfg, cfg.num_layers)

# ---- load the REAL mtp.layers.0.self_attn weights -----------------------------------------------
_idx = json.load(open(glob.glob(os.path.join(CKPT, "*index.json"))[0]))["weight_map"]
_h: dict = {}


def T(name: str) -> torch.Tensor:
    shard = _idx[name]
    if shard not in _h:
        _h[shard] = safe_open(os.path.join(CKPT, shard), framework="pt")
    return _h[shard].get_tensor(name).float()


P = "mtp.layers.0.self_attn."
attn.q_proj.weight = T(P + "q_proj.weight")
attn.k_proj.weight = T(P + "k_proj.weight")
attn.v_proj.weight = T(P + "v_proj.weight")
attn.o_proj.weight = T(P + "o_proj.weight")
attn.q_norm.weight = T(P + "q_norm.weight")
attn.k_norm.weight = T(P + "k_norm.weight")
print(f"loaded real weights: q_proj{tuple(attn.q_proj.weight.shape)} "
      f"o_proj{tuple(attn.o_proj.weight.shape)}")

HD, NQ, NKV = attn._head_dim, attn._num_qo_heads, attn._num_kv_heads
RING = 64                      # small ring so the WRAP is exercised, not just the happy path
SLOT = 1
torch.manual_seed(5)


def dense_reference(x: torch.Tensor) -> torch.Tensor:
    """Plain causal attention over the whole sequence, written straight through.

    Shares only the PROJECTIONS and the rotary with the implementation (those are the backbone's,
    already exercised by every served token). Everything this test is actually about — masking,
    cursor, ring addressing, the GQA contraction — is written independently here.
    """
    S = x.shape[0]
    pos = torch.arange(S, dtype=torch.int32)
    qg = attn.q_proj.forward(x).view(S, NQ, 2 * HD)
    q = qg[..., :HD].reshape(S, NQ * HD)
    gate = qg[..., HD:].reshape(S, NQ * HD)
    k = attn.k_proj.forward(x)
    v = attn.v_proj.forward(x).view(S, NKV, HD)
    attn.q_norm.forward_inplace(q.view(S, NQ, HD))
    attn.k_norm.forward_inplace(k.view(S, NKV, HD))
    q, k = attn.attn.rotary.forward(pos, q, k)
    q = q.view(S, NQ, HD)
    k = k.view(S, NKV, HD)
    rep = NQ // NKV
    # expand K/V to nq heads — the straightforward way, deliberately NOT the implementation's
    # grouped contraction, so a bug in that contraction shows up as a disagreement.
    ke = k.repeat_interleave(rep, dim=1)          # [S, NQ, HD]
    ve = v.repeat_interleave(rep, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, ke) * (float(HD) ** -0.5)
    causal = torch.full((S, S), float("-inf")).triu(1)
    scores = scores + causal.unsqueeze(0)
    probs = scores.softmax(dim=-1)
    o = torch.einsum("hqk,khd->qhd", probs, ve).reshape(S, NQ * HD)
    o = o * torch.sigmoid(gate)
    return attn.o_proj.forward(o)


def ring_stepped(x: torch.Tensor) -> torch.Tensor:
    """`forward_draft_masked` one token at a time, driving the cursor exactly as propose_body does."""
    S = x.shape[0]
    k_buf = torch.zeros(4, RING, NKV, HD)
    v_buf = torch.zeros(4, RING, NKV, HD)
    pos_buf = torch.full((4, RING), -1, dtype=torch.int64)
    slot_rows = torch.tensor([SLOT])
    outs = []
    for t in range(S):
        q_abs = torch.tensor([t], dtype=torch.int64)
        write_col = torch.remainder(q_abs, RING)
        positions = q_abs.to(torch.int32)
        pos_buf[slot_rows, write_col] = q_abs          # publish BEFORE masking (own row visible)
        pa = pos_buf[slot_rows]
        qa = q_abs.unsqueeze(1)
        keep = (pa >= 0) & (pa <= qa) & ((qa - pa) < RING)
        mask_bias = torch.where(keep, 0.0, float("-inf")).to(torch.float32)
        outs.append(attn.forward_draft_masked(
            x[t:t + 1], positions, k_buf, v_buf, slot_rows, write_col, mask_bias))
    return torch.cat(outs, dim=0)


print()
print("INVARIANT: stepping the ring == one dense causal attention over the same tokens")
for S in (1, 4, 17):
    x = torch.randn(S, cfg.hidden_size) * 0.05
    got, ref = ring_stepped(x), dense_reference(x)
    r = rel(got, ref)
    check(f"S={S:<3} (no wrap)  ring == dense", r < 2e-4, f"rel={r:.3e}")

# S > RING forces the ring to WRAP. Only the last RING positions remain addressable, so the
# reference is the dense attention restricted to that same window — which is what the mask encodes.
print()
print("WRAP: past the ring the window slides, and the mask must say so")
S = RING + 9
x = torch.randn(S, cfg.hidden_size) * 0.05
got = ring_stepped(x)[-1:]
q_abs = S - 1
lo = q_abs - RING + 1
sub = x[lo:]
ref_sub = dense_reference(sub)[-1:]   # same window, positions re-based
r = rel(got, ref_sub)
check(f"S={S} wrapped: last row matches the in-window dense attention", r < 5e-2,
      f"rel={r:.3e} (rotary is absolute, so this is a window check, not bit-exact)")

# ---------------------------------------------------------------- SEED vs CHAIN
# `seed_kv_masked` writes the prompt prefix's k/v into the ring WITHOUT running attention, and its
# docstring claims that is "byte-identical to what a per-step forward_draft_masked chain over the
# prompt would have produced". Nothing tested that. It matters more than it looks: the first draft
# of EVERY request is conditioned on the seeded prefix, so a wrong seed costs position-1 acceptance
# on every single request while leaving the text fluent.
print()
print("SEED: seeding a prefix then stepping == stepping the whole sequence")
for P_, D_ in ((5, 3), (11, 2)):
    S_ = P_ + D_
    x = torch.randn(S_, cfg.hidden_size) * 0.05
    # (a) step everything, one token at a time
    full = ring_stepped(x)[-D_:]
    # (b) seed the first P_ rows, then step only the last D_
    k_buf = torch.zeros(4, RING, NKV, HD)
    v_buf = torch.zeros(4, RING, NKV, HD)
    pos_buf = torch.full((4, RING), -1, dtype=torch.int64)
    attn.seed_kv_masked(x[:P_], torch.arange(P_, dtype=torch.int32), k_buf, v_buf, SLOT, 0)
    abs_pos = torch.arange(P_, dtype=torch.int64)
    pos_buf[SLOT, torch.remainder(abs_pos, RING)] = abs_pos
    slot_rows = torch.tensor([SLOT])
    outs = []
    for t in range(P_, S_):
        q_abs = torch.tensor([t], dtype=torch.int64)
        write_col = torch.remainder(q_abs, RING)
        pos_buf[slot_rows, write_col] = q_abs
        pa = pos_buf[slot_rows]; qa = q_abs.unsqueeze(1)
        keep = (pa >= 0) & (pa <= qa) & ((qa - pa) < RING)
        mask_bias = torch.where(keep, 0.0, float("-inf")).to(torch.float32)
        outs.append(attn.forward_draft_masked(
            x[t:t + 1], q_abs.to(torch.int32), k_buf, v_buf, slot_rows, write_col, mask_bias))
    seeded = torch.cat(outs, dim=0)
    r = rel(seeded, full)
    check(f"prefix={P_:<3} then {D_} steps == {S_} steps", r < 2e-4, f"rel={r:.3e}")

# ------------------------------------------------------------------ FALSIFICATION
print()
print("FALSIFICATION: each plausible slip MUST disagree with the implementation")
S = 12
x = torch.randn(S, cfg.hidden_size) * 0.05
base = ring_stepped(x)


def perturbed(**kw) -> torch.Tensor:
    S_ = x.shape[0]
    pos = torch.arange(S_, dtype=torch.int32)
    qg = attn.q_proj.forward(x).view(S_, NQ, 2 * HD)
    if kw.get("swap_gate"):
        q = qg[..., HD:].reshape(S_, NQ * HD); gate = qg[..., :HD].reshape(S_, NQ * HD)
    else:
        q = qg[..., :HD].reshape(S_, NQ * HD); gate = qg[..., HD:].reshape(S_, NQ * HD)
    k = attn.k_proj.forward(x)
    v = attn.v_proj.forward(x).view(S_, NKV, HD)
    if not kw.get("no_norm"):
        attn.q_norm.forward_inplace(q.view(S_, NQ, HD))
        attn.k_norm.forward_inplace(k.view(S_, NKV, HD))
    if not kw.get("no_rope"):
        q, k = attn.attn.rotary.forward(pos, q, k)
    q = q.view(S_, NQ, HD); k = k.view(S_, NKV, HD)
    rep = NQ // NKV
    ke = k.repeat_interleave(rep, dim=1); ve = v.repeat_interleave(rep, dim=1)
    scale = 1.0 if kw.get("no_scale") else (float(HD) ** -0.5)
    scores = torch.einsum("qhd,khd->hqk", q, ke) * scale
    # 2 => each token also attends to its own SUCCESSOR (the off-by-one a cursor slip makes).
    # NOT 0: triu(0) masks row 0 entirely, softmax(all -inf) is NaN, and the arm silently fails
    # as a nan comparison instead of testing anything.
    tri = 2 if kw.get("mask_offbyone") else 1
    scores = scores + torch.full((S_, S_), float("-inf")).triu(tri).unsqueeze(0)
    o = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), ve).reshape(S_, NQ * HD)
    if not kw.get("no_gate"):
        o = o * torch.sigmoid(gate)
    return attn.o_proj.forward(o)


for name, kw in (("gate taken from the WRONG half of q_proj", dict(swap_gate=True)),
                 ("q/k norms omitted", dict(no_norm=True)),
                 ("rotary omitted", dict(no_rope=True)),
                 ("softmax scale dropped", dict(no_scale=True)),
                 ("mask off-by-one (sees its own successor)", dict(mask_offbyone=True)),
                 ("output gate omitted", dict(no_gate=True))):
    r = rel(perturbed(**kw), base)
    check(f"{name} disagrees", r > 1e-3, f"rel={r:.3e}")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
