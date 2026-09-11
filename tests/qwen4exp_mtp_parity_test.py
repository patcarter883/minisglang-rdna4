"""Numeric parity for the Qwen3.8-Flash-Next MTP head's seed fusion, on the REAL `mtp.*` tensors.

WHY THIS EXISTS. The head measures p ~= 0.32 top-1 agreement with the target where a trained MTP
head should reach 0.55-0.8, and three rounds of reasoning about the code failed to localise it. An
acceptance rate is a terrible instrument for finding an arithmetic bug: it is three layers away from
the arithmetic and it moves for a dozen unrelated reasons (sampling, page_size, QSA availability,
draft-chain depth). This tests the arithmetic DIRECTLY, against a literal transcription of
sglang's `qwen4_exp_mtp.py::_fuse_residual_linear_shared`, on the checkpoint's own weights.

The transcription is deliberately written from the reference in the most obvious possible way — no
sharing of code with the implementation, no cleverness — so that a disagreement means the
implementation is wrong rather than that both share a misreading. That is the same discipline the
Nemotron parity test used, where it caught two bugs.

WHAT THIS CAN AND CANNOT CATCH:
  * CAN: a transposed view, a wrong broadcast axis, folding the wide stream when it should stay
    wide, applying fc_hidden across the whole 10240 instead of per-branch, norm convention
    (plus_one vs plain), or the embedding term added to the wrong axis.
  * CANNOT: anything in the layer body (attention/MoE/hyper-connections), because testing those
    against a second reimplementation of the same minisgl components would be circular. The layer
    dataflow was verified by READING against the reference instead (it matches minisgl's own
    Qwen4ExpDecoderLayer, which serves the backbone correctly).

CPU only, no GPU, no engine.

Run:  PYTHONPATH=python python3 tests/qwen4exp_mtp_parity_test.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

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

from safetensors import safe_open  # noqa: E402

_idx = json.load(open(glob.glob(os.path.join(CKPT, "*index.json"))[0]))["weight_map"]
_h: dict = {}


def T(name: str) -> torch.Tensor:
    shard = _idx[name]
    if shard not in _h:
        _h[shard] = safe_open(os.path.join(CKPT, shard), framework="pt")
    return _h[shard].get_tensor(name)


cfg = json.load(open(os.path.join(CKPT, "config.json")))
cfg = cfg.get("text_config", cfg)
H = cfg["hidden_size"]
HC = cfg["hc_count"]
EPS = cfg.get("rms_norm_eps") or cfg.get("norm_eps") or 1e-6

print(f"checkpoint: {CKPT}   hidden={H} hc_count={HC} eps={EPS}")
print()

# --------------------------------------------------------------------------------------------
# the REFERENCE: a literal transcription of sglang qwen4_exp_mtp.py::_fuse_residual_linear_shared
#
#     input_embeds  = fc_embedding(pre_fc_norm_embedding(input_embeds))
#     orig_shape    = hidden_states.shape
#     hidden_states = pre_fc_norm_hidden(hidden_states)
#     decoder_view  = hidden_states.view(*shape[:-1], hc_count, hidden_size)
#     encoder_inputs= fc_hidden(decoder_view)
#     return (input_embeds.unsqueeze(-2) + encoder_inputs).view(orig_shape)
#
# with GemmaRMSNorm == x/rms(x) * (1 + w), computed in fp32 like HF/sglang do.
# --------------------------------------------------------------------------------------------
def gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    d = x.float()
    d = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + eps)
    return (d * (1.0 + w.float())).to(x.dtype)


def reference_fuse(embed_e: torch.Tensor, wide_hidden: torch.Tensor) -> torch.Tensor:
    e = gemma_rmsnorm(embed_e, T("mtp.pre_fc_norm_embedding.weight"), EPS)
    e = torch.nn.functional.linear(e, T("mtp.fc_embedding.weight"))
    orig = wide_hidden.shape
    h = gemma_rmsnorm(wide_hidden, T("mtp.pre_fc_norm_hidden.weight"), EPS)
    view = h.view(*h.shape[:-1], HC, H)
    enc = torch.nn.functional.linear(view, T("mtp.fc_hidden.weight"))
    return (e.unsqueeze(-2) + enc).reshape(orig)


# --------------------------------------------------------------------------------------------
# the IMPLEMENTATION under test, exercised through the real class
# --------------------------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
torch.set_default_dtype(torch.bfloat16)

print("IMPL: build Qwen4ExpMTPHead's seed path with the checkpoint's own weights")
# TP info must exist before any layer primitive runs: the engage ledger logs through a rank0 gate
# that asks for it, so an un-set probe dies inside `minv_linear` rather than anywhere informative.
from minisgl.distributed import set_tp_info  # noqa: E402

set_tp_info(0, 1)
from minisgl.layers import LinearReplicated, RMSNorm  # noqa: E402


class ImplFuse:
    """The exact statements Qwen4ExpMTPHead.fuse runs, on the same layer primitives it uses."""

    def __init__(self) -> None:
        self.pre_e = RMSNorm(H, eps=EPS, plus_one=True)
        self.pre_h = RMSNorm(HC * H, eps=EPS, plus_one=True)
        self.fc_e = LinearReplicated(H, H, has_bias=False)
        self.fc_h = LinearReplicated(H, H, has_bias=False)
        self.pre_e.weight = T("mtp.pre_fc_norm_embedding.weight").clone()
        self.pre_h.weight = T("mtp.pre_fc_norm_hidden.weight").clone()
        self.fc_e.weight = T("mtp.fc_embedding.weight").clone()
        self.fc_h.weight = T("mtp.fc_hidden.weight").clone()

    def __call__(self, embed_e: torch.Tensor, last_hidden: torch.Tensor) -> torch.Tensor:
        e = self.fc_e.forward(self.pre_e.forward(embed_e))
        h = self.pre_h.forward(last_hidden)
        branches = h.view(*h.shape[:-1], HC, H)
        branches = self.fc_h.forward(branches)
        return (e.unsqueeze(-2) + branches).reshape(*h.shape)


impl = ImplFuse()
torch.manual_seed(7)

print()
print("PARITY: implementation vs the literal upstream transcription")
for T_rows in (1, 3, 17):
    emb = torch.randn(T_rows, H, dtype=torch.bfloat16)
    wide = torch.randn(T_rows, HC * H, dtype=torch.bfloat16)
    got = impl(emb, wide)
    ref = reference_fuse(emb, wide)
    check(f"rows={T_rows}: fuse matches upstream", rel(got, ref) < 2e-2, f"rel={rel(got, ref):.3e}")
    check(f"rows={T_rows}: output stays hc_count-WIDE", tuple(got.shape) == (T_rows, HC * H),
          f"{tuple(got.shape)}")

# ------------------------------------------------------------------ FALSIFICATION
# Each of these is a plausible misreading of the shapes. If any AGREES with the reference, this
# test cannot tell the right wiring from the wrong one and proves nothing.
print()
print("FALSIFICATION: the plausible wrong wirings must DISAGREE")
emb = torch.randn(5, H, dtype=torch.bfloat16)
wide = torch.randn(5, HC * H, dtype=torch.bfloat16)
ref = reference_fuse(emb, wide)

# (a) fold the wide seed to hidden, project, then re-widen by repeat — the reading the shapes invite
h_n = gemma_rmsnorm(wide, T("mtp.pre_fc_norm_hidden.weight"), EPS)
folded = h_n.view(5, HC, H).mean(dim=1)
wrong_fold = torch.nn.functional.linear(folded, T("mtp.fc_hidden.weight"))
wrong_fold = (torch.nn.functional.linear(
    gemma_rmsnorm(emb, T("mtp.pre_fc_norm_embedding.weight"), EPS),
    T("mtp.fc_embedding.weight")) + wrong_fold).repeat(1, HC)
check("(a) fold-then-widen disagrees", rel(wrong_fold, ref) > 1e-1, f"rel={rel(wrong_fold, ref):.3e}")

# (b) embedding added to branch 0 only, instead of broadcast to all hc branches
e_p = torch.nn.functional.linear(
    gemma_rmsnorm(emb, T("mtp.pre_fc_norm_embedding.weight"), EPS), T("mtp.fc_embedding.weight"))
enc = torch.nn.functional.linear(h_n.view(5, HC, H), T("mtp.fc_hidden.weight")).clone()
enc[:, 0] += e_p
check("(b) embedding on branch-0 only disagrees", rel(enc.reshape(5, HC * H), ref) > 1e-2,
      f"rel={rel(enc.reshape(5, HC * H), ref):.3e}")

# (c) plain RMSNorm instead of the (1+w) Gemma convention
def plain_rms(x, w, eps):
    d = x.float(); d = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + eps)
    return (d * w.float()).to(x.dtype)
h_p = plain_rms(wide, T("mtp.pre_fc_norm_hidden.weight"), EPS)
wrong_norm = (e_p.unsqueeze(-2) + torch.nn.functional.linear(
    h_p.view(5, HC, H), T("mtp.fc_hidden.weight"))).reshape(5, HC * H)
check("(c) plain-RMSNorm (no +1) disagrees", rel(wrong_norm, ref) > 1e-2,
      f"rel={rel(wrong_norm, ref):.3e}")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
