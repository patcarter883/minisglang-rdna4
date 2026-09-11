"""Phase 2 parity: minisgl's Nemotron-H layer math vs HF's own `modeling_nemotron_h`, on the REAL
checkpoint's weights.

WHY AGAINST HF AND NOT A REFERENCE WE WROTE. Two of the plan's architectural calls were already
falsified once by reading the checkpoint (the experts are ungated relu², and `layers_block_type`
cannot be squeezed into `layer_types`). A reference we author ourselves can repeat the same
misreading in both places and agree with itself. `transformers.models.nemotron_h` is the reference
implementation NVIDIA's checkpoint was published against, so a disagreement here is OUR bug.

WHAT IS AND IS NOT COVERED. This is layer math on real tensors, CPU, no engine and no GPU:

  * ROUTING   — the sigmoid + e_score_correction_bias router, on the checkpoint's own gate weights.
                The plan's Phase 2 exit criterion names this one specifically.
  * EXPERTS   — ungated relu², on real NVFP4-dequantized expert weights. This is the finding a
                gated path would silently corrupt, so it is checked against HF's expert forward
                rather than against our own idea of what relu² means.
  * GATEDNORM — Mamba-2's gate-then-normalise epilogue, which is the opposite order from a
                post-norm gate and is silent when reversed.
  * SPLIT     — the `in_proj` split order [z | x·B·C | dt] and the conv-then-split ordering, which
                give correct shapes and wrong numbers if transposed.
  * NAMES     — every weight the module tree expects exists in the checkpoint under exactly that
                name. A rename here does not crash, it loads a subset and serves noise.

  NOT covered: attention (needs the engine's paged-KV context), the decode path (needs the Phase 3
  Mamba-2 state cache, which is why no decode path is wired at all), and anything fused.

Run:  PYTHONPATH=python python3 tests/nemotron_layer_parity_test.py
"""
from __future__ import annotations

import gzip
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.manual_seed(0)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILED.append(name)


def close(a: torch.Tensor, b: torch.Tensor, tol: float) -> tuple[bool, str]:
    """RELATIVE agreement, scaled by the reference's own magnitude.

    An absolute tolerance is the wrong instrument for bf16 and it fails in the direction that
    wastes time: the gated-norm output here reaches ±31, where one bf16 ulp is already 0.12, so a
    3e-2 absolute bound rejects tensors that agree to HALF AN ULP. Sizing the bound against
    `b.abs().max()` asks the question actually worth asking — do these agree to the precision the
    dtype can represent — and still fails loudly on a real disagreement, which at these magnitudes
    is orders of magnitude larger."""
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    scale = max(b.abs().max().item(), 1e-6)
    d = (a - b).abs().max().item()
    return d <= tol * scale, f"max|Δ|={d:.3e} rel={d / scale:.3e} > {tol:.0e}"


# ---------------------------------------------------------------------------------------------
# checkpoint access
# ---------------------------------------------------------------------------------------------
REPO = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"


def _snapshot() -> str | None:
    root = os.path.expanduser("~/.cache/huggingface/hub/models--" + REPO.replace("/", "--"))
    snaps = os.path.join(root, "snapshots")
    if not os.path.isdir(snaps):
        return None
    for d in sorted(os.listdir(snaps)):
        p = os.path.join(snaps, d)
        if os.path.isfile(os.path.join(p, "config.json")):
            return p
    return None


SNAP = _snapshot()
if SNAP is None:
    # A skip here must be LOUD. A green run that silently tested nothing is the failure mode this
    # repo has already been bitten by (9 of 14 HC parity tests skipping clean without their mounts).
    print(f"SKIPPED: {REPO} is not in the HF cache — this test asserts nothing without it.")
    sys.exit(0)

CFG = json.load(open(os.path.join(SNAP, "config.json")))
BLOCK_TYPES = CFG["layers_block_type"]
MAMBA_ID = BLOCK_TYPES.index("mamba")
MOE_ID = BLOCK_TYPES.index("moe")

from safetensors import safe_open  # noqa: E402

_INDEX = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
_HANDLES: dict = {}


def tensor(name: str) -> torch.Tensor:
    shard = _INDEX[name]
    h = _HANDLES.get(shard)
    if h is None:
        h = _HANDLES[shard] = safe_open(os.path.join(SNAP, shard), framework="pt")
    return h.get_tensor(name)


print(f"checkpoint: {SNAP}")
print(f"layers: {len(BLOCK_TYPES)}  mamba@{MAMBA_ID}  moe@{MOE_ID}  "
      f"attention@{[i for i, t in enumerate(BLOCK_TYPES) if t == 'attention']}")
print()

# ---------------------------------------------------------------------------------------------
# NAMES — the module tree vs the checkpoint
# ---------------------------------------------------------------------------------------------
print("NAMES: every weight the module tree expects exists under exactly that name")
_present = set(_INDEX)
for n in [
    "backbone.embeddings.weight",
    "backbone.norm_f.weight",
    "lm_head.weight",
    f"backbone.layers.{MAMBA_ID}.norm.weight",
    f"backbone.layers.{MAMBA_ID}.mixer.in_proj.weight",
    f"backbone.layers.{MAMBA_ID}.mixer.out_proj.weight",
    f"backbone.layers.{MAMBA_ID}.mixer.conv1d.weight",
    f"backbone.layers.{MAMBA_ID}.mixer.conv1d.bias",
    f"backbone.layers.{MAMBA_ID}.mixer.A_log",
    f"backbone.layers.{MAMBA_ID}.mixer.D",
    f"backbone.layers.{MAMBA_ID}.mixer.dt_bias",
    f"backbone.layers.{MAMBA_ID}.mixer.norm.weight",
    f"backbone.layers.{MOE_ID}.mixer.gate.weight",
    f"backbone.layers.{MOE_ID}.mixer.gate.e_score_correction_bias",
    f"backbone.layers.{MOE_ID}.mixer.experts.0.up_proj.weight",
    f"backbone.layers.{MOE_ID}.mixer.experts.0.down_proj.weight",
    f"backbone.layers.{MOE_ID}.mixer.shared_experts.up_proj.weight",
    f"backbone.layers.{MOE_ID}.mixer.shared_experts.down_proj.weight",
]:
    check(f"present: {n}", n in _present)
# The absence assertion is the load-bearing half: a gate half MUST NOT exist, because a gated MoE
# path would happily read one and this is how we know it cannot.
for n in [
    f"backbone.layers.{MOE_ID}.mixer.experts.0.gate_proj.weight",
    f"backbone.layers.{MOE_ID}.mixer.experts.0.gate_up_proj.weight",
    f"backbone.layers.{MOE_ID}.mixer.shared_experts.gate_proj.weight",
]:
    check(f"ABSENT (ungated relu²): {n}", n not in _present)
check("an attention layer ships ONE norm, not two (input+post)",
      f"backbone.layers.5.norm.weight" in _present
      and f"backbone.layers.5.input_layernorm.weight" not in _present
      and f"backbone.layers.5.post_attention_layernorm.weight" not in _present)

# ---------------------------------------------------------------------------------------------
# ROUTING
# ---------------------------------------------------------------------------------------------
print()
print("ROUTING: sigmoid + e_score_correction_bias, on the checkpoint's own gate")
from transformers.models.nemotron_h import modeling_nemotron_h as HF  # noqa: E402
from transformers.models.nemotron_h.configuration_nemotron_h import (  # noqa: E402
    NemotronHConfig,
)

hf_cfg = NemotronHConfig(**{k: v for k, v in CFG.items() if k != "quantization_config"})
if not hasattr(hf_cfg, "num_local_experts"):
    hf_cfg.num_local_experts = CFG["n_routed_experts"]

gate_w = tensor(f"backbone.layers.{MOE_ID}.mixer.gate.weight")
gate_b = tensor(f"backbone.layers.{MOE_ID}.mixer.gate.e_score_correction_bias")
H = CFG["hidden_size"]
x = torch.randn(11, H, dtype=torch.bfloat16)

hf_router = HF.NemotronHTopkRouter(hf_cfg)
with torch.no_grad():
    hf_router.weight.copy_(gate_w)
    hf_router.e_score_correction_bias.copy_(gate_b)
    _, hf_w, hf_i = hf_router(x)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from minisgl.models.nemotron_h import NemotronHExperts, NemotronHTopkGate, _GatedRMSNorm  # noqa: E402

gate = NemotronHTopkGate(H, CFG["n_routed_experts"])
gate.weight = gate_w.clone()
gate.e_score_correction_bias = gate_b.clone()
gate.configure(CFG["num_experts_per_tok"], CFG["norm_topk_prob"], CFG["routed_scaling_factor"])
my_w, my_i = gate.forward(x)

# Compare as SETS per token: HF passes sorted=False to topk, so the order within a token is an
# implementation detail, but the selected set and each expert's weight are not.
same_set = all(
    set(hf_i[t].tolist()) == set(my_i[t].tolist()) for t in range(x.shape[0])
)
check("selected expert SET matches HF per token", same_set,
      f"hf={hf_i[0].tolist()} mine={my_i[0].tolist()}")
if same_set:
    hf_sorted = torch.stack([hf_w[t][hf_i[t].argsort()] for t in range(x.shape[0])])
    my_sorted = torch.stack([my_w[t][my_i[t].argsort()] for t in range(x.shape[0])])
    ok, d = close(hf_sorted, my_sorted, 1e-6)
    check("routing WEIGHTS match HF (bias steers selection only)", ok, d)
check("weights are scaled by routed_scaling_factor",
      abs(float(my_w.sum(-1).mean()) - CFG["routed_scaling_factor"]) < 1e-3,
      f"mean row sum {float(my_w.sum(-1).mean()):.4f} vs {CFG['routed_scaling_factor']}")

# ---------------------------------------------------------------------------------------------
# EXPERTS — ungated relu², on real NVFP4 weights
# ---------------------------------------------------------------------------------------------
print()
print("EXPERTS: ungated relu² on real NVFP4-dequantized weights")
from minisgl.quant.nvfp4 import dequant_reference  # noqa: E402

E_TEST = 3  # three experts is enough to exercise the gather; 128 would just be slower


def deq(base: str) -> torch.Tensor:
    # `global_field` is NOT decoration: NVFP4_GLOBAL_SCALE_IS_RECIPROCAL says compressed-tensors'
    # `weight_global_scale` is a DIVISOR while modelopt's `weight_scale_2` is a MULTIPLIER. This
    # checkpoint ships `weight_scale_2`, and taking the default inverted the scale — dequantized
    # weights ~1e6 instead of ~1e-1, which surfaced here as a 1e23 parity gap rather than as
    # anything that looked like a scale bug.
    return dequant_reference(
        tensor(base + ".weight"), tensor(base + ".weight_scale"), tensor(base + ".weight_scale_2"),
        global_field="weight_scale_2",
    )


up = torch.stack([deq(f"backbone.layers.{MOE_ID}.mixer.experts.{e}.up_proj") for e in range(E_TEST)])
dn = torch.stack([deq(f"backbone.layers.{MOE_ID}.mixer.experts.{e}.down_proj") for e in range(E_TEST)])
check("up_proj dequantizes to [inter, hidden] with NO gate half",
      up.shape[1:] == (CFG["moe_intermediate_size"], H), f"{tuple(up.shape)}")
check("down_proj dequantizes to [hidden, inter]",
      dn.shape[1:] == (H, CFG["moe_intermediate_size"]), f"{tuple(dn.shape)}")

xe = torch.randn(7, H, dtype=torch.bfloat16)
ids = torch.randint(0, E_TEST, (7, 2))
wts = torch.rand(7, 2).float()

experts = NemotronHExperts(E_TEST, H, CFG["moe_intermediate_size"])
experts.up_proj = up.to(torch.bfloat16)
experts.down_proj = dn.to(torch.bfloat16)
mine = experts.forward(xe, wts, ids)

# The definition, written out per token — deliberately NOT sharing code with the implementation.
ref = torch.zeros(7, H, dtype=torch.float32)
for t in range(7):
    for k in range(2):
        e = int(ids[t, k])
        h = F.linear(xe[t].float(), up[e].float())
        h = F.relu(h).pow(2)
        ref[t] += F.linear(h, dn[e].float()) * float(wts[t, k])
# bf16 weights + a 2688-wide reduction against an fp32 reference: a few ulps, relative.
ok, d = close(mine, ref, 2e-2)
check("relu² expert output matches the definition", ok, d)

# The falsification: a GATED reading of the same weights must NOT agree. If it did, this test
# could not tell the two apart and the Phase 0 finding would be untestable.
half = CFG["moe_intermediate_size"] // 2
gated = torch.zeros(7, H, dtype=torch.float32)
for t in range(7):
    for k in range(2):
        e = int(ids[t, k])
        h = F.linear(xe[t].float(), up[e].float())
        g, u = h[:half], h[half:]
        gated[t] += F.linear(F.silu(g) * u, dn[e][:, :half].float()) * float(wts[t, k])
check("a GATED reading of the same weights DISAGREES (the finding is testable)",
      not close(mine, gated, 2e-2)[0])

# ---------------------------------------------------------------------------------------------
# GATED NORM — gate BEFORE normalise
# ---------------------------------------------------------------------------------------------
print()
print("GATEDNORM: Mamba-2 gates before it normalises")
inner = CFG["mamba_num_heads"] * CFG["mamba_head_dim"]
nw = tensor(f"backbone.layers.{MAMBA_ID}.mixer.norm.weight")
gn = _GatedRMSNorm(inner, eps=CFG["norm_eps"])
gn.weight = nw.clone()
h = torch.randn(5, inner, dtype=torch.bfloat16)
z = torch.randn(5, inner, dtype=torch.bfloat16)
mine_n = gn.forward(h, z)

hf_norm = HF.NemotronHRMSNorm(inner, eps=CFG["norm_eps"])
with torch.no_grad():
    hf_norm.weight.copy_(nw)
    ref_n = hf_norm(h * F.silu(z))                       # gate FIRST, then normalise
    wrong_n = hf_norm(h) * F.silu(z)                     # normalise first — the silent transposition
# ~0.44 bf16 ulp measured; 2e-2 relative is ~5 ulps of headroom and still 100x tighter
# than the gate-order transposition the next check pins.
ok, d = close(mine_n, ref_n, 2e-2)
check("gated norm == RMSNorm(x * silu(gate))", ok, d)
check("it is NOT RMSNorm(x) * silu(gate) (the two differ, so the order is testable)",
      not close(ref_n, wrong_n, 2e-2)[0])

# ---------------------------------------------------------------------------------------------
# SPLIT — in_proj layout and conv-then-split
# ---------------------------------------------------------------------------------------------
print()
print("SPLIT: in_proj emits [z | x·B·C | dt] and the conv runs over x·B·C JOINTLY")
groups_state = CFG["n_groups"] * CFG["ssm_state_size"]
conv_dim = inner + 2 * groups_state
in_w = tensor(f"backbone.layers.{MAMBA_ID}.mixer.in_proj.weight")
conv_w = tensor(f"backbone.layers.{MAMBA_ID}.mixer.conv1d.weight")
check("in_proj out == inner + conv_dim + num_heads",
      in_w.shape[0] == inner + conv_dim + CFG["mamba_num_heads"],
      f"{in_w.shape[0]} vs {inner + conv_dim + CFG['mamba_num_heads']}")
check("conv1d width == conv_dim (x, B and C convolved together)",
      conv_w.shape[0] == conv_dim, f"{conv_w.shape[0]} vs {conv_dim}")
check("conv1d is depthwise with kernel == conv_kernel",
      tuple(conv_w.shape[1:]) == (1, CFG["conv_kernel"]), f"{tuple(conv_w.shape)}")
check("A_log / D / dt_bias are per-head",
      all(tensor(f"backbone.layers.{MAMBA_ID}.mixer.{n}").shape[0] == CFG["mamba_num_heads"]
          for n in ("A_log", "D", "dt_bias")))
check("the mamba norm is over the INNER dim, not hidden",
      nw.shape[0] == inner, f"{nw.shape[0]} vs inner {inner}")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
