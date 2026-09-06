"""Adversarial probe for the SILENT-WRONG-NUMBERS lens over the MoE interposition seam.

WHAT IT CAUGHT (2026-09-03, all now fixed in `weights/moe_interpose.py` with regression tests in
`tests/core/test_moe_interpose.py`):

  A. Every alias of a component was rebound with the CANONICAL component's dtype and shape, so
     `_GroupedFP8Experts`' `_w_op` came back as `f8_e4m3` instead of `uint8` and `_scales_op` came
     back as `(E, N, 1)` instead of `(E, N)`. The names stayed aliased — the shipped alias test
     passed — but every downstream `element_size()`/`numel()`/kernel binding then computes from the
     wrong dtype. FIX: `_Alias` carries each name's own dtype/shape; `_reinterpret` rebuilds ITS
     view.
  B/E. The read-back gate used `torch.equal`, i.e. VALUE equality over tensors that are bit
     patterns. FALSE PASS on a `-0.0`-for-`+0.0` arena (the gate's whole job is to prove the bytes),
     and FALSE FAIL — accusing the arena — on any stack containing a NaN, which
     `quant/mxfp4.convert_mxfp4_moe` can genuinely produce (`e8m0_nan_groups`). FIX:
     `_bitwise_equal` compares the uint8 views, the same decision `granule._rows_0_1_equal` makes.

  C is the end-to-end trace and passed before and after: every expert's weight AND scale row lands
  at its own index, component-major stride `base + e*row_bytes` is preserved, and the containers
  really point into the arena.

CPU only, no GPU, no lease. Run inside the serve image:

    docker run --rm -v <worktree>:/engine -w /engine -e PYTHONPATH=/engine/python \
      -e MINISGL_TAIL_HIP=0 --entrypoint bash minisgl-rdna4:lean-pytest \
      -lc 'python tools/offload/probe_silent_wrong_numbers.py'
"""

from __future__ import annotations

import torch

from minisgl.weights.granule import ExpertContainer, spec_for_container
from minisgl.weights.moe_interpose import MoEWeightSeam
from minisgl.weights.stacks import StackKind, TorchStackAllocator


class _Layer:
    _weight_offload = None

    def __init__(self, n, k, w13, w2):
        self.local_num_experts = n
        self.top_k = k
        self.gate_up_proj = w13
        self.down_proj = w2

    def expert_containers(self):
        return {"gate_up_proj": self.gate_up_proj, "down_proj": self.down_proj}


class _Aliased(ExpertContainer):
    """`_GroupedFP8Experts` shape: `weight` (fp8) and `_w_op` (uint8 view) are ONE storage."""

    def __init__(self, n, out_f, in_f):
        self._num_experts = n
        self.weight = torch.zeros((n, out_f, in_f), dtype=torch.float8_e4m3fn)
        self._w_op = self.weight.view(torch.uint8)
        # `_scales_op` aliases `weight_scale` with a DIFFERENT SHAPE, exactly like the real
        # `_GroupedFP8Experts.post_load` (`weight_scale.squeeze(-1).contiguous().float()` is a no-op
        # on an already-contiguous f32 (E,N,1)).
        self.weight_scale = torch.arange(n * out_f, dtype=torch.float32).reshape(n, out_f, 1)
        self._scales_op = self.weight_scale.squeeze(-1)

    def forward(self, *a, **kw):
        raise RuntimeError


class _Marked(ExpertContainer):
    """Every component's row `e` is filled with a value derived from `e`, so one expert's bytes
    can be traced end to end and a cross-expert scale swap is a countable mismatch."""

    def __init__(self, n, out_f, in_f):
        self._num_experts = n
        self._w_op = torch.zeros((n, out_f, in_f // 8), dtype=torch.int32)
        self._scales_op = torch.zeros((n, in_f // 32, out_f), dtype=torch.float16)
        for e in range(n):
            self._w_op[e] = 0x1000 + e
            self._scales_op[e] = float(e + 1)

    def forward(self, *a, **kw):
        raise RuntimeError


def _alloc():
    handed = []

    def host_alloc(shape, dtype):
        t = torch.empty(shape, dtype=dtype)
        handed.append(t)
        return t

    return TorchStackAllocator(device="cpu", host_alloc=host_alloc), handed


def _seam(layer):
    n = layer.local_num_experts
    return MoEWeightSeam(
        "m.layers[0].mlp",
        layer,
        w13_spec=spec_for_container(layer.gate_up_proj, n),
        w2_spec=spec_for_container(layer.down_proj, n),
    )


fails = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------------------- A: alias identity
layer = _Layer(4, 1, _Aliased(4, 8, 16), _Aliased(4, 16, 8))
pre = {
    "weight": (layer.gate_up_proj.weight.dtype, tuple(layer.gate_up_proj.weight.shape)),
    "_w_op": (layer.gate_up_proj._w_op.dtype, tuple(layer.gate_up_proj._w_op.shape)),
    "weight_scale": (
        layer.gate_up_proj.weight_scale.dtype,
        tuple(layer.gate_up_proj.weight_scale.shape),
    ),
    "_scales_op": (
        layer.gate_up_proj._scales_op.dtype,
        tuple(layer.gate_up_proj._scales_op.shape),
    ),
}
alloc, _ = _alloc()
_seam(layer).bind(StackKind.HOST, alloc)
c = layer.gate_up_proj
post = {
    "weight": (c.weight.dtype, tuple(c.weight.shape)),
    "_w_op": (c._w_op.dtype, tuple(c._w_op.shape)),
    "weight_scale": (c.weight_scale.dtype, tuple(c.weight_scale.shape)),
    "_scales_op": (c._scales_op.dtype, tuple(c._scales_op.shape)),
}
for k in pre:
    check(f"A: alias {k!r} keeps dtype/shape", pre[k] == post[k], f"{pre[k]} -> {post[k]}")
check(
    "A: aliases still share storage",
    c.weight.untyped_storage().data_ptr() == c._w_op.untyped_storage().data_ptr()
    and c.weight_scale.untyped_storage().data_ptr() == c._scales_op.untyped_storage().data_ptr(),
)

# ---------------------------------------------------------------------------- B: read-back is bitwise
class _SignFlipArena:
    """An 'arena' that stores -0.0 where +0.0 was written. Bitwise wrong, value-equal."""

    def __init__(self):
        self.handed = []

    def __call__(self, shape, dtype):
        t = _Poison(torch.empty(shape, dtype=dtype))
        self.handed.append(t)
        return t


class _Poison(torch.Tensor):
    @staticmethod
    def __new__(cls, base):
        return torch.Tensor._make_subclass(cls, base, False)

    def copy_(self, src, *a, **kw):
        out = super().copy_(src, *a, **kw)
        if self.dtype.is_floating_point:
            # +0.0 -> -0.0: identical under `torch.equal`, a different bit pattern.
            with torch.no_grad():
                z = super().__getitem__(slice(None))
                torch.Tensor.copy_(self, torch.where(z == 0, torch.full_like(z, -0.0), z))
        return out


sf = _SignFlipArena()
layer_b = _Layer(4, 1, _Marked(4, 8, 64), _Marked(4, 64, 64))
layer_b.gate_up_proj._scales_op.zero_()  # every scale is +0.0 -> flipped to -0.0 by the arena
alloc_b = TorchStackAllocator(device="cpu", host_alloc=sf)
try:
    _seam(layer_b).bind(StackKind.HOST, alloc_b)
    bits_ok = torch.equal(
        layer_b.gate_up_proj._scales_op.reshape(-1).view(torch.uint8),
        torch.zeros(layer_b.gate_up_proj._scales_op.numel() * 2, dtype=torch.uint8),
    )
    check("B: bake read-back catches a BITWISE-wrong, value-equal arena", bits_ok,
          "bind succeeded and the stored bits are -0.0, not +0.0" if not bits_ok else "")
except Exception as exc:  # noqa: BLE001
    check("B: bake read-back catches a BITWISE-wrong, value-equal arena", True, type(exc).__name__)

# ---------------------------------------------------------------------------- C: trace one expert
layer_c = _Layer(6, 2, _Marked(6, 16, 64), _Marked(6, 64, 64))
alloc_c, handed = _alloc()
rep = _seam(layer_c).bind(StackKind.HOST, alloc_c)
ok = True
for attr in ("gate_up_proj", "down_proj"):
    cc = getattr(layer_c, attr)
    for e in range(6):
        if not bool((cc._w_op[e] == 0x1000 + e).all()):
            ok = False
        if not bool((cc._scales_op[e] == float(e + 1)).all()):
            ok = False
check("C: every expert's weight AND scale row survived at its own index", ok)

# component-major: expert e of a component is at base + e*row_bytes
cm = True
for attr in ("gate_up_proj", "down_proj"):
    cc = getattr(layer_c, attr)
    for nm in ("_w_op", "_scales_op"):
        t = getattr(cc, nm)
        row = t[0].numel() * t.element_size()
        for e in range(6):
            if t[e].data_ptr() != t.data_ptr() + e * row:
                cm = False
check("C: component-major stride preserved (base + e*row_bytes)", cm)
check("C: the containers really came from the arena",
      all(any(h.data_ptr() == getattr(getattr(layer_c, a), n).data_ptr() for h in handed)
          for a in ("gate_up_proj", "down_proj") for n in ("_w_op", "_scales_op")))

# ---------------------------------------------------------------------------- D: fp8 read-back
try:
    a = torch.zeros(4, dtype=torch.float8_e4m3fn)
    torch.equal(a, a.clone())
    check("D: torch.equal works on float8_e4m3fn", True)
except Exception as exc:  # noqa: BLE001
    check("D: torch.equal works on float8_e4m3fn", False, f"{type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------- E: NaN self-test
layer_e = _Layer(2, 1, _Marked(2, 8, 64), _Marked(2, 64, 64))
layer_e.gate_up_proj._scales_op[0, 0, 0] = float("nan")
alloc_e, _ = _alloc()
try:
    _seam(layer_e).bind(StackKind.HOST, alloc_e)
    check("E: a NaN in a scale does not fail a correct bake", True)
except Exception as exc:  # noqa: BLE001
    check("E: a NaN in a scale does not fail a correct bake", False, f"{type(exc).__name__}: {exc}")

print()
print("FAILURES:", fails or "none")
raise SystemExit(1 if fails else 0)
