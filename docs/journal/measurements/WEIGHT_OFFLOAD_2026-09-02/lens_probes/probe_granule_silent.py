"""SILENT-WRONG-NUMBERS lens probes over `weights/granule.py` — the pre-fix observations, kept.

Run (GPU-free, in the serve image because the host torch is broken):

    docker run --rm -v <worktree>:/engine --entrypoint bash minisgl-rdna4:lean-pytest -lc \\
      'source /app/.venv/bin/activate; MINISGL_TAIL_HIP=0 PYTHONPATH=/engine/python \\
       python /engine/docs/measurements/WEIGHT_OFFLOAD_2026-09-02/lens_probes/probe_granule_silent.py'

Each probe prints what the module DOES. The `PRE-FIX` line under each is the output recorded on
2026-09-03 against the derivation as it stood before the lens pass; every one of them is a
descriptor that reported a container as fully described while it was not. They are kept executable
rather than written down so the guards cannot be quietly removed — if one of these starts printing
its PRE-FIX line again, that guard is gone.
"""

import torch
from minisgl.layers.base import BaseOP
from minisgl.weights import granule as G


def show(title: str, prefix_note: str, fn) -> None:
    print(f"\n=== {title} ===")
    print(f"  PRE-FIX (2026-09-03): {prefix_note}")
    try:
        print(f"  NOW    : {fn()}")
    except G.GranuleError as e:
        print(f"  NOW    : GranuleError — {str(e).split('.')[0]}.")
    except Exception as e:  # pragma: no cover - probe
        print(f"  NOW    : {type(e).__name__}: {e}")


# ---------------------------------------------------------------- A: undeclared granule axis
class Silent(G.ExpertContainer, BaseOP):
    """A container that FORGOT `self._num_experts = num_experts`."""

    def __init__(self):
        self.weight = torch.arange(4 * 8 * 16, dtype=torch.float32).reshape(4, 8, 16)
        self.weight_scale = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8, 1)


def a():
    s = G.spec_for_container(Silent())
    return f"num_experts={s.num_experts} granule_bytes={s.granule_bytes} (true one-expert: 544)"


show(
    "A: undeclared granule axis silently read as DENSE",
    "num_experts=None, granule_bytes=2176 (4x the truth); expert_slice(c,3) returned the WHOLE "
    "stack and expert_slice(c,999) was ACCEPTED with no IndexError",
    a,
)


# ---------------------------------------------------------------- B: dense spec ignores `e`
class Dense(G.ExpertContainer, BaseOP):
    _granule_dense = True

    def __init__(self):
        self.weight = torch.zeros(4, 8)


def b():
    d = Dense()
    return {k: tuple(v.shape) for k, v in d.granule_spec().expert_slice(d, 12).items()}


show(
    "B: expert_slice(e) on a DENSE spec ignored e",
    "expert_slice(d, 12) -> {'weight': (4, 8)} — every index returned the whole container",
    b,
)


# ---------------------------------------------------------------- C: strided alias merged
class Strided(G.ExpertContainer, BaseOP):
    def __init__(self):
        self.weight = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4).contiguous()
        self._w_t = self.weight.transpose(1, 2)  # same storage, offset AND numel -> same byte key
        self._num_experts = 4


def c():
    s = G.spec_for_container(Strided())
    return [(x.name, x.aliases) for x in s.components]


show(
    "C: a transposed view merged in as an ALIAS of its contiguous base",
    "components=[('weight', ('_w_t',), (4,4))] — one component, two names, two DIFFERENT readings "
    "of the same storage; the bake rebinds `_w_t` to the row-major arena row and the transpose is "
    "gone (right shapes, wrong numbers)",
    c,
)


# ---------------------------------------------------------------- C2: dense arm, no check at all
class DenseStrided(G.ExpertContainer, BaseOP):
    _granule_dense = True

    def __init__(self):
        self.weight = torch.zeros(8, 16).transpose(0, 1)


def c2():
    return [(x.name, tuple(x.shape)) for x in DenseStrided().granule_spec().components]


show(
    "C2: the DENSE arm had no contiguity check at all",
    "ACCEPTED: [('weight', (16, 8))]",
    c2,
)


# ---------------------------------------------------------------- C3: understated span hides overlap
class DenseOverlap(G.ExpertContainer, BaseOP):
    _granule_dense = True

    def __init__(self):
        base = torch.arange(4 * 12, dtype=torch.float32).reshape(4, 12).contiguous()
        self.a = base[:, :4]                            # strided; numel*itemsize says 64 B
        self.b = base.reshape(-1)[24:36].reshape(3, 4)  # contiguous, [96,144) B — inside a's reach


def c3():
    return [(x.name, tuple(x.shape)) for x in DenseOverlap().granule_spec().components]


show(
    "C3: partial overlap hidden because the byte span is numel*itemsize",
    "ACCEPTED: [('a', (4,4)), ('b', (3,4))] — the overlap scan compared [0,64) vs [96,144) and "
    "called them disjoint, while `a` really reads out to 160 B",
    c3,
)


# ---------------------------------------------------------------- D: decode policy
def d():
    fields = G.GranuleSpec.__dataclass_fields__
    return "GranuleSpec.policy present" if "policy" in fields else "GranuleSpec has NO policy field"


show(
    "D: the compressed-tensors sign convention was invisible to the descriptor",
    "GranuleSpec had NO policy field — w13 XORed to uint4b8 and w2 left two's-complement would "
    "produce identical component sets, identical byte totals and an identical fingerprint",
    d,
)
