"""The prefill expert-staging slab: copy fidelity, alias preservation, and the refusals.

WHY. `weights/prefill_stage.py` substitutes the containers the MoE kernels read — it hands them
views into one shared device slab instead of the pinned host arena. Everything that can go wrong
with that is a WRONG-WEIGHTS bug with no crash, so each property is pinned separately:

  * the twin's tensors must carry the host tensors' VALUES after a refill (a slab that is allocated
    and never filled reads as zeros, which on a MoE is plausible-looking degraded output, not an
    error);
  * they must be DISTINCT STORAGE from the host originals, or the "staging" did nothing and the
    measurement that follows would be host-reads-vs-host-reads;
  * ALIASES must stay aliases. `_GroupedFP8Experts` exposes one storage under two names; a twin that
    gave them two slab views would have one of them go stale the moment the other is written, and
    the granule fingerprint machinery exists precisely because that class of de-aliasing is
    invisible to shape checks;
  * a refill must be REPEATABLE, because one slab serves every layer in turn — layer L+1 overwrites
    layer L, and a twin that cached its first fill would serve L's weights for L+1;
  * the refusals (nested/subscripted names, non-contiguous storage) must return None and leave the
    caller on the in-place path, never raise and never half-stage.

Runs on CPU — the slab is `torch.empty(..., device=...)` and every property above is device-agnostic,
so nothing here needs a GPU or the custom kernels. The end-to-end MoE parity (staged vs in-place
logits on a real checkpoint) is a separate GPU gate and is NOT claimed by this file.

Run:  PYTHONPATH=python python3 tests/moe_prefill_stage_test.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from minisgl.distributed import set_tp_info, try_get_tp_info  # noqa: E402

if try_get_tp_info() is None:
    set_tp_info(0, 1)   # the refusal paths log through *_rank0, which needs it

from minisgl.weights import prefill_stage  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


E, N, K = 4, 16, 32
DEV = torch.device("cpu")


class Experts:
    """Stands in for a grouped-expert container: stacked (E, ...) tensors as plain attributes.

    `e` is parameterised because the granule spec checks dim 0 against the seam's declared
    `num_experts` — a fixture that disagrees is rejected before any staging logic runs, which is
    correct of the spec and merely a fixture bug here."""

    def __init__(self, alias: bool = False, noncontig: bool = False,
                 e: int = E, n: int = N, k: int = K) -> None:
        self.weight = torch.randn(e, n, k)
        self.weight_scale = torch.randn(e, n)
        if noncontig:
            self.weight = torch.randn(e, k, n).transpose(1, 2)   # same shape, not contiguous
        if alias:
            self._w_op = self.weight                              # one storage, two names


class Seam:
    """The two attributes `prefill_stage` reads off a seam."""

    def __init__(self, path: str = "model.layers.0.mlp.experts") -> None:
        self.path = path
        self.num_experts = E


def tensors_of(c):
    """name -> tensor for EVERY name, aliases included (that is what the twin must rebind)."""
    out = {}
    for names, t in prefill_stage.container_bindings(c, E):
        for n in names:
            out[n] = t
    return out


print("component discovery")
w13, w2 = Experts(), Experts()
names13 = sorted(tensors_of(w13))
check("container tensors discovered", set(names13) >= {"weight", "weight_scale"}, f"{names13}")
if FAILED:
    print("\ncannot proceed: the granule walker did not describe the fixture container.")
    raise SystemExit(1)

per_layer = sum((t.numel() * t.element_size() + 255) // 256 * 256
                for c in (w13, w2) for t in tensors_of(c).values())

print()
print("staging: values, distinctness, repeatability")
st = prefill_stage.PrefillStager(per_layer * 2, DEV)
seam = Seam()
twin = st.build_twin(seam, w13, w2)
check("twin built", twin is not None)
if twin is None:
    raise SystemExit(1)

twin.refill()
for label, orig, made in (("w13", w13, twin.w13), ("w2", w2, twin.w2)):
    o, m = tensors_of(orig), tensors_of(made)
    for n in o:
        check(f"{label}.{n}: staged VALUES equal the host tensor", torch.equal(m[n], o[n]))
        check(f"{label}.{n}: staged storage is DISTINCT from the host tensor",
              m[n].data_ptr() != o[n].data_ptr(),
              "same pointer => nothing was staged")

# A second fill must track a changed source: the slab is reused by every layer in turn.
w13.weight.copy_(torch.randn(E, N, K))
twin.refill()
check("refill tracks a changed source (slab is reused across layers)",
      torch.equal(tensors_of(twin.w13)["weight"], w13.weight))

print()
print("alias preservation")
aw13, aw2 = Experts(alias=True), Experts()
at = prefill_stage.PrefillStager(per_layer * 3, DEV).build_twin(Seam(), aw13, aw2)
check("aliased container still stageable", at is not None)
if at is not None:
    at.refill()
    check("the two alias names share ONE slab view",
          at.w13.weight.data_ptr() == at.w13._w_op.data_ptr(),
          "de-aliased: one name would go stale when the other is written")
    check("aliased storage is carved ONCE (not double-counted)",
          at.nbytes <= twin.nbytes + 256,
          f"aliased twin {at.nbytes} B vs plain {twin.nbytes} B")

print()
print("refusals fall back through resolve(), never raise and never half-stage")
# Through `resolve`, not `build_twin`: `resolve` is what the forward calls, and it is the layer that
# has to turn ANY staging failure into "keep the in-place host read". A container the granule spec
# itself rejects (strided storage) raises GranuleError inside the walk — production must absorb it.
st2 = prefill_stage.PrefillStager(per_layer * 2, DEV)
nc_seam = Seam("model.layers.9.mlp.experts")
check("non-contiguous component => resolve() returns None (caller keeps the in-place read)",
      st2.resolve(nc_seam, Experts(noncontig=True), Experts()) is None)
check("the refusal is REMEMBERED, not retried every launch",
      getattr(nc_seam, "_prefill_twin", "missing") is None)

tiny_seam = Seam("model.layers.7.mlp.experts")
tiny = prefill_stage.PrefillStager(256, DEV)
check("a slab too small refuses rather than overlapping two layers onto one region",
      tiny.resolve(tiny_seam, Experts(), Experts()) is None)

print()
print("resolve() on a good seam stages and counts")
ok_seam = Seam("model.layers.3.mlp.experts")
gw13, gw2 = Experts(), Experts()
got = st2.resolve(ok_seam, gw13, gw2)
check("resolve returns a staged pair", got is not None)
if got is not None:
    check("staged pair carries the host values",
          torch.equal(tensors_of(got[0])["weight"], gw13.weight))
    check("staged pair is distinct storage",
          tensors_of(got[0])["weight"].data_ptr() != gw13.weight.data_ptr())
    check("the launch was counted (provenance for the ledger)", st2.staged_launches == 1,
          f"staged_launches={st2.staged_launches}")

print()
print("THE GATE: decode must not stage, a sweep must")
# Called unbound on a duck-typed seam so the gate is tested on its own, without a model. These are
# the only attributes `_stage_for_prefill` reads, and getting this wrong in either direction is the
# whole risk of the change: staging a decode would copy 0.66 GiB to read top_k of 512 experts, and
# failing to stage a prefill leaves the 4.0 tok/s defect in place.
from minisgl.weights.moe_interpose import MoEWeightSeam  # noqa: E402
from minisgl.weights.stacks import ExpertStackTable, StackKind  # noqa: E402


class FakeSeam:
    def __init__(self, kind=StackKind.HOST, top_k_local=8, num_experts=512, bound=True):
        self.path = "model.layers.1.mlp.experts"
        self._bound = bound
        self._table = ExpertStackTable.uniform(num_experts, kind)
        self.top_k_local = top_k_local
        self.num_experts = num_experts


# A container whose dim 0 matches the 512-expert FakeSeam, kept tiny in the other dims.
GE, GN, GK = 512, 2, 2
gate_bytes = 4 * sum((t.numel() * t.element_size() + 255) // 256 * 256
                     for t in (torch.empty(GE, GN, GK), torch.empty(GE, GN)))
prefill_stage.reset()
prefill_stage.install(gate_bytes, DEV)
gate = MoEWeightSeam._stage_for_prefill


def staged(seam, n):
    e = seam.num_experts
    return gate(seam, Experts(e=e, n=GN, k=GK), Experts(e=e, n=GN, k=GK), n) is not None


check("decode (1 token, top_k 8 of 512) does NOT stage", not staged(FakeSeam(), 1))
check("small extend (8 tokens) does NOT stage", not staged(FakeSeam(), 8))
check("just below the sweep threshold (63 tokens) does NOT stage", not staged(FakeSeam(), 63))
check("at the sweep threshold (64 tokens = 512/8) DOES stage", staged(FakeSeam(), 64))
check("a full prefill chunk (2048 tokens) DOES stage", staged(FakeSeam(), 2048))
check("num_tokens=None (boot proof / non-forward caller) never stages",
      not staged(FakeSeam(), None))
check("a DEVICE-tier layer never stages (already in VRAM)",
      not staged(FakeSeam(kind=StackKind.DEVICE), 2048))
check("an UNBOUND seam never stages", not staged(FakeSeam(bound=False), 2048))
# top_k_local, not the global top_k: under EP this rank computes fewer routed slots per token, so
# the sweep threshold moves out. A gate on the global value would stage launches that do not sweep.
check("EP-local top_k moves the threshold out (top_k_local=1 => 512 tokens needed)",
      not staged(FakeSeam(top_k_local=1), 256) and staged(FakeSeam(top_k_local=1), 512))
prefill_stage.reset()

print()
print("install/get/reset")
prefill_stage.reset()
check("no stager installed by default", prefill_stage.get() is None)
check("install(0) stays disabled", prefill_stage.install(0, DEV) is None)
got = prefill_stage.install(per_layer * 2, DEV)
check("install(n) returns a live stager", got is not None and prefill_stage.get() is got)
check("slab is exactly the reserved size", got is not None and got.slab_bytes == per_layer * 2)
prefill_stage.reset()

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    raise SystemExit(1)
print("ALL CHECKS PASS")
