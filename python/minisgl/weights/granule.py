"""The format-agnostic definition of "one expert": which tensors must travel together.

WHY THIS EXISTS. Weight offload moves an expert's bytes between media. A granule is therefore not
"the weight" — it is **one expert's slice of EVERY tensor in the container**: the packed weight, its
per-group scales, and its zero-points. Move the weight without the scale and the kernel dequantizes
expert `e` against expert `e'`'s scale. Shapes stay right, no kernel faults, nothing asserts, and the
model emits plausible text. That silent-corruption class is the entire reason this module exists, and
it is why the descriptor is DERIVED from the live container rather than written down per format:
which buffers survive `post_load` is a function of `format x six env knobs` (`MINISGL_MOE_W4A16`,
`MINISGL_MOE_MXFP4_REGDIRECT`, `MINISGL_MOE_W8A8_REGDIRECT`,
`MINISGL_ZAYA_OLDMOE`, `MINISGL_ZAYA_W8A16`), so any hand-maintained per-format table is wrong the
first time somebody flips a knob — and wrong SILENTLY, by omitting a scale.

THE DESCRIPTOR IS `FrameLayout`, THE PACKING IS NOT.  `FrameLayout`/`FrameComponent`
(`kvcache/host_arena.py`) already encode "these components form one indivisible frame", which is
exactly the enforcement needed here, so they are reused as the granule DESCRIPTOR. They are
deliberately NOT reused as the physical packing. The MoE kernels do per-component implicit-contiguous
pointer arithmetic — `w_base = w_rep + e*(N/16)*ktiles*32`, `ws_e = w_scales + e*G*N`
(`w4a8_fp8_wmma_kernel.hip:1676-1680`), `wq_e = w_fp8 + (long)e*N*K` — i.e. each component's row
stride is ITS OWN row size. A frame-major slab (expert 0's weight, expert 0's scales, expert 1's
weight, ...) would give every component a row stride of `frame_bytes`, and each kernel would read a
wrong-but-well-formed byte range. So `plan_component_major` is the packing, `GranuleSpec.layout` is
only the manifest, and `tests/core/test_granule_placement.py` asserts the two DISAGREE (a guard that
could pass while frame-major happened to coincide would be vacuous).

CLASSIFY BY THE KERNELS' OWN PRECONDITION. A tensor is per-expert iff
`t.dim() >= 1 and t.shape[0] == E and t.is_contiguous()`. That predicate IS what makes `t[e]` a flat
range at `base + e*(numel//E)*itemsize`, so the derivation and the kernels' correctness cannot drift
apart: anything this module calls a granule component is exactly what the kernel indexes.

FAIL CLOSED. Every tensor reachable from the container must be either per-expert, or provably
expert-invariant, or explicitly declared `_residency_shared`. Anything else RAISES at derivation with
the offending attribute path — because the failure mode of a quiet omission is not a crash, it is a
dropped scale.

FAIL CLOSED ON THE GRANULE AXIS TOO. "How many granules is this container?" is a DECLARATION, never a
default. `num_experts=None` (one granule for the whole container) is the dense answer and it is a
legal, useful answer — which is exactly why it must never be what an *unanswered* question resolves
to. A MoE container silently treated as one granule keeps working: `stacked_tensors` returns the same
tensors, every byte total is still right, and nothing raises. What changes is that `expert_slice(c, 7)`
hands back the WHOLE stack and the residency layer prices a 512-expert container as one indivisible
unit — a wrong-bytes bug with no crash, the same family as the dropped scale. So `spec_for_container`
asks the container (`_num_experts` for MoE, `_granule_dense = True` for whole-container) and RAISES
when neither is declared, rather than falling back to the dense reading.

WALK RAW `__dict__`, INCLUDING UNDERSCORE NAMES. `BaseOP.state_dict` / `load_state_dict` /
`post_load` all skip `_`-prefixed attributes (`layers/base.py:59,:76,:101`) and every quantized
`post_load` DELETES the public checkpoint names once it has built `_w_op`/`_scales_op`/`_zeros_op`.
A `state_dict`-driven walk therefore finds ZERO expert weights on a loaded quantized model.

DEDUPE BY BYTE RANGE, NOT BY (shape, stride). `_GroupedFP8Experts.post_load` sets
`_w_op = weight.contiguous().view(torch.uint8)` — the same bytes under a different dtype and shape —
and under `MINISGL_ZAYA_OLDMOE=1` does NOT delete `weight`. Keying on shape/stride would call those
two components and double the granule; worse, a copy-based rebind would then DE-alias them so
`dequant()` and the kernel read different memory. Identical byte range == one component with several
names; a PARTIAL overlap is unrepresentable in component-major placement and raises.

...WHICH IS ONLY SOUND BECAUSE EVERY TENSOR IS CONTIGUOUS, SO CONTIGUITY IS CHECKED FIRST. The byte
key is `(storage, storage_offset, numel*itemsize)`, and `numel*itemsize` is a tensor's true byte span
only when it is contiguous. Checking contiguity late — inside the per-expert branch, as this did
until 2026-09-03 — leaves two silent holes, and both are the wrong-numbers class this module exists
to close:

  * `t` and `t.transpose(1, 2)` share a storage, a storage_offset AND a numel, so they produce the
    SAME key and the strided view is merged in as an *alias* of the contiguous one. Nothing then
    distinguishes them: `moe_interpose._reinterpret` sees matching dtype+shape, hands the alias the
    row-major arena row, and the transpose is simply gone. Right shapes, right dtype, wrong numbers,
    no error anywhere — and the walk had already reported the container as fully described.
  * a strided view's real byte reach EXCEEDS `numel*itemsize`, so the partial-overlap scan below
    compares understated spans and can call two genuinely overlapping tensors disjoint.

The dense arm (`num_experts is None`) had no contiguity check at all, so both holes were open there
unconditionally. The check is therefore in the WALK, before the key is computed, and applies to every
tensor the walk reaches on either arm. There is no component-major placement for a strided tensor
under any granule axis, so this is never a tolerable state — `post_load` must `.contiguous()` it.

THE CLASSIFICATION MUST NOT DEPEND ON THIS RANK'S SHARD CONTENTS (TP determinism). Every rank
derives its own spec with NO collective, and `placement.LayerWeights` turns `granule_bytes` /
`fingerprint()` straight into that rank's byte budget — `placement.py:24` and
`host_capacity.py:178` both spell out the consequence of a disagreement ("ranks place granules at
different offsets and the collectives hang"). Under TP the two ranks hold DIFFERENT SHARDS of the
same tensors (w13 is column-split, w2 row-split) and under EP different EXPERTS, so any decision
read out of the bytes is a decision the two ranks can make differently. Dropping a component
because *this rank's* slice happens to be row-identical is exactly such a decision. So:

  * a tensor leaves the granule ONLY when the container DECLARES it in `_residency_shared` — a
    property of the config (`quant.sym`), which every rank shares;
  * the declaration is then VERIFIED bitwise against the live rows and a stale one RAISES (an
    unverified declaration silently drops a real per-expert scale, the very bug this module
    exists to prevent, and the pre-2026-09-03 code trusted declarations blindly);
  * an UNdeclared tensor whose rows happen to match is kept as a full component and merely
    reported in `GranuleSpec.content_invariant`, which is deliberately NOT hashed into
    `fingerprint()`. Over-counting bytes is safe and rank-identical; under-counting on one rank
    only is not.

NOTHING HERE MAY RUN UNDER GRAPH CAPTURE. The verification compares device tensors with
`torch.equal`, which launches a kernel and synchronizes the host — illegal mid-capture and a
per-step stall on an eager decode path. `derive_granule_spec` / `expert_slice` / `stacked_tensors`
therefore refuse outright while the current stream is capturing (`_assert_not_capturing`). This
module creates no stream and issues no side-stream op, so there is nothing here that could be
recorded into a graph; it is boot-time descriptor code and the guard keeps it that way.

ADDRESSES ARE NOT PART OF `fingerprint()`. A captured graph bakes the pointers it saw at capture
time, so a rebind after capture (arena bind, a second bake pass) leaves replay reading the OLD
memory — plausible text, no crash. `binding_fingerprint()` + `assert_bindings_unchanged()` are the
address-level counterpart the capture path is expected to snapshot before capture and re-assert
after; `assert_spec_still_holds()` additionally catches a rebind that DE-ALIASED two names that
used to share bytes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterator, List, Mapping, Sequence, Tuple

import torch
from minisgl.kvcache.host_arena import FrameComponent, FrameLayout

# `BaseOP` is imported LAZILY, at walk time, never at module scope. `minisgl.layers.__init__` imports
# `MoELayer`, and `layers/moe.py` imports this module — so a module-scope `from minisgl.layers.base
# import BaseOP` here is a genuine import cycle (it resolves to a half-initialised `granule`). The
# walk needs the class only to decide whether to descend, which happens long after both packages are
# loaded, so a cached lazy lookup costs nothing and keeps this module's import graph to torch +
# `kvcache.host_arena`.
_BASEOP: type | None = None


def _baseop_cls() -> type:
    global _BASEOP
    if _BASEOP is None:
        from minisgl.layers.base import BaseOP

        _BASEOP = BaseOP
    return _BASEOP


# Placement alignment for the component-major plan. 256 B matches `host_arena._ALIGN`: DMA-friendly
# and above every dtype's alignment requirement, so any component view reinterpreted out of a uint8
# slab is safely aligned. It pads only BETWEEN components (once per component), never between expert
# rows — padding rows would change the stride the kernels compute and is the frame-major bug again.
DEFAULT_ALIGN = 256


# =====================================================================================
# Descriptor types
# =====================================================================================


@dataclass(frozen=True)
class ExpertComponent(FrameComponent):
    """One tensor that must travel with an expert.

    Inherits the `FrameComponent` contract verbatim — `name`, `dtype`, `shape`, `.nbytes` — with
    `shape` deliberately being the PER-EXPERT slice shape (dim 0 dropped). So `.nbytes` is one
    expert's byte cost of this component, i.e. the component's ROW STRIDE, which is precisely the
    number the kernels' pointer arithmetic uses.
    """

    # Other attribute names bound to the exact same bytes (e.g. `weight` and `_w_op` on a ZAYA fp8
    # container under MINISGL_ZAYA_OLDMOE=1). Sorted, so the value is rank-deterministic.
    aliases: Tuple[str, ...] = ()
    # The full stacked shape, `(E, *shape)`. Kept so a caller can rebuild a view without the container.
    stacked_shape: Tuple[int, ...] = ()

    @property
    def row_bytes(self) -> int:
        """Alias for `.nbytes`, named for the placement arithmetic that consumes it."""
        return self.nbytes

    @property
    def names(self) -> Tuple[str, ...]:
        return (self.name,) + self.aliases


@dataclass(frozen=True)
class ReplicatedTensor:
    """A tensor in the container that is NOT part of the granule, and why.

    Two legitimate reasons, and no third — and BOTH require the container to have DECLARED the name
    in `_residency_shared`, because the decision has to be a function of the config (which every TP
    rank shares) and not of this rank's shard contents (which it does not):
      * `expert-invariant` — declared AND carrying the expert axis, so the declaration is checked:
        every one of the E rows is bitwise identical to row 0 (symmetric compressed-tensors
        `_zeros_op` is E copies of `0x88`, ~3 % of w13). A declaration that does NOT hold raises.
      * `declared-shared` — declared and carrying no expert axis at all, so there is nothing to
        verify: it cannot be a per-expert buffer.
    Anything else raises rather than landing here; see `derive_granule_spec`.
    """

    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...]  # the FULL shape, not a per-expert slice
    reason: str
    nbytes: int
    aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class GranuleSpec:
    """What one expert of one container is, and what the container holds besides.

    `num_experts is None` means "the whole container is ONE granule" — the dense case
    (`_LinearTPImpl` and friends). Everything below then degenerates cleanly: every tensor is a
    component whose `shape` is its full shape, `granule_bytes == stacked_bytes`, and the
    component-major plan is a plain concatenation. Dense and MoE share one mechanism, which is why
    dense offload is a merge gate and not a follow-on.
    """

    kind: str
    num_experts: int | None
    components: Tuple[ExpertComponent, ...]
    replicated: Tuple[ReplicatedTensor, ...] = ()
    # True when derived from meta-device tensors: shapes/dtypes are trustworthy, aliasing and
    # expert-invariance are NOT (every meta tensor reports data_ptr()==0, so dedupe is impossible).
    meta: bool = False
    # DIAGNOSTIC ONLY, and deliberately excluded from `fingerprint()` and from every byte total:
    # undeclared components whose rows 0 and 1 happened to match on THIS rank's shard. Acting on it
    # would make the granule content-dependent, and the two TP ranks hold different shards — see the
    # module docstring's TP-determinism section. It exists so a developer can see "you could declare
    # `_residency_shared = (name,)` in post_load and save these bytes on every rank at once".
    content_invariant: Tuple[str, ...] = ()
    # The container's DECODE POLICY: `(dotted_path, repr(value))` for every path it declared in
    # `_granule_policy`, sorted. Not tensors and not bytes — decisions `post_load` made about how the
    # bytes are to be READ, which the tensor walk cannot see and which are exactly as load-bearing as
    # a scale. See `_policy_values` and `assert_granule_pair_consistent`.
    policy: Tuple[Tuple[str, str], ...] = ()

    # -- byte accounting ------------------------------------------------------

    @property
    def granule_bytes(self) -> int:
        """One expert's bytes across every component. UNPADDED — this is a capacity number, and the
        placement plan's inter-component padding must not inflate it."""
        return sum(c.nbytes for c in self.components)

    @property
    def num_granules(self) -> int:
        return 1 if self.num_experts is None else self.num_experts

    @property
    def stacked_bytes(self) -> int:
        return self.granule_bytes * self.num_granules

    @property
    def replicated_bytes(self) -> int:
        return sum(r.nbytes for r in self.replicated)

    @property
    def total_bytes(self) -> int:
        return self.stacked_bytes + self.replicated_bytes

    # -- descriptor -----------------------------------------------------------

    @property
    def layout(self) -> FrameLayout:
        """The granule DESCRIPTOR — the enforced manifest of what travels together.

        NEVER the physical packing: `FrameLayout` is frame-major by construction, and a frame-major
        weight slab gives every component the wrong row stride (see the module docstring). Use
        `plan_component_major(spec)` for placement. This property exists so the manifest can be
        printed, hashed and compared, and so a future host-frame consumer that genuinely wants one
        contiguous frame (a debug dump, a granule checksum) has the same geometry both sides.
        """
        return FrameLayout(self.components)

    def fingerprint(self) -> str:
        """Stable hash of the manifest — kind, expert count, and every component's name/dtype/shape.

        Two ranks running the same config must produce the same string. A mismatch is a
        format/knob divergence (one rank repacked to `_w_rep`, the other did not), which under
        layer-granular placement would give the two ranks different byte budgets and different
        placement — the exact shape of the TP-divergence class this project refuses to build.

        CONTENT-INDEPENDENT BY CONSTRUCTION, and that is a requirement rather than a nicety: the two
        ranks hold different SHARDS (w13 column-split, w2 row-split) and under EP different EXPERTS,
        so anything hashed here that is read out of the bytes could differ between them with no
        collective to catch it. Hence `content_invariant` is not hashed, and the replicated set is
        declaration-derived. Addresses are not hashed either — see `binding_fingerprint`.
        """
        h = hashlib.sha256()
        h.update(f"{self.kind}|n={self.num_experts}|".encode())
        for c in self.components:
            h.update(f"C:{c.name}:{c.dtype}:{tuple(c.shape)}:{c.aliases}|".encode())
        for r in self.replicated:
            h.update(f"R:{r.name}:{r.dtype}:{tuple(r.shape)}:{r.reason}|".encode())
        # The decode policy IS hashed: it is config-derived (the checkpoint's packing convention),
        # so it is the same on every rank, and two ranks that decoded the same stack differently is
        # precisely the divergence a fingerprint exists to surface. `_policy_values` is responsible
        # for keeping the recorded value a DECISION and not a per-shard diagnostic.
        for name, value in self.policy:
            h.update(f"P:{name}={value}|".encode())
        return h.hexdigest()[:16]

    def describe(self) -> str:
        comps = " ".join(f"{c.name}{tuple(c.shape)}:{_dtype_tag(c.dtype)}" for c in self.components)
        rep = " ".join(f"{r.name}({r.reason})" for r in self.replicated)
        # The hint is printed but never priced: it is this rank's observation about its own shard.
        hint = " ".join(self.content_invariant)
        return (
            f"{self.kind} n={self.num_experts} granule={self.granule_bytes}B "
            f"total={self.total_bytes / (1 << 20):.1f}MiB fp={self.fingerprint()} "
            f"[{comps}]"
            + (f" shared[{rep}]" if rep else "")
            + (f" rows-match-on-this-rank[{hint}]" if hint else "")
        )

    # -- views ----------------------------------------------------------------

    def expert_slice(self, container: Any, e: int) -> Dict[str, torch.Tensor]:
        """The live per-expert views for expert `e`, keyed by component name.

        Component-major by construction: each value is `t[e]` of that component's OWN stacked
        tensor, so its address is `base(c) + e*c.row_bytes` — the kernels' arithmetic, unchanged.

        BOOT-TIME ONLY. It resolves attribute paths and builds fresh views, so the tensors it
        returns are pointers valid NOW; a graph captured around them bakes those pointers, and a
        later rebind leaves replay reading the old memory. `_assert_not_capturing` makes the
        captured case loud; the eager-decode misuse is the caller's to avoid.
        """
        _assert_not_capturing(f"{self.kind}.expert_slice")
        n = self.num_experts
        if n is None:
            # Dense: ONE granule. `e != 0` is a caller that believes this container is stacked, and
            # silently handing it the whole container back is how a mis-declared granule axis stays
            # invisible (see `declared_granule_count`) — so it is an IndexError like any other.
            if e != 0:
                raise IndexError(
                    f"granule {e} out of range for {self.kind}: it is a DENSE container "
                    f"(num_experts is None), i.e. exactly one granule."
                )
            return {c.name: _lookup(container, c.name) for c in self.components}
        if not 0 <= e < n:
            raise IndexError(f"expert {e} out of range for {self.kind} with n={n}")
        return {c.name: _lookup(container, c.name)[e] for c in self.components}

    def stacked_tensors(self, container: Any) -> Dict[str, torch.Tensor]:
        """The stacked (dim-0 == E) tensor behind every component, keyed by component name."""
        _assert_not_capturing(f"{self.kind}.stacked_tensors")
        return {c.name: _lookup(container, c.name) for c in self.components}

    # -- bindings (WHERE the components live right now) ------------------------

    def binding_fingerprint(self, container: Any) -> str:
        """Hash of the ADDRESSES this spec currently resolves to, aliases included.

        `fingerprint()` deliberately says nothing about memory, so two specs can agree while the
        containers point at different buffers. That distinction is the whole graph-capture hazard:
        a captured HIP graph bakes the device pointers it saw, so any rebind after capture (an
        arena bind, a second bake pass, a `copy_`-then-reassign) leaves replay reading the OLD
        allocation — right shapes, freed or stale bytes, plausible text, no crash.

        Every alias is resolved SEPARATELY, so a rebind that de-aliased `weight` from `_w_op`
        (`_GroupedFP8Experts` under `MINISGL_ZAYA_OLDMOE=1`) moves this hash even though both names
        still exist with the right shape.

        WITHIN ONE RANK, ACROSS TIME — never across ranks. It hashes raw device pointers, so two
        ranks ALWAYS disagree (they allocated separately) and comparing it between them would
        report a permanent, meaningless failure. `fingerprint()` is the cross-rank token; this is
        the before-capture/after-bind token.
        """
        h = hashlib.sha256()
        h.update(f"{self.kind}|n={self.num_experts}|".encode())
        for c in self.components:
            for name in c.names:
                t = _lookup(container, name)
                ptr, off, ln = (0, 0, t.numel() * t.element_size()) if t.is_meta else _byte_key(t)
                h.update(f"B:{name}:{t.device}:{ptr}:{off}:{ln}:{t.dtype}|".encode())
        return h.hexdigest()[:16]


# =====================================================================================
# Component-major placement
# =====================================================================================


@dataclass(frozen=True)
class ComponentPlacement:
    name: str
    offset: int  # byte offset of this component's slab inside the container region
    row_bytes: int  # stride between consecutive experts WITHIN this component
    num_rows: int
    dtype: torch.dtype
    row_shape: Tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return self.row_bytes * self.num_rows

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(frozen=True)
class ContainerPlacement:
    """COMPONENT-MAJOR byte layout for one container's expert stack.

    Component `c` owns one contiguous `[offset, offset + E*row_bytes)` slab, so expert `e`'s slice of
    `c` is at `offset + e*row_bytes` — literally `base(c) + e*row_bytes(c)`, which is what every MoE
    kernel computes from the pointer it is handed. Components are padded to `align` only at their
    START; rows are never padded, because a padded row changes the stride the kernel derives from
    shapes and is therefore a wrong-bytes bug, not a waste of space.

    This is the deliberate opposite of `FrameLayout`, which interleaves components inside one frame.
    """

    kind: str
    num_experts: int | None
    components: Tuple[ComponentPlacement, ...]
    nbytes: int
    align: int

    def _get(self, name: str) -> ComponentPlacement:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(f"{name!r} is not a component of {self.kind} ({[c.name for c in self.components]})")

    def base_offset(self, name: str) -> int:
        return self._get(name).offset

    def row_bytes(self, name: str) -> int:
        return self._get(name).row_bytes

    def row_offset(self, name: str, e: int) -> int:
        c = self._get(name)
        if not 0 <= e < c.num_rows:
            raise IndexError(f"expert {e} out of range for {self.kind}.{name} ({c.num_rows} rows)")
        return c.offset + e * c.row_bytes

    def granule_offsets(self, e: int) -> Dict[str, int]:
        """Every component's byte offset for expert `e` — the scatter list a populate/copy pass walks.

        NOTE the offsets are NOT contiguous with each other: that discontiguity IS component-major
        placement. A caller that wants one `memcpy` per expert is asking for a frame-major slab and
        must not get one.
        """
        return {c.name: self.row_offset(c.name, e) for c in self.components}

    @property
    def payload_bytes(self) -> int:
        return sum(c.nbytes for c in self.components)

    @property
    def pad_bytes(self) -> int:
        return self.nbytes - self.payload_bytes


def plan_component_major(spec: GranuleSpec, *, align: int = DEFAULT_ALIGN) -> ContainerPlacement:
    """Lay a container's expert stack out COMPONENT-MAJOR. Pure integer arithmetic, no tensors.

    Component order follows `spec.components`, which follows the container's `__dict__` insertion
    order — construction order, therefore identical on every TP rank, therefore a placement that two
    ranks can independently compute and agree on without a collective.
    """
    if align <= 0 or (align & (align - 1)) != 0:
        raise ValueError(f"align must be a positive power of two; got {align}")
    n = spec.num_granules
    out: List[ComponentPlacement] = []
    off = 0
    for c in spec.components:
        off = (off + align - 1) // align * align
        out.append(
            ComponentPlacement(
                name=c.name,
                offset=off,
                row_bytes=c.nbytes,
                num_rows=n,
                dtype=c.dtype,
                row_shape=tuple(c.shape),
            )
        )
        off += c.nbytes * n
    return ContainerPlacement(
        kind=spec.kind,
        num_experts=spec.num_experts,
        components=tuple(out),
        nbytes=off,
        align=align,
    )


# =====================================================================================
# The walk
# =====================================================================================


@dataclass
class _Found:
    """One tensor found on the container, with every attribute path that reaches its exact bytes."""

    names: List[str] = field(default_factory=list)
    tensor: torch.Tensor | None = None


def _dtype_tag(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def _lookup(container: Any, dotted: str) -> torch.Tensor:
    """Resolve a path produced by the walk back to its tensor.

    Paths are dotted attribute names, with `name[i]` / `name[key]` segments for tensors found inside
    a list/tuple/dict attribute. The subscript form must round-trip exactly — a component name that
    the walk can emit but this cannot resolve would make `expert_slice`/`stacked_tensors` raise on a
    container the descriptor claimed to describe.
    """
    obj: Any = container
    if isinstance(obj, torch.Tensor):
        # Bare stacked-tensor container (the unquantized MoE case) — see `_iter_tensors`.
        if dotted != _BARE_TENSOR_NAME:
            raise KeyError(f"bare tensor container has only {_BARE_TENSOR_NAME!r}; got {dotted!r}")
        return obj
    for part in dotted.split("."):
        head, _, subs = part.partition("[")
        if head:
            obj = getattr(obj, head)
        while subs:
            key, _, subs = subs.partition("]")
            key = key.lstrip("[")
            try:
                obj = obj[int(key)]
            except ValueError:
                obj = obj[key]
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{dotted!r} did not resolve to a tensor on {type(container).__name__}")
    return obj


# The synthetic component name for a container that IS a bare stacked tensor
# (`_UnquantizedMoEMethod.create_experts` returns `torch.empty(E, out, in)`; there is no object to
# hang attributes on). Deliberately `weight`, so that if the plan's `_GroupedUnquantizedExperts`
# wrapper (§6.1 rule 6) ever lands with `self.weight = t`, the descriptor and every fingerprint
# derived from it stay byte-identical across that change.
_BARE_TENSOR_NAME = "weight"


def _iter_tensors(obj: Any, prefix: str, seen: set[int]) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield `(dotted_name, tensor)` for every tensor reachable from `obj`'s OWN attributes.

    Descends into `BaseOP` and `nn.Module` children and into plain list/tuple/dict values, because
    "fail closed" is only closed over what the walk actually reaches. `nn.Module` subtrees matter in
    practice: `GDNLinearAttn._gdn`, ZAYA's `CCAConv`/`ZayaRouter` and the quantized GDN projections
    inside `_MethodLinear` are invisible to a `BaseOP.__dict__` walk, and on Qwen3.5 those are three
    of every four layers' dense weights.
    """
    if id(obj) in seen:
        return
    seen.add(id(obj))
    descend = (_baseop_cls(), torch.nn.Module)

    if isinstance(obj, torch.Tensor):
        yield (prefix or _BARE_TENSOR_NAME), obj
        return

    def _child(name: str, value: Any) -> Iterator[Tuple[str, torch.Tensor]]:
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, torch.Tensor):
            yield path, value
        elif isinstance(value, descend):
            yield from _iter_tensors(value, path, seen)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                if isinstance(v, (torch.Tensor,) + descend):
                    yield from _child(f"{name}[{i}]", v)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                if isinstance(v, (torch.Tensor,) + descend):
                    yield from _child(f"{name}[{k}]", v)

    if isinstance(obj, torch.nn.Module):
        # Parameters/buffers first (they are NOT in __dict__ for an nn.Module), then this node's own
        # underscore tensors, then children.
        for name, p in obj.named_parameters(recurse=False):
            if p is not None:
                yield (f"{prefix}.{name}" if prefix else name), p.data
        for name, b in obj.named_buffers(recurse=False):
            if b is not None:
                yield (f"{prefix}.{name}" if prefix else name), b
        for name, value in list(obj.__dict__.items()):
            if name in ("_parameters", "_buffers", "_modules", "_non_persistent_buffers_set"):
                continue
            yield from _child(name, value)
        for name, m in obj.named_children():
            yield from _iter_tensors(m, f"{prefix}.{name}" if prefix else name, seen)
        return

    for name, value in list(vars(obj).items()):
        yield from _child(name, value)


def _byte_key(t: torch.Tensor) -> Tuple[int, int, int]:
    """(storage id, byte offset, byte length) — the identity that matters for placement.

    Deliberately NOT (shape, stride, dtype): `_w_op = weight.view(torch.uint8)` has the same bytes
    under a different dtype AND a different shape, and calling those two components would double the
    granule and later de-alias it.

    PRECONDITION: `t` is contiguous. `numel*itemsize` is its real byte span only then, and the
    caller (`derive_granule_spec`) raises before reaching here otherwise — a strided view keys
    identically to the contiguous tensor it is a view of, which merges two genuinely different
    readings of the same storage into one component.
    """
    st = t.untyped_storage()
    off = t.storage_offset() * t.element_size()
    return (st.data_ptr(), off, t.numel() * t.element_size())


# Cap on the temporary a row comparison may materialize. `torch.equal` on two same-shaped tensors
# lowers to an elementwise compare plus a reduction, so comparing the WHOLE stack against a
# broadcast row 0 in one call allocates a bool the size of the stack — tens to hundreds of MB, on
# the device, at BOOT, which is precisely when VRAM is tightest and when the allocator's reserved
# pool is what OOMs (see the "OOM is RESERVED, not allocated" note). Chunked, the temporary is
# bounded by this constant no matter how large the stack is.
_ROWCMP_CHUNK_BYTES = 8 << 20


def _policy_values(container: Any) -> Tuple[Tuple[str, str], ...]:
    """The container's declared DECODE POLICY, as sorted `(dotted_path, repr(value))` pairs.

    WHY A NON-TENSOR THING LIVES IN THE GRANULE DESCRIPTOR. Some `post_load`s decide how the bytes
    are to be READ, not just which bytes there are. Compressed-tensors int4 is the shipped case:
    `ct_packed_sign_convention` samples the nibble histogram and either XORs the whole stack to
    uint4b8 or passes it through, and `_zeros_op` is transformed to match. Get that decision
    different for w13 and w2 of the same layer and one GEMM decodes `q+8` while the other decodes
    two's-complement — every weight of that GEMM off by 8 quanta. Shapes right, dtypes right, no
    kernel fault, plausible text: the same failure family as a left-behind scale, and completely
    invisible to a walk that only sees tensors. So the decision is carried on the spec, hashed into
    `fingerprint()` (two TP ranks must decode identically) and compared across the w13/w2 pair.

    DECLARED, DOTTED, AND REDUCED TO THE DECISION. A container lists the paths it owns in
    `_granule_policy` (e.g. `("_ct_sign.uint4b8",)`) — a dotted path rather than a bare attribute
    precisely so the recorded value is the BOOLEAN DECISION and not the whole `CtSignConvention`,
    whose `margin`/`sampled_words` are sample diagnostics that legitimately differ between w13 and
    w2 and between two ranks' shards. Hashing those would make the fingerprint content-dependent,
    which the module docstring's TP-determinism rule forbids. A missing path records `None` rather
    than raising: `_granule_policy` is also read on a meta container, before any `post_load` has run.
    """
    paths = tuple(getattr(container, "_granule_policy", ()) or ())
    out: List[Tuple[str, str]] = []
    for path in paths:
        obj: Any = container
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        out.append((path, repr(obj)))
    return tuple(sorted(out))


def decode_policy(container: Any) -> Tuple[Tuple[str, str], ...]:
    """Public form of `_policy_values`: the container's declared DECODE POLICY, no tensor walk.

    Exists so a caller that only needs the DECISIONS — `MoELayer.post_load`'s w13/w2 cross-check —
    can have them without deriving a whole `GranuleSpec`. Derivation walks every tensor and runs the
    `_residency_shared` verification (`torch.equal` over the full `_zeros_op` stack, synchronizing),
    which is boot-time-only work that a policy comparison has no business paying for; and, worse,
    routing the cross-check through spec derivation is what made it OFFLOAD-ONLY — `granule_specs()`
    is called by `moe_interpose.attach_seams` and by nothing else, so on a serve without weight
    offload the two GEMMs of a layer could decode with opposite sign conventions and nothing looked.
    """
    return _policy_values(container)


def assert_decode_policy_agrees(
    policies: Mapping[str, Tuple[Tuple[str, str], ...]], *, where: str = ""
) -> None:
    """Every container in one layer must have made the SAME decode decisions.

    The failure this closes is the compressed-tensors sign convention. `w13` and `w2` come out of one
    checkpoint and one quantizer, but `post_load` runs per CONTAINER and each one samples its own
    nibble histogram, so a stack whose sample is atypical (an outlier input group, a shard boundary)
    can resolve the opposite way from its partner. Then one GEMM of the layer is dequantized as
    `q + 8` and the other as two's-complement — every weight of that GEMM off by 8 quanta, right
    shapes, right dtypes, no kernel fault, plausible text. It is the same silent-corruption family as
    a left-behind scale and it is invisible to any check that only looks at tensors.

    Raises rather than picking a side: by the time `post_load` has returned, the XOR has already been
    applied to whichever container chose it, so there is nothing to reconcile — only a boot to fail.
    """
    tag = f" ({where})" if where else ""
    distinct = set(policies.values())
    if len(distinct) <= 1:
        return
    raise GranuleError(
        f"decode policy{tag}: the containers of one layer decided DIFFERENTLY — "
        + "; ".join(f"{k}={dict(v)}" for k, v in sorted(policies.items()))
        + ". They came out of one checkpoint and one quantizer, so a disagreement means one of them "
        "is being read with the wrong convention: for compressed-tensors int4 that is `q+8` against "
        "two's-complement, i.e. every weight of one GEMM off by 8 quanta, with right shapes, no "
        "kernel fault and plausible text. It is a detector that resolved two ways on two stacks, "
        "not something to average — pin the convention from the checkpoint's quantization_config "
        "and pass it into both."
    )


def _rows_as_u8(t: torch.Tensor) -> "torch.Tensor | None":
    """`t` viewed as (E, row_bytes) uint8, or None when that view is impossible.

    uint8, not the native dtype: a NaN payload compares unequal to itself under value equality, and
    a packed int4/E8M0/e4m3 tensor is a bit pattern, not a number.
    """
    if t.dim() < 1 or t.shape[0] < 1 or t.is_meta:
        return None
    try:
        return t.reshape(t.shape[0], -1).view(torch.uint8)
    except (RuntimeError, TypeError):
        return None


def _rows_0_1_equal(t: torch.Tensor) -> bool:
    """The CHEAP reject: are rows 0 and 1 bitwise identical? Two rows, never the whole stack.

    Necessary-not-sufficient on purpose. It is the only thing an UNDECLARED tensor is subjected to,
    because its answer is a diagnostic hint and never changes the granule (module docstring, TP
    determinism); a full-stack read to sharpen a hint would be a boot cost for nothing.
    """
    flat = _rows_as_u8(t)
    if flat is None or flat.shape[0] < 2:
        return False
    return bool(torch.equal(flat[0], flat[1]))


def _all_rows_equal(t: torch.Tensor) -> bool:
    """Is EVERY dim-0 row of `t` bitwise identical to row 0? The full proof, in bounded chunks.

    Run only to VERIFY a `_residency_shared` declaration, where a false accept drops a real
    per-expert buffer. It must therefore cover the last row as well as the first — an early-exit
    over the first chunk only would accept a stack that diverges at expert E-1.

    This is the ONE place a content read can still differ between TP ranks, and that is deliberate:
    the outcome is "the declaration holds" or "boot fails with the attribute path". Two ranks
    disagreeing here means one of them CRASHES, loudly, at boot — never that they serve on with
    different byte budgets, which is what a content-driven EXCLUSION would have produced.
    """
    flat = _rows_as_u8(t)
    if flat is None:
        return False
    n, row = int(flat.shape[0]), int(flat.shape[1]) if flat.dim() > 1 else 0
    if n < 2:
        # A single row cannot CONTRADICT the declaration, and cannot confirm it either. Accepting is
        # right here: the declaration is the config's statement, and E==1 gives us no evidence.
        return True
    if not bool(torch.equal(flat[0], flat[1])):
        return False
    rows_per_chunk = max(1, _ROWCMP_CHUNK_BYTES // max(1, row))
    ref = flat[0].unsqueeze(0)
    for i in range(1, n, rows_per_chunk):
        blk = flat[i : i + rows_per_chunk]
        if not bool(torch.equal(ref.expand_as(blk), blk)):
            return False
    return True


class GranuleError(RuntimeError):
    """A container the granule walk cannot describe safely. Always a boot failure, never a warning."""


def _is_capturing() -> bool:
    """Is the CURRENT stream mid HIP/CUDA-graph capture?

    Never initialises a device context: `torch.cuda.is_initialized()` is a Python flag, and a
    process that has never made a context cannot be capturing. That keeps this free on the
    CPU-only hosts the whole `weights/` package is unit-tested on.
    """
    try:
        if not torch.cuda.is_initialized():
            return False
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # pragma: no cover - a torch without the API is not capturing either
        return False


def _assert_not_capturing(where: str) -> None:
    """Refuse to run descriptor code inside a graph capture.

    Three separate reasons, any one of which is disqualifying:
      * the declaration check compares device tensors with `torch.equal`, which launches a kernel
        AND synchronizes the host — illegal mid-capture, and a per-step stall if a residency layer
        ever calls this from an eager decode instead;
      * the walk resolves attribute paths and builds fresh views, so the pointers it hands back are
        valid only until the next rebind, while a captured graph bakes them forever;
      * a spec is a boot-time artifact by design — placement, budgets and the arena bind are all
        decided before the first capture, so reaching this under capture means a consumer moved
        descriptor work onto the hot path.
    """
    if _is_capturing():
        raise GranuleError(
            f"{where}: called while the current stream is CAPTURING a HIP/CUDA graph. Granule "
            f"derivation launches kernels and synchronizes (the `_residency_shared` check), and "
            f"the views it returns are only valid until the next rebind — a captured graph would "
            f"bake pointers this call cannot promise. Derive the spec ONCE after post_load(), "
            f"before capture, and hold it."
        )


def derive_granule_spec(
    container: Any,
    num_experts: int | None,
    *,
    detect_invariant: bool = True,
    allow_meta: bool = False,
    for_offload: bool = False,
) -> GranuleSpec:
    """Derive the granule descriptor from a LIVE container, after `post_load()`.

    `num_experts=None` -> the whole container is one granule (the dense case).

    Raises `GranuleError` for anything it cannot describe: a tensor that is neither per-expert nor
    provably shared, a non-contiguous per-expert tensor (the kernels' `base + e*row` arithmetic is
    then simply false), a partial storage overlap, or a meta-device container without `allow_meta`.
    Every one of those is a wrong-bytes bug if waved through, so none of them warn.

    BOOT-TIME ONLY: refuses to run under graph capture (`_assert_not_capturing`), and its result is
    a pure-Python artifact that says nothing about addresses (`binding_fingerprint` does).
    """
    kind = type(container).__name__
    if isinstance(container, torch.Tensor):
        kind = f"BareStacked[{_dtype_tag(container.dtype)}]"
    _assert_not_capturing(f"derive_granule_spec({kind})")
    shared_names = tuple(getattr(container, "_residency_shared", ()) or ())

    if for_offload:
        why = offload_refusal(container)
        if why is not None:
            raise GranuleError(f"{kind}: cannot be offloaded — {why}")

    found: Dict[Tuple[int, int, int], _Found] = {}
    order: List[Tuple[int, int, int]] = []
    any_meta = False
    for name, t in _iter_tensors(container, "", set()):
        # CONTIGUITY FIRST, before the byte key exists — on BOTH arms, not just the per-expert one.
        # `_byte_key`'s span is `numel*itemsize`, which is the tensor's real byte reach only when it
        # is contiguous, so a strided view (a) keys IDENTICALLY to the contiguous tensor it is a
        # transpose of and is silently absorbed as an alias of it, and (b) understates its span to
        # the partial-overlap scan below. See the module docstring; both are wrong-numbers-no-crash.
        if not t.is_contiguous():
            raise GranuleError(
                f"{kind}.{name}: tensor is NOT contiguous (shape={tuple(t.shape)}, "
                f"stride={tuple(t.stride())}). A strided tensor has no component-major placement at "
                f"any granule axis — the kernels read granule e at `base + e*(numel//E)*itemsize` "
                f"and that address is simply wrong for it — and it also defeats the alias dedupe, "
                f"because `t` and `t.transpose(...)` share a storage, an offset and a numel and so "
                f"key identically: the strided view would be rebound to the row-major arena row and "
                f"the transpose would vanish, with right shapes and wrong numbers. Make it "
                f"contiguous in post_load (`.contiguous()`), or drop the alias."
            )
        if t.is_meta:
            any_meta = True
        key = (id(t), 0, 0) if t.is_meta else _byte_key(t)
        if key not in found:
            found[key] = _Found(names=[], tensor=t)
            order.append(key)
        found[key].names.append(name)

    if any_meta and not allow_meta:
        raise GranuleError(
            f"{kind}: granule derivation ran on META tensors. Shapes are trustworthy but aliasing "
            f"and expert-invariance are not (every meta tensor reports data_ptr()==0), so the "
            f"granule would be wrong in exactly the silent way this module exists to prevent. "
            f"Call after post_load() on materialized tensors, or pass allow_meta=True for a "
            f"shapes-only sizing estimate."
        )

    # Partial-overlap check: two tensors sharing a storage must either be the SAME byte range (an
    # alias, already merged above) or be disjoint. Anything else cannot be placed component-major.
    if not any_meta:
        by_storage: Dict[int, List[Tuple[int, int, str]]] = {}
        for key in order:
            ptr, off, ln = key
            by_storage.setdefault(ptr, []).append((off, off + ln, found[key].names[0]))
        for ptr, spans in by_storage.items():
            spans.sort()
            for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
                if b0 < a1:
                    raise GranuleError(
                        f"{kind}: {an!r} [{a0},{a1}) and {bn!r} [{b0},{b1}) PARTIALLY overlap in one "
                        f"storage. A partial overlap has no component-major placement — moving one "
                        f"would silently rewrite part of the other."
                    )

    components: List[ExpertComponent] = []
    replicated: List[ReplicatedTensor] = []
    content_invariant: List[str] = []
    for key in order:
        rec = found[key]
        t = rec.tensor
        assert t is not None
        primary, aliases = rec.names[0], tuple(sorted(rec.names[1:]))
        nbytes = t.numel() * t.element_size()
        has_expert_axis = num_experts is not None and t.dim() >= 1 and t.shape[0] == num_experts

        if primary in shared_names or any(a in shared_names for a in aliases):
            # DECLARED SHARED. The declaration is what makes the exclusion rank-safe (it comes from
            # the config, which every rank has), but a declaration is exactly the thing that keeps
            # applying after the checkpoint underneath it changed — a symmetric-CT declaration left
            # in place for an asymmetric checkpoint would drop REAL per-group zero-points from every
            # granule, dequantize each expert against expert 0's zeros, and emit plausible text. So
            # verify it whenever there is something to verify: the tensor carries the expert axis,
            # and it is materialized.
            if has_expert_axis and not t.is_meta and not _all_rows_equal(t):
                raise GranuleError(
                    f"{kind}.{primary}: declared in `_residency_shared` but its {num_experts} "
                    f"expert rows are NOT bitwise identical, so it is a REAL per-expert buffer. "
                    f"Honouring the declaration would move every expert's weights without its own "
                    f"{primary!r} and dequantize expert e against expert 0's — plausible text, no "
                    f"error. Remove {primary!r} from `_residency_shared` (the declaration is stale: "
                    f"the checkpoint or the post_load branch that made it constant has changed)."
                )
            reason = "expert-invariant" if has_expert_axis else "declared-shared"
            replicated.append(
                ReplicatedTensor(primary, t.dtype, tuple(t.shape), reason, nbytes, aliases)
            )
            continue

        if num_experts is None:
            # Dense: the whole container is one granule, so every tensor is a component and its
            # "per-expert" shape is its full shape.
            components.append(
                ExpertComponent(
                    name=primary, dtype=t.dtype, shape=tuple(t.shape),
                    aliases=aliases, stacked_shape=tuple(t.shape),
                )
            )
            continue

        if t.dim() >= 1 and t.shape[0] == num_experts:
            # Contiguity was already enforced for EVERY tensor in the walk above, before the byte
            # key was taken — it has to be, or the dedupe that produced `primary`/`aliases` was
            # itself unsound. Re-checking it here would only be a second copy of a rule with one
            # home; the ONE gate is the walk.
            if detect_invariant and not t.is_meta and _rows_0_1_equal(t):
                # A HINT, never an exclusion. Whether this rank's shard happens to have identical
                # rows is a property of the BYTES, and the two TP ranks hold different shards (w13
                # column-split, w2 row-split) and under EP different experts — so acting on it lets
                # rank 0 drop a component rank 1 keeps, giving them different `granule_bytes`,
                # different `fingerprint()`, different budgets and different placement, with no
                # collective anywhere to notice. It stays a full component on every rank; the name
                # is reported so a developer can make it a DECLARATION, which is rank-safe.
                content_invariant.append(primary)
            components.append(
                ExpertComponent(
                    name=primary, dtype=t.dtype, shape=tuple(t.shape[1:]),
                    aliases=aliases, stacked_shape=tuple(t.shape),
                )
            )
            continue

        raise GranuleError(
            f"{kind}.{primary}: shape {tuple(t.shape)} has no expert axis (expected dim 0 == "
            f"{num_experts}) and is not declared shared. A new format buffer must either carry E on "
            f"dim 0 (so it travels with its expert) or be listed in `_residency_shared` with a "
            f"comment saying why it is expert-invariant. Add "
            f"`_residency_shared = (..., {primary!r})` to {kind} if and only if that is true — "
            f"omitting it here would move weights without their {primary!r} and dequantize one "
            f"expert against another's, which produces plausible text and no error."
        )

    return GranuleSpec(
        kind=kind,
        num_experts=num_experts,
        components=tuple(components),
        replicated=tuple(replicated),
        meta=any_meta,
        content_invariant=tuple(content_invariant),
        policy=_policy_values(container),
    )


class _Unset:
    """Sentinel for "the caller did not answer; ask the container". NOT `None` — `None` is the
    dense answer and must stay distinguishable from an unanswered question."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<granule-axis-unset>"


UNSET = _Unset()


def declared_granule_count(container: Any) -> int | None:
    """What the container DECLARES it is: `int` = that many expert granules, `None` = one granule.

    Raises rather than guessing. The two declarations are deliberately tiny — `self._num_experts =
    num_experts` in a MoE container's `__init__`, or `_granule_dense = True` on a whole-container
    one — and a format that makes neither is a format nobody has thought about the granule axis for.
    Defaulting such a container to the dense reading is the failure this function exists to stop: it
    produces a spec that passes every self-consistency check, returns the right tensors, sums to the
    right bytes, and hands `expert_slice(c, e)` the entire stack for every `e`.
    """
    if isinstance(container, torch.Tensor):
        raise GranuleError(
            "a bare stacked tensor carries no granule declaration (there is no object to hang one "
            "on), so the count must be passed explicitly: `spec_for_container(t, E)` for the "
            "unquantized MoE stack, or `spec_for_container(t, None)` for a single dense tensor."
        )
    if getattr(container, "_granule_dense", False):
        return None
    n = getattr(container, "_num_experts", None)
    if isinstance(n, int) and not isinstance(n, bool) and n > 0:
        return n
    raise GranuleError(
        f"{type(container).__name__}: expert count is unset, so the granule axis is undeclared. "
        f"A container must say which it is, because the fallback reading (one granule for the whole "
        f"container) is silently wrong for a stacked one: every byte total still balances and "
        f"`expert_slice(c, e)` just returns the entire stack for every e. Add "
        f"`self._num_experts = num_experts` to {type(container).__name__}.__init__ for a per-expert "
        f"container, or `_granule_dense = True` on the class for a whole-container one — or pass "
        f"`num_experts=` explicitly at the call site."
    )


def spec_for_container(
    container: Any, num_experts: "int | None | _Unset" = UNSET, **kw: Any
) -> GranuleSpec:
    """`derive_granule_spec` that also knows where a container records its own granule axis.

    Omitting `num_experts` means "ask the container" and RAISES when it has not declared; passing
    `None` explicitly means "one granule for the whole container" (dense). Those are different
    questions and the signature keeps them different — see `declared_granule_count`.
    """
    if isinstance(num_experts, _Unset):
        num_experts = declared_granule_count(container)
    return derive_granule_spec(container, num_experts, **kw)


def per_expert_tensors(
    container: Any, num_experts: "int | None | _Unset" = UNSET, **kw: Any
) -> Dict[str, torch.Tensor]:
    """The STACKED tensors that must travel together, keyed by component name.

    The free-function form of `ExpertContainer.per_expert_tensors`, so it also covers the container
    that cannot host a method: a bare stacked `torch.Tensor` (the unquantized MoE case), which must
    be given its `E` because it has nowhere to declare one. Values are the FULL `(E, ...)` tensors;
    use `GranuleSpec.expert_slice(container, e)` for one expert's views.
    """
    return spec_for_container(container, num_experts, **kw).stacked_tensors(container)


def offload_refusal(container: Any) -> str | None:
    """Why this container cannot be host-resident, or None.

    A container whose forward MATERIALIZES the whole `(E, N, K)` stack defeats the point: the bytes
    would be pulled over PCIe every step regardless of which expert routed. Refuse loudly at boot
    rather than serve at a fraction of the projected rate and let somebody conclude the mechanism is
    slow.
    """
    fn = getattr(container, "offload_refusal", None)
    if callable(fn):
        return fn()
    return None


def assert_granule_pair_consistent(a: GranuleSpec, b: GranuleSpec, *, where: str = "") -> None:
    """The w13/w2 pair of one MoE layer must agree on kind, expert count and component NAME SET.

    Their shapes differ (w13 is `2*inter x hidden`, w2 is `hidden x inter`), so shapes are not
    compared — but a mismatched name set means one container repacked and the other did not
    (`_w_rep` on one side, `_w_op` on the other), which is a real half-applied-knob state that would
    place two halves of the same layer on different media.
    """
    tag = f" ({where})" if where else ""
    if a.kind != b.kind:
        raise GranuleError(f"granule pair{tag}: container kinds differ — {a.kind} vs {b.kind}")
    if a.num_experts != b.num_experts:
        raise GranuleError(
            f"granule pair{tag}: expert counts differ — {a.num_experts} vs {b.num_experts}"
        )
    an = tuple(c.name for c in a.components)
    bn = tuple(c.name for c in b.components)
    if an != bn:
        raise GranuleError(
            f"granule pair{tag}: component sets differ — {an} vs {bn}. One half of this layer was "
            f"repacked and the other was not; they cannot be placed as one unit."
        )
    ad = {c.name: c.dtype for c in a.components}
    bd = {c.name: c.dtype for c in b.components}
    if ad != bd:
        raise GranuleError(f"granule pair{tag}: component dtypes differ — {ad} vs {bd}")
    if a.policy != b.policy:
        raise GranuleError(
            f"granule pair{tag}: DECODE POLICY differs — {dict(a.policy)} vs {dict(b.policy)}. The "
            f"two GEMMs of one layer came out of one checkpoint and one quantizer, so a disagreement "
            f"means one of them is being read with the wrong convention: for compressed-tensors int4 "
            f"that is `q+8` against two's-complement, i.e. every weight of one GEMM off by 8 quanta, "
            f"with right shapes, no kernel fault and plausible text. It is a detector that resolved "
            f"two ways on two stacks, not something to average — pin the convention from the "
            f"checkpoint's quantization_config and pass it into both."
        )


# =====================================================================================
# Bindings — the address-level checks a graph capture needs
# =====================================================================================


def binding_fingerprint(spec: GranuleSpec, container: Any) -> str:
    """Free-function form of `GranuleSpec.binding_fingerprint`."""
    return spec.binding_fingerprint(container)


def assert_bindings_unchanged(
    spec: GranuleSpec, container: Any, expected: str, *, where: str = ""
) -> None:
    """The container's components must still live at the addresses `expected` was taken over.

    THIS IS THE GRAPH-CAPTURE CHECK. A captured HIP graph bakes the device pointers its kernels were
    given; every replay reads THOSE addresses, forever. So a rebind after capture — an arena bind
    that ran late, a second bake pass, a `copy_`-then-reassign, a `.contiguous()` that quietly
    allocated — leaves replay reading the previous allocation. Shapes still match, the kernel still
    launches, nothing raises, and the text is merely wrong. `fingerprint()` cannot see any of it (it
    hashes the manifest, not memory) which is exactly why this exists as a separate token: snapshot
    it immediately before capture, re-assert it after capture and after any bind pass.
    """
    got = spec.binding_fingerprint(container)
    if got != expected:
        tag = f" ({where})" if where else ""
        raise GranuleError(
            f"granule bindings{tag}: {spec.kind}'s components have MOVED since the fingerprint was "
            f"taken ({expected} -> {got}). If a graph was captured over the old pointers, every "
            f"replay is now reading the previous allocation with the right shapes and the wrong "
            f"bytes. Re-bind before capture, never after."
        )


def assert_spec_still_holds(
    container: Any, spec: GranuleSpec, *, where: str = "", **kw: Any
) -> None:
    """Re-derive from the LIVE container and require the same manifest.

    Complements `assert_bindings_unchanged`: that one catches "the bytes moved", this one catches
    "the SHAPE of the description changed" — most importantly a rebind that DE-ALIASED two names
    which used to share a byte range. `_GroupedFP8Experts` keeps `weight` and `_w_op` as one
    component with two names under `MINISGL_ZAYA_OLDMOE=1`; a copy-based rebind that gave each its
    own buffer would split them into two components (and double the granule), so the manifest hash
    moves even though every individual name still resolves to a right-shaped tensor.
    """
    fresh = derive_granule_spec(container, spec.num_experts, **kw)
    if fresh.fingerprint() != spec.fingerprint():
        tag = f" ({where})" if where else ""
        raise GranuleError(
            f"granule manifest{tag}: {spec.kind} no longer matches the spec it was planned with "
            f"({spec.fingerprint()} -> {fresh.fingerprint()}).\n"
            f"  was: {spec.describe()}\n  now: {fresh.describe()}\n"
            f"A component appearing, vanishing or DE-ALIASING between planning and use means the "
            f"placement, the byte budget and any captured graph were all computed against weights "
            f"that no longer exist in that form."
        )


# =====================================================================================
# The accessor every expert container presents
# =====================================================================================


class ExpertContainer:
    """MIXIN adding the granule accessor to a weight container — per-expert OR dense.

    A mixin rather than a `BaseOP` subclass purely to keep this module's import graph acyclic (see
    `_baseop_cls`); containers declare `class _GroupedXExperts(ExpertContainer, BaseOP)`, so they are
    still `BaseOP`s in every `isinstance` the loader does. It adds no instance attributes, and its
    class attributes are underscore-prefixed, so `BaseOP.state_dict`/`load_state_dict`/`post_load`
    (which skip `_`-names) are entirely unaffected.

    IT IS THE DENSE SURFACE TOO. `_LinearTPImpl` mixes in this same class with
    `_granule_dense = True`; there is no second dense implementation, so a consumer written against
    `granule_spec` / `per_expert_tensors` / `expert_slice` / `offload_refusal` gets the identical
    five-method surface from a 512-expert MoE stack and from a bf16 QKV projection. That is the
    "MoE and dense land together" rule discharged in code rather than in a doc: a dense-only
    divergence here would show up as an `AttributeError` in the residency layer months later.

    Deliberately ONE derivation shared by every format rather than nine implementations. What a
    format "implements" is only its declarations — the granule axis (`_num_experts` set in
    `__init__`, or `_granule_dense = True`) and, if it genuinely owns an expert-invariant buffer,
    `_residency_shared`. Everything else is derived from the live tensors, so a format that grows a
    buffer gets a loud boot failure instead of a silently omitted scale, and a format that drops one
    (a regdirect repack) needs no edit here at all.
    """

    # Names that are legitimately shared across experts, and the ONLY way a tensor leaves the
    # granule. Set it where the CODE PATH that made the buffer constant runs, not as a hand-written
    # class-level table — `_GroupedCompressedTensorsExperts.post_load` sets it on the instance in the
    # symmetric branch, so an asymmetric checkpoint through the other branch declares nothing and its
    # real zero-points stay in the granule. That keeps the decision config-derived (identical on
    # every TP rank, no collective) instead of content-derived (each rank reading its own shard),
    # while `derive_granule_spec` VERIFIES the declaration bitwise so a stale one raises rather than
    # silently dropping a per-expert buffer. Empty by default.
    _residency_shared: ClassVar[Tuple[str, ...]] = ()

    # THE GRANULE AXIS, declared one of two ways and never defaulted (see declared_granule_count):
    #   * `self._num_experts = num_experts` in __init__  -> E per-expert granules. Explicit rather
    #     than inferred from a tensor's dim 0: E can coincide with N or K, and guessing would
    #     silently reclassify a whole component.
    #   * `_granule_dense = True` on the class            -> the whole container is ONE granule.
    _granule_dense: ClassVar[bool] = False
    _num_experts: int = 0

    # Dotted paths to the DECODE DECISIONS this format's `post_load` made — things that change how
    # the bytes are read but are not themselves tensors, so the walk cannot see them. Empty for a
    # format that makes none. Point at the decision, not at the object holding it
    # (`"_ct_sign.uint4b8"`, not `"_ct_sign"`), so the recorded value is rank-invariant; see
    # `_policy_values`.
    _granule_policy: ClassVar[Tuple[str, ...]] = ()

    def _resolved_n(self, num_experts: "int | None | _Unset") -> int | None:
        if not isinstance(num_experts, _Unset):
            return num_experts
        return declared_granule_count(self)

    def granule_spec(self, num_experts: "int | None | _Unset" = UNSET, **kw: Any) -> GranuleSpec:
        return derive_granule_spec(self, self._resolved_n(num_experts), **kw)

    def per_expert_tensors(
        self, num_experts: "int | None | _Unset" = UNSET, **kw: Any
    ) -> Dict[str, torch.Tensor]:
        """The stacked tensors that must travel with one granule, keyed by component name.

        Named for the expert case, not restricted to it: a dense container has exactly one granule,
        so this is "the tensors that must travel together" — weight/scales/zeros AND the bias, which
        is a real correctness item (a moved weight with a left-behind bias is a wrong-numbers bug of
        exactly the same family as a left-behind scale).
        """
        return self.granule_spec(num_experts, **kw).stacked_tensors(self)

    def expert_slice(
        self, e: int, num_experts: "int | None | _Unset" = UNSET, **kw: Any
    ) -> Dict[str, torch.Tensor]:
        """Granule `e`'s views of every component. Component-major: each is `t[e]` of its own stack.
        A dense container has one granule, so only `e == 0` is in range."""
        spec = self.granule_spec(num_experts, **kw)
        return spec.expert_slice(self, e)

    def offload_refusal(self) -> str | None:
        """Override to refuse host residency for this container. See `granule.offload_refusal`."""
        return None


def total_granule_bytes(specs: Sequence[GranuleSpec]) -> int:
    """Bytes of ONE co-demanded granule across several containers (a layer's w13 + w2).

    This is the unit the oracle counts and the placement plan prices: routing expert `e` in a layer
    demands `e`'s slice of BOTH GEMMs, so they are one granule for every purpose except placement
    arithmetic, where each component still gets its own contiguous slab.
    """
    return sum(s.granule_bytes for s in specs)
