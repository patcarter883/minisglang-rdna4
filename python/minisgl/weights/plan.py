"""`resolve_weight_plan(config)` -- the SINGLE config-side resolver for layer-granular placement.

WHAT THIS FILE OWNS, AND WHAT IT DEFERS TO
    `placement.py` decides placement given a list of layers with byte counts, and it is the whole of
    the placement policy -- greedy integer fill, `OffloadPlan`, the projection, the f-sweep. This
    file is the other half: turning a `ModelConfig`/`EngineConfig` into that list, BEFORE any weight
    exists, and answering the capacity question that decides whether the serve can boot at all.

    Nothing here re-implements `plan_layer_granular`, and there is no parallel layer type: this
    file emits `placement.LayerWeights` directly, so the config-derived path and the granule-derived
    path (`LayerWeights.from_specs`, after `post_load()`) feed ONE planner, one sweep and one
    projection. The only difference between them is provenance -- an estimate from container shapes
    versus measured bytes off live tensors -- which is recorded in `byte_source`, never in a second
    code path that could quietly disagree.

WHY THE PLAN MUST BE RESOLVED BEFORE THE LOAD
    `weight.py:906` etc. open shards with `device=str(device)` and `post_load` repacks the whole
    stack on GPU, so a 68 GB checkpoint OOMs long before anything could be measured. Phase 0 also
    measured the pinned host ceiling at 62 GiB across two ranks on an IDLE box against a 68.8 GB
    requirement -- so the default expectation is that an all-host arena DOES NOT FIT, and the
    difference between a good failure and a terrible one is entirely when it is detected. This
    resolver answers "does it fit, and if not how much VRAM would fix it" from config arithmetic, in
    milliseconds, before the first shard is opened.

WHY IT IS RESOLVED IN ONE PLACE
    Modelled on `engine/config.py::resolve_prefix_cache`, and for the same reason. Prefix caching
    used to be decided independently by the engine (sizing the pool) and the scheduler (building the
    cache), and the two DIVERGED -- a SWA hybrid reserved 0.98 GiB of KV pool for a store that could
    not exist. Weight placement has the identical shape and a worse failure mode: the engine needs
    `plan.device_resident_bytes` while it is SIZING the KV pool, the arena needs to know which
    layers to pin, and the granule walker needs to know which stacks to register. If any two of
    those answer differently, one rank streams a layer another holds resident and the model emits
    plausible wrong text with no error anywhere.

PURITY
    Reads no clock, no environment, no device, no `/proc`. The capacity ceiling comes from the BAKED
    P3b table in `host_capacity.py`, not from a live `MemAvailable` read -- a runtime measurement
    differs between the two rank processes, so letting it into the plan would make placement
    timing-dependent and rank-divergent. `host_capacity.check_capacity()` is the separate, live,
    pass-or-raise gate the arena runs at allocation time; the two answer different questions and
    both are needed.

    Config fields are read with defensive `getattr` (the `config.py:87` pattern): the object handed
    in may be an `EngineConfig`, a `SchedulerConfig`, or a plain stub. This module therefore imports
    no torch and no transformers at module scope, so its unit tests run on a host with neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .chunk_plan import (
    ALIGN,
    CHUNK_GRANULE,
    DEFAULT_CHUNK_BYTES,
    RegionRequest,
    headroom_chunks,
    plan_regions,
    round_up,
    suggest_chunk_bytes,
    torch_charged_rows,
)
from .host_capacity import (
    DEFAULT_FLOOR_BYTES,
    P3B_NOTE,
    P3B_PINNED_CEILING_BYTES,
)
from .placement import (
    LayerWeights,
    OffloadPlan,
    PlacementError,
    PostLoadCorrection,
    ep_local_top_k,
    format_sweep,
    plan_layer_granular,
    plan_three_tier,
    project_plan,
    sweep_device_fraction,
)
from .prior import PHASE0_PRIOR, GiB, OffloadPrior, Projection  # NOTE: prior.GB is NOT imported
from .sizing import (
    SCHEME_MXFP4,
    cpu_wload_policy,
    cpu_wload_policy_for_kind,
    expert_stack_bytes,
    post_load_delta_bytes,
    scheme_from_quant,
    scheme_supports_ep,
)
from .stacks import StackKind

__all__ = [
    "MoELayerShape",
    "WeightPlanResolution",
    "arena_reservation_bytes",
    "effective_chunk_bytes",
    "exact_arena_reservation_bytes",
    "build_planned_layers",
    "host_arena_ceiling_bytes",
    "is_mtp_path",
    "moe_layer_indices",
    "moe_layer_shapes",
    "observed_planned_layers",
    "plan_arena_reservation_bytes",
    "required_device_bytes",
    "resolve_expert_parallel",
    "resolve_weight_plan",
    "CpuTierGate",
    "cpu_tier_gate",
    "cpu_tier_sweep",
    "format_cpu_tier_sweep",
]

# `prior.GB` (decimal 1e9, the unit BANDWIDTHS are quoted in) is deliberately not imported above:
# every byte figure this module handles is binary, and having both names in scope is exactly how the
# operator's `--weight-offload-gb` came to be parsed at 1e9 while `_gb()` printed it back as GiB.
#
# `--weight-offload-gb` and `--weight-offload-device-gb` are read in GiB, NOT
# decimal 1e9 GB: every other memory figure on this boot path is binary (P3b's 34/62 GiB ceiling
# table, the 12 GiB pinning floor, the 2 GiB arena chunk, `engine.py:_graph_capture_bytes`'s
# `int(float(margin) * (1 << 30))` for its own `..._MARGIN_GB` env knob, and `_gb()` below, which
# RENDERS in GiB). Parsing the operator's number as 1e9 while printing it back as GiB silently
# under-grants by 7.4% on the two knobs that decide whether the box boots, and the log looks right.
GIB_PER_UNIT = GiB

# The MTP / next-token-prediction draft head carries its own `MoELayer`, built REPLICATED
# (`force_no_ep=True`) because it is tiny. It is also read on EVERY draft step of every speculative
# decode -- `num_draft` extra times per accepted token. Streaming it from host would multiply the one
# thing spec decode exists to make cheap. Excluded by construction; if a checkpoint ever ships an MTP
# head large enough to matter, that is a measurement, not a default.
OFFLOAD_MTP_HEAD = False


# =================================================================================================
# Config -> structure
# =================================================================================================


def moe_layer_indices(model_config: Any, *, include_mtp: bool = False) -> Tuple[int, ...]:
    """Which decoder layers own a `MoELayer`, derived from config the way the builders do.

    One rule per family, each transcribed from the model file that builds the block:

      * ZAYA (`is_cca_hybrid`): even layers are CCA attention, ODD layers are the EDA/MOD MoE
        (`zaya.py:718-722` -- `is_cca` picks `ZayaCCAAttn`, else `ZayaMoEBlock`).
      * everything else: `layer_id >= first_k_dense_replace` (`glm4_moe_lite.py:339`,
        `laguna.py:281`). Gemma4 and Qwen2/Qwen3.5-MoE build experts on every layer and carry
        `first_k_dense_replace == 0`, so the same rule covers them without a model-name branch.

    This is INFERRED, not observed -- the seven builders are the ground truth and they live in seven
    files. `resolve_weight_plan(..., layer_indices=...)` exists precisely so the engine can hand in
    the truth once the meta model is built (`engine.py:218`, before the `load_state_dict` at `:239`),
    and the registry re-derives it from live containers afterwards. Do not let this function be the
    only thing that ever decides.
    """
    if not bool(getattr(model_config, "is_moe", False)):
        return ()
    if int(getattr(model_config, "num_experts", 0) or 0) <= 0:
        return ()
    n_layers = int(getattr(model_config, "num_layers", 0) or 0)
    if bool(getattr(model_config, "is_cca_hybrid", False)):
        ids = [i for i in range(n_layers) if i % 2 == 1]
    else:
        first_dense = int(getattr(model_config, "first_k_dense_replace", 0) or 0)
        ids = [i for i in range(n_layers) if i >= first_dense]
    if include_mtp:
        n_mtp = int(getattr(model_config, "num_nextn_predict_layers", 0) or 0) + int(
            getattr(model_config, "mtp_num_hidden_layers", 0) or 0
        )
        ids.extend(range(n_layers, n_layers + n_mtp))
    return tuple(ids)


def resolve_expert_parallel(config: Any, *, method_supports_ep: bool = True) -> Tuple[bool, int]:
    """(enable_ep, ep_size) exactly as `Engine.__init__` + `set_dp_info` + `MoELayer` resolve them.

    Transcribed from `engine.py:172-200`: EP-over-TP is `enable_ep and dp_size == 1 and tp_size > 1`
    (experts sharded across the TP group); otherwise EP means DP+EP and shards across DP replicas;
    with `dp_size == 1` and `tp_size == 1` EP is inert. Getting this wrong is a factor-of-tp error in
    every byte count on the page, in the direction that turns an infeasible plan feasible.

    `method_supports_ep` is the THIRD conjunct, and leaving it out was a real defect. The engine
    toggle is only two thirds of the decision -- `MoELayer.__init__` (moe.py:1004) is

        self.enable_ep = is_ep_enabled() and self._moe_method.supports_ep and not force_no_ep

    so the quant method holds a VETO. `_RXFMoEMethod` (`supports_ep = False`, "no precomputed-topk
    shard route") and `_UnquantizedMoEMethod` (inherits the base-class `False`) both exercise it. On
    such a checkpoint served `--enable-ep --tp 2` the layer is built REPLICATED with the intermediate
    tensor-split, and a planner that shards it anyway describes a layer that does not exist: it
    reports `num_experts = E/ep` and `top_k_local = ceil(k/ep)` where the real layer has `E` and `k`,
    which halves every `distinct_experts()` traffic figure feeding the projection, the K4 gate and
    the f-sweep. The resident-byte totals happen to coincide (`E/2 x 2I` and `E x 2(I/2)` multiply
    out the same), so nothing else would have caught it -- and the EP branch also skips the
    `intermediate % tp_size` divisibility check that the layer will actually assert on.

    Callers that know the checkpoint should pass the answer from `sizing.scheme_supports_ep`;
    `moe_layer_shapes` does. The default stays True so the two-conjunct engine-level question can
    still be asked on its own.
    """
    tp_size = int(getattr(getattr(config, "tp_info", None), "size", 1) or 1)
    dp_size = int(getattr(getattr(config, "dp_info", None), "dp_size", 1) or 1)
    if not bool(getattr(config, "enable_ep", False)) or not method_supports_ep:
        return False, 1
    if dp_size == 1 and tp_size > 1:
        return True, tp_size
    if dp_size > 1:
        return True, dp_size
    return False, 1


def _fp8_experts_signal(mc: Any) -> bool:
    """The `fp8_experts=` argument the builder passes to `MoELayer`, from the DECLARED scheme only.

    This used to read `is_cca_hybrid AND quant.is_fp8_w8a8` -- a model-family branch, in a file whose
    whole claim is that it derives from config, and against `create_moe_quant_method`'s own stated
    contract ("Selection is purely config-driven: no model-name branch, and no env that substitutes a
    different scheme than the checkpoint declares"). It was also a no-op by construction: the extra
    conjunct can only be true when `quant.is_fp8_w8a8` is already true, and that predicate wins the
    dispatch on its own (`create_moe_quant_method` line 1: `if fp8_experts or quant.is_fp8_w8a8`). So
    it changed no byte on any checkpoint and only encoded "fp8 experts means ZAYA", which is exactly
    the assumption that breaks the next family to ship fp8 experts -- `zaya.py:620` passes the flag
    from `config.quant.is_fp8_w8a8`, so any family may.
    """
    return bool(getattr(getattr(mc, "quant", None), "is_fp8_w8a8", False))


def local_arena_count(config: Any) -> int:
    """How many independent expert stacks this NODE holds -- the host-capacity multiplier.

    Host RAM is a node resource: at TP=2 both ranks pin from the same pool and P3b's 62 GiB ceiling
    is their SUM, so a per-rank check passes twice while the box dies. `tp_size * dp_size` is right
    for all three sharding modes this engine supports, which is not obvious:

      * plain TP, dp=1     -- per-rank bytes are full/tp; node total = full.               (x tp)
      * EP-over-TP, dp=1   -- per-rank bytes are full/tp (E/tp whole experts); total full. (x tp)
      * DP(+EP), dp>1      -- each DP replica has its own tp ranks, and under EP the experts are
        sharded across DP while the intermediate is NOT tp-split, so the tp ranks within a replica
        hold DUPLICATE copies of the same shard. Node total = full * tp.                (x tp * dp)

    Single-node by assumption, which is what this box is. Override for multi-node.
    """
    tp_size = int(getattr(getattr(config, "tp_info", None), "size", 1) or 1)
    dp_size = int(getattr(getattr(config, "dp_info", None), "dp_size", 1) or 1)
    return max(1, tp_size * dp_size)


@dataclass(frozen=True)
class MoELayerShape:
    """One MoE layer's per-rank shape, resolved from config before any allocation."""

    layer_index: int
    path: str
    label: str
    num_experts: int  # global expert count
    num_local_experts: int  # what THIS rank holds after EP sharding
    top_k: int
    top_k_local: int  # routed slots THIS rank computes per token (see moe_layer_shapes)
    hidden_size: int
    intermediate_size_per_partition: int
    offloadable: bool
    note: str = ""
    # Whether THIS rank's containers are EP-sharded, and by how much. Recorded rather than
    # re-derived by every consumer: it is the third conjunct of `MoELayer.__init__` (moe.py:1004)
    # and the quant method can veto it (see `resolve_expert_parallel`), so "did EP apply" is not
    # answerable from `config.enable_ep` alone.
    expert_parallel: bool = False
    ep_size: int = 1


def moe_layer_shapes(
    config: Any, *, layer_indices: Optional[Sequence[int]] = None
) -> Tuple[MoELayerShape, ...]:
    """Resolve every MoE layer's per-rank shape, mirroring `MoELayer.__init__` (moe.py:930-1030).

    Sharding:
      * EP on  -> this rank owns `num_experts // ep_size` WHOLE experts, at the FULL intermediate
                  size (EP-over-TP keeps the full intermediate; only plain TP splits it).
      * EP off -> all `num_experts` experts, each split to `intermediate_size // tp_size`.
    Either way `w13` is (E_local, 2*I_part, H) and `w2` is (E_local, H, I_part).

    `top_k_local` is `ceil(top_k / ep_size)` under EP, not `top_k / ep_size`. A mean would be right
    for a long-run throughput average but wrong for a step-time floor, which is what the projection
    is: the step waits for the SLOWEST rank, so it is set by the rank that drew more than its share
    of routed experts, not by the average. Rounding up is the conservative direction.

    EP is asked as all THREE conjuncts of `MoELayer.__init__` (moe.py:1004), not just the engine
    toggle: the quant method vetoes it for RXF and for unquantized experts, and a planner that
    misses the veto halves every traffic figure on a `--enable-ep --tp 2` serve of such a
    checkpoint. See `resolve_expert_parallel`.
    """
    mc = getattr(config, "model_config", config)
    tp_size = int(getattr(getattr(config, "tp_info", None), "size", 1) or 1)
    method_ep = scheme_supports_ep(
        getattr(mc, "quant", None), fp8_experts=_fp8_experts_signal(mc)
    )
    enable_ep, ep_size = resolve_expert_parallel(config, method_supports_ep=method_ep)

    num_experts = int(getattr(mc, "num_experts", 0) or 0)
    top_k = int(getattr(mc, "num_experts_per_tok", 0) or 0)
    hidden = int(getattr(mc, "hidden_size", 0) or 0)
    inter = int(getattr(mc, "moe_intermediate_size", 0) or 0)

    n_layers_cfg = int(getattr(mc, "num_layers", 0) or 0)
    n_mtp_cfg = int(getattr(mc, "num_nextn_predict_layers", 0) or 0) + int(
        getattr(mc, "mtp_num_hidden_layers", 0) or 0
    )
    ids = tuple(layer_indices) if layer_indices is not None else moe_layer_indices(mc)
    if layer_indices is not None:
        # `is_mtp` below is decided by `lid >= num_layers`, which is only meaningful if the caller
        # numbers the MTP head's MoELayer past the decoder stack the way `moe_layer_indices` does.
        # A caller using any other numbering would classify the DRAFT HEAD as an ordinary layer and
        # make it offloadable -- and the draft head is re-read `num_draft` times per accepted token
        # inside the captured spec-decode graphs, which is precisely the one layer whose streaming
        # would multiply the cost spec decode exists to avoid. Refuse the ambiguous input instead of
        # silently choosing the bad reading.
        bad = [i for i in ids if not 0 <= int(i) < n_layers_cfg + n_mtp_cfg]
        if bad:
            raise PlacementError(
                f"layer_indices {bad} fall outside [0, {n_layers_cfg + n_mtp_cfg}) "
                f"(num_layers={n_layers_cfg}, mtp={n_mtp_cfg}). MTP layers MUST be numbered from "
                f"num_layers upward, because that is the only signal distinguishing the draft head "
                f"-- which is re-read on every draft step and must never be offloaded -- from an "
                f"ordinary decoder layer."
            )
        if len(set(ids)) != len(ids):
            raise PlacementError(f"layer_indices contains duplicates: {ids}")

    if not ids or num_experts <= 0 or top_k <= 0 or hidden <= 0 or inter <= 0:
        return ()

    if enable_ep:
        if num_experts % ep_size:
            raise PlacementError(
                f"EP needs num_experts ({num_experts}) divisible by ep_size ({ep_size}); "
                "MoELayer.__init__ asserts the same thing"
            )
        local_experts = num_experts // ep_size
        inter_part = inter
        # `placement.ep_local_top_k` is the ONE implementation of this (ceil, clamped); the model
        # path below and `moe_interpose.MoEWeightSeam` call the same function, which is what stops
        # the config path and the observed path from producing different plans for one model.
        top_k_local = ep_local_top_k(top_k, ep_size, local_experts)
    else:
        if inter % tp_size:
            raise PlacementError(
                f"plain TP needs moe_intermediate_size ({inter}) divisible by tp_size ({tp_size}); "
                "div_even in MoELayer.__init__ asserts the same thing"
            )
        local_experts = num_experts
        inter_part = inter // tp_size
        top_k_local = min(local_experts, top_k)

    # The MTP head is built `force_no_ep=True` (`qwen3_5_moe.py:200`), which is the THIRD conjunct of
    # `MoELayer.__init__` (moe.py:1004) and applies per LAYER, not per model. So it is replicated at the
    # tp-split intermediate even on a serve where every decoder layer is EP-sharded. Reported
    # correctly rather than "it does not matter because it is excluded": `OFFLOAD_MTP_HEAD` is a
    # policy constant somebody may flip after a measurement, and it would then flip onto a shape that
    # was silently wrong by a factor of ep_size.
    mtp_experts = num_experts
    mtp_inter_part = inter // tp_size if inter % tp_size == 0 else inter
    mtp_top_k = min(mtp_experts, top_k)

    n_layers = int(getattr(mc, "num_layers", 0) or 0)
    shapes = []
    for lid in ids:
        is_mtp = lid >= n_layers
        # STRUCTURAL path, never a construction counter: the MTP draft head builds its own MoELayer,
        # and a counter would renumber every layer after it -- silently placing the wrong layers on
        # the wrong stack. `placement.plan_layer_granular` refuses duplicates for the same reason.
        path = (
            f"mtp.layers.{lid - n_layers}.mlp.experts" if is_mtp
            else f"model.layers.{lid}.mlp.experts"
        )
        shapes.append(
            MoELayerShape(
                layer_index=lid,
                path=path,
                label="moe_mtp" if is_mtp else "moe",
                num_experts=num_experts,
                num_local_experts=mtp_experts if is_mtp else local_experts,
                top_k=top_k,
                top_k_local=mtp_top_k if is_mtp else top_k_local,
                hidden_size=hidden,
                intermediate_size_per_partition=mtp_inter_part if is_mtp else inter_part,
                offloadable=(not is_mtp) or OFFLOAD_MTP_HEAD,
                note="MTP draft head: re-read every draft step, never offloaded" if is_mtp else "",
                expert_parallel=enable_ep and not is_mtp,
                ep_size=1 if is_mtp else ep_size,
            )
        )
    return tuple(shapes)


def _expert_quant_for(mc: Any, layer_index: int) -> Any:
    """The quant scheme the ROUTED experts of `layer_index` are built with: the HEADLINE `mc.quant`.

    This deliberately does NOT consult `QuantConfig.for_module`, and that is a correction. An earlier
    version probed three guessed module-name spellings per layer and used whatever group matched,
    modelling a per-layer refinement that **no builder performs**. Every MoE family passes the
    headline config straight through -- `expert_quant = config.quant` in `qwen3_5_moe.py:187`,
    `glm4_moe_lite.py`, `laguna.py`, `gemma4.py:265`, `zaya.py:630` -- and `create_moe_quant_method`
    then builds ONE container class for the whole expert stack of that layer. `for_module` is used
    only for DENSE linears (`qwen3_5.py:125/293`, `models/utils.py:106-112`), which is exactly where
    a checkpoint's mixed-precision `config_groups` actually bite.

    On a real mixed checkpoint the removed probe was WRONG, not merely redundant. Qwen3.8-27B-NVFP4
    lists group_0 as `layers.(56|...|63).mlp.(gate|up|down)_proj` (fp8) and group_1 as the catch-all
    `.*mlp.(gate|up|down)_proj` (NVFP4); `for_module("model.layers.60.mlp.experts.0.gate_proj")`
    matches group_0 by substring, so the planner sized those layers fp8 (1 byte/elem) while the layer
    allocates NVFP4 (0.5 byte + scale). Its own fallback was mis-argued too: returning the headline
    quant for a layer that is really unquantized UNDER-sizes the arena ~4x, which is the direction
    that makes an infeasible plan look feasible -- the docstring claimed the opposite.

    The one per-stack quant difference the builders DO implement is the MTP draft head
    (`qwen3_5_moe.py:194`, `glm4_moe_lite.py:707`: `mtp_quant = None` when
    `mtp.layers.0.mlp.experts.0.gate_proj` is not `is_module_quantized`). That head is never
    offloaded (`OFFLOAD_MTP_HEAD`), so it never reaches this function's result.

    `layer_index` is kept in the signature because it is the natural hook if a family ever does
    thread a per-layer expert quant; today it is unused, and unused-by-construction is the honest
    state rather than a probe that fabricates one.
    """
    del layer_index  # see docstring: no builder varies the expert quant per decoder layer
    return getattr(mc, "quant", None)


# =================================================================================================
# Model -> planner input  (the OBSERVED path: nothing is transcribed, nothing is guessed)
# =================================================================================================


def is_mtp_path(path: str) -> bool:
    """Is this structural path inside the MTP / next-token-prediction draft head?

    Both families that ship one attach it as `self.mtp` (`qwen3_5.py:735`, `glm4_moe_lite.py:711`),
    and `BaseOP.state_dict` names children by attribute, so the head's `MoELayer`s are the ones with
    an `mtp` path SEGMENT. Segment equality, not `in`: a decoder layer called `mtp_adapter` would
    otherwise be excluded from offload and nobody would notice a whole layer had gone quiet.
    """
    return "mtp" in path.split(".")


def _path_layer_index(path: str) -> Optional[int]:
    """`"model.layers.31.mlp.experts"` -> 31; None when the path carries no decoder index.

    Structural, never a construction counter — the same rule `moe_interpose.discover_moe_layers`
    keys on. A path with no index (a bare MoE head) is never excluded by index: it cannot be named
    by one.
    """
    parts = path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def observed_planned_layers(
    model: Any,
    *,
    priorities: Optional[Dict[Any, int]] = None,
    allow_meta: bool = True,
    layer_indices: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[LayerWeights, ...], Dict[str, Any]]:
    """Size every offloadable MoE layer by WALKING THE BUILT MODEL. Returns (layers, diagnostics).

    THIS IS THE PATH THAT MAKES THE ANSWER TO "what must a new model or quant format implement?"
    BE "NOTHING". Everything it needs it reads off objects that already exist:

      * which layers own a `MoELayer`      -- `moe_interpose.discover_moe_layers` (the same walk the
        seam binder uses), instead of `moe_layer_indices`' transcription of seven builder files;
      * how this rank is sharded          -- `layer.local_num_experts` / `layer.enable_ep` /
        `layer.ep_size`, i.e. the THREE-conjunct answer `MoELayer.__init__` actually computed,
        including the quant method's EP veto and `force_no_ep`;
      * how many bytes an expert is       -- `MoELayer.granule_specs()`, the one granule walker,
        which is format-agnostic by construction (it requires only that a per-expert tensor carries
        E on dim 0) and needs no arm per container class.

    The config path (`build_planned_layers`) has to GUESS all three, and each guess has a live
    counterexample in this repo: `models/utils.py:145`'s `MoEMLP` builds `MoELayer(...)` with **no
    `quant=` argument at all**, so its experts are unquantized bf16 no matter what `config.quant`
    says -- the config path sizes that stack int4 and under-reserves the arena ~4x, in the direction
    that makes an infeasible plan look feasible. There is no config field that distinguishes it.

    So: use this whenever a model object exists. `engine.py` builds the model on meta at `:218`,
    before `load_state_dict` at `:239`, which is exactly the point where the capacity abort must
    happen -- so `allow_meta=True` (shapes only; aliasing and expert-invariance are not provable on
    meta, see `derive_granule_spec`). After `post_load()` call it again with `allow_meta=False` for
    the measured figure and compare.

    A container that REFUSES host residency (`granule.offload_refusal`, e.g. ZAYA's fp8 experts under
    `MINISGL_ZAYA_OLDMOE=1`, whose forward materialises the whole stack) is recorded as skipped
    rather than planned. That does make this path env-sensitive where the config path is not -- but
    the alternative is a plan that promises host residency the bake will then refuse, and both ranks
    are forked from one environment. The skip is reported, never silent.
    """
    from .granule import offload_refusal
    from .moe_interpose import discover_moe_layers, ep_size_of

    def _meta_corrections(layer: Any, e_local: int) -> Dict[str, PostLoadCorrection]:
        """What `post_load()` will add to this layer's two containers. See `PostLoadCorrection`.

        ONLY for a spec derived on META. `engine.py` builds the model on meta and resolves the plan
        there — which is the whole reason the capacity abort can land before `load_state_dict` — so
        the specs this path reads are `__init__` shapes and `post_load()` has not run. Two of the
        nine shipped containers grow across it (`sizing.PostLoadDelta`): compressed-tensors
        SYMMETRIC materialises a real `_zeros_op` (+18 MiB/layer, ~2.7%, on the target shape) and
        MXFP4 widens its E8M0 scale to fp16 (+6.25% at g=32). The config path
        (`expert_stack_bytes`) has charged both since 2026-09-03; this path did not, and the
        anonymous-headroom reservation's +28% slop was hiding it. With the reservation exact those
        bytes are rows with nowhere to go, i.e. `hipMalloc` VRAM booked as host RAM.

        Shapes are read off `MoELayer`, which stores every term (`hidden_size`,
        `intermediate_size`, `tp_size`, `enable_ep`) — no transcription, and the EP branch is the
        layer's own answer rather than a re-derivation of it. Anything unreadable yields a zero
        correction, which is exactly today's behaviour.
        """
        try:
            hidden = int(layer.hidden_size)
            inter = int(layer.intermediate_size)
            i_part = inter if bool(layer.enable_ep) else inter // int(layer.tp_size)
            scheme = scheme_from_quant(
                getattr(layer, "quant", None),
                fp8_experts=bool(getattr(layer, "fp8_experts", False)),
                strict=False,
            )
            out = {}
            for attr, (n, k) in (
                ("gate_up_proj", (2 * i_part, hidden)),
                ("down_proj", (hidden, i_part)),
            ):
                d = post_load_delta_bytes(scheme, e_local, n, k)
                # MXFP4 grows the EXISTING u8 group-scale row in place; everything else that grows
                # does so by allocating a tensor `__init__` never made, which is a row of its own.
                widen = (
                    e_local * n * (k // scheme.group_size)
                    if scheme.kind == SCHEME_MXFP4 and scheme.group_size
                    else 0
                )
                out[attr] = PostLoadCorrection(d.resident, d.granule, widen)
            return out
        except Exception:  # pragma: no cover - an unreadable layer keeps today's behaviour
            return {}

    prio = priorities or {}
    want = None if layer_indices is None else {int(i) for i in layer_indices}
    layers: List[LayerWeights] = []
    skipped: List[str] = []
    kinds: List[str] = []
    ep_sizes: List[int] = []
    saw_meta = False
    for path, layer in discover_moe_layers(model):
        # `layer_indices` RESTRICTS the plan, on this path exactly as it does on the config path.
        # It used to be accepted by `resolve_weight_plan`, forwarded to `build_planned_layers`, and
        # then silently dropped the moment a model was passed — i.e. every production call, since
        # `engine.py` always has one. A caller that excludes layers (the weight-offload STREAM tier
        # excludes the ones it will re-read from the checkpoint per forward) would have got a plan
        # naming layers it never intended to place, an arena reserved for them, and a KV pool sized
        # against that reservation.
        if want is not None and _path_layer_index(path) not in want:
            skipped.append(f"{path} (excluded by layer_indices)")
            continue
        if is_mtp_path(path) and not OFFLOAD_MTP_HEAD:
            skipped.append(f"{path} (MTP draft head: re-read every draft step, never offloaded)")
            continue
        refusal = next(
            (
                r
                for r in (offload_refusal(c) for c in layer.expert_containers().values())
                if r is not None
            ),
            None,
        )
        if refusal is not None:
            skipped.append(f"{path} (container refuses host residency: {refusal})")
            continue
        specs = layer.granule_specs(allow_meta=allow_meta)
        is_meta = any(getattr(s, "meta", False) for s in specs.values())
        saw_meta = saw_meta or is_meta
        # Applied ONLY on meta. Off a live post-load container the tensors are already there and
        # adding the delta again would double-charge the arena.
        corrections = _meta_corrections(layer, int(layer.local_num_experts)) if is_meta else {}
        kinds.append(next(iter(specs.values())).kind)
        e_local = int(layer.local_num_experts)
        # ceil, and for the same reason as the config path: the step waits for the SLOWEST rank.
        # `ep_size_of` + `ep_local_top_k` are the SAME two functions `MoEWeightSeam` calls, so this
        # path and the standalone `build_layer_weights(attach_seams(model))` path cannot produce
        # different `top_k` — and therefore cannot produce different `OffloadPlan.digest()`es — for
        # one model. They did: this site did the EP correction and the seam did not.
        ep_sizes.append(ep_size_of(layer))
        top_k_local = ep_local_top_k(int(layer.top_k), ep_size_of(layer), e_local)
        # Priorities may be keyed by path (structural, what M2's --weight-prior file will use) or by
        # decoder index (what the config path uses). Accept both; never invent one from a counter.
        p = prio.get(path)
        if p is None:
            tail = [s for s in path.split(".") if s.isdigit()]
            p = prio.get(int(tail[-1]), 0) if tail else 0
        layers.append(
            LayerWeights.from_specs(
                path,
                num_experts=e_local,
                top_k=top_k_local,
                w13=specs["gate_up_proj"],
                w2=specs["down_proj"],
                priority=int(p),
                w13_post_load=corrections.get("gate_up_proj"),
                w2_post_load=corrections.get("down_proj"),
            )
        )

    diagnostics: Dict[str, Any] = {
        "byte_source": "observed-meta" if saw_meta else "observed",
        "schemes": sorted(set(kinds)),
        "n_offloadable_layers": len(layers),
        "skipped": tuple(skipped),
        "sizing_agreement_min": None,
        "sizing_agreement_max": None,
        "compute_dtype_bytes": None,
        # Same two keys the config path emits, read off the LIVE layers rather than re-derived.
        # `ep_size_of` asks all three conjuncts `MoELayer.__init__` collapses into `enable_ep`,
        # so a checkpoint whose quant method vetoed EP reports 1 here even under `--enable-ep`.
        # The CPU-COMPUTE tier is refused when this is True; see `cpu_tier_gate`.
        "expert_parallel": max(ep_sizes, default=1) > 1,
        "ep_size": max(ep_sizes, default=1),
    }
    return tuple(layers), diagnostics


# =================================================================================================
# Config -> planner input  (the pre-build fallback: everything here is a transcription)
# =================================================================================================


def build_planned_layers(
    config: Any,
    *,
    layer_indices: Optional[Sequence[int]] = None,
    compute_dtype_bytes: Optional[int] = None,
    prefer_meta: bool = True,
    priorities: Optional[Dict[int, int]] = None,
    model: Any = None,
) -> Tuple[Tuple[LayerWeights, ...], Dict[str, Any]]:
    """Size every offloadable MoE layer. Returns (layers, diagnostics).

    `model` -- a built model object (meta or materialized). When given, everything is read off it via
    `observed_planned_layers` and NOTHING below runs: no family rule, no quant guess, no per-format
    byte arm. Prefer it whenever a model exists; the config arithmetic here is the pre-build
    fallback that answers the capacity question before anything is constructed.
    """
    if model is not None:
        return observed_planned_layers(model, priorities=priorities, layer_indices=layer_indices)
    mc = getattr(config, "model_config", config)
    if compute_dtype_bytes is None:
        compute_dtype_bytes = int(getattr(getattr(config, "dtype", None), "itemsize", 2) or 2)
    fp8_experts = _fp8_experts_signal(mc)

    layers: List[LayerWeights] = []
    skipped: List[str] = []
    sources: List[str] = []
    schemes: List[str] = []
    agreements: List[float] = []
    shapes = list(moe_layer_shapes(config, layer_indices=layer_indices))
    for shape in shapes:
        if not shape.offloadable:
            skipped.append(f"{shape.path} ({shape.note})")
            continue
        sized = expert_stack_bytes(
            quant=_expert_quant_for(mc, shape.layer_index),
            num_local_experts=shape.num_local_experts,
            hidden_size=shape.hidden_size,
            intermediate_size_per_partition=shape.intermediate_size_per_partition,
            fp8_experts=fp8_experts,
            compute_dtype_bytes=compute_dtype_bytes,
            prefer_meta=prefer_meta,
        )
        layers.append(
            LayerWeights(
                path=shape.path,
                num_experts=shape.num_local_experts,
                top_k=shape.top_k_local,
                granule_bytes=sized.granule_bytes,
                resident_bytes=sized.total,
                # Identifies WHICH weights these bytes describe, so the cross-rank digest catches a
                # divergence in the SIZING (one rank resolving a different quant scheme) and not
                # only in the placement. Derived from config, so it is rank-identical by
                # construction; `LayerWeights.from_specs` fills the same field from the live
                # container fingerprints after post_load, and the two are NOT interchangeable --
                # which is exactly what makes a swapped-provenance plan detectable.
                fingerprint=f"est:{sized.scheme}:{sized.source}:{sized.w13}:{sized.w2}",
                priority=(priorities or {}).get(shape.layer_index, 0),
                # PACKING bound for the arena's chunk reservation. The config path cannot see
                # individual components, but a single arena row is one component of ONE container,
                # so the larger of the two containers is a sound upper bound -- and a tight-ish one
                # (the weight stack is ~90% of a quantized container, and w13 is 2x w2). Without
                # this the reservation assumes perfect next-fit packing and comes up ~25-30% short
                # on this shape; see `arena_reservation_bytes`.
                max_row_bytes=max(int(sized.w13), int(sized.w2)),
                # And the ROWS themselves when either byte model could break the containers down.
                # With them the arena is reserved by enumeration (`exact_arena_reservation_bytes`)
                # instead of against `max_row_bytes`' worst-case packing bound; without them nothing
                # changes and the bound still applies. Note the bound above stays as the container
                # total even when rows are known -- it is the FALLBACK, and a fallback that quietly
                # got tighter would hide a lost enumeration.
                rows=tuple(sized.rows),
            )
        )
        sources.append(sized.source)
        schemes.append(sized.scheme)
        if sized.agreement is not None:
            agreements.append(sized.agreement)

    if sources and all(s == "meta" for s in sources):
        byte_source = "meta"
    elif sources and all(s == "analytic" for s in sources):
        byte_source = "analytic"
    else:
        byte_source = "mixed" if sources else "none"
    diagnostics: Dict[str, Any] = {
        "byte_source": byte_source,
        "schemes": sorted(set(schemes)),
        "n_offloadable_layers": len(layers),
        "skipped": tuple(skipped),
        "sizing_agreement_min": min(agreements) if agreements else None,
        "sizing_agreement_max": max(agreements) if agreements else None,
        "compute_dtype_bytes": compute_dtype_bytes,
        # Recorded because the CPU-COMPUTE tier is REFUSED under EP: `MoELayer._ep_dispatch`
        # all_gathers and re-orders rows across ranks, so a CPU partial computed from pre-gather
        # rows would be added to the wrong tokens (`layers/moe.py` asserts this). `cpu_tier_gate`
        # reads it rather than re-deriving "did EP apply", which is a three-conjunct question the
        # engine-level toggle alone cannot answer.
        "expert_parallel": any(sh.expert_parallel for sh in shapes if sh.offloadable),
        "ep_size": max((sh.ep_size for sh in shapes if sh.offloadable), default=1),
    }
    return tuple(layers), diagnostics


# =================================================================================================
# The CPU-COMPUTE tier gate and sweep
#
# Everything here answers "MAY this plan put layers on the CPU, and what would it buy?" -- never
# "turn the feature on". The tier is opt-in at the call site (`resolve_weight_plan(num_cpu_layers=)`)
# because a CPU layer needs a NATIVE EXECUTOR that does not exist as a loadable `.so` yet
# (`cpu_worker.NativeBackend` declares its ABI and refuses to guess). The sweep runs on every boot
# regardless, so the lever is visible in the banner rather than discovered by reading this file.
# =================================================================================================


@dataclass(frozen=True)
class CpuTierGate:
    """May the CPU tier be used for this model on this box, and with which WLoad policy."""

    allowed: bool
    reason: str
    wload: Optional[str]
    layout_fraction: float
    max_layers: int
    threads_per_rank: int
    total_threads: int

    def describe(self) -> str:
        head = "ALLOWED" if self.allowed else "REFUSED"
        return (
            f"cpu-compute tier {head}: {self.reason}"
            + (
                f" [wload={self.wload}, layout x{self.layout_fraction:.2f}, "
                f"{self.threads_per_rank}T/rank = {self.total_threads} physical cores]"
                if self.allowed
                else ""
            )
        )


def cpu_tier_gate(
    diagnostics: Dict[str, Any],
    *,
    local_ranks: int,
    n_offloadable: int,
    repacked: bool = False,
    threads_per_rank: Optional[int] = None,
    cpu_prior: Any = None,
) -> CpuTierGate:
    """Four independent refusals, each of which is a real failure mode rather than a policy.

    1. NO CPU CORE FOR THIS FORMAT. `sizing.cpu_wload_policy` returns None for a scheme no
       `tools/cpu_moe/wload.hpp` policy can decode. Placing such a layer would bake valid tensors
       into pageable memory that nothing is able to read -- a dead layer at the first token rather
       than a boot error. A mixed-scheme checkpoint is refused unless EVERY scheme has a core.
    2. EXPERT PARALLEL IS ON. `MoELayer._ep_dispatch` all_gathers and re-orders rows across ranks,
       so a CPU partial computed from PRE-gather rows lands on the wrong tokens -- fluent, plausible
       and wrong. `layers/moe.py` asserts this at the seam; refusing here means the operator sees it
       at plan time with the reason, instead of at the first forward with an assert.
    3. THE CORE BUDGET DOES NOT FIT. `threads_per_rank x local_ranks` physical cores must be free
       after the engine and the OS. This is a refusal and not a clamp because the native pool's spin
       barrier does not degrade gracefully when starved -- see `cpu_tier.CoreBudget`.
    4. NOTHING TO PLACE.
    """
    from .cpu_tier import CPU_TIER_PRIOR, CpuTierError

    cp = cpu_prior or CPU_TIER_PRIOR
    tpr = int(threads_per_rank or cp.default_threads_per_rank)
    total = tpr * max(1, int(local_ranks))
    schemes = [str(k) for k in diagnostics.get("schemes", ())]
    # BOTH SPELLINGS. `diagnostics["schemes"]` is scheme kinds from the config path and container
    # class names from the model path (`observed_planned_layers` fills it from `GranuleSpec.kind`),
    # and the model path is the one every real serve takes. See `sizing._CPU_WLOAD_BY_CONTAINER`.
    wloads = {sc: cpu_wload_policy_for_kind(sc) for sc in schemes}
    missing = sorted(sc for sc, w in wloads.items() if w is None)
    if not n_offloadable:
        return CpuTierGate(False, "no offloadable MoE layers", None, 1.0, 0, tpr, total)
    if missing:
        return CpuTierGate(
            False,
            f"no CPU expert core reads {', '.join(missing)} -- sizing.cpu_wload_policy has no "
            f"wload.hpp policy for it. Adding one is a table row in sizing.py plus the policy in "
            f"tools/cpu_moe/wload.hpp (KERNEL_CORE_POLICY: a weight format is a WLoad policy, "
            f"never a new kernel and never a new tier)",
            None, 1.0, 0, tpr, total,
        )
    if bool(diagnostics.get("expert_parallel")):
        return CpuTierGate(
            False,
            f"expert parallel is active (ep_size={diagnostics.get('ep_size')}). "
            f"MoELayer._ep_dispatch re-orders rows across ranks, so a CPU partial computed from "
            f"pre-gather rows would be added to the WRONG TOKENS. Serve without --enable-ep to use "
            f"the CPU tier, or pack the CPU partial into the gather first",
            None, 1.0, 0, tpr, total,
        )
    try:
        cp.cores.assert_fits(total, what="the CPU MoE tier")
    except CpuTierError as e:
        return CpuTierGate(False, str(e), None, 1.0, 0, tpr, total)
    wload = next(iter(wloads.values()))
    return CpuTierGate(
        True,
        f"{n_offloadable} layer(s) eligible",
        wload,
        cp.layout_fraction_for(wload, repacked=repacked),
        int(n_offloadable),
        tpr,
        total,
    )


def cpu_tier_sweep(
    layers: Sequence[LayerWeights],
    *,
    device_budget_bytes: int,
    gate: CpuTierGate,
    prior: OffloadPrior = PHASE0_PRIOR,
    local_ranks: int = 1,
    arena_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    batch: int = 1,
    loaded: bool = True,
    handoff_us: Optional[float] = None,
    cpu_prior: Any = None,
) -> List[Dict[str, Any]]:
    """Project the plan at every K in {0, 25%, 50%, 75%, 100%} of the offloadable layers.

    Both the CAPACITY answer (pinned arena bytes relieved) and the THROUGHPUT answer, because they
    are different levers and on this box only one of them has ever been the binding constraint.
    Every row carries the RAW projection and the same figure multiplied by
    `CpuTierPrior.projection_fidelity` -- the one measured-vs-projected ratio this model has, and
    the reason no row here may be quoted as a prediction.

    `handoff_us` defaults to the PESSIMISTIC end of the unmeasured bracket. That direction is
    deliberate: the optimistic end would make a sweep row look better than anything that has been
    run, and the whole point of the bracket is that it has not been measured on a card.
    """
    from .cpu_tier import CPU_TIER_PRIOR, CpuTierMode, project_cpu_tier

    cp = cpu_prior or CPU_TIER_PRIOR
    hus = cp.handoff_us_bracket[1] if handoff_us is None else float(handoff_us)
    n = len(layers)
    if not n:
        return []
    ks = sorted({0, n // 4, n // 2, (3 * n) // 4, n})
    if not gate.allowed:
        ks = [0]
    rows: List[Dict[str, Any]] = []
    for k in ks:
        plan = plan_three_tier(
            layers,
            device_budget_bytes=device_budget_bytes,
            num_cpu_layers=k,
            cpu_layout_fraction=gate.layout_fraction if k else 1.0,
        )
        proj = project_cpu_tier(
            host_bytes_per_rank=plan.host_active_bytes(batch),
            device_bytes_per_rank=plan.device_active_bytes(batch),
            cpu_bytes_per_rank=plan.cpu_active_bytes(batch),
            num_cpu_layers=plan.num_cpu_layers,
            mode=CpuTierMode.BLOCK if k else CpuTierMode.OFF,
            handoff_us_per_layer=hus if k else 0.0,
            threads_per_rank=gate.threads_per_rank,
            prior=prior,
            cpu_prior=cp,
            num_ranks=local_ranks,
            loaded=loaded,
            num_layers=n,
            check_cores=bool(k),
        )
        rows.append(
            {
                "cpu_layers": k,
                "device_layers": plan.num_device_layers,
                "host_layers": plan.num_host_layers,
                "pinned_node_bytes": plan_arena_reservation_bytes(plan, arena_chunk_bytes)
                * local_ranks,
                "pageable_node_bytes": plan.cpu_resident_bytes * local_ranks,
                "pinned_saved_node_bytes": plan.pinned_arena_bytes_saved_vs_host * local_ranks,
                "step_ms": proj.step_ms,
                "tok_s": proj.tok_s,
                "calibrated_tok_s": proj.calibrated_tok_s,
                "graph_segments": proj.graph_segments,
                "rel_rms": proj.rel_rms if k else 0.0,
            }
        )
    return rows


def format_cpu_tier_sweep(rows: Sequence[Dict[str, Any]], gate: CpuTierGate) -> str:
    if not rows:
        return "cpu-compute tier sweep: n/a (no offloadable layers)"
    out = [gate.describe(),
           "  cpuL  devL  hostL   pinned/node  pageable/node    step ms   tok/s   tok/s*  segs"]
    for r in rows:
        out.append(
            f"  {r['cpu_layers']:>4}  {r['device_layers']:>4}  {r['host_layers']:>5}  "
            f"{_gb(r['pinned_node_bytes']):>12}  {_gb(r['pageable_node_bytes']):>13}  "
            f"{r['step_ms']:>9.2f}  {r['tok_s']:>6.2f}  {r['calibrated_tok_s']:>6.2f}  "
            f"{r['graph_segments']:>4}"
        )
    out.append(
        "  tok/s is the RAW projection; tok/s* is that figure x the ONE measured "
        "model-fidelity ratio (0.637). segs = separately-captured device segments this placement "
        "forces; anything above 1 means today's whole-forward CUDAGraph capture cannot be used."
    )
    return "\n".join(out)


# =================================================================================================
# Capacity arithmetic -- the part that decides whether the serve can boot
# =================================================================================================


def arena_reservation_bytes(
    host_payload_bytes: int,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    max_row_bytes: int = 0,
) -> int:
    """Host RAM one rank's arena will actually PIN to hold `host_payload_bytes` of weights.

    NOT the same number as the payload, and the difference is what decides a boot. The arena is
    reserved in whole chunks: `StageARuntime.attach_host_arena` calls
    `arena.reserve([], extra_bytes=plan.host_resident_bytes,
    extra_max_region_bytes=plan.max_host_row_bytes)`, `chunk_plan.plan_regions` turns that into
    `extra_chunks = headroom_chunks(extra, chunk, max_row)` (whole chunks, because next-fit
    abandons a partial tail), and `PinnedWeightArena.attach` then `hipHostMalloc`s **every chunk at
    full `chunk_bytes`** and first-touches it. So the pinned footprint is the payload rounded UP to
    the 2 GiB chunk -- up to 2 GiB per rank, 4 GiB across this box's two ranks, of host RAM that the
    plan-time inequality used to ignore.

    `max_row_bytes` IS THE SECOND HALF OF THE SAME MISTAKE, and it is the larger half. A region may
    never straddle two chunks (they are independent `hipHostMalloc` mappings with non-contiguous
    device pointers), so `BumpAllocator` abandons a chunk's tail the moment the next row does not
    fit in it. With rows of up to `m` bytes only `chunk - m` bytes per chunk are GUARANTEED
    placeable, which is `chunk_plan.headroom_chunks`' arithmetic and is why that function's
    docstring tells callers who know their largest row to pass it. Omitting it (`m = 0`) charges
    `ceil(payload / chunk)` -- i.e. it assumes perfect packing, which next-fit does not do. On this
    feature's shape the error is not marginal: a fused w13 weight stack is a ~1 GiB row in a 2 GiB
    chunk, so a chunk holds one layer and abandons ~0.5 GiB, ~25-30% of every chunk. A 55 GiB
    payload reserved as 28 chunks then holds ~40 GiB and the remaining ~15 GiB of rows fall out of
    the arena entirely -- coming back through `ArenaMemPool`'s `hipMalloc` fallback as **VRAM**,
    on a 16 GB card, budgeted as host RAM. `seal()`'s `assert_arena_clean()` refuses that boot, but
    it refuses it after the arena is pinned and the checkpoint is loaded, which is precisely the
    "terrible failure" `host_capacity.py` exists to convert into a millisecond config error.

    Passing it makes the reservation a GUARANTEE rather than an estimate, so an infeasible shape is
    refused by `raise_if_infeasible()` before a page is pinned -- and the operator is told to raise
    the device tier (or `MINISGL_WEIGHT_ARENA_CHUNK_MIB`, which is the lever that shrinks `m/chunk`)
    instead of watching a bind die with an unrelated OOM.

    That gap is not cosmetic in the regime this feature lives in. On the target-shaped fixture,
    granting exactly the old `required_device_bytes` produced a plan the resolver called FEASIBLE at
    27.84 GiB/rank while the arena went on to pin 28.00 GiB/rank = 56.00 GiB node against a 55.80 GiB
    usable ceiling: the resolver's own "raise the device tier to >= X" advice returned a tier that
    still aborts mid-pin, with the box in swap. That is exactly the "terrible failure" mode
    `host_capacity.py`'s docstring exists to prevent, produced by the planner itself.

    `chunk_bytes` is a parameter and not an env read, so the planner stays a pure function of its
    arguments (two ranks must derive the identical plan). The engine passes
    `ArenaSettings.chunk_bytes`, so an operator who moves `MINISGL_WEIGHT_ARENA_CHUNK_MIB` moves this
    charge with it.
    """
    if host_payload_bytes <= 0:
        return 0
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    chunk = effective_chunk_bytes(chunk_bytes, max_row_bytes)
    return headroom_chunks(int(host_payload_bytes), chunk, int(max_row_bytes)) * chunk


def exact_arena_reservation_bytes(
    rows: Sequence[Any],
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    *,
    align: int = ALIGN,
) -> int:
    """Host RAM one rank's arena will pin for an ENUMERATED row list. The exact answer, not a bound.

    THIS IS THE FIX `arena_reservation_bytes`' docstring could not make. That function is a
    GUARANTEE derived from one number (`max_row_bytes`): next-fit may abandon up to `m` bytes at
    every chunk boundary, so only `chunk - m` per chunk is provably placeable. It has to assume the
    worst row lands at the worst offset in EVERY chunk because it has nothing else to go on. On the
    target shape -- 48 layers of six rows (384/48/12 MiB for w13, 192/24/6 MiB for w2), 666 MiB per
    layer, 2 GiB chunks -- that charges 20 chunks (40.00 GiB/rank, 80.00 GiB/node) where the real
    allocator fits three whole layers per chunk and uses 16 (32.00 GiB/rank). The +28% is not a
    rounding artefact: at a 55.80 GiB node budget it is the difference between needing 11.06 GiB/rank
    of device tier on a 16 GiB card and needing far less, i.e. ~1.4M KV tokens per rank.

    `sizing.meta_gemm_spec` builds the real container under `torch.device("meta")` before
    `load_state_dict`, so the row sizes ARE knowable at reserve() time; there was never a reason to
    reserve anonymously. This runs the SAME `chunk_plan.plan_regions` / `BumpAllocator` that
    `PinnedWeightArena.reserve` and `allocate_raw` use, so the planner cannot drift from the arena.

    `suggest_chunk_bytes` (not `effective_chunk_bytes`) mirrors `reserve()`'s growth rule for NAMED
    regions: a named region only needs a chunk it fits in, while the anonymous-headroom bound needs
    one strictly larger than any row or `chunk - m` is zero.
    """
    rows = list(rows)
    if not rows:
        return 0
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    chunk = suggest_chunk_bytes(rows, int(chunk_bytes))
    return plan_regions(rows, chunk, align=align).reserved_bytes


def plan_arena_reservation_bytes(plan: OffloadPlan, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> int:
    """Host RAM ONE rank's arena pins for `plan`. THE definition — every caller goes through here.

    Exact when the plan enumerated its rows, the `arena_reservation_bytes` bound when it did not.
    Deliberately one function and not two call sites: `resolve_weight_plan` computes the feasibility
    verdict from this and `WeightPlanResolution.host_reservation_bytes_per_rank` reports it, and the
    two disagreeing produces the worst available outcome — a resolution that says FEASIBLE while its
    own printed reservation exceeds the ceiling, or vice versa.
    """
    rows = plan.host_row_requests()
    if rows:
        return exact_arena_reservation_bytes(rows, chunk_bytes)
    return arena_reservation_bytes(
        plan.host_resident_bytes, chunk_bytes, plan.max_host_row_bytes
    )


def effective_chunk_bytes(chunk_bytes: int, max_row_bytes: int = 0) -> int:
    """The chunk size `PinnedWeightArena.reserve()` will actually use, given a row bound.

    Mirrors `reserve()`'s own `suggest_chunk_bytes` growth so the planner charges the same chunks
    the arena pins. A row may never straddle, so the chunk must be STRICTLY larger than the largest
    row -- equal is not enough: `headroom_chunks` would then compute `usable = chunk - m == 0` and
    the reservation would be unbounded. Growing by one `CHUNK_GRANULE` past the row is the smallest
    size at which the guarantee is non-vacuous.

    A pure function of its two arguments, like everything else in the planner, so two ranks derive
    the same number.
    """
    m = int(max_row_bytes)
    if m <= 0:
        return int(chunk_bytes)
    return max(int(chunk_bytes), round_up(m + CHUNK_GRANULE, CHUNK_GRANULE))


def host_arena_ceiling_bytes(local_ranks: int, prior: OffloadPrior = PHASE0_PRIOR) -> int:
    """The NODE-WIDE pinned-host budget the plan may commit, after the headroom derate.

    The raw ceiling is P3b's measured table (`host_capacity.P3B_PINNED_CEILING_BYTES`) -- 34 GiB at
    one rank, 62 GiB across two. Beyond two ranks it stays at the two-rank figure: that number is a
    node `MemAvailable` floor, not a per-rank allowance, so adding ranks does not add RAM.

    The derate is not conservatism for its own sake. P3b measured 62 GiB with NO ENGINE LOADED and
    was ALREADY SWAPPING when it got there (114,813 pages). A live serve additionally holds the
    tokenizer and scheduler processes, whatever page cache the checkpoint load just filled, and the
    engine's own host allocations. `prior.host_arena_headroom_fraction` is the single explicit knob
    that decides "does it boot", so it is a named field rather than a magic number in an inequality.
    """
    if local_ranks < 1:
        raise ValueError(f"local_ranks must be >= 1, got {local_ranks}")
    known = max(P3B_PINNED_CEILING_BYTES)
    raw = P3B_PINNED_CEILING_BYTES.get(local_ranks, P3B_PINNED_CEILING_BYTES[known])
    return int(raw * prior.host_arena_headroom_fraction)


def required_device_bytes(
    layers: Sequence[Any],
    ceiling_total_bytes: int,
    local_ranks: int,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> int:
    """Smallest per-rank device tier that brings the NODE host arena under the ceiling.

    Walks the SAME `(-priority, declaration index)` order `plan_layer_granular` uses, so the answer
    is the plan the operator would actually get if they granted that many bytes -- not an abstract
    byte count they cannot act on. Returns 0 when the all-host plan already fits.

    The inequality is over what the arena will PIN, not over the payload. Comparing the payload here
    is how the "raise the device tier to >= X" message came to name a tier that still aborts
    mid-pin: the number an operator is told to grant has to be a number that boots, or the message
    sends them round a loop with the box in swap each time.

    PINNED IS `exact_arena_reservation_bytes` WHENEVER THE ROWS ARE ENUMERATED, and only
    `arena_reservation_bytes`' worst-case bound when they are not. The bound over-charges by the
    tail next-fit *might* abandon at every boundary; the enumeration charges the tails it *does*
    abandon. Since this function answers "how much VRAM must the operator surrender", the difference
    lands directly on a 16 GiB card.

    This is the arithmetic the whole feature turns on. Phase 0 3.4: all-host is 68.8 GB against a
    62 GiB two-rank ceiling, so on the target checkpoint this returns a non-zero number and the
    device tier is a CAPACITY PREREQUISITE, not an optimisation.
    """
    if local_ranks < 1:
        raise ValueError(f"local_ranks must be >= 1, got {local_ranks}")

    order = sorted(range(len(layers)), key=lambda i: (-layers[i].priority, i))

    def _row_bound(host_idx: Sequence[int]) -> int:
        """Packing bound over the layers still on HOST at this point in the walk.

        Recomputed rather than fixed, because it SHRINKS as layers move to the device: the greedy
        fill takes them in `(-priority, declaration index)` order, so the answer this returns is
        the bound the arena will actually face at the tier it is recommending. Using the whole
        model's largest row throughout would over-charge every candidate tier and hand the operator
        a bigger device tier than they need."""
        return max((layers[i].row_bound for i in host_idx), default=0)

    def _rows(host_idx: Sequence[int]) -> Optional[List[Any]]:
        """The host rows in CARVE order, or None when any host layer could not enumerate.

        Sorted by DECLARATION index, not by the priority order this walk iterates in: the bake
        binds seams in declaration order (`discover_moe_layers`), and the arena's allocator is
        order-sensitive by construction, so a reservation laid out in priority order would not be
        the reservation the bake consumes.
        """
        idx = sorted(host_idx)
        if not idx or any(not layers[i].rows for i in idx):
            return None
        # Charged as ONE ordered sequence — see `chunk_plan.torch_charged_rows`. This is the
        # feasibility side of the same arithmetic `OffloadPlan.host_row_requests` reserves with, so
        # the two must go through the same function or the walk picks a tier the arena then refuses.
        flat = [
            (f"{layers[i].path}.{name}", nb) for i in idx for name, nb in layers[i].rows
        ]
        return [
            RegionRequest(name, charge, forecast=True)
            for name, charge in torch_charged_rows(flat)
            if charge > 0
        ]

    def _fits(host_payload: int, host_idx: Sequence[int]) -> bool:
        rows = _rows(host_idx)
        if rows is not None:
            pinned = exact_arena_reservation_bytes(rows, chunk_bytes)
        else:
            pinned = arena_reservation_bytes(host_payload, chunk_bytes, _row_bound(host_idx))
        return pinned * local_ranks <= ceiling_total_bytes

    host = sum(lw.resident_bytes for lw in layers)
    remaining = list(order)
    if _fits(host, remaining):
        return 0
    used = 0
    for pos, i in enumerate(order):
        size = layers[i].resident_bytes
        used += size
        host -= size
        remaining = order[pos + 1 :]
        if _fits(host, remaining):
            return used
    return used


# =================================================================================================
# The resolution
# =================================================================================================


@dataclass(frozen=True)
class WeightPlanResolution:
    """The resolved plan plus everything a caller needs to log it, gate on it, and bill it.

    `plan` is `placement.OffloadPlan` -- the placement decision itself, unmodified, so every
    consumer (arena, registry, accounting) reads the same object the placement module produced.
    """

    plan: OffloadPlan
    layers: Tuple[LayerWeights, ...]
    prior: OffloadPrior
    tp_size: int
    dp_size: int
    local_ranks: int
    device_budget_bytes: int
    host_ceiling_bytes: int
    required_device_bytes: int
    feasible: bool
    reason: str
    projection: Projection
    all_host_projection: Projection
    sweep_text: str
    byte_source: str
    diagnostics: Dict[str, Any]
    warnings: Tuple[str, ...] = ()
    # Arena chunk size the capacity arithmetic was charged against. Carried so the boot log names the
    # granularity the reservation was rounded to; a plan resolved under one chunk size and attached
    # under another is a silent under-charge, which is why the engine passes ArenaSettings.chunk_bytes
    # rather than letting this default drift away from the arena's.
    arena_chunk_bytes: int = DEFAULT_CHUNK_BYTES
    # -- CPU-COMPUTE tier ------------------------------------------------------------------------
    # Always populated, even on a two-tier plan: `cpu_gate` says WHY the tier is or is not usable
    # and `cpu_sweep_text` prices it at every K. That is deliberate — a lever that only appears in
    # the log once somebody has already enabled it is a lever nobody finds.
    cpu_gate: Optional[CpuTierGate] = None
    cpu_sweep: Tuple[Dict[str, Any], ...] = ()
    cpu_sweep_text: str = ""
    cpu_projection: Optional[Any] = None

    # -- the numbers other subsystems bill against ----------------------------------------------
    @property
    def device_bytes(self) -> int:
        """Per-rank VRAM the device-resident expert stacks occupy. The VRAM-accounting subject."""
        return self.plan.device_resident_bytes

    @property
    def host_bytes_per_rank(self) -> int:
        return self.plan.host_resident_bytes

    @property
    def host_bytes_per_node(self) -> int:
        return self.plan.host_resident_bytes * self.local_ranks

    @property
    def host_reservation_bytes_per_rank(self) -> int:
        """What one rank's arena will PIN -- the payload rounded up to a whole arena chunk.

        This, not `host_bytes_per_rank`, is the number that has to clear the ceiling: `attach()`
        `hipHostMalloc`s and first-touches every chunk at full size.

        EXACT when the plan enumerated its rows (`OffloadPlan.host_row_requests`) -- the same
        `plan_regions` call `PinnedWeightArena.reserve` will make, so the feasibility answer and the
        reservation cannot disagree. Only when the rows are unknown does this fall back to
        `arena_reservation_bytes`' worst-case packing BOUND, which over-charges +28% on the target
        shape (20 chunks against a real 16) and buys that with device tier the card does not have.
        """
        return plan_arena_reservation_bytes(self.plan, self.arena_chunk_bytes)

    @property
    def host_reservation_bytes_per_node(self) -> int:
        return self.host_reservation_bytes_per_rank * self.local_ranks

    @property
    def cpu_bytes_per_rank(self) -> int:
        """PAGEABLE host bytes the CPU tier holds. Not pinned, so not against the arena ceiling."""
        return self.plan.cpu_resident_bytes

    @property
    def pinned_bytes_saved_per_node(self) -> int:
        """THE CAPACITY WIN. Pinned-arena bytes this plan does not need because layers went to CPU.

        Priced in the DEVICE layout on purpose: it is what those same layers WOULD have cost on the
        host tier, which is the counterfactual an operator is comparing against. The extra 10% the
        repack removes shows up separately, as `total_host_bytes` being smaller than the two-tier
        plan's `host_bytes_per_node` by more than this.
        """
        return self.plan.pinned_arena_bytes_saved_vs_host * self.local_ranks

    @property
    def total_host_bytes_per_node(self) -> int:
        """All host RAM the expert weights occupy: pinned arena + pageable CPU tier.

        `host_capacity.check_capacity` gates on MemAvailable, which does not care whether a page is
        pinned; only the `hipHostMalloc` ceiling does. So `host_reservation_bytes_per_node` gates
        FEASIBILITY and this gates the box not swapping.
        """
        return self.plan.total_host_bytes * self.local_ranks

    @property
    def shortfall_bytes(self) -> int:
        return max(0, self.required_device_bytes - self.device_budget_bytes)

    @property
    def enabled(self) -> bool:
        """False means the code path is a genuine no-op -- nothing is host-resident."""
        return not self.plan.is_empty

    # -- gates -----------------------------------------------------------------------------------
    def clears_kill_gate(self) -> bool:
        """K4: GPU-side streaming must beat llama.cpp/1.57 (3.178 tok/s) or ship nothing."""
        return self.projection.tok_s >= self.prior.kill_tok_s

    @property
    def acceptance_threshold_tok_s(self) -> float:
        """A1.7: the measured serve must reach this. PROJECTED input -- never quote it as achieved."""
        return self.projection.tok_s * self.prior.accept_fraction

    def raise_if_infeasible(self) -> None:
        """Abort the boot with the numbers and the fix, before the first shard is opened."""
        if not self.feasible:
            raise PlacementError(self.failure_message())

    def failure_message(self) -> str:
        return (
            f"weight offload cannot fit this model: "
            f"{_gb(self.host_reservation_bytes_per_node)} of pinned host arena across "
            f"{self.local_ranks} rank(s) ({_gb(self.host_bytes_per_rank)}/rank of weights, pinned "
            f"as whole {_gb(self.arena_chunk_bytes)} chunks) exceeds the usable ceiling "
            f"{_gb(self.host_ceiling_bytes)} "
            f"({P3B_NOTE}; derated x{self.prior.host_arena_headroom_fraction:.2f}). "
            f"NOTE the separate live gate: host_capacity.check_capacity() additionally requires "
            f"MemAvailable to stay {_gb(DEFAULT_FLOOR_BYTES)} above the arena at ALLOCATION time, "
            f"and this baked ceiling does not include it.\n"
            f"  device tier granted : {_gb(self.device_budget_bytes)}/rank\n"
            f"  device tier required: {_gb(self.required_device_bytes)}/rank "
            f"(short by {_gb(self.shortfall_bytes)})\n"
            f"  fixes: raise the device tier to >= {_gb(self.required_device_bytes)}/rank (each GiB "
            f"is ~200k KV tokens surrendered), or serve a smaller checkpoint, or add host RAM."
        )

    # -- cross-rank agreement --------------------------------------------------------------------
    def sizing_digest(self) -> str:
        """Hash of WHICH WEIGHTS these bytes describe, not just how many there are.

        `OffloadPlan.digest()` deliberately hashes only the placement decision -- path, stack, bytes,
        expert count, top_k -- because that is what has to match for two ranks to agree on where a
        layer lives. It does NOT carry `LayerWeights.fingerprint`: `plan_layer_granular` builds
        `LayerPlacement`s and the fingerprint is not one of their fields. So a comment claiming the
        plan digest catches a divergence in the SIZING was wrong, and this method is the missing
        half: it hashes the provenance string too, so two ranks that resolved DIFFERENT quant
        schemes, or different `meta`/`analytic` byte sources, that happen to total the same bytes are
        still distinguishable.

        Kept separate from the plan digest on purpose. The post-`post_load()` reconciliation
        re-plans from `LayerWeights.from_specs`, whose fingerprints are content hashes of live
        tensors and can never equal the `est:` strings built here -- so folding provenance into the
        PLAN digest would make estimate-vs-measured comparison permanently fail. Placement agreement
        and sizing agreement are two questions; they get two hashes.
        """
        import hashlib

        h = hashlib.sha256()
        for lw in self.layers:
            h.update(
                f"{lw.path}|{lw.fingerprint}|{lw.resident_bytes}|{lw.granule_bytes}|"
                f"{lw.num_experts}|{lw.top_k}|{lw.priority}|".encode()
            )
        return h.hexdigest()[:16]

    def agreement_digest(self) -> str:
        """Everything a second rank must derive identically. One string, comparable in one gather.

        Covers the placement, the sizing provenance, and the INPUTS the resolver could not vet for
        itself -- above all `device_budget_bytes`, which the engine supplies per-rank and which this
        module's docstring can only ASK to be a stable quantity. A caller who wires it to a live
        `mem_get_info` delta produces a budget that differs by tens of MiB between the two rank
        processes; that is enough to move a layer across the greedy fill boundary, and from there the
        two ranks hold different device tiers, `_determine_num_pages` bills different `model_memory`
        (it corrects with the per-rank `self._woff.model_memory_correction()`, and `num_pages` is NOT
        cross-rank reduced), and the two KV pools end up different sizes. Nothing downstream detects
        that. This does.
        """
        import hashlib

        h = hashlib.sha256()
        h.update(
            (
                f"plan={self.plan.digest()}|sizing={self.sizing_digest()}|"
                f"budget={self.device_budget_bytes}|ceiling={self.host_ceiling_bytes}|"
                f"ranks={self.local_ranks}|tp={self.tp_size}|dp={self.dp_size}|"
                f"required={self.required_device_bytes}|feasible={int(self.feasible)}|"
                f"prior={self.prior.name}|bytes={self.byte_source}|"
                f"chunk={self.arena_chunk_bytes}|enabled={int(self.enabled)}"
            ).encode()
        )
        return h.hexdigest()[:16]

    def assert_rank_agreement(self, group: Any = None, *, gather: Any = None) -> Tuple[str, ...]:
        """Prove -- not assume -- that every rank resolved the same plan. Boot-time, CPU-only.

        `placement.py`'s docstring says the digest exists "so a caller can prove agreement across
        ranks with one tiny collective rather than trusting that the derivation was pure". Nothing
        called it. Purity is an argument, and this box has burned people four separate times on this
        stack reporting success over wrong state; the argument also has a real hole, because
        `device_budget_bytes` is an input the resolver cannot vet (see `agreement_digest`).

        THE COLLECTIVE ITSELF MUST NOT BE THE DESYNC. Every rank enters the gather unconditionally
        and the comparison happens strictly AFTER it returns, so a rank that is about to raise still
        participates and its peer never blocks forever waiting for it. For the same reason this must
        be called on EVERY rank, including ranks whose plan is empty -- `bake._resolve_driver` calls
        it BEFORE the `if not resolution.enabled` early return for exactly that reason. Moving it
        after that branch would hang the serve the first time the two ranks disagreed about whether
        offload is on at all, which is the very condition it exists to catch.

        `gather` is an injection point for tests: any callable taking this rank's digest and
        returning one digest per rank. With neither `group` nor `gather`, or on a single-rank group,
        this is a no-op returning just this rank's digest.
        """
        mine = self.agreement_digest()
        if gather is not None:
            digests = tuple(str(d) for d in gather(mine))
        elif group is None:
            return (mine,)
        else:
            try:
                import torch.distributed as dist
            except Exception:  # pragma: no cover - torch-free host
                return (mine,)
            if not dist.is_available() or not dist.is_initialized():
                return (mine,)
            world = dist.get_world_size(group)
            if world <= 1:
                return (mine,)
            box: List[Any] = [None] * world
            dist.all_gather_object(box, mine, group=group)
            digests = tuple(str(d) for d in box)

        if len(set(digests)) > 1:
            raise PlacementError(
                "weight-offload plan DIVERGED across ranks: "
                + ", ".join(f"rank{i}={d}" for i, d in enumerate(digests))
                + ".\nThe plan is supposed to be a pure integer function of config, so a divergence "
                "means one of its inputs was not: almost always `device_budget_bytes` derived from "
                "a live per-rank free-memory reading, or a config field that differs between the "
                "rank processes. Ranks that disagree place different layers on the host stack, size "
                "different KV pools (num_pages is not cross-rank reduced) and serve different "
                "weights with no other error anywhere. Derive the device tier from a STABLE "
                "quantity -- card total memory or a configured tier -- never from mem_get_info.\n"
                f"  this rank: plan={self.plan.digest()} sizing={self.sizing_digest()} "
                f"budget={self.device_budget_bytes} ceiling={self.host_ceiling_bytes} "
                f"tp={self.tp_size} dp={self.dp_size} enabled={self.enabled}"
            )
        return digests

    # -- VRAM accounting -------------------------------------------------------------------------
    def assert_device_accounting(
        self, observed_delta_bytes: int, *, tolerance_bytes: int = 64 << 20
    ) -> None:
        """Boot assertion: the VRAM the device tier actually took == what the plan promised.

        `observed_delta_bytes` MUST be an independent MEASUREMENT off the live post-load containers
        (`StageADriver.observed_device_bytes` / `LayerWeights.from_specs` re-planned through
        `plan_layer_granular`). It must NOT be derived as `offloadable_total - copied_bytes`: with
        `total == host + device` by construction that reduces algebraically to `host - copied`, i.e.
        the copied-bytes check restated, and it can never catch a plan whose byte model disagrees
        with what `post_load()` actually produced. Nor can it be a raw free-VRAM delta — a
        device-resident layer is never reallocated (it is left exactly where `load_state_dict` put
        it), so there is no allocation event, and the delta it sits inside also carries the dense
        weights and the HIP context.

        Call site: between `post_load()` (`engine.py:241`) and `_determine_num_pages`
        (`:245`). There is deliberately NO sixth subtrahend in the KV-sizing arithmetic:
        device-located expert stacks
        are allocated before the free-memory delta is taken, so they land inside `model_memory` and
        are billed exactly once. Subtracting them again is a DOUBLE subtraction that drives
        `available_memory` negative at a 16 GB nominal tier and trips "Not enough memory for KV
        cache" with a misleading cause. This assertion is what makes relying on that safe -- if the
        plan and the allocator ever disagree, the KV pool is being sized against a fiction, and
        Phase 0 produced four separate cases of this stack reporting success over wrong state, so
        the check is on the DATA and never on a return code.
        """
        delta = abs(observed_delta_bytes - self.device_bytes)
        if delta > tolerance_bytes:
            raise AssertionError(
                "weight-offload VRAM accounting mismatch: plan promised "
                f"{_gb(self.device_bytes)} of device-resident expert stacks over "
                f"{self.plan.num_device_layers} layers, allocator moved "
                f"{_gb(observed_delta_bytes)} (delta {_gb(delta)} > tolerance "
                f"{_gb(tolerance_bytes)}). The KV pool is about to be sized against whichever of "
                f"the two is wrong. Plan digest={self.plan.digest()}."
            )

    # -- logging ---------------------------------------------------------------------------------
    def summary_line(self) -> str:
        """The one-line form for the `KV sizing:` annotation and the `[serve]` banner."""
        if not self.enabled:
            return f"weight-offload: OFF ({self.reason})"
        return (
            f"weight-arena={_gb(self.device_bytes)} (dev tier, inside model) "
            f"host={_gb(self.host_bytes_per_rank)}x{self.local_ranks} "
            f"[{self.plan.num_device_layers}/"
            f"{self.plan.num_device_layers + self.plan.num_host_layers} layers device, "
            f"f={self.plan.device_fraction:.3f}]; step floor "
            f"{self.projection.step_ms:.1f} ms -> {self.projection.tok_s:.1f} tok/s projected"
        )

    def render_lines(self) -> List[str]:
        """Multi-line boot log: the decision, the capacity arithmetic, the sweep, the gates."""
        out = [
            f"[weight-offload] {self.summary_line()}",
            f"[weight-offload] reason: {self.reason}",
            f"[weight-offload] prior={self.prior.name} bytes={self.byte_source} "
            f"tp={self.tp_size} dp={self.dp_size} arenas={self.local_ranks} "
            f"digest={self.plan.digest()}",
            f"[weight-offload] {self.plan.describe()}",
        ]
        if self.enabled or not self.feasible:
            out.append(
                f"[weight-offload] capacity: host {_gb(self.host_bytes_per_node)} of weights -> "
                f"{_gb(self.host_reservation_bytes_per_node)} PINNED across "
                f"{self.local_ranks} arena(s) at {_gb(self.arena_chunk_bytes)} chunks, "
                f"vs usable budget {_gb(self.host_ceiling_bytes)} -> "
                + ("FITS" if self.feasible else "DOES NOT FIT")
            )
        if not self.feasible:
            out.extend(f"[weight-offload] {line}" for line in self.failure_message().splitlines())
        for w in self.warnings:
            out.append(f"[weight-offload] WARNING: {w}")
        out.extend(f"[weight-offload] {line}" for line in self.sweep_text.splitlines())
        if self.cpu_sweep_text:
            out.extend(f"[weight-offload] {line}" for line in self.cpu_sweep_text.splitlines())
        if self.plan.num_cpu_layers:
            out.append(
                f"[weight-offload] cpu-tier capacity: "
                f"{_gb(self.pinned_bytes_saved_per_node)} of pinned arena RELIEVED, "
                f"{_gb(self.cpu_bytes_per_rank * self.local_ranks)} held as pageable instead "
                f"(total host {_gb(self.total_host_bytes_per_node)} against MemAvailable, not "
                f"against the {_gb(self.host_ceiling_bytes)} pinned ceiling)"
            )
        if self.cpu_projection is not None:
            out.append(f"[weight-offload] {self.cpu_projection.describe()}")
        out.append(
            f"[weight-offload] gates: projected {self.projection.tok_s:.2f} tok/s vs K4 "
            f"{self.prior.kill_tok_s:.3f} -> {'PASS' if self.clears_kill_gate() else 'KILL'}; "
            f"A1.7 threshold {self.acceptance_threshold_tok_s:.2f} tok/s "
            "(PROJECTED -- A1.7 is measured on a served A/B with graphs on)"
        )
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "feasible": self.feasible,
            "reason": self.reason,
            "digest": self.plan.digest(),
            "device_bytes": self.device_bytes,
            "host_bytes_per_rank": self.host_bytes_per_rank,
            "host_bytes_per_node": self.host_bytes_per_node,
            # The PINNED figures are the ones the ceiling is compared against; the payload figures
            # above are what the weights weigh. An artifact that records only the payload cannot be
            # used to explain a boot that aborted mid-pin.
            "host_reservation_bytes_per_rank": self.host_reservation_bytes_per_rank,
            "host_reservation_bytes_per_node": self.host_reservation_bytes_per_node,
            "arena_chunk_bytes": self.arena_chunk_bytes,
            "total_resident_bytes": self.plan.total_resident_bytes,
            "unused_device_bytes": self.plan.unused_device_bytes,
            "device_budget_bytes": self.device_budget_bytes,
            "host_ceiling_bytes": self.host_ceiling_bytes,
            "required_device_bytes": self.required_device_bytes,
            "shortfall_bytes": self.shortfall_bytes,
            "device_fraction": self.plan.device_fraction,
            "num_device_layers": self.plan.num_device_layers,
            "num_host_layers": self.plan.num_host_layers,
            "device_layers": [p.path for p in self.plan.placements if p.kind is StackKind.DEVICE],
            "host_layers": [p.path for p in self.plan.placements if p.kind is StackKind.HOST],
            # CPU-COMPUTE tier. Reported alongside rather than folded into `host_layers`: those
            # bytes are PAGEABLE, so they are relieved against MemAvailable and NOT against the
            # pinned `hipHostMalloc` ceiling that `host_ceiling_bytes` above is about. Zero and []
            # on every two-tier plan, so the diagnostics shape is a superset of the old one.
            "num_cpu_layers": self.plan.num_cpu_layers,
            "cpu_layers": [p.path for p in self.plan.placements if p.kind is StackKind.CPU],
            "cpu_resident_bytes": self.plan.cpu_resident_bytes,
            "cpu_pinned_arena_bytes_saved": self.plan.pinned_arena_bytes_saved_vs_host,
            "cpu_pinned_arena_bytes_saved_per_node": self.pinned_bytes_saved_per_node,
            "total_host_bytes_per_node": self.total_host_bytes_per_node,
            "cpu_block_contiguous": self.plan.cpu_block_is_contiguous,
            "cpu_tier_allowed": bool(self.cpu_gate and self.cpu_gate.allowed),
            "cpu_tier_reason": self.cpu_gate.reason if self.cpu_gate else "",
            "cpu_tier_wload": self.cpu_gate.wload if self.cpu_gate else None,
            "cpu_tier_threads_per_rank": self.cpu_gate.threads_per_rank if self.cpu_gate else 0,
            "cpu_tier_sweep": list(self.cpu_sweep),
            "cpu_projected_tok_s": (
                self.cpu_projection.tok_s if self.cpu_projection is not None else None
            ),
            "cpu_calibrated_tok_s": (
                self.cpu_projection.calibrated_tok_s if self.cpu_projection is not None else None
            ),
            "cpu_graph_segments": (
                self.cpu_projection.graph_segments if self.cpu_projection is not None else None
            ),
            "projected_step_ms": self.projection.step_ms,
            "projected_tok_s": self.projection.tok_s,
            "all_host_tok_s": self.all_host_projection.tok_s,
            "host_gbps": self.projection.host_gbps,
            "acceptance_threshold_tok_s": self.acceptance_threshold_tok_s,
            "clears_kill_gate": self.clears_kill_gate(),
            "prior": self.prior.name,
            "byte_source": self.byte_source,
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "local_ranks": self.local_ranks,
            "diagnostics": {k: v for k, v in self.diagnostics.items() if k != "skipped"},
            "skipped": list(self.diagnostics.get("skipped", ())),
            "warnings": list(self.warnings),
        }


def resolve_weight_plan(
    config: Any,
    *,
    device_budget_bytes: Optional[int] = None,
    prior: OffloadPrior = PHASE0_PRIOR,
    layer_indices: Optional[Sequence[int]] = None,
    priorities: Optional[Dict[int, int]] = None,
    prefer_meta: bool = True,
    batch: int = 1,
    project_loaded: bool = False,
    arena_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    model: Any = None,
    num_cpu_layers: int = 0,
    cpu_repacked: bool = False,
    cpu_threads_per_rank: Optional[int] = None,
) -> WeightPlanResolution:
    """Decide, once, which MoE layers keep their expert stack in VRAM. Pure function of config.

    `model` -- the built model (meta at `engine.py:218`, or materialized after `post_load()`). Pass
    it whenever one exists: the layer set, the EP sharding and the per-expert byte counts are then
    READ off the model instead of transcribed from seven builder files and nine container
    `__init__`s, which is what makes a new model family or quant format require no edit here. See
    `observed_planned_layers`.

    `device_budget_bytes` is how much VRAM PER RANK the expert tier may occupy. Pass it explicitly
    from the engine, which is the only component that knows the memory figure -- and derive it from
    a STABLE quantity (the card's total memory, a configured tier size), NEVER from a live
    `mem_get_info` delta. A live delta differs between the two rank processes, so the ranks would
    resolve different plans from the same config and the model would emit plausible wrong text with
    no error anywhere. When omitted it falls back to `config.weight_offload_device_gb`, then to 0
    (pure T1, all-host).

    `config.weight_offload_gb`, if set, SETS the per-rank pinned-host budget, in GiB, replacing the
    baked P3b default in BOTH directions. It is never an enable switch: the plan is derived either
    way, and when the whole stack fits the device budget the returned plan is empty and costs
    nothing. That is what keeps this resolver on the boot path for every serve, so it cannot rot.

    It has to be able to raise the budget, not only lower it. `host_capacity.py` states outright that
    `P3B_PINNED_CEILING_BYTES` is "Advisory only -- never a gate, because it is a property of that
    box on that day, while MemAvailable is the live truth", and this resolver turns it into THE boot
    gate. With a clamp-only knob there was no way to serve on a box with more RAM than the one P3b
    ran on: the plan aborted with "add host RAM" while the box had tens of GiB free and the operator
    had no override. Raising it is not a way to wave a bad plan through -- the live, pass-or-raise
    `check_capacity()` against real `MemAvailable` (plus the per-chunk floor check and the swap
    tripwire in `PinnedWeightArena.attach`) still stands between this number and a pinned page. All
    an override moves is WHERE the failure is detected, and a warning is emitted whenever the budget
    exceeds anything ever demonstrated on this box.

    `arena_chunk_bytes` is the granularity capacity is charged in -- pass `ArenaSettings.chunk_bytes`
    so the plan charges the same chunks `PinnedWeightArena` will pin. See `arena_reservation_bytes`.

    `num_cpu_layers` places that many of the DEEPEST offloadable layers on the CPU-COMPUTE tier
    (`StackKind.CPU`): computed by AVX-512 cores on the host instead of streamed to the GPU. It
    defaults to 0 and it is EXPLICIT rather than derived, for one reason: the tier needs a native
    executor that is not a loadable `.so` yet (`cpu_worker.NativeBackend` declares its ABI and
    refuses to guess), so a planner that switched itself on would place layers on a tier with
    nothing able to run them. `cpu_tier_gate` still runs on every boot and `cpu_sweep_text` still
    prices every K, so the lever is in the banner rather than in this docstring. A request the gate
    refuses (no CPU core for the format, EP active, core budget exceeded) RAISES here rather than
    being silently downgraded to 0 -- a silent downgrade is how a capacity plan comes to describe a
    residency the process does not have.

    `cpu_repacked` asserts that a repacker will actually run, which is what lets the CPU tier be
    charged 0.9x the device-layout bytes (the checkpoint's own e4m3 group-scale byte against the
    fp16-folded one). Default False charges the full device-layout bytes. See
    `cpu_tier.CpuTierPrior.layout_fraction_for`.

    The returned resolution is not self-enforcing. Call `raise_if_infeasible()` before allocating.
    """
    tp_size = int(getattr(getattr(config, "tp_info", None), "size", 1) or 1)
    dp_size = int(getattr(getattr(config, "dp_info", None), "dp_size", 1) or 1)
    ranks = local_arena_count(config)

    if device_budget_bytes is None:
        # GiB, not decimal 1e9 GB -- see GIB_PER_UNIT. Reading 1e9 while `_gb()` prints GiB
        # under-grants the operator's tier by 7.4% and the log reads as if it had been honoured.
        device_budget_bytes = int(
            float(getattr(config, "weight_offload_device_gb", 0.0) or 0.0) * GIB_PER_UNIT
        )
    device_budget_bytes = max(0, int(device_budget_bytes))

    layers, diag = build_planned_layers(
        config, layer_indices=layer_indices, prefer_meta=prefer_meta, priorities=priorities,
        model=model,
    )

    warnings: List[str] = []
    ceiling = host_arena_ceiling_bytes(ranks, prior)
    host_cap_gb = float(getattr(config, "weight_offload_gb", 0.0) or 0.0)
    if host_cap_gb > 0:
        # SET the node budget from the operator's per-rank GiB figure, in both directions. Expressed
        # as the single ceiling rather than as a second, independent constraint so exactly ONE
        # inequality decides feasibility and the "raise the device tier to >= X" message stays
        # correct under both. See the docstring for why this must be able to raise as well as lower.
        ceiling = int(host_cap_gb * GIB_PER_UNIT) * ranks
        demonstrated = host_arena_ceiling_bytes(ranks, prior)
        if ceiling > demonstrated:
            warnings.append(
                f"--weight-offload-gb raises the pinned-host budget to {_gb(ceiling)} across "
                f"{ranks} rank(s), above the {_gb(demonstrated)} this box has ever demonstrated "
                f"({P3B_NOTE}). The live MemAvailable check, the per-chunk floor and the swap "
                "tripwire still gate the actual pinning -- but the plan-time answer is now an "
                "operator assertion, not a measurement."
            )

    total = sum(lw.resident_bytes for lw in layers)

    # ---- CPU-COMPUTE tier: gate first, then place. -------------------------------------------
    gate = cpu_tier_gate(
        diag,
        local_ranks=ranks,
        n_offloadable=len(layers),
        repacked=cpu_repacked,
        threads_per_rank=cpu_threads_per_rank,
    )
    num_cpu_layers = max(0, int(num_cpu_layers))
    if num_cpu_layers and not gate.allowed:
        raise PlacementError(
            f"--weight-offload-cpu-layers={num_cpu_layers} was asked for but the CPU-COMPUTE tier "
            f"is not usable here. {gate.reason}. This raises instead of falling back to 0 because "
            f"a silent downgrade would leave every capacity number in this resolution describing a "
            f"residency the process does not have."
        )
    if num_cpu_layers > len(layers):
        raise PlacementError(
            f"--weight-offload-cpu-layers={num_cpu_layers} exceeds the {len(layers)} offloadable "
            f"MoE layer(s) this model has"
        )
    cpu_fraction = gate.layout_fraction if num_cpu_layers else 1.0

    def _plan_for(budget: int) -> OffloadPlan:
        return plan_three_tier(
            layers,
            device_budget_bytes=budget,
            num_cpu_layers=num_cpu_layers,
            cpu_layout_fraction=cpu_fraction,
        )

    if not layers:
        plan = plan_layer_granular((), device_budget_bytes=device_budget_bytes)
        reason = "no offloadable expert stacks (dense model, or every MoE layer excluded)"
        # "MoE and dense land together" (CLAUDE.md) is NOT satisfied yet, and the reason string
        # above reads as "nothing to offload" when for a dense checkpoint the truth is "this planner
        # does not look at dense linears". The primitive is already general -- `granule.GranuleSpec`
        # takes `num_experts=None` (the whole container is one granule) and
        # `granule.per_expert_tensors` documents covering a dense `_LinearTPImpl` -- so what is
        # missing is enumeration here, not a second mechanism. Say so on the banner rather than let
        # an operator read OFF as "there was nothing to gain".
        if not bool(getattr(getattr(config, "model_config", config), "is_moe", False)):
            warnings.append(
                "this checkpoint is DENSE and weight offload currently plans MoE expert stacks "
                "only, so nothing was offloaded. That is a gap in this planner's enumeration, not "
                "a property of the mechanism: granule.GranuleSpec already describes a dense "
                "container (num_experts=None). Do not read 'OFF' here as 'dense cannot benefit'."
            )
    elif total <= device_budget_bytes and not num_cpu_layers:
        plan = _plan_for(device_budget_bytes)
        reason = (
            f"whole expert stack ({_gb(total)}/rank) fits the device budget "
            f"({_gb(device_budget_bytes)}) -- offload is a no-op"
        )
    else:
        plan = _plan_for(device_budget_bytes)
        cpu_note = (
            f", {plan.num_cpu_layers} CPU-computed "
            f"({_gb(plan.cpu_resident_bytes)}/rank PAGEABLE, "
            f"{_gb(plan.pinned_arena_bytes_saved_vs_host * ranks)} of pinned arena NOT needed)"
            if plan.num_cpu_layers
            else ""
        )
        reason = (
            f"expert stack {_gb(total)}/rank exceeds the {_gb(device_budget_bytes)} device budget; "
            f"{plan.num_device_layers} of {len(layers)} layers placed on device, "
            f"{_gb(plan.host_resident_bytes)}/rank streams from pinned host{cpu_note}"
        )

    needed = required_device_bytes(layers, ceiling, ranks, arena_chunk_bytes)
    # Charged against what the arena will PIN (payload rounded up to whole chunks), not the payload.
    # `attach()` hipHostMallocs and first-touches every chunk at full size, so the payload figure was
    # under-charging by up to one chunk per rank -- and it did so precisely at the ceiling, where the
    # module's whole reason for existing is to fail from config in milliseconds instead of mid-pin
    # with the box in swap. See `arena_reservation_bytes`.
    # Charged against the PACKING bound too (`max_host_row_bytes`), not only the payload: a region
    # may never straddle two chunks, so with rows of up to `m` bytes only `chunk - m` per chunk is
    # guaranteed placeable. Omitting it assumed perfect packing, and on this feature's shape (a
    # ~1 GiB w13 row in a 2 GiB chunk) that under-reserved the arena by ~25-30% -- the overflow rows
    # then land in `hipMalloc` VRAM at bind time, which `seal()` refuses AFTER the arena is pinned
    # and the checkpoint is loaded. See `arena_reservation_bytes`.
    # ...and against the ENUMERATED rows whenever the sizing model could produce them, which turns
    # that bound into the answer: `plan_arena_reservation_bytes` runs the real next-fit allocator
    # over the real row list instead of assuming the worst row lands at the worst offset in every
    # chunk. On the target shape that is 16 chunks rather than 20 (32.00 vs 40.00 GiB/rank).
    reserved_node = plan_arena_reservation_bytes(plan, arena_chunk_bytes) * ranks
    feasible = reserved_node <= ceiling
    if not feasible:
        reason = (
            f"INFEASIBLE: {_gb(reserved_node)} of pinned host arena across {ranks} rank(s) "
            f"({_gb(plan.host_resident_bytes)}/rank of weights, rounded up to whole "
            f"{_gb(arena_chunk_bytes)} arena chunks) exceeds the usable pinned budget "
            f"{_gb(ceiling)}. Raise the device tier to >= {_gb(needed)}/rank"
        )
    elif layers and reserved_node > ceiling * 0.95:
        warnings.append(
            f"host arena {_gb(reserved_node)} is within 5% of the usable "
            f"budget {_gb(ceiling)}. {P3B_NOTE} -- that was an IDLE box and it was already "
            "swapping. Expect a tight or failed boot under any other load."
        )
    # Only meaningful while the budget is actually BINDING. In the no-op plan (whole stack on
    # device) the leftover budget is just VRAM the operator offered and the model did not need, not
    # a quantisation residual -- warning about it would train people to ignore this line.
    if (
        not plan.is_empty
        and plan.total_resident_bytes
        and plan.unused_device_bytes > plan.total_resident_bytes // 20
    ):
        warnings.append(
            f"{_gb(plan.unused_device_bytes)} of the device budget is unused because layers are "
            "indivisible and this stack's layers are uneven. That residual is exactly what a "
            "per-expert split would capture, and P2' priced that capture at ~1%."
        )

    # BANDWIDTH GATE: the number of CARDS streaming on this node, i.e. `local_ranks` (= tp x dp),
    # never `tp_size`. `prior.slow_host_gbps(n)` returns the slowest of the first n cards because the
    # links are independent (P4, efficiency 0.999) and the step waits for the last rank to finish.
    # Passing tp_size made a dp=2/tp=1 replica pair project against card 0's 28.93 GB/s while rank 1
    # runs on the Gen4 card at 14.48 -- a 1.94x over-projection of the mechanism ceiling on exactly
    # the sharding mode where the two arenas are ALREADY charged to both ranks by
    # `local_arena_count`. Both gates read off this projection: K4 could pass a plan that should be
    # killed, and A1.7 (0.75 x this) would set an acceptance threshold no serve can reach, which
    # reads as the mechanism failing rather than as the arithmetic being wrong.
    bandwidth_ranks = ranks
    projection = project_plan(
        plan, prior, num_ranks=bandwidth_ranks, batch=batch, loaded=project_loaded
    )
    all_host = project_plan(
        plan_layer_granular(layers, device_budget_bytes=0),
        prior, num_ranks=bandwidth_ranks, batch=batch, loaded=project_loaded,
    )
    sweep_text = (
        format_sweep(
            sweep_device_fraction(layers, prior, num_ranks=bandwidth_ranks, batch=batch), prior
        )
        if layers
        else "device-fraction sweep: n/a (no offloadable layers)"
    )

    agreement_min = diag.get("sizing_agreement_min")
    agreement_max = diag.get("sizing_agreement_max")
    if agreement_min is not None and (agreement_min < 0.99 or (agreement_max or 1.0) > 1.01):
        warnings.append(
            "weights/sizing.py's analytic byte model disagrees with the real containers built on "
            f"meta ({agreement_min:.4f}..{agreement_max:.4f}x). A container in layers/moe.py "
            "changed shape and sizing.py did not follow it -- one of the two is now wrong."
        )
    if diag.get("byte_source") == "analytic" and layers:
        warnings.append(
            "expert bytes came from the ANALYTIC model only (torch or minisgl.layers.moe was not "
            "importable). The plan is still deterministic, but it MUST be re-checked against the "
            "granule walker's post-load bytes before anything is allocated against it."
        )
    if diag.get("skipped"):
        warnings.append(
            "excluded from offload by policy: " + ", ".join(diag["skipped"])
        )

    # ---- CPU-COMPUTE tier: the sweep runs whether or not the tier is in use ------------------
    cpu_rows = cpu_tier_sweep(
        layers,
        device_budget_bytes=device_budget_bytes,
        gate=gate,
        prior=prior,
        local_ranks=ranks,
        arena_chunk_bytes=arena_chunk_bytes,
        batch=batch,
        loaded=True,  # a serving box, not an idle one -- see OffloadPrior.host_read_gbps_loaded
    )
    cpu_projection = None
    if num_cpu_layers:
        from .cpu_tier import CPU_TIER_PRIOR, CpuTierMode, project_cpu_tier

        cpu_projection = project_cpu_tier(
            host_bytes_per_rank=plan.host_active_bytes(batch),
            device_bytes_per_rank=plan.device_active_bytes(batch),
            cpu_bytes_per_rank=plan.cpu_active_bytes(batch),
            num_cpu_layers=plan.num_cpu_layers,
            mode=CpuTierMode.BLOCK,
            handoff_us_per_layer=CPU_TIER_PRIOR.handoff_us_bracket[1],
            threads_per_rank=gate.threads_per_rank,
            prior=prior,
            num_ranks=bandwidth_ranks,
            loaded=True,
            num_layers=len(layers),
        )
        warnings.append(
            f"the CPU-COMPUTE tier is ACTIVE on {plan.num_cpu_layers} layer(s). Three things this "
            f"buys and one it costs, all measured: (a) {_gb(plan.pinned_arena_bytes_saved_vs_host * ranks)} "
            f"of pinned arena is no longer needed; (b) the same layers cost "
            f"{_gb(plan.cpu_resident_bytes * ranks)} of PAGEABLE host RAM instead, which is not "
            f"against the hipHostMalloc ceiling; (c) projected {cpu_projection.tok_s:.2f} tok/s raw "
            f"/ {cpu_projection.calibrated_tok_s:.2f} at the one measured model-fidelity ratio. "
            f"COST: the expert MLP runs with int8 activations at rel_rms "
            f"{cpu_projection.rel_rms:.2e} (against the fp32 CPU core's 2.4e-07 -- but against the "
            f"4.09e-02 the GPU's per-token fp8 path already serves, so it is ~4.9x MORE accurate "
            f"than what ships today), and the forward needs "
            f"{cpu_projection.graph_segments} separately-captured device segments, which today's "
            f"whole-forward CUDAGraph capture cannot express AT ALL."
        )
        # THE HEADLINE `projection` MUST BECOME THE THREE-TIER ONE. `project_plan` sums only the
        # HOST and DEVICE terms, so on a plan with CPU layers it silently drops the largest term in
        # the step and reports a tok/s the box cannot reach — at 48 CPU layers it read 133 tok/s,
        # because every byte had left both of the tiers it knows about. Every downstream consumer
        # (the K4 kill gate, the A1.7 acceptance threshold, `summary_line`, the engine banner and
        # the accounting) reads `projection`, so the substitution happens HERE, once, rather than
        # by teaching each of them about a third tier.
        projection = Projection(
            step_ms=cpu_projection.step_ms,
            tok_s=cpu_projection.tok_s,
            host_ms=cpu_projection.host_ms + cpu_projection.cpu_ms + cpu_projection.handoff_ms,
            device_ms=cpu_projection.device_ms,
            compute_ms=cpu_projection.compute_ms,
            host_gbps=cpu_projection.host_gbps,
            active_bytes_per_rank=(
                plan.host_active_bytes(batch)
                + plan.device_active_bytes(batch)
                + plan.cpu_active_bytes(batch)
            ),
        )
        if not cpu_projection.handoff_measured:
            warnings.append(
                "the CPU tier's per-layer activation round trip (D2H hidden + H2D partial + the "
                "fence) is UNMEASURED -- no card was available when it was designed. The "
                "projection above used the PESSIMISTIC end of the 10-40 us bracket "
                f"({cpu_projection.handoff_ms:.2f} ms total). It is the one term a GPU run must "
                "close before any tok/s figure here is quoted."
            )

    return WeightPlanResolution(
        plan=plan,
        layers=layers,
        prior=prior,
        tp_size=tp_size,
        dp_size=dp_size,
        local_ranks=ranks,
        device_budget_bytes=device_budget_bytes,
        host_ceiling_bytes=ceiling,
        required_device_bytes=needed,
        feasible=feasible,
        reason=reason,
        projection=projection,
        all_host_projection=all_host,
        sweep_text=sweep_text,
        byte_source=str(diag.get("byte_source", "none")),
        diagnostics=diag,
        warnings=tuple(warnings),
        arena_chunk_bytes=int(arena_chunk_bytes),
        cpu_gate=gate,
        cpu_sweep=tuple(cpu_rows),
        cpu_sweep_text=format_cpu_tier_sweep(cpu_rows, gate),
        cpu_projection=cpu_projection,
    )


def _gb(n: int) -> str:
    """Bytes -> a short human string. GiB, because P3b's ceiling was measured in GiB."""
    if n < GiB // 10:
        return f"{n / (1 << 20):.0f} MiB"
    return f"{n / GiB:.2f} GiB"
