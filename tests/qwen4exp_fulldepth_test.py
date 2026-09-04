"""ALL 48 LAYERS of qwen4_exp, real weights, on ONE 16 GB card — by streaming the expert tier.

WHY THIS EXISTS
---------------
`qwen4exp_gpu_forward_test.py` and `qwen4exp_engine_test.py` both run a 4-LAYER SUBSET, and both say
in their own docstrings that the text they produce is meaningless by construction. That is the
honest state of the bring-up: every structural seam has been exercised and NOTHING has been checked
against the one thing that would falsify a wrong port — whether the model, at full depth, predicts
sensible tokens. `docs/QWEN4EXP_BRINGUP_PLAN.md` KILL-1 says a full-depth run is blocked because the
model is ~85 GiB and the reachable tiers are ~63.7 GiB.

KILL-1 is right about SERVING and wrong about CORRECTNESS, and the difference is measured here:

    routed experts      70.312 GiB      <- 1.465 GiB per layer, and only ONE layer is live at a time
    everything else      9.216 GiB      <- fits the card outright, with room for a KV pool

So a full-depth forward needs 9.2 + 1.5 = ~10.7 GiB resident, not 85. Depth was never the blocker;
holding all 48 layers' experts AT ONCE was. This harness keeps the whole non-expert body resident,
aliases all 48 layers' expert containers onto ONE shared pair of device buffers, and refills those
buffers from the layer's own four shards immediately before the layer runs.

WHAT THIS IS AND IS NOT
-----------------------
  * IS: a correctness vehicle. It produces REAL logits from ALL 48 layers with ALL real weights, so
    for the first time the output text is a statement about the port rather than about a truncation.
  * IS NOT: a serving path, and nothing here should be read as one.

TWO EXPERT-TIER MODES
---------------------
  `--streamed` (default): restage a layer's WHOLE 512-expert tier before it runs. 70 GiB per forward,
      ~178 s/token. Simple, and it is the reference the routed mode is validated against.
  `--routed`: stage only the experts the layer's tokens ROUTE to. `num_experts_per_tok` is 10, so this
      reads 10 of 512 — MEASURED at full depth: 10.0 experts/layer, 1.236 GiB/token, ~1.3 s/token, a
      48-layer 12-token greedy run in 52 s instead of ~40 min. The route is taken from
      `quant.kernels._route_align` (the op the kernel itself routes with) and NOT re-derived; see
      `RoutedExpertGather` for the bf16-tie failure that requirement comes from.

Neither is a serve. The routed gather reads through `safetensors` mmap at ~1257 MiB/s; a serve wants
the O_DIRECT granule reader (`docs/WEIGHT_OFFLOAD_PLAN.md` §9, 6.9-9.0 GB/s) and a prefetch, which is
what makes 1.236 GiB/token a throughput claim instead of a traffic measurement.

THE PROVENANCE GUARD (`--validate`)
-----------------------------------
A streamer that silently stages the WRONG layer's experts produces fluent-looking garbage and no
error — the exact failure this repo has been burned by. So the streaming path is not trusted on
inspection: `--validate` builds the SAME 4-layer subset twice, once with every layer's experts
resident (the round-2 path) and once streamed, and requires the logits to be **bit-identical**. Any
off-by-one in the layer->shard mapping, any stale buffer, any missed restage moves at least one
logit. Run it before believing any full-depth number.

Run (card 0, in the serve image):

    docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
      --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES=0 \
      -v <worktree>:/engine -v /home/pat/.cache/hf-q4e:/model:ro -v /home/pat/.cache/hf-ple:/ple:ro \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_fulldepth_test.py --validate'
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL = os.environ.get("Q4E_MODEL", "/model")
PLE_DIR = os.environ.get("Q4E_PLE", "/ple")

_failures = 0


def check(name: str, got, want) -> None:
    global _failures
    ok = got == want
    _failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name:54s} got={got!r:<24} want={want!r}", flush=True)


def check_true(name: str, cond, detail: str = "") -> None:
    global _failures
    _failures += not cond
    print(f"  {'ok  ' if cond else 'FAIL'} {name:54s} {detail}", flush=True)


def _gib(nbytes: float) -> str:
    return f"{nbytes / 2**30:.3f} GiB"


# --------------------------------------------------------------------------------------------
# Checkpoint layout
# --------------------------------------------------------------------------------------------

_EXPERT_LEAVES = (
    "mlp.experts.gate_up_proj.weight_packed",
    "mlp.experts.gate_up_proj.weight_scale",
    "mlp.experts.down_proj.weight_packed",
    "mlp.experts.down_proj.weight_scale",
)


def _layer_shards(model_dir: str, layer: int) -> "list[str]":
    """The four per-expert-range shards holding layer `layer`'s 512 NVFP4 experts.

    The name carries the layer index, so this mapping is the ONE place a full-depth run can go
    silently wrong (stage layer L's buffers from layer L' != L and the logits are plausible and
    meaningless). It is a pure function of the index, asserted to exist, and `--validate` checks the
    composition end to end.
    """
    out = []
    for lo in (0, 128, 256, 384):
        p = f"{model_dir}/layer-{layer:05d}-experts-{lo:04d}-{lo + 127:04d}.safetensors"
        if not os.path.exists(p):
            raise FileNotFoundError(f"layer {layer}: missing expert shard {p}")
        out.append(p)
    return out


class _ExpertShardDirs:
    """One symlink-only directory per layer, so the real loader's own glob sees EXACTLY that layer.

    `_load_qwen4_exp_weight` globs `*.safetensors` and asserts every expert stack completed. Pointing
    it at the full 196-shard checkpoint would stack all 48 layers (70 GiB); pointing it at a
    four-symlink directory reuses the ENTIRE verified loader — the modelopt `weight_scale_2` fold
    (and its reciprocal convention), the gate/up merge, the stack over E — for one layer at a time.
    Nothing about the fold is re-implemented here, which is the point.
    """

    def __init__(self, model_dir: str, layers: "list[int]") -> None:
        self.root = tempfile.mkdtemp(prefix="q4e-shards-")
        self.dirs = {}
        for lid in layers:
            d = os.path.join(self.root, f"L{lid:05d}")
            os.makedirs(d)
            for s in _layer_shards(model_dir, lid):
                os.symlink(s, os.path.join(d, os.path.basename(s)))
            self.dirs[lid] = d

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _bf16_only_dir(model_dir: str) -> str:
    """A symlink dir holding ONLY the four `model-bf16-*` shards — the whole non-expert body.

    Those four shards carry every one of the 48 layers' bf16 tensors (plus the mtp/vision tensors the
    loader's ignore ledger drops), and no expert shard, so the loader emits the non-expert body and
    leaves `expert_buf` empty — which its own completeness assert then passes trivially.
    """
    d = tempfile.mkdtemp(prefix="q4e-bf16-")
    files = sorted(glob.glob(f"{model_dir}/model-bf16-*.safetensors"))
    assert files, f"no model-bf16-*.safetensors under {model_dir}"
    for f in files:
        os.symlink(f, os.path.join(d, os.path.basename(f)))
    return d


@contextlib.contextmanager
def _quiet():
    """Swallow the loader's per-call tqdm bars — 48 stagings per forward pass, 13 passes.

    It does NOT silence `minisgl`'s own logger: those records go through a handler that captured the
    real stream when it was installed, so `redirect_stderr` cannot reach them and the per-layer
    "1536 NVFP4 modules / ignored 1536 tensors" ledger still lands in the log. Left that way on
    purpose — it is the only per-staging evidence in the transcript that each layer really was
    re-read — but said out loud here so the next reader does not conclude the suppression is broken.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


def _load_layer_experts(shard_dir: str, mc, dev: torch.device, layer: int) -> "dict[str, torch.Tensor]":
    """The four stacked (E, ...) expert tensors for one layer, through the REAL loader.

    Returned keyed by the model-relative leaf name, with the `model.layers.{L}.` prefix stripped —
    and the prefix is CHECKED against `layer` rather than assumed, because the checkpoint file name
    and the tensor names inside it are two independent sources of the layer index and a mismatch
    between them is exactly the silent staging bug `--validate` exists to catch.
    """
    from minisgl.models.weight import _load_qwen4_exp_weight

    want_prefix = f"model.layers.{layer}."
    out = {}
    with _quiet():
        for k, v in _load_qwen4_exp_weight(shard_dir, dev, mc):
            if not k.startswith(want_prefix):
                raise RuntimeError(
                    f"expert shard dir for layer {layer} yielded {k!r}, which is not layer {layer}: "
                    f"the file-name->layer mapping and the tensor names disagree"
                )
            out[k[len(want_prefix) :]] = v
    missing = [leaf for leaf in _EXPERT_LEAVES if leaf not in out]
    if missing or len(out) != len(_EXPERT_LEAVES):
        raise RuntimeError(f"layer {layer}: expected {list(_EXPERT_LEAVES)}, got {sorted(out)}")
    return out


# --------------------------------------------------------------------------------------------
# The streamer
# --------------------------------------------------------------------------------------------


class ExpertStreamer:
    """Alias all layers' expert containers onto one shared pair of op buffers; refill per layer.

    The substitution is at the CONTAINER level, not inside any kernel: after `post_load` a
    `_GroupedNvFp4Experts` is exactly `_w_op` (E,N,K//8 int32) + `_scales_op` (E,K//16,N fp16), and
    `kernels.w4a8_moe` reads those two. Every layer's container is pointed at the SAME two tensors,
    and `stage(L)` copies layer L's converted weights into them in place, so the MoE path,the kernel
    selection and the op layout are untouched and identical to the resident build.

    The conversion itself is `_GroupedNvFp4Experts.post_load`'s own two lines, re-used rather than
    re-derived (`convert_nvfp4_moe` + the group-major transpose), so a change to the op layout
    cannot leave this staging behind silently — it would raise on the `copy_` shape.
    """

    def __init__(self, model, mc, dev, shard_dirs: "_ExpertShardDirs") -> None:
        self.mc = mc
        self.dev = dev
        self.shard_dirs = shard_dirs
        self.layers = model.model.layers.op_list
        self.pairs = [(l.mlp.experts.gate_up_proj, l.mlp.experts.down_proj) for l in self.layers]
        self.staged: "list[int]" = []
        self.bytes_staged = 0
        self.seconds = 0.0

    def alias_all_to(self, src_idx: int) -> None:
        """Point every layer's containers at layer `src_idx`'s post_load'd op buffers."""
        s13, s2 = self.pairs[src_idx]
        for i, (c13, c2) in enumerate(self.pairs):
            if i == src_idx:
                continue
            for dst, src in ((c13, s13), (c2, s2)):
                dst._w_op = src._w_op
                dst._scales_op = src._scales_op
                # Drop the checkpoint-shaped copies these containers were loaded with. They are the
                # SAME aliased tensors on every layer, so the last delete is what actually frees the
                # 1.56 GiB — and leaving them would also let a stale `weight_packed` be mistaken for
                # live state by anything walking the container.
                for attr in ("weight_packed", "weight_scale"):
                    if hasattr(dst, attr):
                        delattr(dst, attr)

    def stage(self, layer: int) -> None:
        from minisgl.quant import nvfp4

        t0 = time.time()
        got = _load_layer_experts(self.shard_dirs.dirs[layer], self.mc, self.dev, layer)
        c13, c2 = self.pairs[layer]
        for cont, base in ((c13, "mlp.experts.gate_up_proj"), (c2, "mlp.experts.down_proj")):
            conv = nvfp4.convert_nvfp4_moe(got[f"{base}.weight_packed"], got[f"{base}.weight_scale"])
            # Exactly `_GroupedNvFp4Experts.post_load`'s layout: packed int32 codes, and the scale
            # transposed to GROUP-MAJOR (E, K//16, N) for the op's coalesced scale read.
            cont._w_op.copy_(conv["w_packed"])
            cont._scales_op.copy_(conv["scales"].transpose(1, 2))
            del conv
        self.bytes_staged += sum(v.numel() * v.element_size() for v in got.values())
        del got
        torch.cuda.synchronize()
        self.seconds += time.time() - t0
        self.staged.append(layer)

    def install_hooks(self) -> None:
        """Restage immediately before each layer's forward.

        An instance attribute shadows the bound method, so the model file is untouched — a streaming
        hack that lives in the model is a streaming hack that ships.
        """
        for lid, layer in enumerate(self.layers):
            inner = layer.forward

            def staged_forward(hidden, _lid=lid, _inner=inner):
                self.stage(_lid)
                return _inner(hidden)

            layer.forward = staged_forward


# --------------------------------------------------------------------------------------------
# The ROUTED gather
# --------------------------------------------------------------------------------------------


class RoutedExpertGather:
    """Stage only the experts a layer's tokens actually ROUTE to — top-10 of 512, not all 512.

    WHY
    ---
    `ExpertStreamer` above restages a layer's ENTIRE 512-expert tier before the layer runs: 1.465 GiB
    per layer, 70 GiB per forward, ~178 s/token. But `num_experts_per_tok` is **10**, so a decode step
    only ever READS 10 of the 512 — 51x less. `docs/QWEN4EXP_BRINGUP_PLAN.md` §2.2 names that gap and
    §4 names the routed gather as "the shape that fits this model", by analogy with the gather
    `weights/row_table.py` already implements for the 51.2 GB PLE table. This is that gather, built
    against the real checkpoint, so the plan's 1.373 GiB/token stops being arithmetic.

    HOW IT STAYS INSIDE THE EXISTING OP LAYOUT
    ------------------------------------------
    Nothing about the MoE dispatch, the kernel, the expert ids or the op buffers changes. The shared
    `_w_op` (E,N,K//8) / `_scales_op` (E,K//16,N) pair is still E=512 wide and still indexed by the
    GLOBAL expert id; this only declines to FILL the 502 rows the kernel will not read. So there is no
    id remap, no compaction, and no second code path for the grouped GEMM to get wrong.

    The route is taken from `quant.kernels._route_align` — THE op the served e2m1 path routes with
    (`moe_hip.moe_route_align`: softmax + top-k + renormalize + align, one launch). Not from
    `MoELayer._ep_route`, and that distinction was measured, not assumed: `_ep_route`'s torch
    `softmax().topk()` and the kernel disagree on **exact bf16 ties at the k-th boundary**, which are
    common (the gate logits are bf16 — 8 mantissa bits — over 512 experts). Measured on this
    checkpoint, layer 0, token row 5: experts 324 and 366 BOTH have logit -5.09375 at ranks 9 and 10,
    straddling k=10; the kernel keeps the lower index (324), `torch.topk` keeps the higher (366).
    Two of eight MoE calls in one 8-token prefill disagreed by exactly one expert that way. Staging
    `_ep_route`'s answer therefore left the expert the GEMM actually read unstaged.

    WHY UNSTAGED ROWS ARE POISONED, NOT LEFT STALE
    ----------------------------------------------
    All 48 layers alias ONE buffer pair, so a row left over from the previous layer holds a DIFFERENT
    layer's expert — real NVFP4 weights of the right shape and magnitude. If the kernel ever read an
    expert this gather did not stage (a route disagreement, a tie broken the other way, a kernel that
    touches more experts than it is routed), the result would be fluent, plausible and wrong, with no
    error. So every row outside the live set is filled with NaN: the same mistake now produces NaN
    logits and trips the finiteness checks instead of a quotable number. This is the cheap half of the
    guard; the expensive half is `--validate`, which requires bit-identical logits against the
    full-tier streamer.
    """

    def __init__(self, model, mc, dev, model_dir: str, layer_ids: "list[int] | None" = None) -> None:
        """`layer_ids=None` (the default) is the whole-model streamed build, unchanged.

        Passing a SUBSET is what `qwen4exp_hybrid_test.py` needs: there only the tail of the model
        streams, and the device-resident and arena-resident layers keep their OWN op buffers — so
        the aliasing invariant below is asserted over the STREAMING layers and anchored on the first
        of them, instead of over every layer anchored on layer 0. A streaming layer that is not
        aliased still raises: the check is narrowed to the layers it is true of, never dropped.
        """
        self.mc = mc
        self.dev = dev
        self.model_dir = model_dir
        self.layers = model.model.layers.op_list
        self.layer_ids = (
            sorted(int(i) for i in layer_ids)
            if layer_ids is not None
            else list(range(len(self.layers)))
        )
        if not self.layer_ids:
            raise RuntimeError("RoutedExpertGather needs at least one streaming layer")
        self.num_experts = int(mc.num_experts)
        self._handles: "dict[tuple[int, int], object]" = {}
        self._live: "set[int]" = set()
        self.bytes_read = 0
        self.seconds = 0.0
        self.experts_staged = 0
        self.staged: "list[int]" = []
        self.calls: "list[tuple[int, int, int]]" = []  # (layer, n_experts, bytes) per staging

        # Every STREAMING layer must already be aliased onto the first one's op buffers.
        # Asserted on the DATA POINTER, not on `is`, because that is what the kernel dereferences.
        anchor = self.layer_ids[0]
        c13, c2 = self._containers(anchor)
        for lid in self.layer_ids:
            a, b = self._containers(lid)
            if (a._w_op.data_ptr() != c13._w_op.data_ptr()
                    or b._w_op.data_ptr() != c2._w_op.data_ptr()):
                raise RuntimeError(
                    f"layer {lid}'s expert buffers are not aliased onto layer {anchor}'s — the "
                    f"routed gather requires the streamed build, which is what makes ONE 512-row "
                    f"buffer pair serve all {len(self.layer_ids)} streaming layers."
                )
        self.w13, self.sc13 = c13._w_op, c13._scales_op
        self.w2, self.sc2 = c2._w_op, c2._scales_op
        # Nothing is valid until a layer stages: start fully poisoned.
        self.sc13.fill_(float("nan"))
        self.sc2.fill_(float("nan"))

    def _containers(self, lid: int):
        mlp = self.layers[lid].mlp
        return mlp.experts.gate_up_proj, mlp.experts.down_proj

    def _handle(self, layer: int, lo: int):
        key = (layer, lo)
        h = self._handles.get(key)
        if h is None:
            import safetensors

            path = f"{self.model_dir}/layer-{layer:05d}-experts-{lo:04d}-{lo + 127:04d}.safetensors"
            if not os.path.exists(path):
                raise FileNotFoundError(f"layer {layer}, experts {lo}..{lo + 127}: missing {path}")
            h = safetensors.safe_open(path, framework="pt", device="cpu")
            h.__enter__()
            self._handles[key] = h
        return h

    def _leaf(self, layer: int, eid: int, projs: "tuple[str, ...]"):
        """One expert's (packed uint8, folded fp16 scale) for one merged MoE GEMM.

        Mirrors `_load_qwen4_exp_weight` exactly and deliberately: fold at the LEAF with modelopt's
        `weight_scale_2` convention, THEN merge gate/up on dim 0 (NVFP4 packs along input K, so the
        gate/up concat is the OUTPUT dim for every leaf). Getting either wrong is silent, which is
        why `--validate` requires bit-identical logits rather than an eyeball.
        """
        from minisgl.quant import nvfp4

        f = self._handle(layer, (eid // 128) * 128)
        pre = f"model.language_model.layers.{layer}.mlp.experts.{eid}."
        packed, scales = [], []
        for p in projs:
            w = f.get_tensor(pre + p + ".weight")
            s = f.get_tensor(pre + p + ".weight_scale")
            g = f.get_tensor(pre + p + ".weight_scale_2")
            self.bytes_read += w.numel() * w.element_size() + s.numel() * s.element_size() + 4
            packed.append(w.to(self.dev, non_blocking=True))
            scales.append(
                nvfp4.fold_nvfp4_scale(
                    s.to(self.dev), g.to(self.dev), global_field="weight_scale_2"
                )
            )
        if len(projs) == 1:
            return packed[0], scales[0]
        return torch.cat(packed, dim=0), torch.cat(scales, dim=0)

    def stage_routed(self, layer: int, ids: "list[int]") -> None:
        from minisgl.quant import nvfp4

        t0 = time.time()
        bytes0 = self.bytes_read
        want = sorted({int(i) for i in ids})
        # Poison whatever the PREVIOUS layer left behind that this layer will not overwrite. Every
        # row in `want` is rewritten below, so the live set is exactly `want` afterwards.
        stale = self._live - set(want)
        if stale:
            idx = torch.tensor(sorted(stale), dtype=torch.long, device=self.dev)
            self.sc13.index_fill_(0, idx, float("nan"))
            self.sc2.index_fill_(0, idx, float("nan"))
        self._live = set()

        p13, s13, p2, s2 = [], [], [], []
        for e in want:
            a, b = self._leaf(layer, e, ("gate_proj", "up_proj"))
            c, d = self._leaf(layer, e, ("down_proj",))
            p13.append(a)
            s13.append(b)
            p2.append(c)
            s2.append(d)
        # `convert_nvfp4_moe` is `_GroupedNvFp4Experts.post_load`'s own conversion, run over a stack of
        # `len(want)` experts instead of 512. It is per-expert internally, so a subset stack is
        # bit-identical to the same experts inside a full one.
        conv13 = nvfp4.convert_nvfp4_moe(torch.stack(p13), torch.stack(s13))
        conv2 = nvfp4.convert_nvfp4_moe(torch.stack(p2), torch.stack(s2))
        idx = torch.tensor(want, dtype=torch.long, device=self.dev)
        self.w13.index_copy_(0, idx, conv13["w_packed"])
        self.sc13.index_copy_(0, idx, conv13["scales"].transpose(1, 2).contiguous())
        self.w2.index_copy_(0, idx, conv2["w_packed"])
        self.sc2.index_copy_(0, idx, conv2["scales"].transpose(1, 2).contiguous())
        self._live = set(want)
        self.experts_staged += len(want)
        self.staged.append(layer)
        # Per-CALL, so prefill and decode can be reported separately. Averaging bytes over all
        # forwards and calling the result "per token" would overstate decode by however much more the
        # prefill routes (an 8-token prefill touches ~60 experts/layer against decode's 10).
        self.calls.append((layer, len(want), self.bytes_read - bytes0))
        torch.cuda.synchronize()
        self.seconds += time.time() - t0

    def install_hooks(self) -> None:
        """Hook `mlp.forward`, not `layer.forward` — the route is not known until the block's input is.

        `ExpertStreamer` can hook the whole layer because it stages unconditionally. A routed gather
        cannot: the route is a function of the post-attention hidden state, which only exists once the
        layer has run its attention and its mlp-side hyper-connection mix. So the interposition point
        is one level deeper, at the MoE block, where `x` is exactly what the router will see.
        The router gate and the route op are each evaluated twice per layer (once here, once inside
        the block) — a [T,2560]x[2560,512] GEMM and one fused route launch, negligible against the
        gather they schedule. Both are deterministic functions of `x`, so the second evaluation
        reproduces the first; `--validate` is what actually holds that.
        """
        from minisgl.quant import kernels as qk

        for lid in self.layer_ids:
            mlp = self.layers[lid].mlp
            inner = mlp.forward

            def routed_forward(x, _lid=lid, _mlp=mlp, _inner=inner):
                experts = _mlp.experts
                router_logits = _mlp.gate.forward(x)
                # `block_size` only sets how many BLOCKS the aligner emits; the set of experts with at
                # least one token — which is what has to be staged — is block-size independent
                # (checked at block_m 16/32/64), so this need not track the layer's own choice.
                _, topk_ids, _, expert_ids, ntp = qk._route_align(
                    router_logits.contiguous(), experts.top_k, experts.renormalize,
                    experts.num_experts, 16,
                )
                # `expert_ids[:live]` is the block->expert map the grouped GEMM iterates, i.e. the rows
                # it will actually dereference. Union it with `topk_ids` rather than trusting either
                # alone: they agreed in every measurement, and a future aligner that emits a padding
                # block under some expert id would be staged rather than read as poison.
                live = int(ntp.item()) // 16
                ids = topk_ids.flatten().tolist() + expert_ids[:live].tolist()
                self.stage_routed(_lid, ids)
                return _inner(x)

            mlp.forward = routed_forward

    def close(self) -> None:
        for h in self._handles.values():
            with contextlib.suppress(Exception):
                h.__exit__(None, None, None)
        self._handles.clear()


# --------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------


def build_model(mc, dev, model_dir: str, *, streamed: bool, n_layers: int):
    """Meta-build, load the non-expert body, then either stream or resident-load the expert tier."""
    from minisgl.models import cast_checkpoint_tensor, create_model
    from minisgl.models.weight import _load_qwen4_exp_weight

    with torch.device("meta"):
        model = create_model(mc)

    want = set(model.state_dict())
    expert_keys = {k for k in want if ".mlp.experts." in k}
    body_keys = want - expert_keys

    # ---- non-expert body: one pass over the four bf16 shards -------------------------------
    bf16_dir = _bf16_only_dir(model_dir)
    sd, dropped = {}, 0
    t0 = time.time()
    with _quiet():
        for k, v in _load_qwen4_exp_weight(bf16_dir, dev, mc):
            if k in body_keys:
                sd[k] = cast_checkpoint_tensor(k, v, torch.bfloat16)
            else:
                dropped += 1
                del v
    shutil.rmtree(bf16_dir, ignore_errors=True)
    missing = sorted(body_keys - set(sd))
    if missing:
        raise RuntimeError(
            f"{len(missing)} non-expert parameters the model declares were NOT produced by the "
            f"loader (e.g. {missing[:5]})"
        )
    body_bytes = sum(v.numel() * v.element_size() for v in sd.values())
    print(f"  [body] {len(sd)} params, {_gib(body_bytes)} resident, {dropped} out-of-model keys "
          f"dropped, {time.time() - t0:.1f} s", flush=True)

    # ---- expert tier ------------------------------------------------------------------------
    shard_dirs = _ExpertShardDirs(model_dir, list(range(n_layers)))
    streamer = None
    if streamed:
        # Every layer is loaded with layer 0's tensors, so `load_state_dict`'s shape assert still
        # runs on real shapes and all 48 containers end up aliased to ONE 1.56 GiB set. Only layer
        # 0's `post_load` is allowed to allocate the op buffers; the rest are suppressed and then
        # pointed at layer 0's. Contents are irrelevant — `install_hooks` restages EVERY layer,
        # including layer 0, before it runs.
        seed = _load_layer_experts(shard_dirs.dirs[0], mc, dev, 0)
        for lid in range(n_layers):
            for leaf, t in seed.items():
                sd[f"model.layers.{lid}.{leaf}"] = t
        model.load_state_dict(sd)
        pairs = [(l.mlp.experts.gate_up_proj, l.mlp.experts.down_proj)
                 for l in model.model.layers.op_list]
        for i, (c13, c2) in enumerate(pairs):
            if i == 0:
                continue
            c13.post_load = lambda: None
            c2.post_load = lambda: None
        model.post_load()
        streamer = ExpertStreamer(model, mc, dev, shard_dirs)
        streamer.alias_all_to(0)
        del seed
    else:
        for lid in range(n_layers):
            for leaf, t in _load_layer_experts(shard_dirs.dirs[lid], mc, dev, lid).items():
                sd[f"model.layers.{lid}.{leaf}"] = t
        model.load_state_dict(sd)
        model.post_load()
    torch.cuda.empty_cache()
    return model, streamer, shard_dirs


# --------------------------------------------------------------------------------------------
# Forward driver (Context built by hand, exactly as the round-2 harness does)
# --------------------------------------------------------------------------------------------


def _sample(row: torch.Tensor, *, temperature: float, top_k: int, top_p: float, gen) -> int:
    """The checkpoint's OWN declared sampler, not a bare multinomial.

    `generation_config.json` ships `temperature=1.0, top_k=20, top_p=0.95`. A raw multinomial over
    this model's 248,320-token vocabulary draws from a tail that the served sampler never sees, so
    the "token noise" degeneration signature would be manufactured HERE and misread as a quant or
    kernel bug. `top_k<=0`/`top_p>=1` disable the respective filter; `temperature<=0` is greedy —
    which is only ever a determinism probe, never a quality read.
    """
    if temperature <= 0:
        return int(row.argmax().item())
    logits = row / temperature
    if top_k and top_k > 0 and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        srt, idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(srt, dim=-1)
        # Keep the first index whose cumulative mass crosses top_p, so the set is never empty.
        drop = cum - srt > top_p
        srt = srt.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter_(0, idx, srt)
    probs = (probs / probs.sum()).cpu()
    return int(torch.multinomial(probs, 1, generator=gen).item())


def run(mc, model, dev, prompt_ids: np.ndarray, *, n_new: int, max_seq: int,
        attn_backend: str, real_ple: bool, temperature: float, seed: int,
        streamer, tok=None, top_k: int = 0, top_p: float = 1.0, eos_ids=()):
    """Prefill then `n_new` decode steps. Returns (all logits per step, generated ids)."""
    from qwen4exp_gpu_forward_test import _make_batch, _make_ple_runtime, _make_req

    from minisgl import core
    from minisgl.attention import create_attention_backend
    from minisgl.gdn.metadata import build_gdn_metadata
    from minisgl.kvcache import create_kvcache_pool
    from minisgl.kvcache.gdn_state import GDNStateCache
    from minisgl.moe import create_moe_backend

    saved = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    ctx = core.Context(page_size=16)
    core.set_global_ctx(ctx)
    max_running = 1
    page_size = ctx.page_size
    ctx.page_table = page_table = torch.zeros(
        (max_running + 1, max_seq), dtype=torch.int32, device=dev
    )
    num_pages = 1 + max_running * (max_seq // page_size)
    ctx.kv_cache = create_kvcache_pool(
        model_config=mc, num_pages=num_pages, page_size=page_size, dtype=torch.bfloat16, device=dev
    )
    ctx.attn_backend = create_attention_backend(attn_backend, mc)
    if mc.is_moe:
        ctx.moe_backend = create_moe_backend("fused")
    ctx.gdn_state = GDNStateCache(
        num_gdn_layers=mc.num_gdn_layers, num_slots=max_running + 2, conv_dim=mc.gdn_conv_dim,
        conv_kernel=mc.linear_conv_kernel_dim, num_v_heads=mc.linear_num_value_heads,
        head_v_dim=mc.linear_value_head_dim, head_k_dim=mc.linear_key_head_dim,
        dtype=torch.float32, ssm_dtype=torch.bfloat16, device=dev,
    )
    for gdn in model.iter_gdn_layers():
        gdn.warmup_conv(8)
    # The staging buffer must cover the PREFILL, which is the widest forward here; a decode is 1.
    ple_rt = _make_ple_runtime(
        model, mc, dev, real=real_ple, max_seqs=max_running + 1,
        max_tokens=max(64, int(prompt_ids.shape[0])),
    )
    ctx.ple = ple_rt
    page_table[0, :].copy_(
        torch.arange(page_size, page_size + max_seq, dtype=torch.int32, device=dev)
    )

    req = _make_req(prompt_ids, table_idx=0)
    batch = _make_batch([req], "prefill", page_table, dev)
    ctx.attn_backend.prepare_metadata(batch)
    batch.gdn_metadata = build_gdn_metadata(
        batch, torch.tensor([1], dtype=torch.int32, device=dev), dev
    )
    if ple_rt is not None:
        ple_rt.prepare([1], [prompt_ids])
    t0 = time.time()
    with ctx.forward_batch(batch):
        logits = model.forward()
    torch.cuda.synchronize()
    if ple_rt is not None:
        ple_rt.commit([1], [prompt_ids])
    print(f"  [prefill] {len(prompt_ids)} tok in {time.time() - t0:.1f} s"
          + (f" (staging {streamer.seconds:.1f} s of it)" if streamer else ""), flush=True)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    all_logits = [logits.detach().clone()]
    new_ids: "list[int]" = []
    decode_secs: "list[float]" = []
    for step in range(n_new):
        row = logits[-1].float()
        nxt = _sample(row, temperature=temperature, top_k=top_k, top_p=top_p, gen=gen)
        new_ids.append(nxt)
        if nxt in eos_ids:
            print(f"  [decode {step:2d}] EOS tok={nxt}", flush=True)
            break
        req.append_host(torch.tensor([nxt], dtype=torch.int64))
        req.complete_one()
        batch = _make_batch([req], "decode", page_table, dev)
        ctx.attn_backend.prepare_metadata(batch)
        batch.gdn_metadata = build_gdn_metadata(
            batch, torch.tensor([1], dtype=torch.int32, device=dev), dev
        )
        t = np.array([nxt], dtype=np.int64)
        if ple_rt is not None:
            ple_rt.prepare([1], [t])
        ts = time.time()
        with ctx.forward_batch(batch):
            logits = model.forward()
        torch.cuda.synchronize()
        if ple_rt is not None:
            ple_rt.commit([1], [t])
        all_logits.append(logits.detach().clone())
        dt = time.time() - ts
        decode_secs.append(dt)
        shown = tok.decode(new_ids) if tok is not None else ""
        print(f"  [decode {step:2d}] {dt:5.1f} s  tok={nxt:<7d} {shown!r}", flush=True)
    if decode_secs:
        # Steady-state decode: drop step 0, which pays first-touch page-cache and allocator warmup.
        warm = decode_secs[1:] or decode_secs
        print(f"  [rate] {len(decode_secs)} decode steps, mean {sum(decode_secs)/len(decode_secs):.3f} s"
              f"  |  steady-state {len(warm)/sum(warm):.3f} tok/s ({sum(warm)/len(warm):.3f} s/tok)",
              flush=True)
    core._GLOBAL_CTX = saved
    return all_logits, new_ids, decode_secs


# --------------------------------------------------------------------------------------------


def build_and_hook(mc, dev, model_dir: str, mode: str, n_layers: int):
    """`(model, stager, shard_dirs)` for one of the three expert-tier modes.

    `resident` holds all N layers' experts at once (only viable for a small subset); `streamed`
    restages a whole 512-expert tier per layer; `routed` stages only the top-10 the layer's tokens
    select. The latter two share the SAME aliased build — the difference is purely which hooks go on.
    """
    model, streamer, sd_dirs = build_model(
        mc, dev, model_dir, streamed=(mode != "resident"), n_layers=n_layers
    )
    if mode == "routed":
        stager = RoutedExpertGather(model, mc, dev, model_dir)
        stager.install_hooks()
        return model, stager, sd_dirs
    if streamer is not None:
        streamer.install_hooks()
    return model, streamer, sd_dirs


def _subset_dir(src: str, n_layers: int) -> str:
    from qwen4exp_gpu_forward_test import _subset_config

    d = tempfile.mkdtemp(prefix="q4e-cfg-")
    _subset_config(src, d, n_layers, 512, ple_1based=2)
    return d


@torch.inference_mode()
def main() -> int:
    """NOTE the decorator, for the same reason `qwen4exp_gpu_forward_test.main` carries it: the
    engine runs every forward under `torch.inference_mode()` (`server/launch.py:20`), and
    `Qwen3_5Attn` splits qkv (a multi-view op) then q_norm/k_norm write back IN PLACE, which autograd
    forbids on a multi-view output. Without it the FIRST full-attention layer raises "Output 0 of
    View is a view and is being modified inplace" — a harness artefact, not a model bug."""
    global _failures
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=48, help="48 = full depth")
    # TENSOR PARALLEL. Unlike the serve harness, this file does NOT go through `Engine`: it builds
    # the model directly and drives a hand-rolled forward, so "thread tp through" here means
    # supplying the two things `Engine` would otherwise supply — `set_tp_info(rank, tp)`, which is
    # what every layer shards off, and a real `torch.distributed` process group, which is what
    # `TorchDistributedImpl.all_reduce` (the DEFAULT communicator plugin) calls into. Without the
    # group the TP=2 all-reduces raise; with `set_tp_info` but no group they would be the far worse
    # failure — each rank keeps its own partial sum and the model produces fluent, wrong text.
    #
    # One process per rank, spawned exactly as `server/launch.py`, `tools/kv_fp8_calibrate.py` and
    # this repo's other TP>1 offline runs do. No second mechanism.
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size (1 or 2 on this box)")
    ap.add_argument("--_rank", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--validate", action="store_true",
                    help="bit-exactness A/B of streamed (and, with --routed, routed) vs resident")
    ap.add_argument("--routed", action="store_true",
                    help="stage only the top-k routed experts per layer instead of all 512")
    ap.add_argument("--validate-layers", type=int, default=4)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy, which is a DETERMINISM probe only. Any quality read must be "
                         "sampled: greedy manufactures loops that mimic a quant bug.")
    ap.add_argument("--top-k", type=int, default=0, help="0 = off; the checkpoint declares 20")
    ap.add_argument("--top-p", type=float, default=1.0, help="1 = off; the checkpoint declares 0.95")
    ap.add_argument("--gen-config", action="store_true",
                    help="take temperature/top_k/top_p/eos from the checkpoint's generation_config"
                         ".json — the sampling the model was actually tuned for")
    ap.add_argument("--chat", action="store_true",
                    help="wrap --prompt in the checkpoint's chat template")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--max-seq", type=int, default=128)
    ap.add_argument("--attn-backend", default="rdna4")
    ap.add_argument("--real-ple", action="store_true")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no HIP device visible (is_rocm false?) — check device passthrough")
        return 1
    if args.tp > torch.cuda.device_count():
        print(f"FAIL: --tp {args.tp} but only {torch.cuda.device_count()} device(s) visible. "
              f"TP=2 needs ROCR_VISIBLE_DEVICES=0,1 with HIP_VISIBLE_DEVICES UNSET.")
        return 1
    rank, tp = args._rank, args.tp
    # cuda:{rank}, matching `EngineConfig.device_index` at dp_size=1. Card 1's root port is Gen4 x8
    # (14.48 GB/s vs card 0's 28.93 GB/s) and this file's whole cost is host->device streaming, so
    # WHICH card a rank got is part of every number below — hence the per-rank print.
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    total = torch.cuda.mem_get_info(dev)[1]
    print(f"[gpu] rank {rank}/{tp} -> {dev} {torch.cuda.get_device_name(dev)} total={_gib(total)}",
          flush=True)

    from minisgl.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank, tp)
    if tp > 1:
        # THE COLLECTIVE. `TorchDistributedImpl` (the default `DistributedCommunicator` plugin) calls
        # bare `dist.all_reduce`, so the default WORLD group has to exist and has to be nccl-backed —
        # the reduces are on device tensors. Same rendezvous address `EngineConfig.distributed_addr`
        # uses at dp_rank=0, so this cannot collide with a served engine on the same box by accident
        # of picking a different port.
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl", init_method="tcp://127.0.0.1:2333", rank=rank, world_size=tp
            )
        print(f"[dist] rank {rank}: nccl world={dist.get_world_size()}", flush=True)
    from minisgl.layers.rotary import set_rope_device

    set_rope_device(dev)
    torch.set_default_dtype(torch.bfloat16)

    from minisgl.models.config import ModelConfig
    from minisgl.utils import cached_load_hf_config

    # ---------------- [1] validate the streamer ------------------------------------------
    if args.validate:
        n = args.validate_layers
        print(f"\n[1] PROVENANCE: streamed vs resident, {n}-layer subset, must be BIT-IDENTICAL",
              flush=True)
        cfg_dir = _subset_dir(args.model, n)
        mc = ModelConfig.from_hf(cached_load_hf_config(cfg_dir), spec_algorithm="none")
        rng = np.random.default_rng(11)
        prompt = rng.integers(0, mc.vocab_size, size=8, dtype=np.int64)
        outs = {}
        modes = ("resident", "streamed", "routed") if args.routed else ("resident", "streamed")
        for mode in modes:
            free0 = torch.cuda.mem_get_info(dev)[0]
            model, streamer, sd_dirs = build_and_hook(mc, dev, args.model, mode, n)
            used = free0 - torch.cuda.mem_get_info(dev)[0]
            print(f"  [{mode}] resident after build: {_gib(used)}", flush=True)
            logits, ids, _ = run(
                mc, model, dev, prompt, n_new=2, max_seq=64, attn_backend=args.attn_backend,
                real_ple=args.real_ple, temperature=0.0, seed=args.seed, streamer=streamer,
            )
            outs[mode] = ([x.float().cpu() for x in logits], ids, used)
            if streamer is not None:
                check(f"{mode}: staged every layer, in order", streamer.staged[:n], list(range(n)))
            if mode == "routed":
                # The whole claim of the routed mode is that k << E rows are read. Assert it, so a
                # gather that quietly degenerated into "stage everything" cannot report a speedup.
                per = streamer.experts_staged / max(len(streamer.staged), 1)
                check_true("routed: staged << 512 experts per layer", per <= 8 * mc.num_experts_per_tok,
                           f"{per:.1f} experts/layer of {mc.num_experts} (top_k={mc.num_experts_per_tok})")
                streamer.close()
            del model, streamer
            sd_dirs.close()
            torch.cuda.empty_cache()
        for mode in modes[1:]:
            a, b = outs["resident"][0], outs[mode][0]
            maxdiff = max(float((x - y).abs().max()) for x, y in zip(a, b))
            check_true(f"{mode} logits bit-identical to resident", maxdiff == 0.0,
                       f"max|delta|={maxdiff:.6g} over {len(a)} steps")
            check(f"{mode} greedy ids == resident greedy ids", outs[mode][1], outs["resident"][1])
        saved = outs["resident"][2] - outs["streamed"][2]
        print(f"  [mem] streaming saved {_gib(saved)} at {n} layers "
              f"(={_gib(saved / max(n - 1, 1))} per additional layer)", flush=True)
        shutil.rmtree(cfg_dir, ignore_errors=True)
        if _failures:
            print(f"\nFAIL ({_failures} checks) — NOT running full depth on an unvalidated streamer")
            return 1

    # ---------------- [2] full depth ------------------------------------------------------
    mode = "routed" if args.routed else "streamed"
    print(f"\n[2] FULL DEPTH: {args.layers} layers, real weights, {mode} expert tier", flush=True)
    cfg_dir = args.model if args.layers == 48 else _subset_dir(args.model, args.layers)
    mc = ModelConfig.from_hf(cached_load_hf_config(cfg_dir), spec_algorithm="none")
    check("decoder layers", mc.num_layers, args.layers)
    print(f"  gdn={len(mc.gdn_layer_ids)} attn={len(mc.full_attn_layer_ids)} "
          f"ple={mc.ple_layer_ids} experts={mc.num_experts} top_k={mc.num_experts_per_tok}",
          flush=True)

    tok = None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:  # noqa: BLE001
        print(f"  WARN no tokenizer ({e}); falling back to random ids — text will be meaningless")

    # Sampling. The checkpoint's own generation_config is the only sampler this model was tuned
    # for; anything else (including a bare temperature multinomial over 248,320 tokens) measures
    # the harness. `--gen-config` takes it verbatim, EXCEPT that an explicit --temperature on the
    # command line still wins, so a determinism probe stays available.
    temperature, top_k, top_p = args.temperature, args.top_k, args.top_p
    eos_ids: "tuple[int, ...]" = ()
    if args.gen_config:
        gc_path = os.path.join(args.model, "generation_config.json")
        with open(gc_path) as fh:
            gcfg = json.load(fh)
        if not any(a.startswith("--temperature") for a in sys.argv[1:]):
            temperature = float(gcfg.get("temperature", 1.0))
        if not any(a.startswith("--top-k") for a in sys.argv[1:]):
            top_k = int(gcfg.get("top_k", 0) or 0)
        if not any(a.startswith("--top-p") for a in sys.argv[1:]):
            top_p = float(gcfg.get("top_p", 1.0) or 1.0)
        raw_eos = gcfg.get("eos_token_id", [])
        eos_ids = tuple(raw_eos) if isinstance(raw_eos, list) else (int(raw_eos),)
        print(f"  [sampling] from {gc_path}: temperature={temperature} top_k={top_k} "
              f"top_p={top_p} eos={list(eos_ids)}", flush=True)
    if temperature <= 0:
        print("  [sampling] GREEDY — a determinism probe, NOT a quality read (greedy manufactures "
              "loops that mimic a quant bug)", flush=True)
    else:
        print(f"  [sampling] temperature={temperature} top_k={top_k} top_p={top_p} "
              f"seed={args.seed}", flush=True)

    if tok is not None:
        text_in = args.prompt
        if args.chat:
            text_in = tok.apply_chat_template(
                [{"role": "user", "content": args.prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            print(f"  [chat] templated prompt ({len(text_in)} chars)", flush=True)
        prompt = np.array(tok(text_in)["input_ids"], dtype=np.int64)
        print(f"  prompt {args.prompt!r} -> {len(prompt)} tokens {prompt.tolist()[:24]}", flush=True)
    else:
        prompt = np.random.default_rng(3).integers(0, mc.vocab_size, size=8, dtype=np.int64)

    free0 = torch.cuda.mem_get_info(dev)[0]
    t0 = time.time()
    model, streamer, sd_dirs = build_and_hook(mc, dev, args.model, mode, args.layers)
    used = free0 - torch.cuda.mem_get_info(dev)[0]
    print(f"  [build] {time.time() - t0:.1f} s, resident {_gib(used)} of {_gib(total)}", flush=True)
    check_true("full-depth model is resident on one card", used < total, _gib(used))

    logits, new_ids, decode_secs = run(
        mc, model, dev, prompt, n_new=args.max_new_tokens, max_seq=args.max_seq,
        attn_backend=args.attn_backend, real_ple=args.real_ple,
        temperature=temperature, seed=args.seed, streamer=streamer, tok=tok,
        top_k=top_k, top_p=top_p, eos_ids=eos_ids,
    )
    # One staging call per layer per FORWARD, and there is one forward for the prefill plus one for
    # every decode step that actually ran (an EOS ends the loop before its forward).
    check("staged layer count", len(streamer.staged), args.layers * (1 + len(decode_secs)))
    if mode == "routed":
        nb, secs = streamer.bytes_read, streamer.seconds
        per = streamer.experts_staged / len(streamer.staged)
        print(f"  [routed] {streamer.experts_staged} expert-stagings "
              f"({per:.1f}/layer of {mc.num_experts}, top_k={mc.num_experts_per_tok}); "
              f"{_gib(nb)} read in {secs:.1f} s ({nb / secs / 2**20:.0f} MiB/s)", flush=True)
        # DECODE only, measured per staging call — not the all-forwards average. The prefill routes
        # many more distinct experts per layer than a 1-token decode does, so averaging over both and
        # calling it "per token" inflates the decode figure, which is the one a serve is sized on.
        pre, dec = streamer.calls[:args.layers], streamer.calls[args.layers:]
        if dec:
            steps = max(len(decode_secs), 1)
            dbytes = sum(b for _, _, b in dec)
            print(f"  [routed] prefill {sum(n for _, n, _ in pre) / len(pre):.1f} experts/layer, "
                  f"{_gib(sum(b for _, _, b in pre))}; "
                  f"decode {sum(n for _, n, _ in dec) / len(dec):.1f} experts/layer, "
                  f"{_gib(dbytes / steps)}/token over {steps} tokens", flush=True)
    else:
        print(f"  [stream] {_gib(streamer.bytes_staged)} staged in {streamer.seconds:.1f} s "
              f"({streamer.bytes_staged / streamer.seconds / 2**20:.0f} MiB/s)", flush=True)

    for i, lg in enumerate(logits):
        f = lg.float()
        tag = "prefill" if i == 0 else f"decode[{i - 1}]"
        check_true(f"{tag} finite", bool(f.isfinite().all()),
                   f"min={f.min():.3f} max={f.max():.3f} std={f.std():.3f}")
        check_true(f"{tag} not constant", float(f.max() - f.min()) > 1e-3, "")

    if tok is not None:
        text = tok.decode(new_ids)
        print(f"\n  PROMPT      {args.prompt!r}")
        print(f"  CONTINUATION{text!r}")
        print(f"  FULL        {args.prompt + text!r}")
        # Reported, never asserted: whether 12 tokens read as English is a human judgement and a
        # test that asserted it would be asserting a substring.
        top = logits[0][-1].float().topk(8)
        print("  top-8 next-token after the prompt:")
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"      {p:8.3f}  {i:7d}  {tok.decode([i])!r}")

    if mode == "routed":
        streamer.close()
    sd_dirs.close()
    if args.layers != 48:
        shutil.rmtree(cfg_dir, ignore_errors=True)
    print(f"\n{'PASS' if not _failures else f'FAIL ({_failures} checks)'}")
    return 1 if _failures else 0


def _spawn_rank(rank: int, argv: "list[str]") -> int:
    """`mp.Process` target: re-parse `argv` with `--_rank` appended and run `main()` in this
    process. Re-parsing rather than passing the parsed `Namespace` keeps ONE definition of the
    defaults — a Namespace pickled from the parent would silently freeze whatever the parent's
    parser did, and the two would drift the first time a flag is added."""
    sys.argv = [sys.argv[0]] + argv + ["--_rank", str(rank)]
    return main()


def _run() -> int:
    # `--tp` is read here rather than through the real parser because the parser lives inside
    # `main()`, which is exactly what the spawned children call. A two-line pre-scan is cheaper than
    # hoisting the parser and keeps the TP=1 path byte-identical to what it was before the flag.
    argv = sys.argv[1:]
    tp = 1
    for i, a in enumerate(argv):
        if a == "--tp" and i + 1 < len(argv):
            tp = int(argv[i + 1])
        elif a.startswith("--tp="):
            tp = int(a.split("=", 1)[1])
    if tp <= 1:
        # INLINE. The TP=1 path must not acquire a process boundary it never had, or no TP=1 result
        # this file has produced is comparable with the next one.
        return main()

    import multiprocessing as mp

    # spawn, never fork: a forked child inherits the parent's HIP context, and every rank here calls
    # `torch.cuda.set_device` on a DIFFERENT card.
    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(tp):
        p = mp.Process(target=_spawn_rank, args=(rank, argv), name=f"q4e-fd-TP{rank}")
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    codes = [p.exitcode for p in procs]
    print(f"\n[parent] rank exit codes {codes}", flush=True)
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(_run())
