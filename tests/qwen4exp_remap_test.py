"""qwen4_exp weight-name mapping: every one of the checkpoint's 296,475 tensors is accounted for.

Bring-up plan T1.2 (leaf-name remap) at the pure-string level. No GPU, no weights, no torch tensors —
just `qwen4_exp_remap` run over the checked-in, verbatim tensor-name list of
`RadixArk/Qwen3.8-Flash-Next-NVFP4` (`tests/fixtures/qwen4exp/ckpt_keys.txt.gz`).

What it proves:
  1. **No silent drops.** Every name resolves to a plan or to an explicit ("skip", reason). The
     function never returns None and RAISES on an unrecognised `model.language_model.*` key, so
     "covered" is enforced, not sampled.
  2. **The ignore ledger is real.** What gets skipped is counted per reason and printed, so
     "we ignored 294,xxx tensors" is a stated fact rather than an unnoticed hole.
  3. **The renames are the ones the repo's NVFP4 path expects** — modelopt's
     `.weight`/`.weight_scale_2` -> `.weight_packed`/`.weight_global_scale`, applied only to modules
     the checkpoint actually ships quantized (keyed on the tensors, not the config's ignore list).
  4. Negative cases: an unknown leaf raises rather than passing through.

`native_key_plan()` here is also the pipeline `qwen4exp_build_test.py` compares against the model's
`state_dict()`. Run in the serve image:

    docker run --rm -v <worktree>:/engine --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_remap_test.py'
"""

from __future__ import annotations

import collections
import gzip
import os
import sys
from typing import Dict, List, Set, Tuple

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "qwen4exp")
KEYS_GZ = os.path.join(FIXTURES, "ckpt_keys.txt.gz")

_failures = 0


def check(name: str, got, want) -> None:
    global _failures
    ok = got == want
    _failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name:52s} got={got!r:<24} want={want!r}")


def load_ckpt_names() -> List[str]:
    with gzip.open(KEYS_GZ, "rt") as f:
        return [ln.strip() for ln in f if ln.strip()]


def native_key_plan(names) -> Tuple[Set[str], Dict[str, int], Dict[str, int]]:
    """Run the full name pipeline a streaming loader would: remap -> NVFP4 two-level-scale fold ->
    MoE gate/up merge -> per-expert stack. Returns (native key set, skip counts, concat-group sizes).

    This mirrors `_load_qwen3_5_weight` exactly in ORDER, which matters: the fold has to happen at
    the leaf, before the merge and the stack, or a per-tensor scalar reaches a concat (and the
    resulting fused matrix is scaled by one member's global).
    """
    from minisgl.models.weight import (
        _gate_up_merge,
        _get_expert_stack_info,
        qwen4_exp_nvfp4_modules,
        qwen4_exp_remap,
    )

    nvfp4_modules = qwen4_exp_nvfp4_modules(names)
    skips: Dict[str, int] = collections.Counter()
    mapped: List[str] = []
    concat_groups: Dict[str, int] = collections.Counter()
    for n in names:
        plan = qwen4_exp_remap(n, nvfp4_modules=nvfp4_modules)
        if plan[0] == "skip":
            skips[plan[1]] += 1
        elif plan[0] == "direct":
            mapped.append(plan[1])
        else:
            _, merged, _slot, n_slots, _dim = plan
            concat_groups[merged] += 1
            if concat_groups[merged] == n_slots:
                mapped.append(merged)

    # NVFP4 fold: `weight_scale` (e4m3 per-16-group) and `weight_global_scale` (per-tensor) collapse
    # into ONE fp16 per-group scale spelled `weight_scale`. Assert the pairing rather than just
    # dropping, so a half-shipped pair is caught here instead of at the loader's completeness assert.
    have = set(mapped)
    folded: List[str] = []
    for k in mapped:
        if k.endswith(".weight_global_scale"):
            partner = k[: -len(".weight_global_scale")] + ".weight_scale"
            if partner not in have:
                raise AssertionError(f"{k} has no {partner} to fold with")
            continue
        folded.append(k)

    out: Set[str] = set()
    stacks: Dict[str, Set[int]] = collections.defaultdict(set)
    for k in folded:
        mm = _gate_up_merge(k)
        if mm is not None:
            k = mm[0]
        einfo = _get_expert_stack_info(k)
        if einfo is not None:
            packed, idx = einfo
            stacks[packed].add(idx)
            out.add(packed)
        else:
            out.add(k)
    # Every expert stack must be complete over 0..E-1 — a hole would silently ship a zero expert.
    for packed, idxs in stacks.items():
        if idxs != set(range(len(idxs))):
            raise AssertionError(f"expert stack {packed} is not contiguous 0..{len(idxs) - 1}")
    return out, dict(skips), {"expert_stacks": len(stacks), "stack_width": len(next(iter(stacks.values())))}


def main() -> int:
    global _failures
    from minisgl.models.weight import (
        QWEN4EXP_SKIP_REASONS,
        log_qwen4_exp_ignored,
        qwen4_exp_ignored_summary,
        qwen4_exp_nvfp4_modules,
        qwen4_exp_remap,
    )

    names = load_ckpt_names()
    print(f"[fixture] {len(names)} checkpoint tensor names (RadixArk/Qwen3.8-Flash-Next-NVFP4)")
    check("tensor count", len(names), 296475)

    nvfp4_modules = qwen4_exp_nvfp4_modules(names)
    print(f"\n[1] NVFP4 modules, keyed structurally on a shipped `.weight_scale_2`")
    check("nvfp4 module count", len(nvfp4_modules), 48 * 512 * 3)
    check(
        "all are routed-expert projections",
        all(".mlp.experts." in m for m in nvfp4_modules),
        True,
    )
    check(
        "shared expert NOT quantized",
        any(".shared_expert." in m for m in nvfp4_modules),
        False,
    )
    check(
        "self_attn NOT quantized",
        any(".self_attn." in m for m in nvfp4_modules),
        False,
    )
    check(
        "linear_attn NOT quantized",
        any(".linear_attn." in m for m in nvfp4_modules),
        False,
    )

    print("\n[2] coverage: every name resolves (an unknown LM key would have raised)")
    plans = [qwen4_exp_remap(n, nvfp4_modules=nvfp4_modules) for n in names]
    kinds = collections.Counter(p[0] for p in plans)
    check("no None plans", sum(1 for p in plans if p is None), 0)
    check("kinds", set(kinds), {"skip", "direct", "concat"})
    lm_keys = [n for n in names if n.startswith("model.language_model.")]
    check("model.language_model.* tensors", len(lm_keys), 296475 - 333 - 31 - 1)
    # An LM-namespace tensor may only be skipped for a reason we have DECIDED on. Anything else
    # would be a tensor going missing.
    lm_skip_reasons = collections.Counter(
        p[1] for n, p in zip(names, plans) if n.startswith("model.language_model.") and p[0] == "skip"
    )
    check(
        "LM-namespace skips are only the two intended kinds",
        set(lm_skip_reasons),
        {"ple-ngram-table", "act-calibration"},
    )
    lm_mapped = sum(
        1 for n, p in zip(names, plans) if n.startswith("model.language_model.") and p[0] != "skip"
    )
    check(
        "every LM tensor is mapped or intentionally skipped",
        lm_mapped + sum(lm_skip_reasons.values()),
        len(lm_keys),
    )

    print("\n[3] ignore ledger")
    skips = qwen4_exp_ignored_summary(names, nvfp4_modules=nvfp4_modules)
    log_qwen4_exp_ignored(skips, print)
    check("every skip reason is documented", set(skips) <= set(QWEN4EXP_SKIP_REASONS), True)
    check("vision skipped", skips.get("vision"), 27 * 12 + 6 + 2 + 1)
    check("mtp skipped", skips.get("mtp-head"), 31)
    # 128 shards + weight_scale + ngram_heads_offsets + ngram_heads_vocab_sizes
    check("ple table skipped", skips.get("ple-ngram-table"), 131)
    check("act calibration skipped", skips.get("act-calibration"), 48 * 512 * 3)
    check("no quant-metadata / kv-scale in this ckpt",
          (skips.get("quant-metadata"), skips.get("fp8-kv-scale")), (None, None))

    print("\n[4] representative renames")
    L = "model.language_model.layers"
    cases = [
        ("lm_head.weight", ("direct", "lm_head.weight")),
        ("model.language_model.embed_tokens.weight", ("direct", "model.embed_tokens.weight")),
        (
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            ("direct", "model.hyper_connection_mixer.hc_norm.weight"),
        ),
        (
            f"{L}.10.attn_hyper_connection.block_inject_weight.weight",
            ("direct", "model.layers.10.attn_hyper_connection.block_inject_weight.weight"),
        ),
        (
            f"{L}.10.linear_attn.conv1d.weight",
            ("direct", "model.layers.10.linear_attn.conv1d_weight"),
        ),
        (
            f"{L}.10.linear_attn.in_proj_qkv.weight",
            ("concat", "model.layers.10.linear_attn.in_proj_qkvz.weight", 0, 2, 0),
        ),
        (
            f"{L}.10.linear_attn.in_proj_z.weight",
            ("concat", "model.layers.10.linear_attn.in_proj_qkvz.weight", 1, 2, 0),
        ),
        (
            f"{L}.10.linear_attn.in_proj_b.weight",
            ("concat", "model.layers.10.linear_attn.in_proj_ba.weight", 0, 2, 0),
        ),
        (
            f"{L}.10.linear_attn.in_proj_a.weight",
            ("concat", "model.layers.10.linear_attn.in_proj_ba.weight", 1, 2, 0),
        ),
        (f"{L}.10.linear_attn.A_log", ("direct", "model.layers.10.linear_attn.A_log")),
        (
            f"{L}.11.self_attn.indexer.index_qk_proj.weight",
            ("direct", "model.layers.11.self_attn.indexer.index_qk_proj.weight"),
        ),
        (f"{L}.1.ple.conv1d.weight", ("direct", "model.layers.1.ple.conv1d_weight")),
        (
            f"{L}.1.ple.ple_embedding.layer_multipliers",
            ("direct", "model.layers.1.ple.ple_embedding.layer_multipliers"),
        ),
        (
            f"{L}.1.ple.ple_embedding.ngram_embedding.shard_63.weight",
            ("skip", "ple-ngram-table"),
        ),
        (
            f"{L}.1.ple.ple_embedding.ngram_heads_offsets",
            ("skip", "ple-ngram-table"),
        ),
        # bf16 shared expert: a bare `.weight` is NOT renamed.
        (
            f"{L}.0.mlp.shared_expert.gate_proj.weight",
            ("direct", "model.layers.0.mlp.shared_expert.gate_proj.weight"),
        ),
        # NVFP4 routed expert: the SAME leaf spelling IS the packed blob.
        (
            f"{L}.0.mlp.experts.7.gate_proj.weight",
            ("direct", "model.layers.0.mlp.experts.7.gate_proj.weight_packed"),
        ),
        (
            f"{L}.0.mlp.experts.7.gate_proj.weight_scale",
            ("direct", "model.layers.0.mlp.experts.7.gate_proj.weight_scale"),
        ),
        (
            f"{L}.0.mlp.experts.7.gate_proj.weight_scale_2",
            ("direct", "model.layers.0.mlp.experts.7.gate_proj.weight_global_scale"),
        ),
        (f"{L}.0.mlp.experts.7.gate_proj.input_scale", ("skip", "act-calibration")),
        ("mtp.layers.0.self_attn.q_proj.weight", ("skip", "mtp-head")),
        ("model.visual.blocks.3.attn.qkv.weight", ("skip", "vision")),
    ]
    for key, want in cases:
        check(key.replace("model.language_model.", "…"), qwen4_exp_remap(key, nvfp4_modules=nvfp4_modules), want)

    print("\n[5] a bare `.weight` on a routed expert is bf16 when the checkpoint says so")
    # Same key, empty nvfp4 set -> NOT renamed. This is the distinction the function refuses to
    # guess: it is invisible in the key string and wrong either way if assumed.
    check(
        "no nvfp4 set -> .weight stays .weight",
        qwen4_exp_remap(f"{L}.0.mlp.experts.7.gate_proj.weight", nvfp4_modules=()),
        ("direct", "model.layers.0.mlp.experts.7.gate_proj.weight"),
    )

    print("\n[6] negative cases raise instead of passing through")
    for bad in (
        f"{L}.0.self_attn.foo_proj.weight",
        f"{L}.0.mlp.experts.7.gate_proj.some_new_scale",
        "some.other.namespace.weight",
    ):
        raised = False
        try:
            qwen4_exp_remap(bad, nvfp4_modules=nvfp4_modules)
        except ValueError:
            raised = True
        check(f"raises on {bad[-40:]!r}", raised, True)

    print("\n[7] full pipeline (remap -> fold -> gate/up merge -> expert stack)")
    native, skips2, stats = native_key_plan(names)
    check("skip counts stable", skips2, skips)
    check("expert stacks", stats["expert_stacks"], 48 * 4)  # {gate_up,down} x {packed,scale}
    check("stack width", stats["stack_width"], 512)
    # 1140 = the model's state_dict size when the routed experts (and ONLY they) are NVFP4.
    # qwen4exp_build_test.py asserts set equality against the meta build; this pins the count here
    # too so a name-level change shows up in the cheap test first.
    check("distinct native params", len(native), 1140)

    print(f"\n{'PASS' if not _failures else f'FAIL ({_failures} checks)'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
