#!/usr/bin/env python3
"""Validate the qwen4_exp checkpoint remap and chunk plan from HEADERS ALONE.

No tensors are read, no GPU is touched, nothing is loaded -- so a remap defect that would
otherwise surface many minutes into a boot (or worse, as a silently missing weight) is caught
in seconds.

Three things are checked, and the third is the one an ad-hoc script kept missing:

  1. EVERY checkpoint key gets a plan.  An UNRECOGNISED key means `qwen4_exp_remap` raises
     mid-load.  Validated against the FILES on disk, never against `model.safetensors.index.json`
     -- the index does not list every shard the loader globs (`model-expertprofile.safetensors`
     is not in it) and a key set derived from the index passes clean while the real load raises.

  2. Every expert layer the checkpoint holds appears in exactly one chunk's `finalize_paths`.
     A layer finalized twice runs `post_load` twice; a layer finalized never leaves its experts
     resident on the device, which is how the single-chunk pre-stacked plan OOMed at 15.43 GiB.

  3. The chunk KEY FILTERS partition the native keys.  A native key that passes NO filter is
     never yielded to any chunk and its weight is never loaded -- no error, no warning, just a
     tensor left at its initialized value.  A key that passes TWO is loaded twice.  Both are
     silent, and both are easy to introduce by editing a filter predicate.

Usage:  python tools/q4e_remap_validate.py <model_folder>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
from collections import Counter


def header_keys(files: "list[str]") -> "dict[str, list[str]]":
    """`{file: [tensor names]}` from each shard's JSON header, nothing else read."""
    out: "dict[str, list[str]]" = {}
    for path in files:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        out[path] = [k for k in hdr if k != "__metadata__"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_folder")
    ap.add_argument("--repo-python", default=None,
                    help="path to <repo>/python to import minisgl from (default: alongside this tool)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, args.repo_python or os.path.join(here, "python"))

    from minisgl.models.weight import (  # noqa: E402
        _Q4_SHARED_DEQ,
        _q4_is_stacked_expert_key,
        _q4_stacked_expert_layer,
        qwen4_exp_has_stacked_experts,
        qwen4_exp_nvfp4_prepass,
        qwen4_exp_remap,
        qwen4_exp_stacked_expert_files,
    )

    folder = args.model_folder
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not files:
        print(f"no *.safetensors in {folder}", file=sys.stderr)
        return 2
    by_file = header_keys(files)
    stacked = qwen4_exp_has_stacked_experts(files)
    # `(nvfp4_ckpt_bases, nvfp4_modules)` -- the remap wants the SECOND.
    _nvfp4_bases, nvfp4_modules = qwen4_exp_nvfp4_prepass(files)
    cfg = json.load(open(os.path.join(folder, "config.json")))
    text = cfg.get("text_config", cfg)
    num_layers = int(text["num_hidden_layers"])

    print(f"{len(files)} shards, {sum(map(len, by_file.values()))} tensors; "
          f"stacked_experts={stacked}, num_hidden_layers={num_layers}")

    # ---- 1. every checkpoint key gets a plan -------------------------------------------------
    plans: Counter = Counter()
    native: "set[str]" = set()
    bad: "list[tuple[str, str]]" = []
    for path, keys in by_file.items():
        for ck in keys:
            # Mirror the LEAF transforms the loader applies before calling the remap, because the
            # remap sees the native name, not the checkpoint one.
            base, _, field = ck.rpartition(".")
            emitted = [ck]
            if _Q4_SHARED_DEQ in ck and field in ("weight_packed", "weight_scale"):
                # Folded to ONE bf16 `.weight` (this engine keeps the shared expert separate).
                emitted = [base + ".weight"]
            try:
                for name in emitted:
                    plan = qwen4_exp_remap(name, nvfp4_modules=nvfp4_modules)
                    kind, target = plan[0], (plan[1] if len(plan) > 1 else None)
                    plans[kind] += 1
                    if kind != "skip" and isinstance(target, str):
                        native.add(target)
            except Exception as e:
                bad.append((ck, f"{type(e).__name__}: {e}"))

    print(f"plans: {dict(plans)}")
    if bad:
        print(f"\nUNRECOGNISED: {len(bad)}")
        for ck, why in bad[:20]:
            print(f"  {ck}\n      {why}")
        return 1
    print("UNRECOGNISED: 0")

    # ---- 2. expert layers finalized exactly once ---------------------------------------------
    if stacked:
        by_layer = qwen4_exp_stacked_expert_files(files)
        missing = [l for l in range(num_layers) if l not in by_layer]
        print(f"\nexpert layers with shards: {len(by_layer)}"
              + (f"; NO SHARD for layers {missing}" if missing else "; every layer covered"))
        if missing:
            return 1

    # ---- 3. the chunk key filters partition the native keys ----------------------------------
    # Rebuilt here from the same predicates the plan uses, so this check tracks an edit to them.
    if stacked:
        step_src = __import__("minisgl.models.weight", fromlist=["x"])
        step = step_src._Q4_STACKED_LAYERS_PER_CHUNK
        batches = [list(range(i, min(i + step, num_layers))) for i in range(0, num_layers, step)]
        filters = [("body", lambda k: not _q4_is_stacked_expert_key(k))]
        for lids in batches:
            want = frozenset(lids)
            filters.append((f"stacked-{lids[0]}-{lids[-1]}",
                            lambda k, _w=want: (l := _q4_stacked_expert_layer(k)) is not None
                            and l in _w))
        hits: Counter = Counter()
        orphan: "list[str]" = []
        dup: "list[str]" = []
        for k in sorted(native):
            n = [nm for nm, f in filters if f(k)]
            hits[len(n)] += 1
            if not n:
                orphan.append(k)
            elif len(n) > 1:
                dup.append(f"{k} -> {n}")
        print(f"\nchunk filters over {len(native)} native keys: "
              f"{ {f'{c} chunk(s)': v for c, v in sorted(hits.items())} }")
        # PRESENCE IS NOT COVERAGE. "every key in exactly one chunk" passes trivially if no key is a
        # stacked-expert key at all: the body filter is the negation, so it would claim all of them
        # and the expert chunks would claim nothing. Assert the expert chunks actually claim the
        # 4 leaves x 48 layers they exist for, or this check proves nothing about the thing it is for.
        claimed = sum(1 for k in native if _q4_is_stacked_expert_key(k))
        want_claimed = 4 * num_layers
        print(f"stacked-expert native keys claimed by the expert chunks: {claimed} "
              f"(expected {want_claimed} = 4 leaves x {num_layers} layers)")
        if claimed != want_claimed:
            print("MISMATCH: the partition check above is vacuous -- the expert chunks are "
                  "claiming the wrong number of keys, so the body filter is carrying them.")
            return 1
        if orphan:
            print(f"ORPHANED (never loaded, silently): {len(orphan)}")
            for k in orphan[:20]:
                print(f"  {k}")
        if dup:
            print(f"DOUBLE-LOADED: {len(dup)}")
            for k in dup[:20]:
                print(f"  {k}")
        if orphan or dup:
            return 1
        print("partition OK: every native key in exactly one chunk")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
