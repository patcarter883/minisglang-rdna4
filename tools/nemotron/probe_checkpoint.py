#!/usr/bin/env python3
"""Read a HF safetensors checkpoint's TENSOR INVENTORY without downloading the weights.

Phase 0 of the Nemotron-H bring-up (docs/NEMOTRON35_LIGHTNING_PLAN.md) needs three facts that the
config.json does not carry: which modules actually ship FP8 vs NVFP4 vs bf16, the exact tensor NAMES
(the loader maps against them), and the real shapes. Downloading 20 GiB to answer that is absurd —
a safetensors file states its whole inventory in a JSON header at byte 0, so two HTTP range requests
per shard is enough.

    safetensors layout:  [u64 LE header_len][header_len bytes of JSON][ ...tensor data... ]
    header JSON:         {"name": {"dtype": "F8_E4M3", "shape": [...], "data_offsets": [a, b]}, ...}

Host-side only. No GPU, no torch, no huggingface_hub — stdlib urllib, so it runs anywhere.

    python3 tools/nemotron/probe_checkpoint.py nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
        --out docs/measurements/NEMOTRON_INVENTORY/target.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict

HF = "https://huggingface.co"
UA = {"User-Agent": "minisgl-nemotron-probe/1"}


def _get(url: str, rng: tuple[int, int] | None = None, tries: int = 3) -> bytes:
    headers = dict(UA)
    if rng is not None:
        headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:               # noqa: PERF203 — retry is the point
            if e.code in (401, 403, 404):
                raise
            last = e
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"GET {url} failed: {last}")


def resolve(repo: str, rev: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/{rev}/{path}"


def shard_header(repo: str, rev: str, path: str) -> dict:
    """The tensor inventory of ONE shard, from two range reads."""
    url = resolve(repo, rev, path)
    n = struct.unpack("<Q", _get(url, (0, 7)))[0]
    if n > 200 << 20:
        raise RuntimeError(f"{path}: implausible header length {n}")
    return json.loads(_get(url, (8, 8 + n - 1)).decode())


def shard_list(repo: str, rev: str) -> list[str]:
    """Shard paths from the index, or the single-file name when there is no index."""
    try:
        idx = json.loads(_get(resolve(repo, rev, "model.safetensors.index.json")).decode())
    except urllib.error.HTTPError:
        return ["model.safetensors"]
    return sorted(set(idx["weight_map"].values()))


# Collapse `backbone.layers.31.mixer.experts.7.up_proj.weight` -> a pattern, so 128 experts x 23
# layers become ONE row. The point of the summary is the module KINDS and their dtypes.
_NUM = re.compile(r"\.\d+\.")


def pattern(name: str) -> str:
    prev = None
    while prev != name:
        prev, name = name, _NUM.sub(".N.", name)
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--out", default=None, help="write the full inventory here as JSON")
    ap.add_argument("--config", action="store_true", help="also fetch and print config.json")
    a = ap.parse_args()

    shards = shard_list(a.repo, a.revision)
    print(f"{a.repo}@{a.revision}: {len(shards)} shard(s)", file=sys.stderr)

    inv: dict[str, dict] = {}
    for i, s in enumerate(shards, 1):
        hdr = shard_header(a.repo, a.revision, s)
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            inv[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "shard": s}
        print(f"  [{i}/{len(shards)}] {s}: {len(hdr) - ('__metadata__' in hdr)} tensors",
              file=sys.stderr, flush=True)

    by_pat: dict[str, Counter] = defaultdict(Counter)
    shapes: dict[str, set] = defaultdict(set)
    for name, m in inv.items():
        p = pattern(name)
        by_pat[p][m["dtype"]] += 1
        shapes[p].add(tuple(m["shape"]))

    print(f"\n{'=' * 100}\n{a.repo}\n{'=' * 100}")
    print(f"tensors: {len(inv)}   dtypes: {dict(Counter(m['dtype'] for m in inv.values()))}\n")
    print(f"{'module pattern':<62} {'n':>5}  {'dtype':<10} shape(s)")
    print("-" * 100)
    for p in sorted(by_pat):
        dts = by_pat[p]
        sh = sorted(shapes[p])
        sh_s = str(sh[0]) if len(sh) == 1 else f"{len(sh)} distinct e.g. {sh[0]}"
        print(f"{p:<62} {sum(dts.values()):>5}  {'/'.join(dts):<10} {sh_s}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"repo": a.repo, "revision": a.revision, "tensors": inv}, f, indent=1)
        print(f"\nwrote {a.out} ({len(inv)} tensors)", file=sys.stderr)

    if a.config:
        cfg = _get(resolve(a.repo, a.revision, "config.json")).decode()
        print("\n--- config.json ---")
        print(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
