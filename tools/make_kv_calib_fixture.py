"""Build the fp8-KV calibration fixture — deterministically, from recorded sources.

WHAT A CALIBRATION FIXTURE IS. `tools/kv_fp8_calibrate.py` takes `amax|K|`/`amax|V|` over this text
and turns it into the served cache's descale. That scale is a PROMISE about the range of everything
the model will ever store: text whose activations are quieter than production's yields a scale that
CLIPS (a real store above 448*scale saturates), and text that is wilder than production's wastes
range and flushes quiet heads to zero. So the fixture has to look like the traffic this box actually
serves — coding-agent context and tool-call JSON, long-context retrieval, ordinary prose — not a
generic wikitext dump.

WHY A BUILDER SCRIPT AND NOT A CHECKED-IN BLOB. Repo rule: measurement fixtures must be DURABLE and
RECORDED. The generated text lives outside the repo (it is derived, ~MB, and partly from local
caches), so what is committed is this script plus the manifest it writes: source list, per-source
byte contribution, and the sha256 of the output. `kv_fp8_calibrate.py` then stamps that same
size+hash into the sidecar metadata, so any scale table on this box can be traced fixture -> sources.

INTERLEAVED, NOT CONCATENATED. The calibrator consumes the FIRST `--max-chunks` chunks, so a
concatenated fixture would silently calibrate on nothing but its first source. Segments are
round-robined by weight, which makes every prefix a mixture in roughly the target proportions.

Deterministic: same sources on disk -> byte-identical output (fixed seed, sorted file order).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys

# Anything that looks like a credential is dropped rather than baked into a fixture that gets copied
# around. This is a hygiene filter on LOCAL cache files, not a security control.
_SECRET = re.compile(
    r"(sk-[A-Za-z0-9]{16,}|api[_-]?key\"?\s*[:=]\s*\"?[A-Za-z0-9_\-]{16,}|Bearer\s+[A-Za-z0-9._\-]{16,}"
    r"|password\"?\s*[:=]|hf_[A-Za-z0-9]{20,})",
    re.I,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS = "/home/pat/code/rdna4-hip-kernels"


def _segments(text: str, size: int) -> list[str]:
    """Split into ~`size`-byte segments on line boundaries (never mid-line: a torn line is a token
    distribution this model never sees in production)."""
    out, buf, n = [], [], 0
    for line in text.splitlines(keepends=True):
        buf.append(line)
        n += len(line)
        if n >= size:
            out.append("".join(buf))
            buf, n = [], 0
    if buf:
        out.append("".join(buf))
    return [s for s in out if len(s) > 64 and not _SECRET.search(s)]


def _read(path: str) -> str:
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def _src_code(seg: int) -> list[str]:
    """Engine + kernel source. The dominant traffic class on this box is a coding agent reading and
    writing exactly these files."""
    files: list[str] = []
    for pat in ("python/minisgl/**/*.py", "tools/*.py"):
        files += sorted(glob.glob(os.path.join(REPO, pat), recursive=True))
    if os.path.isdir(KERNELS):
        for pat in ("**/*.hip", "**/*.cpp", "**/*.h", "**/*.py"):
            files += sorted(glob.glob(os.path.join(KERNELS, pat), recursive=True))[:60]
    out: list[str] = []
    for p in files:
        out += _segments(f"# file: {os.path.relpath(p, os.path.dirname(REPO))}\n" + _read(p), seg)
    return out


def _src_docs(seg: int) -> list[str]:
    """Markdown: specs, continuances, READMEs — long technical prose with code fences, which is what
    an agent's context is mostly made of between the code itself."""
    files = sorted(glob.glob(os.path.join(REPO, "*.md")))
    files += sorted(glob.glob(os.path.join(REPO, "docs/**/*.md"), recursive=True))
    if os.path.isdir(KERNELS):
        files += sorted(glob.glob(os.path.join(KERNELS, "*.md")))
    out: list[str] = []
    for p in files:
        out += _segments(_read(p), seg)
    return out


def _src_toolcall(seg: int) -> list[str]:
    """Real tool/MCP JSON from this box's agent caches. Tool schemas and tool RESULTS are a large,
    structurally odd slice of served tokens (deep JSON, long identifier runs) and are exactly the
    traffic a chat-only fixture misses."""
    # Schemas AND results: the bulk of tool tokens is RESULTS (bench dumps, config blobs), not the
    # schema block, so both are drawn from — the agent caches for schemas, the repo's own JSON
    # artifacts for result-shaped payloads.
    paths = sorted(glob.glob(os.path.expanduser("~/.hermes/cache/*.json")))
    paths += sorted(glob.glob(os.path.join(REPO, "control-panel/*.json")))
    paths += sorted(glob.glob(os.path.join(REPO, "*.json")))
    paths += sorted(glob.glob(os.path.join(REPO, "tools/*.json")))
    paths += sorted(glob.glob(os.path.join(REPO, "docs/**/*.json"), recursive=True))
    out: list[str] = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            pretty = json.dumps(json.load(open(p)), indent=2)
        except Exception:
            pretty = _read(p)
        out += _segments(pretty, seg)
    return out


def _src_longctx(seg: int) -> list[str]:
    """BABILong: a fact buried in a long distractor context, then a question. This is the shape of
    the ONE case fp8-KV was measured to lose (mid-context fact retrieval at 7.7k tokens), so the
    fixture must contain it — each sample is emitted whole, not segmented, so the long context stays
    long."""
    out: list[str] = []
    for qa in ("qa1", "qa2", "qa3", "qa5"):
        for size in ("4k", "16k"):
            for p in sorted(
                glob.glob(
                    f"/home/pat/.cache/huggingface/hub/datasets--RMT-team--babilong/snapshots/*/data/{qa}/{size}.json"
                )
            ):
                try:
                    rows = json.load(open(p))
                except Exception:
                    continue
                for r in rows[:8]:
                    out.append(f"{r.get('input','')}\n\nQuestion: {r.get('question','')}\nAnswer: {r.get('target','')}\n")
    _ = seg
    return out


def _src_prose(seg: int) -> list[str]:
    """Ordinary encyclopedic prose (wikitext-2). A baseline register: no code, no JSON, no
    long-range structure — present so the scale is not fitted only to machine-shaped text."""
    files = sorted(
        glob.glob(
            "/home/pat/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/train-*.parquet"
        )
    )
    if not files:
        return []
    import pyarrow.parquet as pq

    txt = "".join(str(x) for x in pq.read_table(files[0]).column("text").to_pylist()[:20000])
    return _segments(txt, seg)


SOURCES = [
    # (name, builder, weight) — weights are the target byte mixture, applied by round-robin so any
    # PREFIX of the fixture is also a mixture (the calibrator reads only the first --max-chunks).
    ("code", _src_code, 0.30),
    ("docs", _src_docs, 0.20),
    ("toolcall_json", _src_toolcall, 0.10),
    ("longctx_babilong", _src_longctx, 0.25),
    ("prose_wikitext", _src_prose, 0.15),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/pat/fixtures/minisgl-kv-calib/kv_calib_v1.txt")
    ap.add_argument("--manifest", default=None, help="default: <out>.manifest.json")
    ap.add_argument("--bytes", type=int, default=2_000_000, help="target fixture size")
    ap.add_argument("--segment", type=int, default=4096, help="bytes per interleaved segment")
    ap.add_argument("--seed", type=int, default=20260804)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pools, weights, names = [], [], []
    for name, fn, w in SOURCES:
        segs = fn(args.segment)
        if not segs:
            print(f"WARN: source {name} produced nothing — it is MISSING from this fixture", file=sys.stderr)
            continue
        rng.shuffle(segs)  # seeded: deterministic, but not "the first 30 files of the repo"
        pools.append(segs)
        weights.append(w)
        names.append(name)
    if not pools:
        print("FAIL: no sources", file=sys.stderr)
        return 2

    cursor = [0] * len(pools)
    used = [0] * len(pools)
    chosen: list[str] = []
    total = 0
    while total < args.bytes:
        # Pick the source that is furthest BELOW its target share — deterministic, self-correcting,
        # and it degrades gracefully when a source runs out of segments (it just stops being behind).
        best, best_deficit = -1, None
        for i in range(len(pools)):
            if cursor[i] >= len(pools[i]):
                continue
            deficit = weights[i] - (used[i] / total if total else 0.0)
            if best_deficit is None or deficit > best_deficit:
                best, best_deficit = i, deficit
        if best < 0:
            print(f"WARN: sources exhausted at {total} bytes (< target {args.bytes})", file=sys.stderr)
            break
        seg = pools[best][cursor[best]]
        cursor[best] += 1
        chosen.append(seg)
        used[best] += len(seg)
        total += len(seg)

    text = "\n\n".join(chosen)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    raw = text.encode()
    digest = hashlib.sha256(raw).hexdigest()

    try:
        head = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        head = ""
    manifest = {
        "fixture": os.path.abspath(args.out),
        "bytes": len(raw),
        "sha256": digest,
        "sha256_16": digest[:16],
        "seed": args.seed,
        "segment_bytes": args.segment,
        "builder": os.path.abspath(__file__),
        "builder_repo_head": head,
        "sources": [
            {"name": n, "weight": w, "bytes": u, "share": round(u / len(raw), 4), "segments": c}
            for n, w, u, c in zip(names, weights, used, cursor)
        ],
    }
    mpath = args.manifest or (args.out + ".manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out}: {len(raw)} bytes, sha256[:16]={digest[:16]}")
    print(f"wrote {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
