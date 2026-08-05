"""Locate a prefix-reuse defect by diffing two serves' per-layer KV state digests.

The block-diffusion SWA-radix gate (docs/…BLOCK_DIFFUSION.md §D5) found a PARTIAL prefix hit whose
generated text differs from the same prompt served cold. Text is the worst place to debug that: it
has been through 30 layers, a 16-step denoising trajectory and an entropy bound that SORTS 256
values, so one flipped bit anywhere rewrites the whole answer and no amount of staring at it says
where the bit flipped.

This is the `tools/cca_chunk_bisect.py` move applied to state instead of activations. Both serves run
`MINISGL_STATE_DIGEST=1`, which makes `_canvas_encode` hash — per layer, per pool — the KV the
encoder pass just produced (kvcache/state_digest.py). A prefill is correct iff that state is
bit-identical to a cold prefill's, because everything downstream is a pure function of it. So:

    digests MATCH      -> the reused prefill is EXACT; the divergence is downstream (canvas/sampler),
                          and the prefix cache is exonerated.
    digests DIFFER     -> located. The report names the first differing LAYER, which POOL, and — the
                          half that decides the story — WHICH POSITIONS:
                            pool=main -> the reused radix PAGES are wrong (content, address, or
                                         mutated after insert) — a full-attention-layer problem;
                            pool=swa  -> the sliding-window state is wrong;
                            segments entirely inside the REUSED span [0, hit)  -> what the cache
                                         handed over is not what a cold prefill produces;
                            segments only at/after `hit`                        -> the cache was
                                         right and the EXTEND that continued from it miscomputed.

Usage:  python3 tools/canvas_state_bisect.py <hit-serve.log> <cold-serve.log>

The two logs must carry the SAME prompt at the same boundary; the tool refuses to compare boundaries
that differ, because a different `device_len` means a different sequence and any diff would be
meaningless rather than informative.
"""
from __future__ import annotations

import re
import sys
from collections import OrderedDict

LINE = re.compile(
    r"\[state-digest\] (?P<tag>\S+) uid=(?P<uid>\d+) boundary=(?P<b>\d+) hit=(?P<hit>\d+) "
    r"pool=(?P<pool>\S+) layer=(?P<layer>\d+) seg=(?P<a>\d+):(?P<z>\d+) k=(?P<k>\S+) v=(?P<v>\S+)"
)


def parse(path: str):
    """((boundary, pool, layer, seg_start) -> (k, v),  boundary -> hit) for the LAST request at each
    boundary.

    Last-wins on purpose: a probe warms the shared prefix and then measures, so the same boundary can
    appear more than once and the measured request is the one that ran last."""
    out: "OrderedDict" = OrderedDict()
    hits = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = LINE.search(line)
        if m:
            b = int(m["b"])
            out[(b, m["pool"], int(m["layer"]), int(m["a"]))] = (m["k"], m["v"])
            hits[b] = int(m["hit"])
    return out, hits


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    hit, hit_h = parse(sys.argv[1])
    cold, cold_h = parse(sys.argv[2])
    if not hit or not cold:
        print(f"no [state-digest] lines in {'hit log' if not hit else 'cold log'} — "
              f"was MINISGL_STATE_DIGEST=1 set on both serves?")
        return 2

    b_hit = sorted({k[0] for k in hit})
    b_cold = sorted({k[0] for k in cold})
    common = sorted(set(b_hit) & set(b_cold))
    if not common:
        print(f"REFUSING to compare: no shared boundary. hit={b_hit} cold={b_cold}. A different "
              f"device_len is a different sequence, so a diff would be noise, not evidence.")
        return 2

    rc = 0
    for b in common:
        keys = [k for k in hit if k[0] == b and k in cold]
        diffs = [k for k in keys if hit[k] != cold[k]]
        hb, cb = hit_h.get(b, 0), cold_h.get(b, 0)
        print(f"\n=== boundary {b} — {len(keys)} (pool, layer, seg) cells; "
              f"reused span: hit-leg [0,{hb}) vs cold-leg [0,{cb}) ===")
        if not diffs:
            print("  IDENTICAL — the reused prefill left byte-identical KV. The prefix cache is "
                  "EXONERATED at this boundary; look downstream (canvas trajectory / sampler).")
            continue
        rc = 1
        pools = sorted({d[1] for d in diffs})
        print(f"  DIFFER in {len(diffs)}/{len(keys)} cells, pools={pools}")
        for pool in pools:
            dp = [d for d in diffs if d[1] == pool]
            layers = sorted({d[2] for d in dp})
            segs = sorted({d[3] for d in dp})
            nl = len({k[2] for k in keys if k[1] == pool})
            print(f"    pool={pool:5s} first differing layer={layers[0]} ({len(layers)}/{nl} layers) "
                  f"first differing segment={segs[0]} (segments: {segs})")
            # THE question: does anything differ strictly BELOW the reuse boundary?
            below = [d for d in dp if d[3] < hb]
            if below:
                print(f"      {len(below)} cells differ INSIDE the reused span (< {hb}) — what the "
                      f"cache restored is NOT what a cold prefill produces.")
            else:
                print(f"      nothing differs below {hb}: the reused span is EXACT and the defect is "
                      f"in the EXTEND that continued from it.")
        print("  pool=main -> the FULL-attention paged KV.  pool=swa -> the sliding-window ring.")
    print(f"\n{'PASS (state identical)' if rc == 0 else 'FAIL (state differs — see above)'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
