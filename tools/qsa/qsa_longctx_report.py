#!/usr/bin/env python3
"""Turn a `qwen4exp_qsa_longctx_test.py` JSON into the tables QSA_INDEXER.md quotes.

Kept as a script rather than done by hand because every number in the doc has to be re-derivable
from the artifact, and a table typed out of a log is a number with no provenance. Prints markdown.
"""
from __future__ import annotations

import json
import sys


def main(paths) -> int:
    for path in paths:
        with open(path) as fh:
            d = json.load(fh)
        r0 = next((r for r in d.get("ranks", []) if r.get("rank") == 0), None)
        if r0 is None:
            print(f"## {path}: no rank-0 record")
            continue
        print(f"\n## {path}")
        print(f"- tp={d.get('tp')} layers={d.get('layers')} sampler={d.get('sampler')}")
        print(f"- cards: {r0.get('card')} (rank0 device {r0.get('device_index')})")
        print(f"- boot {r0.get('boot_seconds')} s")
        print(f"- kv_pages={r0.get('kv_pages')} page_size={r0.get('page_size')} "
              f"pool_tokens={r0.get('kv_pool_tokens')} "
              f"kv_bytes_per_token_per_rank={r0.get('kv_bytes_per_token_per_rank')} "
              f"pool_bytes_per_rank={r0.get('kv_pool_bytes_per_rank')}")
        print(f"- engine_max_seq_len={r0.get('engine_max_seq_len')} "
              f"checkpoint_max_position={r0.get('checkpoint_max_position')}")
        print(f"- qsa_active={r0.get('qsa_active')} {r0.get('qsa_cache')}")
        print(f"- max_context_served_tokens={r0.get('max_context_served_tokens')}")
        if r0.get("ladder_aborted_after"):
            print(f"- LADDER ABORTED after target {r0['ladder_aborted_after']}")
        print()
        # DECODE WALL, derived here rather than in the harness, because the harness's
        # `wall_tok_per_s` is tokens over the WHOLE generate and therefore includes the prefill —
        # at ctx=4k that is 67 s of prefill against 3.5 s of decode and the figure reads as
        # 0.5 tok/s, which is a true end-to-end number and a useless decode one. The comparable
        # quantity to the published 13.93 eager wall tok/s is decode tokens over decode WALL time,
        # i.e. forward + the PCIe expert residual that sits outside the timed forward.
        hdr = ("| ctx (actual) | w | ok | prefill s | prefill tok/s | decode ms/step (fwd) | "
               "decode tok/s (fwd) | decode tok/s (wall) | e2e tok/s | sparsity | early needle "
               "| late needle | gen tok |")
        print(hdr)
        print("|" + "---|" * (hdr.count("|") - 1))
        for r in r0.get("ladder", []):
            if not r.get("ok"):
                print(f"| {r.get('actual_tokens')} | {r.get('batch_width', 1)} | "
                      f"**REFUSED{'/TERMINAL' if r.get('terminal') else ''}** | "
                      f"{r.get('error_type')} | | | | | | | | |")
                continue
            gen = sum(r.get("batch_gen_tokens") or [r.get("gen_tokens", 0)])
            dwall = (r.get("wall_seconds", 0) or 0) - (r.get("prefill_seconds", 0) or 0)
            dtps = round(gen / dwall, 2) if dwall > 0 and gen else None
            print(f"| {r['actual_tokens']} | {r.get('batch_width', 1)} | yes | "
                  f"{r.get('prefill_seconds')} | {r.get('prefill_tok_per_s')} | "
                  f"{r.get('decode_ms_per_step_median')} | {r.get('decode_tok_per_s_median')} | "
                  f"{dtps} | {r.get('wall_tok_per_s')} | "
                  f"{r.get('sparsity_visited_over_dense')} | "
                  f"{r.get('retrieved_early_needle')} | {r.get('retrieved_late_needle')} | "
                  f"{gen} |")
        print()
        for r in r0.get("ladder", []):
            if r.get("ok"):
                print(f"### ctx={r['actual_tokens']} w={r.get('batch_width',1)} "
                      f"degeneration={r.get('degeneration')}")
                print(f"```\n{r.get('text','')}\n```")
                for i, t in enumerate(r.get("batch_texts", [])[1:], start=1):
                    print(f"(concurrent request {i})\n```\n{t}\n```")
            else:
                print(f"### ctx={r['actual_tokens']} REFUSED\n```\n{r.get('error','')}\n```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["out/qsa_longctx/r2.json"]))
