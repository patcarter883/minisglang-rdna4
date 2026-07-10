#!/usr/bin/env python
"""Phase-3 validation: /stats observability + snapshot/restore persistence of the editable banks."""
from __future__ import annotations

import argparse
import torch

from minisgl.core import SamplingParams
from minisgl.llm import LLM

FACTS = [("Lionel Jospin", "The mother tongue of Lionel Jospin is", "Dutch"),
         ("Oleg Kotov",    "The mother tongue of Oleg Kotov is",    "English"),
         ("Marie NDiaye",  "The mother tongue of Marie NDiaye is",  "Russian")]
SNAP = "/tmp/cam_snapshot.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--graph", type=int, default=2)
    ap.add_argument("--max-running-req", type=int, default=2)
    ap.add_argument("--memory-ratio", type=float, default=0.8)
    args = ap.parse_args()

    llm = LLM(model_path=args.model, dtype=torch.bfloat16, cuda_graph_max_bs=args.graph, page_size=16,
              memory_ratio=args.memory_ratio, attention_backend="hip",
              max_running_req=args.max_running_req)
    cam = llm.engine.cam
    tok = llm.tokenizer
    encsp = lambda s: list(tok(" " + s, add_special_tokens=False).input_ids)
    sp = SamplingParams(temperature=0.0, max_tokens=10)

    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    enc = lambda s: list(tok(s, add_special_tokens=False).input_ids)

    def ask(s, p):
        return llm.generate([p], SamplingParams(temperature=0.0, max_tokens=10,
                            mem_subject=s))[0]["text"].replace("\n", " ").strip()

    for s, p, o in FACTS:
        # gated remember (updates the side index) via the SERVED model's base logits — the /cam/remember
        # path. _write alone populates the bank but not the _facts side index that stats/snapshot use.
        pl = llm.base_logits(bos + enc(p))
        cam.set_pending_object(encsp(o))
        cam.remember(encsp(s), pl)
    st = cam.stats()
    print(f"[stats] total_edits={st['total_edits']} max_bank_load={st['max_bank_load']} "
          f"imbalance={st['imbalance']:.2f} crowded={st['crowded_banks']}", flush=True)
    assert st["total_edits"] == len(FACTS), "stats miscount"

    n = cam.snapshot(SNAP)
    print(f"[snapshot] saved {n} edits -> {SNAP}", flush=True)

    cam.reset()
    print(f"[reset] total_edits={cam.stats()['total_edits']}", flush=True)
    assert cam.stats()["total_edits"] == 0
    d0 = ask(FACTS[1][0], FACTS[1][1])
    print(f"[after-reset ask] {d0!r}   [delivers 'English': {'english' in d0.lower()}]", flush=True)

    n = cam.restore(SNAP)
    print(f"[restore] loaded {n} edits; total_edits={cam.stats()['total_edits']}", flush=True)
    assert cam.stats()["total_edits"] == len(FACTS)

    n_hit = 0
    for s, p, o in FACTS:
        d = ask(s, p)
        hit = o.lower() in d.lower()
        n_hit += hit
        print(f"[after-restore] {p!r} -> {d!r}   [delivered '{o}': {hit}]", flush=True)
    print(f"\nCAM-PHASE3 stats=OK snapshot/restore delivery {n_hit}/{len(FACTS)}", flush=True)


if __name__ == "__main__":
    main()
