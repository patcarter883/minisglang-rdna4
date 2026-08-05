"""Capture REAL Gemma4 router inputs, so a reduction-order change on the router GEMM can be judged on
expert SELECTION rather than on a value delta.

WHY THE EXACT TENSOR MATTERS. The router input is not the residual: Gemma4Router does
RMSNormNoScale(residual) * scale * hidden_size**-0.5 and feeds THAT to minv_linear. Only the
post-norm, post-scale tensor has the right magnitude AND the right cross-dimension correlation, and
correlation is what sets the density of near-ties between adjacent experts — which is the entire
risk. So the hook captures `h` at the exact call site, not the block input.

WHY TP=2. The checkpoint is 17.2 GB and does not fit one 16 GB card. A TP>1 offline run here is ONE
PROCESS PER RANK (the caller spawns them, exactly as server/launch.py and tools/kv_fp8_calibrate.py
do). The router itself is REPLICATED and the hidden dim is not sharded, so every rank sees the same
`h` and the same weight — rank 0's capture is the whole truth, and the other rank exists only to hold
half the model.

Writes a fixture consumed by rdna4-hip-kernels dense_gemm/local/splitk_router_flips.py:
    {"layers": [{"layer": int, "W": [E,H] bf16 cpu, "h": [N,H] bf16 cpu}], "top_k": int,
     "provenance": str}
Durable by default — a fixture on tmpfs cannot be re-checked later.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_OUT = "/home/pat/code/_fixtures/gemma4_router.pt"

# Real prompts, deliberately varied: near-tie density is a property of the hidden-state distribution,
# and one prompt's tokens are far too correlated to stand in for a serve.
PROMPTS = [
    "Explain, in careful detail, how a tensor-parallel transformer splits its attention and MLP "
    "weights across two GPUs, and what has to be all-reduced and why.",
    "Write a Python function that merges two sorted lists in linear time, then explain the "
    "invariant that makes it correct.",
    "The capital of France is Paris, and the capital of Japan is",
    "Q: A train leaves Chicago at 3pm travelling 60 mph. A second leaves at 4pm at 80 mph. "
    "When does the second catch the first? A:",
    "Translate to French, then to German, then back to English: 'The quick brown fox jumps over "
    "the lazy dog near the river bank at dawn.'",
    "Summarise the causes of the 1929 stock market crash in three paragraphs, then list the "
    "policy responses in order of effectiveness.",
]


def _rank_main(rank: int, args, result_q) -> None:
    import torch
    from minisgl.core import SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.llm import LLM
    from minisgl.models import gemma4

    cap: dict[int, list] = {}
    wref: dict[int, "torch.Tensor"] = {}
    tk: dict[int, int] = {}
    order: dict[int, int] = {}      # id(router instance) -> ordinal, i.e. layer index in call order
    calls = {"n": 0}
    orig = gemma4.Gemma4Router.forward

    def patched(self, x):
        # Structure-agnostic: the first forward visits routers in layer order, so first-sight order
        # IS layer order. That avoids assuming anything about how blocks nest.
        key = id(self)
        if key not in order:
            order[key] = len(order)
        lid = order[key]
        if rank == 0 and lid in args.want:
            h = self._norm.forward(x)
            h = h * self.scale * self._scalar_root_size
            buf = cap.setdefault(lid, [])
            if sum(t.shape[0] for t in buf) < args.max_rows:
                buf.append(h.detach().reshape(-1, h.shape[-1]).to("cpu", torch.bfloat16).clone())
            if lid not in wref:
                wref[lid] = self.weight.detach().to("cpu", torch.bfloat16).clone()
            tk[lid] = self._top_k
        calls["n"] += 1
        return orig(self, x)

    gemma4.Gemma4Router.forward = patched

    with torch.inference_mode():
        if rank == 0:
            print(f"[cap] loading {args.model} TP={args.tp} ...", flush=True)
        llm = LLM(
            model_path=args.model,
            dtype=torch.bfloat16,
            tp_info=DistributedInfo(rank, args.tp),
            attention_backend=args.attn_backend,
            cuda_graph_max_bs=0,       # eager: capture costs boot time and changes nothing here
            memory_ratio=args.memory_ratio,
            max_running_req=args.max_running_req,
            gdn_radix=False,
        )
        # An instruction-tuned model fed RAW completion prompts decodes off-distribution, and
        # off-distribution hidden states have the wrong near-tie density — which is the one property
        # this capture exists to measure. Apply the checkpoint's own chat template when it has one.
        prompts = PROMPTS
        if not args.raw_prompts:
            try:
                tok = llm.tokenizer
                prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                                   tokenize=False, add_generation_prompt=True)
                           for p in PROMPTS]
                if rank == 0:
                    print(f"[cap] chat template applied; first prompt starts "
                          f"{prompts[0][:60]!r}", flush=True)
            except Exception as e:
                if rank == 0:
                    print(f"[cap] WARNING no chat template ({e}); using raw prompts", flush=True)
        t0 = time.perf_counter()
        outs = llm.generate(prompts,
                            SamplingParams(temperature=0.0, max_tokens=args.max_tokens))
        dt = time.perf_counter() - t0

    if rank != 0:
        result_q.put({"rank": rank, "ok": True})
        return

    print(f"[cap] {len(PROMPTS)} prompts in {dt:.1f}s; {calls['n']} router calls over "
          f"{len(order)} router instances", flush=True)
    try:
        print("[cap] first completion:", repr(str(outs[0])[:110]), flush=True)
    except Exception:
        pass
    if not cap:
        result_q.put({"rank": 0, "error": "router hook never fired"})
        return

    import torch
    recs = []
    for lid in sorted(cap):
        h = torch.cat(cap[lid], dim=0)[: args.max_rows]
        recs.append({"layer": lid, "W": wref[lid], "h": h})
        print(f"[cap] layer {lid}: h {tuple(h.shape)}  W {tuple(wref[lid].shape)}", flush=True)
    payload = {
        "layers": recs,
        "top_k": int(next(iter(tk.values()))),
        "provenance": (f"model={args.model} tp={args.tp} prompts={len(PROMPTS)} "
                       f"max_new_tokens={args.max_tokens} rows<={args.max_rows}; h captured at "
                       f"Gemma4Router.forward AFTER RMSNormNoScale*scale*hidden**-0.5"),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(payload, args.out)
    print(f"[cap] wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB), "
          f"top_k={payload['top_k']}", flush=True)
    result_q.put({"rank": 0, "ok": True, "out": args.out})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--layers", default="0,7,15,22,29")
    ap.add_argument("--max-rows", type=int, default=4096)
    ap.add_argument("--memory-ratio", type=float, default=0.78)
    ap.add_argument("--max-running-req", type=int, default=8)
    ap.add_argument("--attn-backend", default="hip")
    ap.add_argument("--raw-prompts", action="store_true", help="skip the chat template")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    assert not args.out.startswith("/tmp"), "fixture must be durable, not tmpfs"
    args.want = {int(x) for x in args.layers.split(",")}

    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    result_q: mp.Queue = mp.Queue()
    procs = []
    for rank in range(args.tp):
        p = mp.Process(target=_rank_main, args=(rank, args, result_q), name=f"routercap-TP{rank}")
        p.start()
        procs.append(p)
    results = []
    alive = list(procs)
    while alive:
        while not result_q.empty():
            results.append(result_q.get())
        alive = [p for p in alive if p.is_alive()]
        if alive:
            time.sleep(0.5)
    for p in procs:
        p.join()
    while not result_q.empty():
        results.append(result_q.get())
    codes = [p.exitcode for p in procs]
    err = [r for r in results if r.get("error")]
    if err:
        print(f"FAIL: {err}", file=sys.stderr)
        return 1
    if any(c != 0 for c in codes):
        print(f"FAIL: rank exit codes {codes}", file=sys.stderr)
        return 1
    print(f"OK: ranks {codes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
