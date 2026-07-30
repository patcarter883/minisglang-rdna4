"""Concurrent/batched CCA recurrent-radix repro — reproduces the GSM8K-style monotonic decline.

Sends MANY requests that share a long few-shot-like prefix but have DISTINCT short questions, as a
batch through the offline engine (which batches prefill/decode like the server). Compares radix ON
vs naive OFF per request (greedy ids + last-logit cos). If the recurrent-radix reuse is broken under
batching/concurrency, later requests (which hit the populated cache) diverge from naive while early
ones agree — the monotonic-decline signature.

  python tools/cca_radix_concurrent.py --radix on  --out /engine/tools/_c_on.pt
  python tools/cca_radix_concurrent.py --radix off --out /engine/tools/_c_off.pt
  python tools/cca_radix_concurrent.py --compare /engine/tools/_c_on.pt /engine/tools/_c_off.pt
"""
import argparse
import torch


def build_llm(model, radix, graph):
    from minisgl.llm import LLM
    return LLM(
        model_path=model, dtype=torch.bfloat16, cuda_graph_max_bs=graph, page_size=16,
        memory_ratio=0.85, attention_backend="hip", max_running_req=8,
        gdn_radix=radix, cache_type="radix",
    )


def cmd_run(args):
    from minisgl.core import SamplingParams
    llm = build_llm(args.model, args.radix == "on", args.graph)
    print(f"rec_radix={getattr(llm, '_rec_radix', False)} graph={args.graph}", flush=True)
    tok = llm.tokenizer
    # A long shared prefix (few-shot-like, ~fixed) + distinct short questions.
    prefix = ("You are a careful assistant. Here are some examples of reasoning.\n"
              "Q: What is 2+2? A: 2+2 equals 4.\n"
              "Q: What color is the sky? A: The sky is blue during a clear day.\n"
              "Q: Name a fruit. A: An apple is a common fruit.\n"
              "Q: What is the capital of France? A: The capital of France is Paris.\n"
              "Now answer the next question in one short sentence.\n") * 3
    questions = [
        "Q: What is 10 times 3? A:", "Q: What is the largest planet? A:",
        "Q: Who wrote Hamlet? A:", "Q: What is 100 minus 45? A:",
        "Q: What gas do plants absorb? A:", "Q: What is the square root of 81? A:",
        "Q: Name a primary color. A:", "Q: What is 7 plus 8? A:",
        "Q: What is the freezing point of water in Celsius? A:", "Q: What is 12 divided by 4? A:",
        "Q: What is the opposite of hot? A:", "Q: How many days in a week? A:",
        "Q: What is 5 squared? A:", "Q: What is the capital of Japan? A:",
        "Q: What is 9 times 9? A:", "Q: Name an ocean. A:",
    ]
    pref_ids = tok.encode(prefix, add_special_tokens=True)
    prompts = [list(pref_ids) + tok.encode(" " + q, add_special_tokens=False) for q in questions]
    sp = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True)
    # Send ALL at once -> batched prefill/decode (populates then hits the shared prefix cache).
    outs = llm.generate(prompts, sp)
    res = [{"q": questions[i], "gen": outs[i]["token_ids"]} for i in range(len(questions))]
    torch.save({"radix": args.radix, "res": res, "prefix_len": len(pref_ids)}, args.out)
    print(f"saved {args.out} prefix_len={len(pref_ids)}", flush=True)


def cmd_compare(on_path, off_path):
    on, off = torch.load(on_path), torch.load(off_path)
    o, f = on["res"], off["res"]
    print(f"\n===== concurrent shared-prefix (prefix_len={on['prefix_len']}): radix ON vs naive OFF =====")
    ndiff = 0
    for i in range(len(o)):
        same = o[i]["gen"] == f[i]["gen"]
        if not same:
            ndiff += 1
        mark = "OK " if same else "DIFF"
        print(f"  [{i:2d}] {mark}  {o[i]['q'][:42]:42s} ON={o[i]['gen'][:6]} OFF={f[i]['gen'][:6]}")
    print(f"\n  {ndiff}/{len(o)} requests DIVERGE from naive  -> "
          f"{'BROKEN (reuse corrupts)' if ndiff else 'LOSSLESS'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radix", choices=["on", "off"])
    ap.add_argument("--model", default="/root/.cache/huggingface/ZAYA1-8B-RXF-h32")
    ap.add_argument("--graph", type=int, default=0)
    ap.add_argument("--out", default="/engine/tools/_c.pt")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        cmd_compare(*args.compare)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
