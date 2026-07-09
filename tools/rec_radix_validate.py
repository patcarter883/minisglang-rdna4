"""Recurrent-state radix (GDN) prefix-cache validation.

Two sequential requests on ONE offline LLM instance (the radix tree persists across generate calls):

  * Turn A: a prompt whose token length is a MULTIPLE OF 16 (page_size), so the prefill-commit
    snapshot lands exactly on the inserted radix node boundary. Greedy generate.
  * Turn B: prompt = A's full prompt + a DIFFERENT continuation. B shares A's whole prompt, so its
    prefix match lands on A's node@prompt_len, restores the recurrent-state snapshot, and skips
    re-prefilling the shared prefix (a "recurrent-radix HIT" is logged).

Run this twice — once with MINISGL_GDN_RADIX=1 (recurrent radix) and once WITHOUT (naive) — and diff
the printed token ids. Byte-identical output => lossless. The second run's Turn-B prefill wall time
shows the shared-prefix speedup. Use GDN_HIP_WMMA_PREFILL=0 for the losslessness diff (the recurrent
kernel is decomposition-invariant; the default WMMA prefill is ~1e-3 chunk-nondeterministic, exactly
as for existing chunked prefill).
"""
import argparse
import os
import time

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prompt-pages", type=int, default=16, help="A-prompt length in 16-token pages")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--memory-ratio", type=float, default=0.8)
    args = ap.parse_args()

    llm = LLM(
        model_path=args.model, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16,
        memory_ratio=args.memory_ratio, attention_backend="hip", max_running_req=4,
    )
    tok = llm.tokenizer
    rec = getattr(llm, "_rec_radix", False)
    wmma = os.environ.get("GDN_HIP_WMMA_PREFILL", "1") != "0"
    print(f"[cfg] recurrent_radix={rec}  WMMA_prefill={wmma}  model={args.model}", flush=True)

    # A base text, tokenized and trimmed to an exact multiple of 16 tokens.
    base = ("The history of scientific discovery is a long chain of careful observation, bold "
            "hypothesis, and patient experiment. Consider the following detailed account. ") * 60
    a_ids = tok.encode(base, add_special_tokens=True)
    n = (args.prompt_pages * 16)
    assert len(a_ids) >= n, f"base too short: {len(a_ids)} < {n}"
    a_ids = a_ids[:n]
    assert len(a_ids) == n and n % 16 == 0, (len(a_ids), n)

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)

    # Turn A
    t0 = time.perf_counter()
    outA = llm.generate([a_ids], sp)[0]
    tA = time.perf_counter() - t0
    oA = outA["token_ids"]
    print(f"[A] prompt_len={len(a_ids)} gen={len(oA)} time={tA*1000:.1f}ms", flush=True)

    # Turn B: same prompt + a DIFFERENT continuation (so B diverges from A right after the prompt and
    # matches exactly A's node@prompt_len). Use tokens that differ from A's generated continuation.
    cont = tok.encode(" Meanwhile, in a completely different field of study, researchers found",
                      add_special_tokens=False)
    assert cont[: len(oA)] != oA[: len(cont)], "continuation accidentally equals A's output; pick another"
    b_ids = list(a_ids) + list(cont)

    t0 = time.perf_counter()
    outB = llm.generate([b_ids], sp)[0]
    tB = time.perf_counter() - t0
    oB = outB["token_ids"]
    print(f"[B] prompt_len={len(b_ids)} gen={len(oB)} time={tB*1000:.1f}ms  (shared_prefix={len(a_ids)})",
          flush=True)

    print("RESULT_A_IDS " + ",".join(map(str, oA)), flush=True)
    print("RESULT_B_IDS " + ",".join(map(str, oB)), flush=True)
    print(f"RESULT_TIMING A={tA*1000:.1f} B={tB*1000:.1f}", flush=True)


if __name__ == "__main__":
    main()
