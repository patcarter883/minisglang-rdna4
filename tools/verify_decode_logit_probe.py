"""Direct verify(prefill/extend-kernel) vs decode(decode-kernel) logit divergence probe on ZAYA.

Isolates the CROSS-KERNEL question — with NO drafter, NO spec machinery, NO reasoning parser — that the
DFlash losslessness MISMATCH raised: does the token computed by the decode kernel (attn_decode, M=1)
differ from the same token computed by the prefill/extend kernel (attn_prefill_paged, the verify path)?
And is the difference near-tie-concentrated (which, on ZAYA's flat RSA distribution, flips greedy argmax
and — per the observed pattern — systematically sends the verify path into reasoning mode)?

Method (same tokens, two kernels), all in one process on a leased card:
  1. DECODE path: one greedy generate(max_tokens=N). The sampler hook captures each step's last-position
     logits. Step 0 is the PREFILL (predicts tok[0]); steps >=1 use the DECODE kernel (predict tok[i]).
  2. VERIFY/PREFILL path: for each i, a 1-token generate([prompt + tok[:i]]) — last-position logits via
     the PREFILL/extend kernel for the SAME prefix.
  3. Compare position-matched: argmax-flip?, top1-top2 gap, max|Δlogit|. Position 0 is prefill-vs-prefill
     (built-in control — MUST match). i>=1 is decode-kernel vs prefill-kernel — the real test. The FIRST
     flip at small i is the reason-vs-answer branch that cascades.

Toggle MINISGL_MINV_GEMM=0/1 across runs to attribute divergence to the dense_gemm fix vs the residual
(attention / fp8-KV / MoE routing). Run under a 1-card lease via tools/run_verify_decode_probe.sh.
"""
import argparse, os, torch


def build(model):
    from minisgl.llm import LLM
    # radix OFF + graph OFF: pure per-forward numerics, no prefix reuse, minv active (eager).
    return LLM(model_path=model, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16,
               memory_ratio=0.82, attention_backend="hip", max_running_req=2,
               gdn_radix=False, cache_type="naive")


def run(a):
    from minisgl.core import SamplingParams
    llm = build(a.model)
    tok = llm.tokenizer
    # Chat-template the prompt so the reasoning-vs-answer behavior matches the served path. Template to
    # a STRING then encode (the wrapper's tokenize=True path is unreliable); add_special_tokens=False
    # since the jinja template already emits the special tokens.
    try:
        text = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < 4:  # template returned nothing useful -> fall back
            raise ValueError(f"templated ids too short ({len(ids)})")
    except Exception as e:  # noqa: BLE001
        print(f"[probe] chat template unusable ({e}); raw encode", flush=True)
        ids = tok.encode(a.prompt, add_special_tokens=True)
    ids = [int(x) for x in ids]
    print(f"[probe] prompt head: {tok.decode(ids[:24])!r}", flush=True)
    N = a.n
    print(f"[probe] MINV={os.environ.get('MINISGL_MINV_GEMM','1')} prompt_tokens={len(ids)} N={N}", flush=True)

    cap = []
    orig = llm.engine.sampler.sample

    def hook(logits, args):
        cap.append(logits.detach().float().cpu().clone())  # [num_reqs, vocab] last position/req
        return orig(logits, args)

    llm.engine.sampler.sample = hook  # type: ignore[method-assign]
    try:
        # 1) DECODE path — one greedy rollout; cap[i] predicts dec_tokens[i] (i=0 prefill, i>=1 decode).
        cap.clear()
        out = llm.generate([ids], SamplingParams(temperature=0.0, max_tokens=N, ignore_eos=True))
        dec_tokens = list(out[0]["token_ids"])[:N]
        dec_logits = [c[0] for c in cap][:len(dec_tokens)]
        # 2) VERIFY/PREFILL path — per-position prefill-kernel logits for the SAME prefixes.
        pre_logits = []
        for i in range(len(dec_tokens)):
            cap.clear()
            llm.generate([ids + dec_tokens[:i]], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
            pre_logits.append(cap[0][0])
    finally:
        llm.engine.sampler.sample = orig  # type: ignore[method-assign]

    # 3) Compare
    print(f"\n{'i':>3} {'kern':>7} {'dec_argmax':>28} {'pre_argmax':>28} {'flip':>4} {'gap':>8} {'max|Δ|':>9}")
    n_flip = 0; first_flip = None
    for i in range(len(dec_tokens)):
        d, p = dec_logits[i], pre_logits[i]
        da, pa = int(d.argmax()), int(p.argmax())
        gap = float(d.topk(2).values[0] - d.topk(2).values[1])
        maxd = float((d - p).abs().max())
        flip = da != pa and i >= 1  # i==0 is prefill-vs-prefill control
        kern = "prefill" if i == 0 else "decode"
        if flip:
            n_flip += 1
            if first_flip is None:
                first_flip = i
        ds = repr(tok.decode([da]))[:26]; ps = repr(tok.decode([pa]))[:26]
        mark = "FLIP" if flip else ("ctl" if i == 0 else "")
        print(f"{i:>3} {kern:>7} {ds:>28} {ps:>28} {mark:>4} {gap:>8.4f} {maxd:>9.4f}")

    ctl_ok = int(dec_logits[0].argmax()) == int(pre_logits[0].argmax())
    print(f"\n[probe] position-0 control (prefill vs prefill) argmax match: {ctl_ok}  (must be True)")
    print(f"[probe] decode-vs-prefill argmax FLIPS (i>=1): {n_flip}/{len(dec_tokens)-1}"
          f"  first flip @ i={first_flip}")
    if first_flip is not None:
        d = dec_logits[first_flip]
        gap = float(d.topk(2).values[0] - d.topk(2).values[1])
        print(f"[probe] first-flip near-tie gap (decode top1-top2)={gap:.4f}  "
              f"decode->{tok.decode([int(d.argmax())])!r}  prefill->{tok.decode([int(pre_logits[first_flip].argmax())])!r}")
    # near-tie correlation: flip rate among small-gap vs large-gap positions
    gaps = torch.tensor([float(dec_logits[i].topk(2).values[0] - dec_logits[i].topk(2).values[1])
                         for i in range(1, len(dec_tokens))])
    flips = torch.tensor([int(dec_logits[i].argmax() != pre_logits[i].argmax())
                          for i in range(1, len(dec_tokens))], dtype=torch.float)
    if gaps.numel():
        med = float(gaps.median())
        lo = flips[gaps <= med].mean() if (gaps <= med).any() else torch.tensor(0.0)
        hi = flips[gaps > med].mean() if (gaps > med).any() else torch.tensor(0.0)
        print(f"[probe] flip-rate by near-tie: gap<=median {float(lo):.2f}  vs  gap>median {float(hi):.2f}"
              f"  (median gap={med:.4f}) -> flips concentrate at near-ties if left>>right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/.cache/huggingface/ZAYA1-8B-RXF-h32")
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one word.")
    ap.add_argument("--n", type=int, default=24)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
