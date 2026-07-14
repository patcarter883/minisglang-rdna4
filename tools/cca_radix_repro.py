"""CCA recurrent-radix fidelity repro (isolates the snapshot/restore gap).

Runs ONE config per process (recurrent radix ON or naive OFF) — two engines can't share a process
(CUDA re-init). For each config, and for two prompt shapes:

  UNALIGNED (len_A % page != 0, prefilled under a SMALL max_extend_tokens so the recurrent-radix
    chunker forces >=2 page-aligned segments + a sub-page tail -> the multi-chunk STASH path that the
    single-pass validation tool never covered), and
  ALIGNED   (len_A % page == 0, single pass -> the FALLBACK capture path the validation tool DID cover),

it does Turn A (populate radix + snapshot) then Turn B = A's whole prompt + a different continuation
(so B's prefix match RESTORES A's snapshot), and saves B's last-position logits + a short greedy
continuation to a .pt file. Compare the ON vs OFF files: if snapshot/restore is lossless they match.

  python tools/cca_radix_repro.py --radix on  --model <ckpt> --out /engine/tools/_repro_on.pt
  python tools/cca_radix_repro.py --radix off --model <ckpt> --out /engine/tools/_repro_off.pt
  python tools/cca_radix_repro.py --compare /engine/tools/_repro_on.pt /engine/tools/_repro_off.pt
"""
import argparse
import torch


def build_llm(model, gdn_radix, extra_kwargs, max_extend):
    from minisgl.llm import LLM
    return LLM(
        model_path=model, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16,
        memory_ratio=0.85, attention_backend="hip", max_running_req=4,
        gdn_radix=gdn_radix, cache_type="radix", max_extend_tokens=max_extend, **extra_kwargs,
    )


def greedy(llm, ids, n):
    from minisgl.core import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)
    return llm.generate([list(ids)], sp)[0]["token_ids"]


def run_shape(llm, tag, a_ids, b_ids, page):
    print(f"[{tag}] len_A={len(a_ids)} align_down={len(a_ids)//page*page} len_B={len(b_ids)}", flush=True)
    oa = greedy(llm, a_ids, 4)                       # Turn A: populate radix + snapshot
    lb = llm.base_logits(list(b_ids)).float().cpu()  # Turn B last-pos logits (restores on hit)
    ob = greedy(llm, b_ids, 8)                        # Turn B greedy continuation
    print(f"[{tag}] A_gen={oa} B_gen={ob}", flush=True)
    return {"a_gen": oa, "b_gen": ob, "b_logits": lb}


def cmd_run(args):
    extra = {}
    for tok in args.extra:
        if tok.startswith("--"):
            extra[tok[2:].replace("-", "_")] = True
    page = 16
    llm = build_llm(args.model, args.radix == "on", extra, args.max_extend)
    print(f"rec_radix={getattr(llm, '_rec_radix', False)} shape={args.shape}", flush=True)
    tok = llm.tokenizer
    base = ("The quick brown fox jumps over the lazy dog near the riverbank while the sun sets. "
            "In distant mountains, snow falls softly over ancient pines and quiet valleys. ") * 6
    ids = tok.encode(base, add_special_tokens=True)
    cont = tok.encode(" However, the story took an unexpected turn when", add_special_tokens=False)

    if args.shape == "unaligned":
        # STASH path; B shares ALL of A then diverges -> B re-treads A's tokens [align_down,lenA).
        len_a = (args.max_extend * 2) + 7
        assert len_a % page != 0 and len(ids) >= len_a + 20
        a_ids = ids[:len_a]
        b_ids = list(a_ids) + list(cont)
    elif args.shape == "stash_imm":
        # STASH path but B diverges IMMEDIATELY at the snapshot boundary (align_down(lenA)): B shares
        # only A's first `boundary` tokens, then the continuation. Discriminates capture-path vs
        # immediate-divergence-at-boundary.
        len_a = (args.max_extend * 2) + 7
        assert len_a % page != 0 and len(ids) >= len_a + 20
        boundary = len_a // page * page
        a_ids = ids[:len_a]
        b_ids = list(ids[:boundary]) + list(cont)
    elif args.shape == "aligned_long":
        # FALLBACK path at a LARGER boundary: A length == 2*max_extend (page-aligned) chunks into
        # [0,ext)[ext,2ext) where the LAST segment is a normal Req ending exactly at the full prompt
        # -> fallback capture @2ext. B diverges immediately. Tests fallback independent of boundary.
        len_a = args.max_extend * 2
        assert len_a % page == 0 and len(ids) >= len_a + 20
        a_ids = ids[:len_a]
        b_ids = list(a_ids) + list(cont)
    else:  # aligned: FALLBACK path, single-pass, B diverges immediately at boundary==lenA
        len_a = page * 2
        assert len_a % page == 0 and len_a <= args.max_extend and len(ids) >= len_a + 20
        a_ids = ids[:len_a]
        b_ids = list(a_ids) + list(cont)

    out = {"radix": args.radix, "shape": args.shape,
           "res": run_shape(llm, f"{args.shape}/{args.radix}", a_ids, b_ids, page)}
    torch.save(out, args.out)
    print(f"saved {args.out}", flush=True)


def cmd_compare(on_path, off_path):
    on, off = torch.load(on_path), torch.load(off_path)
    o, f = on["res"], off["res"]
    dl = (o["b_logits"] - f["b_logits"]).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(o["b_logits"], f["b_logits"], dim=0).item()
    match = o["b_gen"] == f["b_gen"]
    print(f"\n===== shape={on['shape']} (ON vs OFF) =====")
    print(f"  greedy_ids_match={match}")
    print(f"  ON ={o['b_gen']}")
    print(f"  OFF={f['b_gen']}")
    print(f"  max|logit_on-logit_off|={dl:.4f}  cos={cos:.6f}")
    print(f"  VERDICT: {'LOSSLESS' if (match and dl < 0.5) else 'DIVERGENT (fidelity loss)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radix", choices=["on", "off"])
    ap.add_argument("--shape", choices=["unaligned", "aligned", "stash_imm", "aligned_long"], default="aligned")
    ap.add_argument("--model", default="/root/.cache/huggingface/ZAYA1-8B-RXF-h32")
    ap.add_argument("--out", default="/engine/tools/_repro.pt")
    ap.add_argument("--max-extend", type=int, default=48)
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args()
    if args.compare:
        cmd_compare(*args.compare)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
