"""Pin down the reuse-vs-fresh CCA prefix-KV seam.

Captures the pre-quantization K/V the model feeds to pool.store_kv (bf16, post-RoPE) for every CCA
layer, per token position, during the FIRST prefill forward of a request. Two runs (separate
processes to avoid radix cross-contamination):

  --mode populate : request = P (the shared prefix, len==boundary). radix ON. Captures the keys that
                    get STORED and later REUSED on a hit.
  --mode fresh    : request = P + cont (naive cache). Captures the keys a FRESH full-prefix forward
                    computes for the SAME prefix tokens [0:boundary).

Compare the two .pt files: if the prefix keys are bit-identical, the model's key COMPUTATION is
deterministic and the reuse divergence is purely in the ATTENTION KERNEL (paged-fp8 prefill vs full
prefill) — a kernel-reassociation issue fp8 amplifies. If they DIFFER, there is a structural
key-computation seam (recurrent-state/conv-window/alignment) to fix.
"""
import argparse
import torch


def build_llm(model, radix, max_extend):
    from minisgl.llm import LLM
    return LLM(
        model_path=model, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16,
        memory_ratio=0.85, attention_backend="hip", max_running_req=4,
        gdn_radix=radix, cache_type="radix", max_extend_tokens=max_extend,
    )


def cmd_run(args):
    from minisgl.core import SamplingParams
    llm = build_llm(args.model, args.mode == "populate", args.max_extend)
    print(f"rec_radix={getattr(llm, '_rec_radix', False)} mode={args.mode}", flush=True)
    pool = llm.engine.ctx.kv_cache
    cap = {"layers": {}, "off": {}}   # layers[lid] = list of per-forward k rows; off tracks token offset
    orig_store = pool.store_kv

    def hook(k, v, out_loc, layer_id):
        # Accumulate pre-quant bf16 K per layer across ALL prefill forwards (chunked -> several), in
        # token order. Stop once we have >= boundary rows for layer 0.
        cap["layers"].setdefault(layer_id, [])
        if sum(t.shape[0] for t in cap["layers"][layer_id]) < args.boundary + 8:
            cap["layers"][layer_id].append(k.detach().float().cpu().clone())
        return orig_store(k, v, out_loc, layer_id)

    pool.store_kv = hook
    tok = llm.tokenizer
    base = ("The quick brown fox jumps over the lazy dog near the riverbank while the sun sets. "
            "In distant mountains, snow falls softly over ancient pines and quiet valleys. ") * 6
    ids = tok.encode(base, add_special_tokens=True)
    b = args.boundary
    if args.mode == "populate":
        prompt = ids[:b]
    else:
        cont = tok.encode(" However, the story took an unexpected turn when", add_special_tokens=False)
        prompt = list(ids[:b]) + list(cont)
    # After the first prefill forward stores all layers, freeze the capture (later decode forwards
    # would append more layer-0 entries).
    n_layers = pool.num_layers

    def maybe_freeze(*a, **k):
        pass
    llm.generate([list(prompt)], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
    pool.store_kv = orig_store
    out = {lid: (torch.cat(chunks, 0)[:b].clone(),) for lid, chunks in cap["layers"].items()}
    torch.save({"mode": args.mode, "boundary": b, "layers": out, "n_kv_layers": n_layers}, args.out)
    print(f"saved {args.out}  captured_layers={sorted(out)} rows_per_layer={ {lid: tuple(v[0].shape) for lid,v in list(out.items())[:1]} }", flush=True)


def cmd_compare(pop_path, fresh_path):
    pop, fresh = torch.load(pop_path), torch.load(fresh_path)
    b = pop["boundary"]
    print(f"\n===== reuse(populate) vs fresh KV, prefix [0:{b}) =====")
    worst_k = worst_v = 0.0
    first_diff = None
    for lid in sorted(pop["layers"]):
        kp = pop["layers"][lid][0]; kf = fresh["layers"][lid][0]
        n = min(kp.shape[0], kf.shape[0]); kp, kf = kp[:n], kf[:n]
        dk = (kp - kf).abs(); mk = dk.max().item()
        worst_k = max(worst_k, mk)
        if mk > 0 and first_diff is None:
            pos = (dk.reshape(n, -1).max(dim=1).values > 0).nonzero()[0].item()
            first_diff = (lid, pos, mk)
    print(f"  max|K_reuse - K_fresh| over all CCA layers = {worst_k:.6e}")
    if first_diff:
        print(f"  FIRST divergence: layer={first_diff[0]} token_pos={first_diff[1]} |dK|={first_diff[2]:.6e}")
    verdict = ("KEYS BIT-IDENTICAL -> divergence is ATTENTION-KERNEL (paged fp8 prefill), not keys"
               if worst_k == 0 else
               "KEYS DIFFER -> structural key-computation SEAM (recurrent/conv/alignment)")
    print(f"  VERDICT: {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["populate", "fresh"])
    ap.add_argument("--model", default="/root/.cache/huggingface/ZAYA1-8B-RXF-h32")
    ap.add_argument("--boundary", type=int, default=32)
    ap.add_argument("--max-extend", type=int, default=256)
    ap.add_argument("--out", default="/engine/tools/_kv.pt")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        cmd_compare(*args.compare)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
