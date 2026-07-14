"""Does dense_gemm (MINV) shift the DFlash AUX the drafter trains on? i.e. is the OPD corpus (captured
pre-dense_gemm = the MINV=0/rocBLAS path) the same as what the CURRENT serve (MINV=1/dense_gemm) would
produce? The drafter's input is the target residual hidden at the capture layers [1,39,76]; if that is
ULP-stable across MINV, the stored corpus is still on-policy. If it shifts materially, re-capture.

Runs the SAME fixed prompt through a greedy rollout and captures the residual-stream hidden at the
capture layers for the generated positions, once per process. Run twice (MINV=1, MINV=0) and diff the
saved tensors: cosine + relative-L2 per layer. MINV=0 == the pre-dense_gemm capture-time numerics.
"""
import argparse, os, torch


def build(model):
    from minisgl.llm import LLM
    return LLM(model_path=model, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16,
               memory_ratio=0.82, attention_backend="hip", max_running_req=2,
               gdn_radix=False, cache_type="naive")


def run(a):
    from minisgl.core import SamplingParams
    llm = build(a.model)
    tok = llm.tokenizer
    inner = llm.engine.model.model  # ZayaModel
    layers = inner.layers.op_list
    lids = [int(x) for x in a.layers.split(",")]
    print(f"[aux] MINV={os.environ.get('MINISGL_MINV_GEMM','1')} capture layers={lids} of {len(layers)}", flush=True)

    cap = {lid: [] for lid in lids}

    def wrap(lid, orig):
        def f(*args, **kw):
            o = orig(*args, **kw)
            h = o[0] if isinstance(o, tuple) else o
            cap[lid].append(h.detach().float().cpu().clone())  # [rows, hidden] this forward
            return o
        return f

    for lid in lids:
        layers[lid].forward = wrap(lid, layers[lid].forward)

    try:
        text = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                       add_generation_prompt=True, tokenize=False)
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) < 4:
            raise ValueError("short")
    except Exception:
        ids = tok.encode(a.prompt, add_special_tokens=True)
    ids = [int(x) for x in ids]
    llm.generate([ids], SamplingParams(temperature=0.0, max_tokens=a.n, ignore_eos=True))

    # cap[lid] = [prefill hidden [P,H], then N decode rows [1,H] each]. Keep the N GENERATED positions'
    # hidden (the last prompt row + each decode row) — exactly the residual the drafter's aux fuses.
    out = {}
    for lid in lids:
        chunks = cap[lid]
        gen_rows = [chunks[0][-1:]] + [c[-1:] for c in chunks[1:]]  # [N, H]
        out[lid] = torch.cat(gen_rows, 0)[: a.n].clone()
    torch.save({"layers": lids, "n": a.n, "aux": out, "minv": os.environ.get("MINISGL_MINV_GEMM", "1")}, a.out)
    print(f"[aux] saved {a.out}  rows={out[lids[0]].shape[0]} hidden={out[lids[0]].shape[1]}", flush=True)


def cmp(p1, p2):
    A, B = torch.load(p1), torch.load(p2)
    print(f"\n===== AUX delta: MINV={A['minv']} ({p1})  vs  MINV={B['minv']} ({p2}) =====")
    print("  (MINV=0 == pre-dense_gemm capture-time path; MINV=1 == current serve)")
    for lid in A["layers"]:
        a_, b_ = A["aux"][lid], B["aux"][lid]
        n = min(a_.shape[0], b_.shape[0]); a_, b_ = a_[:n], b_[:n]
        cos = torch.nn.functional.cosine_similarity(a_, b_, dim=-1)  # per row
        rel = (a_ - b_).norm(dim=-1) / a_.norm(dim=-1).clamp_min(1e-9)
        maxabs = (a_ - b_).abs().max().item()
        print(f"  layer {lid:>2}: cos min={cos.min():.6f} mean={cos.mean():.6f}  "
              f"relL2 mean={rel.mean():.4f} max={rel.max():.4f}  max|Δ|={maxabs:.4f}")
    print("\n  verdict: cos~1.0000 & relL2<~0.01 -> corpus still on-policy (dense_gemm aux shift negligible);"
          "\n           relL2 >~0.05 -> the stored aux is OFF the current serve -> re-capture before re-distill.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/.cache/huggingface/ZAYA1-8B-RXF-h32")
    ap.add_argument("--prompt", default="Explain in one sentence why the sky is blue.")
    ap.add_argument("--layers", default="1,39,76")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="/engine/tools/_aux.pt")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        cmp(*a.compare)
    else:
        run(a)


if __name__ == "__main__":
    main()
