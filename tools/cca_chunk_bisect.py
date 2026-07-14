"""Decisive chunk-boundary bisect for the CCA prefix-reuse divergence.

Runs ONE fixed prompt through prefill TWICE in the same process:
  * single : max_extend huge  -> one contiguous prefill forward.
  * chunked: max_extend small -> the SAME prompt split into several extend forwards, each seeded
             from the prior chunk's recurrent state (has_initial_state path).

For every CCA layer it captures, per token:
  (A) qk_out  -- the conv front-end output BEFORE RoPE (hook ZayaCCAAttn.forward via the qk_out
                 tensor handed to attn.forward)  -> isolates the CONV / recurrent handoff.
  (B) stored K -- post-RoPE key at pool.store_kv                                  -> conv + RoPE.

If (A) diverges, the two-stage conv reconstruction across the chunk boundary is not bit-exact
(the real seam). If (A) matches but (B) diverges, it is RoPE/positions. The FIRST divergent token
tells us whether it starts exactly at the chunk boundary (conv receptive-field spillover) or earlier.

Two separate processes (--mode single|chunked) write .pt captures; --compare diffs them.
"""
import argparse, torch


def build(model, max_extend):
    from minisgl.llm import LLM
    # radix OFF: we want the pure chunked-vs-single prefill numerics, not the radix layer.
    return LLM(model_path=model, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16,
               memory_ratio=0.85, attention_backend="hip", max_running_req=2,
               gdn_radix=False, cache_type="radix", max_extend_tokens=max_extend)


def run(a):
    from minisgl.core import SamplingParams
    llm = build(a.model, a.max_extend)
    pool = llm.engine.ctx.kv_cache
    tok = llm.tokenizer
    base = ("The quick brown fox jumps over the lazy dog near the riverbank while the sun sets. "
            "In distant mountains, snow falls softly over ancient pines and quiet valleys. ") * 4
    ids = tok.encode(base, add_special_tokens=True)[:a.ntok]
    print(f"mode={a.mode} max_extend={a.max_extend} ntok={len(ids)}", flush=True)

    # Cap EXACTLY at len(ids): prefill fills exactly that many tokens before any decode. Capping
    # per-append + slicing to N keeps decode rows out and guarantees single/chunked token alignment.
    N = len(ids)
    from minisgl.models.zaya import ZayaCCAAttn
    cca_layers = []
    inner = llm.engine.model.model  # ZayaForCausalLM -> ZayaModel
    for layer in inner.layers.op_list:
        sa = getattr(layer, "self_attn", None)
        if isinstance(sa, ZayaCCAAttn):
            cca_layers.append(sa)
    print(f"found {len(cca_layers)} CCA layers", flush=True)

    capQK = {}   # conv output (pre-attention), k-cols:  cca_layer_id -> list[tensor]
    capO = {}    # attention output (post-attention):    cca_layer_id -> list[tensor]

    def rows(d, lid):
        return sum(t.shape[0] for t in d.get(lid, []))

    for m in cca_layers:
        lid = m._cca_layer_id
        attn = m.attn
        latent_q = m._cca._latent_q
        latent_k = m._cca._latent_k
        orig_fwd = attn.forward

        def make(lid, latent_q, latent_k, orig_fwd):
            def wrapped(qkv):
                if rows(capQK, lid) < N:
                    capQK.setdefault(lid, []).append(
                        qkv[:, latent_q:latent_q + latent_k].detach().float().cpu().clone())
                o = orig_fwd(qkv)
                if rows(capO, lid) < N:
                    capO.setdefault(lid, []).append(o.detach().float().cpu().clone())
                return o
            return wrapped
        attn.forward = make(lid, latent_q, latent_k, orig_fwd)

    llm.generate([list(ids)], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    outQK = {lid: torch.cat(ch, 0)[:N].clone() for lid, ch in capQK.items()}
    outO = {lid: torch.cat(ch, 0)[:N].clone() for lid, ch in capO.items()}
    torch.save({"mode": a.mode, "ntok": N, "QK": outQK, "O": outO}, a.out)
    print(f"saved {a.out} layers_QK={len(outQK)} layers_O={len(outO)} rows={next(iter(outQK.values())).shape[0]}", flush=True)


def cmp(p1, p2):
    s, c = torch.load(p1), torch.load(p2)
    print(f"\n===== single vs chunked =====")
    for name in ("QK", "O"):
        label = "conv-output PRE-attention (A)" if name == "QK" else "attention-output POST-attention (B)"
        worst = 0.0; worst_lid = -1; first = None
        per_layer_max = []
        for lid in sorted(s[name]):
            a_ = s[name][lid]; b_ = c[name][lid]
            n = min(a_.shape[0], b_.shape[0])
            a_ = a_[:n].reshape(n, -1); b_ = b_[:n].reshape(n, -1)
            w = min(a_.shape[1], b_.shape[1])
            d = (a_[:, :w] - b_[:, :w]).abs()
            m = d.max().item()
            per_layer_max.append((lid, m))
            if m > worst:
                worst = m; worst_lid = lid
            per_tok = d.max(dim=1).values
            posnz = (per_tok > 1e-6).nonzero()
            if posnz.numel():
                p0 = posnz[0].item()
                if first is None or (lid, p0) < (first[0], first[1]):
                    first = (lid, p0, per_tok[p0].item())
        print(f"\n  [{name}] {label}")
        print(f"    max|Δ| = {worst:.6e}  (layer {worst_lid})")
        if first:
            print(f"    FIRST divergent token: layer={first[0]} token={first[1]} |Δ|={first[2]:.3e}")
        else:
            print(f"    BIT-IDENTICAL across all layers/tokens")
        # per-layer max trend (does |Δ| grow with depth? -> compounding through residual)
        trend = " ".join(f"L{lid}:{m:.1e}" for lid, m in per_layer_max[:8])
        trend2 = " ".join(f"L{lid}:{m:.1e}" for lid, m in per_layer_max[-4:])
        print(f"    per-layer max|Δ| (first 8): {trend}")
        print(f"    per-layer max|Δ| (last 4):  {trend2}")
        # layer-0 per-token divergence set (isolates WHERE it first enters, vs true boundary)
        l0 = min(s[name])
        a0 = s[name][l0]; b0 = c[name][l0]; n0 = min(a0.shape[0], b0.shape[0])
        a0 = a0[:n0].reshape(n0, -1); b0 = b0[:n0].reshape(n0, -1); w0 = min(a0.shape[1], b0.shape[1])
        pt = (a0[:, :w0] - b0[:, :w0]).abs().max(dim=1).values
        nz = (pt > 1e-6).nonzero().flatten().tolist()
        print(f"    layer{l0} divergent tokens ({len(nz)}): {nz[:24]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "chunked"])
    ap.add_argument("--model", default="/root/.cache/huggingface/ZAYA1-8B-RXF-h32")
    ap.add_argument("--ntok", type=int, default=128)
    ap.add_argument("--max-extend", type=int, default=8192)
    ap.add_argument("--out", default="/engine/tools/_cbz.pt")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        cmp(*a.compare)
    else:
        run(a)


if __name__ == "__main__":
    main()
