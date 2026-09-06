"""Which prefill attention is closer to EXACT: the dense WMMA tile core, or the QSA sparse path?

The kernel-swap control came back at 0.0 (attn_hip.flash_prefill and attn_prefill_paged.
flash_prefill_paged agree bit-for-bit), so "reordering noise" is not an explanation the harness has
earned. This measures both against a float64 host reference computed from the SAME paged KV, for
sampled query rows of the same prefill. If the sparse path is no further from f64 than the dense
one, the difference between them is the two kernels' accumulation, not a QSA defect.
"""
import os, sys
sys.path.insert(0, "/engine/python"); sys.path.insert(0, "/engine/tests")
import numpy as np, torch
import qwen4exp_qsa_gate_test as G
class A: pass
args=A(); args.model="/model"; args.layers=4; args.experts=16; args.max_seq=2048
args.prompt_len=1024; args.steps=0
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
from minisgl.distributed import set_tp_info, try_get_tp_info
if try_get_tp_info() is None: set_tp_info(0,1)
from minisgl.layers.rotary import set_rope_device; set_rope_device(dev)

def grab(ctx, store):
    b = ctx.attn_backend
    f, fs = b.forward, getattr(b, "forward_sparse", None)
    def w(q,k,v,lid,batch,**kw):
        o = f(q,k,v,lid,batch,**kw)
        if lid == 0: store.append((q.detach().clone(), o.detach().clone()))
        return o
    b.forward = w
    if fs is not None:
        def ws(q,k,v,lid,batch,slots,lens):
            o = fs(q,k,v,lid,batch,slots,lens)
            if lid == 0: store.append((q.detach().clone(), o.detach().clone()))
            return o
        b.forward_sparse = ws

def f64_ref(kvpool, page_table, rows, q, scale):
    """Exact causal attention for the sampled rows, from the stored paged KV, in float64."""
    kc = kvpool.k_cache(0); vc = kvpool.v_cache(0)
    ns = kc.shape[0]*kc.shape[1]; kvh = kc.shape[2]; hd = kc.shape[3]
    K = kc.view(ns, kvh, hd).double(); V = vc.view(ns, kvh, hd).double()
    out = {}
    for m in rows:
        slots = page_table[0, :m+1].long()
        k = K[slots]; v = V[slots]                       # [L, kvh, hd]
        qi = q[m].double()                               # [Hq, hd]
        gq = qi.shape[0]//kvh
        k = k.repeat_interleave(gq, dim=1); v = v.repeat_interleave(gq, dim=1)
        p = torch.softmax(torch.einsum("hd,lhd->hl", qi, k)*scale, dim=-1)
        out[m] = torch.einsum("hl,lhd->hd", p, v)
    return out

with torch.inference_mode():
    h = G._Harness(args, dev)
    rng = np.random.default_rng(11)
    prompt = rng.integers(0, h.mc.vocab_size, size=args.prompt_len, dtype=np.int64)
    res = {}
    for leg in (False, True):
        st = []
        h.build(qsa=leg); grab(h.ctx, st); h.run(prompt, 0)
        q, o = st[0]
        scale = h.ctx.attn_backend._softmax_scale(q) if hasattr(h.ctx.attn_backend,"_softmax_scale") \
                else 1.0/np.sqrt(q.shape[-1])
        rows = [0, 1, 7, 63, 255, 511, 1023]
        ref = f64_ref(h.ctx.kv_cache, h.ctx.page_table, rows, q, scale)
        errs = {m: float(((o[m].double()-ref[m]).norm()/ref[m].norm()).item()) for m in rows}
        res["sparse" if leg else "dense"] = (errs, o.double().cpu())
        h.teardown()
    de, do = res["dense"]; se, so = res["sparse"]
    print(f"{'row':>6} {'dense rel(f64)':>16} {'sparse rel(f64)':>17} {'winner':>8}")
    for m in de:
        w = "dense" if de[m] < se[m] else ("sparse" if se[m] < de[m] else "tie")
        print(f"{m:6d} {de[m]:16.3e} {se[m]:17.3e} {w:>8}")
    print(f"\nattn out dense-vs-sparse max|d| = {float((do-so).abs().max()):.3e}")
    print(f"mean rel(f64): dense={np.mean(list(de.values())):.3e} sparse={np.mean(list(se.values())):.3e}")
