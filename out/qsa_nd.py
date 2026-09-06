"""Where does the harness's prefill become non-reproducible? Name the first divergent block."""
import os, sys
sys.path.insert(0, "/engine/python"); sys.path.insert(0, "/engine/tests")
import numpy as np, torch
import qwen4exp_qsa_gate_test as G
class A: pass
args=A(); args.model="/model"; args.layers=4; args.experts=16; args.max_seq=1024
args.prompt_len=128; args.steps=0
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
from minisgl.distributed import set_tp_info, try_get_tp_info
if try_get_tp_info() is None: set_tp_info(0,1)
from minisgl.layers.rotary import set_rope_device; set_rope_device(dev)

def trace(model, store):
    for i, lyr in enumerate(model.model.layers.op_list):
        for attr in ("ple", "linear_attn", "self_attn", "mlp",
                     "attn_hyper_connection", "mlp_hyper_connection"):
            ob = getattr(lyr, attr, None)
            if ob is None: continue
            orig = ob.forward
            def w(*a, _o=orig, _k=f"L{i}.{attr}", _s=store, **kw):
                r = _o(*a, **kw)
                t = r[0] if isinstance(r, tuple) else r
                if isinstance(t, torch.Tensor):
                    _s.append((_k, t.detach().float().cpu().clone()))
                return r
            ob.forward = w

with torch.inference_mode():
    h = G._Harness(args, dev)
    rng = np.random.default_rng(11)
    prompt = rng.integers(0, h.mc.vocab_size, size=args.prompt_len, dtype=np.int64)
    a=[]; h.build(qsa=False); trace(h.model,a); h.run(prompt,0); h.teardown()
    b=[]; h.build(qsa=False); trace(h.model,b); h.run(prompt,0); h.teardown()
    print(f"traced {len(a)} vs {len(b)} block outputs")
    for (ka,ta),(kb,tb) in zip(a,b):
        d = float((ta-tb).abs().max())
        print(f"  {ka:34s} max|d|={d:.3e} {'BITEXACT' if torch.equal(ta,tb) else 'DIFFERS'}")
