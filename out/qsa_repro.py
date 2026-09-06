"""Is the gate harness reproducible at all? Digest the weights of two identical builds."""
import os, sys, hashlib
sys.path.insert(0, "/engine/python"); sys.path.insert(0, "/engine/tests")
import numpy as np, torch
import qwen4exp_qsa_gate_test as G
class A: pass
args=A(); args.model="/model"; args.layers=4; args.experts=16; args.max_seq=1024
args.prompt_len=128; args.steps=2
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
from minisgl.distributed import set_tp_info, try_get_tp_info
if try_get_tp_info() is None: set_tp_info(0,1)
from minisgl.layers.rotary import set_rope_device; set_rope_device(dev)

def digest(model):
    h = {}
    for n,p in model.state_dict().items():
        if isinstance(p, torch.Tensor) and p.device.type != "meta":
            h[n] = hashlib.sha256(p.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
    return h

with torch.inference_mode():
    h = G._Harness(args, dev)
    rng = np.random.default_rng(11)
    prompt = rng.integers(0, h.mc.vocab_size, size=args.prompt_len, dtype=np.int64)
    h.build(qsa=False); d1 = digest(h.model); o1,i1,_ = h.run(prompt, 2); h.teardown()
    h.build(qsa=False); d2 = digest(h.model); o2,i2,_ = h.run(prompt, 2); h.teardown()
    diff = [k for k in d1 if d1[k]!=d2.get(k)]
    print(f"weights: {len(d1)} tensors (meta skipped), {len(diff)} DIFFER")
    for k in diff[:12]: print("   ", k)
    print("ids", i1, i2)
    for (a,b) in zip(o1,o2):
        print(f"  {a[0]}: max|d|={float((a[1]-b[1]).abs().max()):.3e} bitexact={torch.equal(a[1],b[1])}")
