"""Targeted QSA probe: is the SELECTION complete at <=budget, and does the sparse ATTENTION agree?

Splits the two questions the end-to-end logit gate conflates.
"""
import os, sys, json, tempfile, shutil
sys.path.insert(0, "/engine/python"); sys.path.insert(0, "/engine/tests")
import numpy as np, torch
import qwen4exp_qsa_gate_test as G

class A: pass
args = A(); args.model="/model"; args.layers=4; args.experts=16; args.max_seq=1024
args.prompt_len=128; args.steps=2
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
from minisgl.distributed import set_tp_info, try_get_tp_info
if try_get_tp_info() is None: set_tp_info(0,1)
from minisgl.layers.rotary import set_rope_device; set_rope_device(dev)

with torch.inference_mode():
    h = G._Harness(args, dev)
    rng = np.random.default_rng(11)
    prompt = rng.integers(0, h.mc.vocab_size, size=args.prompt_len, dtype=np.int64)

    # capture per-layer attention inputs/outputs on both legs
    def instrument(model, store):
        from minisgl.layers.attention import AttentionLayer
        for lyr in model.model.layers.op_list:
            sa = getattr(lyr, "self_attn", None)
            if sa is None: continue
            orig = sa.attn.forward
            def wrap(qkv, selection=None, _o=orig, _s=store):
                out = _o(qkv, selection)
                _s.setdefault("out", []).append(out.detach().float().cpu().clone())
                _s.setdefault("qkv", []).append(qkv.detach().float().cpu().clone())
                if selection is not None:
                    _s["sel"] = (selection.slots.cpu().clone(), selection.lens.cpu().clone(),
                                 selection.visited, selection.dense)
                return out
            sa.attn.forward = wrap

    os.environ["MINISGL_QSA_TAP"] = "1"
    sd = {}
    h.build(qsa=False); instrument(h.model, sd)
    d_out, d_ids, _ = h.run(prompt, args.steps); h.teardown()
    ss = {}
    h.build(qsa=True); instrument(h.model, ss)
    s_out, s_ids, _ = h.run(prompt, args.steps)
    rt = h.ctx.qsa

    print(f"visited={rt.total_visited} dense={rt.total_dense} ratio={rt.total_visited/rt.total_dense:.4f}")
    slots, lens, vis, den = ss["sel"]
    print(f"last selection: rows={slots.shape[0]} width={slots.shape[1]} lens[:6]={lens[:6].tolist()} visited={vis} dense={den}")
    # PREFILL layer-0 attention comparison
    print("qkv identical (prefill):", torch.equal(sd['qkv'][0], ss['qkv'][0]),
          " max|d|=", float((sd['qkv'][0]-ss['qkv'][0]).abs().max()))
    print("attn out (prefill) max|d|:", float((sd['out'][0]-ss['out'][0]).abs().max()))
    for i in range(1, min(len(sd['out']), len(ss['out']))):
        print(f"  call {i}: qkv eq={torch.equal(sd['qkv'][i], ss['qkv'][i])} "
              f"attn max|d|={float((sd['out'][i]-ss['out'][i]).abs().max()):.3e}")
    # selection completeness for the LAST forward (a decode step): expect every visible token
    t = rt._last[0]
    toks = t["tokens"].cpu(); pos = t["logical_pos"].cpu()
    for m in range(min(3, toks.shape[0])):
        got = sorted(int(x) for x in toks[m] if int(x) >= 0)
        want = list(range(int(pos[m]) + 1))
        print(f" row {m}: pos={int(pos[m])} n_sel={len(got)} complete={got==want} "
              f"missing={sorted(set(want)-set(got))[:6]} extra={sorted(set(got)-set(want))[:6]}")
