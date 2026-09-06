"""CHUNKED prefill above the budget — the path a real serve actually takes.

The single-shot probe stopped at 16384 because an unchunked prefill materialises a
[rows, index_width] selection per forward; a scheduler never does that. This drives the SAME
prompt through group-aligned chunks (cached_len advances, so the compressed keys of earlier chunks
are read back out of the cache rather than recomputed) and reports the reach + sparsity.
"""
import os, sys, time
sys.path.insert(0, "/engine/python"); sys.path.insert(0, "/engine/tests")
import numpy as np, torch
import qwen4exp_qsa_gate_test as G
class A: pass
args=A(); args.model="/model"; args.layers=4; args.experts=16; args.steps=2
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
from minisgl.distributed import set_tp_info, try_get_tp_info
if try_get_tp_info() is None: set_tp_info(0,1)
from minisgl.layers.rotary import set_rope_device; set_rope_device(dev)
from minisgl.gdn.metadata import build_gdn_metadata

CHUNK = int(os.environ.get("CHUNK", "2048"))
LENS = [int(x) for x in os.environ.get("LENS", "8192,32768,131072,262144").split(",")]

def run_chunked(h, prompt, steps):
    ctx, dev = h.ctx, h.dev
    L = len(prompt)
    req = G._make_req(prompt, 0, steps + 1)
    req.device_len = L; req.max_device_len = L + steps + 1
    logits = None
    for start in range(0, L, CHUNK):
        end = min(start + CHUNK, L)
        req.cached_len = start; req.device_len = end
        b = G._make_batch([req], "prefill", h.page_table, dev)
        ctx.attn_backend.prepare_metadata(b)
        b.gdn_metadata = build_gdn_metadata(b, torch.tensor([1],dtype=torch.int32,device=dev), dev)
        if h.ple is not None: h.ple.prepare([1], [prompt[start:end]])
        with ctx.forward_batch(b):
            logits = h.model.forward()
        torch.cuda.synchronize()
        if h.ple is not None: h.ple.commit([1], [prompt[start:end]])
    ids = []
    for _ in range(steps):
        nxt = int(logits[-1].float().argmax().item()); ids.append(nxt)
        req.append_host(torch.tensor([nxt], dtype=torch.int64)); req.complete_one()
        b = G._make_batch([req], "decode", h.page_table, dev)
        ctx.attn_backend.prepare_metadata(b)
        b.gdn_metadata = build_gdn_metadata(b, torch.tensor([1],dtype=torch.int32,device=dev), dev)
        t = np.array([nxt], dtype=np.int64)
        if h.ple is not None: h.ple.prepare([1], [t])
        with ctx.forward_batch(b):
            logits = h.model.forward()
        torch.cuda.synchronize()
        if h.ple is not None: h.ple.commit([1], [t])
    return logits, ids

with torch.inference_mode():
    for L in LENS:
        args.max_seq = L + 64; args.prompt_len = min(CHUNK, L)
        h = G._Harness(args, dev)
        rng = np.random.default_rng(5)
        p = rng.integers(0, h.mc.vocab_size, size=L, dtype=np.int64)
        try:
            t0 = time.time(); h.build(qsa=True)
            lg, ids = run_chunked(h, p, 2)
            dt = time.time() - t0; rt = h.ctx.qsa
            print(f"ctx={L:7d} chunk={CHUNK} OK finite={bool(torch.isfinite(lg).all())} "
                  f"ratio={rt.total_visited/rt.total_dense:.4f} ids={ids} {dt:6.1f}s "
                  f"peak={torch.cuda.max_memory_allocated()>>20} MiB", flush=True)
        except Exception as e:
            print(f"ctx={L:7d} chunk={CHUNK} FAILED {type(e).__name__}: {str(e)[:180]}", flush=True)
        finally:
            try: h.teardown()
            except Exception: pass
            torch.cuda.reset_peak_memory_stats()
