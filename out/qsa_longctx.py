"""How far past the old 2048 bound does this actually go, and what does capture do?

Two questions, one lease:
  (1) MAX CONTEXT SERVED — prefill + decode at increasing lengths until it stops working, with the
      measured sparsity at each. "It ran" is the claim; the f64 selection check in the gate test is
      what makes it a correct run.
  (2) CAPTURE — the sparse ATTENTION is capture-safe (one kernel, fixed pointers, fixed index_width)
      but the SELECTION is not (host reads + a .max() sync). Assert the refusal is the NAMED one
      rather than a confusing capture failure, and that it names the workaround.
"""
import os, sys, time
sys.path.insert(0, "/engine/python"); sys.path.insert(0, "/engine/tests")
import numpy as np, torch
import qwen4exp_qsa_gate_test as G
class A: pass
args=A(); args.model="/model"; args.layers=4; args.experts=16; args.prompt_len=8
args.steps=2; args.max_seq=4096
dev=torch.device("cuda:0"); torch.cuda.set_device(dev)
from minisgl.distributed import set_tp_info, try_get_tp_info
if try_get_tp_info() is None: set_tp_info(0,1)
from minisgl.layers.rotary import set_rope_device; set_rope_device(dev)

LENS = [int(x) for x in os.environ.get("LENS", "4096,16384,65536,262144").split(",")]
with torch.inference_mode():
    for L in LENS:
        args.max_seq = L + 64
        args.prompt_len = L
        h = G._Harness(args, dev)
        rng = np.random.default_rng(5)
        p = rng.integers(0, h.mc.vocab_size, size=L, dtype=np.int64)
        try:
            t0 = time.time()
            h.build(qsa=True)
            outs, ids, _ = h.run(p, 2)
            torch.cuda.synchronize()
            dt = time.time() - t0
            rt = h.ctx.qsa
            fin = all(bool(torch.isfinite(o[1]).all()) for o in outs)
            print(f"ctx={L:7d}  OK finite={fin}  visited={rt.total_visited:12d} "
                  f"dense={rt.total_dense:12d} ratio={rt.total_visited/rt.total_dense:.4f}  "
                  f"{dt:6.1f}s  peak={torch.cuda.max_memory_allocated()>>20} MiB", flush=True)
        except Exception as e:
            print(f"ctx={L:7d}  FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        finally:
            try: h.teardown()
            except Exception: pass
            torch.cuda.reset_peak_memory_stats()

    # ---- (2) capture behaviour ----
    args.max_seq = 1024; args.prompt_len = 32
    h = G._Harness(args, dev)
    rng = np.random.default_rng(5)
    p = rng.integers(0, h.mc.vocab_size, size=32, dtype=np.int64)
    h.build(qsa=True); h.run(p, 1)
    from minisgl.gdn.metadata import build_gdn_metadata
    from minisgl.core import Batch
    req = G._make_req(p, 0, 8)
    b = G._make_batch([req], "decode", h.page_table, dev)
    h.ctx.attn_backend.prepare_metadata(b)
    g = torch.cuda.CUDAGraph()
    pool = torch.cuda.graphs.graph_pool_handle()
    msg = None
    try:
        with h.ctx.forward_batch(b):
            with torch.cuda.graph(g, pool=pool):
                h.model.forward()
        msg = "CAPTURED (unexpected)"
    except NotImplementedError as e:
        msg = f"NAMED REFUSAL: {str(e)[:180]}"
    except Exception as e:
        msg = f"UNNAMED {type(e).__name__}: {str(e)[:180]}"
    print(f"\ncapture: {msg}")
