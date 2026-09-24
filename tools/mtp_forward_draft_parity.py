#!/usr/bin/env python
"""Parity + timing for the MTP draft attention on the HIP decode kernel (spec/draft_attn.py).

The MTP heads used to attend in plain torch: gather the slot's whole draft-KV ring
(`k_buf[slot_rows]`, a copy), einsum over all R columns, add a -inf keep-mask, softmax, einsum with V.
They now hand the ring IN PLACE to attn_decode.flash_decode_paged through a page_size-1 block table
built from the same keep-mask (DraftAttnBuilder). Checked against a torch reference of the SAME masked
attention, on the cases that make the ring hard:
  * rows on different slots at different cursors;  * ring WRAPAROUND;
  * REJECTED drafts: K+1 future positions written, then the cursor rewinds past them;
  * a SEEDED tail with an unwritten HOLE below it (radix hit: [0, origin] never written);
  * a REUSED slot whose columns still hold the previous owner's K/V.
Not bit-identical to torch (different fp32 reduction order); the bar is a tight tolerance — a draft
only affects ACCEPTANCE, the target verifies every token. Also: eager == graph replay for the kernel
path, and old-torch-core vs kernel timing at the served head shapes.

    gpu-lease -n 1 -- ... PYTHONPATH=/engine/python:/opt/kernels python tools/mtp_forward_draft_parity.py
"""
import statistics as st

import torch

from minisgl.spec.draft_attn import DraftAttnBuilder, paged_draft_attention

DEV = "cuda"


def torch_masked_ref(q, k_buf, v_buf, slots, keep, scale):
    """The old forward_draft_masked core, same math: gather, grouped einsum, -inf mask, softmax."""
    B, nq, hd = q.shape
    nkv = k_buf.shape[2]
    rep = nq // nkv
    Ks, Vs = k_buf[slots], v_buf[slots]
    scores = torch.einsum("bgrd,bsgd->bgrs", q.view(B, nkv, rep, hd), Ks) * scale
    scores = scores + torch.where(keep, 0.0, float("-inf")).view(B, 1, 1, -1)
    probs = scores.softmax(dim=-1).to(Vs.dtype)
    return torch.einsum("bgrs,bsgd->bgrd", probs, Vs).reshape(B, nq, hd)


class Ring:
    """A miniature of spec/mtp.py's bookkeeping: pos_buf + keep-mask, exactly as the proposer."""

    def __init__(self, slots, R, nkv, hd, dt):
        self.R = R
        self.k = torch.randn(slots, R, nkv, hd, device=DEV, dtype=dt)   # = a previous owner's data
        self.v = torch.randn(slots, R, nkv, hd, device=DEV, dtype=dt)
        self.pos = torch.full((slots, R), -1, device=DEV, dtype=torch.int64)

    def write(self, slot, p):
        c = p % self.R
        self.k[slot, c] = torch.randn_like(self.k[slot, c])
        self.v[slot, c] = torch.randn_like(self.v[slot, c])
        self.pos[slot, c] = p

    def keep(self, slots, q_abs):
        pa = self.pos[slots]
        qa = q_abs.unsqueeze(1)
        return (pa >= 0) & (pa <= qa) & ((qa - pa) < self.R)


def check_parity(nq, nkv, hd, R, K, dt, tol):
    torch.manual_seed(0)
    scale = hd ** -0.5
    ring = Ring(4, R, nkv, hd, dt)
    b = DraftAttnBuilder(R, DEV)
    # slot 0 cold start growing past the ring; slot 1 seeded tail over a HOLE (origin 40 unwritten);
    # slot 2 reused slot, cold at 0 over the previous owner's data; slot 3 long history.
    cursor = {0: 0, 1: 61, 2: 0, 3: 3 * R}
    for p in range(41, 61):
        ring.write(1, p)
    for p in range(0, 3 * R):
        ring.write(3, p)
    worst, n = 0.0, 0
    for it in range(3 * R // (K + 1) + 4):
        slots = torch.tensor([0, 1, 2, 3], device=DEV)
        base = torch.tensor([cursor[s] for s in range(4)], device=DEV)
        for j in range(K + 1):                           # the proposer's K+1-step chain
            q_abs = base + j
            for s in range(4):
                ring.write(s, int(q_abs[s]))
            keep = ring.keep(slots, q_abs)
            q = torch.randn(4, nq, hd, device=DEV, dtype=dt)
            ref = torch_masked_ref(q, ring.k, ring.v, slots, keep, scale)
            got = paged_draft_attention(q, ring.k, ring.v, b.meta(slots, q_abs, keep), scale)
            d = (ref.float() - got.float()).abs().max().item()
            worst = max(worst, d)
            n += 1
            assert d <= tol, f"MISMATCH nq={nq} nkv={nkv} hd={hd} R={R} {dt} it={it} j={j}: max|d|={d:.3e}"
        for s in range(4):                               # accept a varying number of drafts per slot
            cursor[s] += 1 + ((it + s) % (K + 1))
    return worst, n


def graphed(fn, calls=1):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            out = fn()
    return g, out


def check_capture(nq, nkv, hd, R, dt):
    torch.manual_seed(1)
    ring = Ring(2, R, nkv, hd, dt)
    for p in range(R + 37):
        ring.write(0, p)
        ring.write(1, p // 2)
    b = DraftAttnBuilder(R, DEV)
    slots = torch.tensor([0, 1], device=DEV)
    q_abs = torch.tensor([R + 36, (R + 36) // 2], device=DEV)
    q = torch.randn(2, nq, hd, device=DEV, dtype=dt)
    fn = lambda: paged_draft_attention(q, ring.k, ring.v, b.meta(slots, q_abs, ring.keep(slots, q_abs)), hd ** -0.5)  # noqa: E731
    eager = fn().clone()
    g, out = graphed(fn)
    g.replay()
    torch.cuda.synchronize()
    return torch.equal(eager, out)


def time_arms(B, nq, nkv, hd, R, dt, calls=20, replays=20, windows=7):
    torch.manual_seed(2)
    k = torch.randn(8, R, nkv, hd, device=DEV, dtype=dt)
    v = torch.randn(8, R, nkv, hd, device=DEV, dtype=dt)
    slots = torch.arange(B, device=DEV)
    q_abs = torch.full((B,), 5 * R, device=DEV)                     # a long history: every column live
    pos = (q_abs.unsqueeze(1) - R + 1 + torch.arange(R, device=DEV)).remainder(R) + (q_abs.unsqueeze(1) - R + 1)
    keep = torch.ones(B, R, dtype=torch.bool, device=DEV)
    q = torch.randn(B, nq, hd, device=DEV, dtype=dt)
    b = DraftAttnBuilder(R, DEV)
    arms = {"torch": lambda: torch_masked_ref(q, k, v, slots, keep, hd ** -0.5),
            "hip": lambda: paged_draft_attention(q, k, v, b.meta(slots, q_abs, keep), hd ** -0.5),
            "torch2": lambda: torch_masked_ref(q, k, v, slots, keep, hd ** -0.5)}
    graphs = {n: graphed(fn, calls)[0] for n, fn in arms.items()}
    samp = {n: [] for n in graphs}
    names = list(graphs)
    for w in range(windows + 1):
        for n in names[w % 3:] + names[:w % 3]:
            a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(replays):
                graphs[n].replay()
            e.record()
            e.synchronize()
            if w:
                samp[n].append(a.elapsed_time(e) * 1e3 / (calls * replays))
    del pos
    return {n: st.median(x) for n, x in samp.items()}


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"card: {p.name} WGP={p.multi_processor_count}")
    # (nq, nkv, hd) per rank for the served MTP heads: Qwen3.6-35B-A3B 16/2/256 and Qwen3.8-27B 24/4/256,
    # at TP=2 and TP=1.
    shapes = [(8, 1, 256), (16, 2, 256), (12, 2, 256), (24, 4, 256)]
    for (nq, nkv, hd) in shapes:
        for dt, tol in ((torch.bfloat16, 1.6e-2), (torch.float16, 4e-3)):
            worst, n = check_parity(nq, nkv, hd, 64, 4, dt, tol)
            print(f"[parity] nq={nq:2d} nkv={nkv} hd={hd} R=64 K=4 {str(dt)[6:]:8s}: {n} draft steps "
                  f"(wrap, rejected overwrite, seeded hole, reused slot) max|d|={worst:.2e}  OK")
        print(f"[capture] nq={nq} nkv={nkv}: eager == graph replay: {check_capture(nq, nkv, hd, 128, torch.bfloat16)}")
    print("[timing] one draft-step attention, graph replay, us/call (torch = the old gather+einsum core)")
    for (nq, nkv, hd) in shapes[:3]:
        for R in (512, 2064):
            for B in (1, 4):
                m = time_arms(B, nq, nkv, hd, R, torch.bfloat16)
                ctrl = abs(m["torch2"] - m["torch"]) / m["torch"] * 100
                print(f"  nq={nq:2d} nkv={nkv} R={R:4d} B={B}: torch {m['torch']:7.1f}  hip {m['hip']:6.1f}  "
                      f"({m['torch'] / m['hip']:5.2f}x)  ctrl {ctrl:4.1f}%", flush=True)
    print("PARITY_DONE")


if __name__ == "__main__":
    main()
