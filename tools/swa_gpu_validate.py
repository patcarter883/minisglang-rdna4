"""GPU validation of the Sliding-Window-Attention (SWA) mechanism used for Laguna, exercised through
the REAL HIP kernels (attn_hip.flash_prefill + attn_decode.flash_decode_paged) with the exact ring
addressing minisgl's SWA backend uses. No full model needed — validates items 2 (window-bounded ring
pool addressing) + 3 (kernel window masking) directly against a pure-torch fp32 reference.

Run inside the lean image with a GPU lease:
    gpu-lease -n 1 -- bash -c '... python tools/swa_gpu_validate.py'
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

import attn_decode
import attn_hip

DEV = "cuda"
torch.manual_seed(0)
FAILS = []

# Laguna sliding-layer shape (TP=1): 64 QO heads, 8 KV heads, head_dim 128, window 512.
HQ, HK, D, W = 64, 8, 128, 512
SCALE = D ** -0.5


def _record(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:42s} {extra}")
    if not ok:
        FAILS.append(name)


def ref_windowed_causal_prefill(q, k, v, window):
    """q/k/v: [S, Hq|Hk, D] single seq -> [S, Hq, D]. Causal + sliding-window fp32 reference."""
    S, Hq = q.shape[0], q.shape[1]
    Hk = k.shape[1]
    rep = Hq // Hk
    qf = q.float().permute(1, 0, 2)                              # [Hq,S,D]
    kf = k.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    vf = v.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    attn = torch.matmul(qf, kf.transpose(-1, -2)) * SCALE        # [Hq,S,S]
    i = torch.arange(S, device=q.device)
    causal = i[:, None] < i[None, :]                             # key after query
    old = (i[:, None] - i[None, :]) >= window                    # older than window
    attn = attn.masked_fill((causal | old)[None], float("-inf"))
    out = torch.matmul(F.softmax(attn, dim=-1), vf)              # [Hq,S,D]
    return out.permute(1, 0, 2).contiguous()


def ref_decode(q, k, v):
    """q:[1,Hq,D] k/v:[1,S,Hk,D] -> [1,Hq,D] fp32, GQA, full attention over the given keys."""
    Hq, Hk = q.shape[1], k.shape[2]
    rep = Hq // Hk
    qf = q.float().unsqueeze(2)
    kf = k.float().repeat_interleave(rep, dim=2).permute(0, 2, 1, 3)
    vf = v.float().repeat_interleave(rep, dim=2).permute(0, 2, 1, 3)
    attn = torch.matmul(qf, kf.transpose(-1, -2)) * SCALE
    return torch.matmul(F.softmax(attn, dim=-1), vf).squeeze(2).contiguous()


def test_prefill_window():
    """Cold prefill over a prompt longer than the window: kernel's sliding_window arg must mask keys
    older than W (this is the SWA sliding-layer cold-prefill path)."""
    S = 700
    q = torch.randn(S, HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(S, HK, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(S, HK, D, device=DEV, dtype=torch.bfloat16)
    got = attn_hip.flash_prefill(q.contiguous(), k.contiguous(), v.contiguous(), SCALE, 1, W).float()
    ref = ref_windowed_causal_prefill(q, k, v, W)
    d = (got - ref).abs().max().item()
    # bf16 in / fp32 accumulate over a 700-length windowed softmax -> a few e-3 is expected (the
    # canonical attn_hip SWA test uses 4e-3 at S=160; ~4.4x longer here). 1.5e-2 is the bf16 floor.
    _record("cold prefill windowed causal (S=700,W=512)", d <= 1.5e-2, f"max|Δ|={d:.3e}")

    # OFF-BY-ONE DISCRIMINATOR: the kernel's window boundary must match the W reference MUCH better
    # than the W-1 / W+1 references. A boundary bug would flip which side matches. (This is what
    # proves the ~8e-3 above is bf16 noise, not a masked-key-count error.)
    d_wm1 = (got - ref_windowed_causal_prefill(q, k, v, W - 1)).abs().max().item()
    d_wp1 = (got - ref_windowed_causal_prefill(q, k, v, W + 1)).abs().max().item()
    _record("boundary == W (not W±1)", d < d_wm1 * 0.5 and d < d_wp1 * 0.5,
            f"Δ(W)={d:.3e}  Δ(W-1)={d_wm1:.3e}  Δ(W+1)={d_wp1:.3e}")

    # Sanity: with sliding_window=0 (no window) the last query attends to ALL keys, so its output
    # must DIFFER from the windowed one (proves the window arg actually changes behaviour).
    got_full = attn_hip.flash_prefill(q.contiguous(), k.contiguous(), v.contiguous(), SCALE, 1, 0).float()
    diff = (got_full[-1] - got[-1]).abs().max().item()
    _record("window actually masks (full vs windowed last row)", diff > 1e-2, f"Δlast={diff:.3e}")


def build_ring(k_all, v_all, S_stored):
    """Store positions 0..S_stored-1 into a per-seq ring of W slots (slot = pos % W), exactly as
    RDNA4Backend._build_swa_metadata addresses swa_out_loc. Returns (k_ring,v_ring)[W,1,Hk,D]."""
    k_ring = torch.zeros(W, 1, HK, D, device=DEV, dtype=torch.bfloat16)
    v_ring = torch.zeros(W, 1, HK, D, device=DEV, dtype=torch.bfloat16)
    for p in range(S_stored):
        s = p % W
        k_ring[s, 0] = k_all[p]
        v_ring[s, 0] = v_all[p]
    return k_ring, v_ring


def test_ring_decode():
    """Ring store (long seq) + decode read: the ring must present exactly the last W positions, and
    the kernel over the ring block must equal a reference windowed decode."""
    S = 700  # prefill positions 0..699; decode token is position 700
    k_all = torch.randn(S + 1, HK, D, device=DEV, dtype=torch.bfloat16)
    v_all = torch.randn(S + 1, HK, D, device=DEV, dtype=torch.bfloat16)
    q_dec = torch.randn(1, HQ, D, device=DEV, dtype=torch.bfloat16)

    # store positions 0..700 into the ring (S+1 = 701 stores; last W survive)
    k_ring, v_ring = build_ring(k_all, v_all, S + 1)
    device_len = S + 1  # 701
    cnt = min(device_len, W)  # 512
    block_table = torch.arange(0, cnt, device=DEV, dtype=torch.int32).unsqueeze(0)  # [1, cnt]
    ctx = torch.tensor([cnt], device=DEV, dtype=torch.int32)
    got = attn_decode.flash_decode_paged(q_dec, k_ring, v_ring, block_table, ctx, SCALE, 0).float()

    # reference: the newest query (position 700) attends to positions [700-W+1, 700] = last W keys.
    lo = device_len - W  # 701-512 = 189
    k_win = k_all[lo:device_len].unsqueeze(0)  # positions 189..700
    v_win = v_all[lo:device_len].unsqueeze(0)
    ref = ref_decode(q_dec, k_win, v_win)
    d = (got - ref).abs().max().item()
    _record("ring decode == windowed decode (S=701,W=512)", d <= 5e-3, f"max|Δ|={d:.3e}")

    # Cross-check: the ring holds exactly {189..700}. Build the SAME key set in positional order and
    # confirm the ring-order kernel output matches (softmax is permutation-invariant over keys).
    survivors = sorted({p % W: p for p in range(device_len)}.values())
    _record("ring survivors == last-W positions", survivors == list(range(lo, device_len)),
            f"[{survivors[0]}..{survivors[-1]}], n={len(survivors)}")

    # Short sequence (< W): ring holds all S positions, decode attends to all of them (no masking).
    Ss = 100
    ks = torch.randn(Ss, HK, D, device=DEV, dtype=torch.bfloat16)
    vs = torch.randn(Ss, HK, D, device=DEV, dtype=torch.bfloat16)
    qd = torch.randn(1, HQ, D, device=DEV, dtype=torch.bfloat16)
    kr, vr = build_ring(ks, vs, Ss)
    bt = torch.arange(0, Ss, device=DEV, dtype=torch.int32).unsqueeze(0)
    ctx2 = torch.tensor([Ss], device=DEV, dtype=torch.int32)
    got2 = attn_decode.flash_decode_paged(qd, kr, vr, bt, ctx2, SCALE, 0).float()
    ref2 = ref_decode(qd, ks.unsqueeze(0), vs.unsqueeze(0))
    d2 = (got2 - ref2).abs().max().item()
    _record("ring decode (S=100<W, full attend)", d2 <= 5e-3, f"max|Δ|={d2:.3e}")


def main():
    print("== SWA prefill window masking (item 3) ==")
    test_prefill_window()
    print("== SWA ring-pool addressing + decode (items 2+3) ==")
    test_ring_decode()
    print()
    if FAILS:
        print(f"FAILED: {FAILS}")
        return 1
    print("ALL SWA GPU CHECKS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
