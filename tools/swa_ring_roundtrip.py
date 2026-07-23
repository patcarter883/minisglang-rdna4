"""Does the SWA serve sliding path stay byte-identical THROUGH the ring? The serve extend gathers the
window from the ring (written by tail_hip.store_kv), not from the inline K the cold path uses. Test:
  (A) store_kv(K) -> gather == K  (bf16 round-trip identity?)
  (B) full serve sliding path: cold dense _swa_prefill_cold  vs  extend over [pad | ring-gathered
      window | new]  -> byte-identical for the new tokens?
If (A) or (B) DIFFERS, that ULP is the SWA serve reuse-vs-cold late-token divergence.

Run: MINISGL_CMD='python /engine/tools/swa_ring_roundtrip.py' gpu-lease -n 1 -- docker compose --profile run run --rm run
"""
from __future__ import annotations

import sys

import torch

import attn_hip

DEV = "cuda"
torch.manual_seed(0)
HQ, HK, D, W = 64, 8, 128, 512
SCALE = D ** -0.5
BC = 32

_STORE = None
try:
    import tail_hip
    _STORE = tail_hip.store_kv
except Exception as e:
    print("tail_hip.store_kv unavailable:", e)


def store_gather(k, v, nslots):
    """Mimic MHAKVCache.store_kv(bf16, scale 1.0) into a ring, then gather positions [0, len(k))."""
    kc = torch.zeros(nslots, HK, D, device=DEV, dtype=torch.bfloat16)
    vc = torch.zeros(nslots, HK, D, device=DEV, dtype=torch.bfloat16)
    loc = torch.arange(k.shape[0], device=DEV, dtype=torch.int32)
    if _STORE is not None:
        _STORE(k.contiguous(), v.contiguous(), kc, vc, loc, 1.0, 1.0)
    else:  # torch fallback (matches mha_pool bf16 path)
        kc[loc.long()] = k.to(torch.bfloat16)
        vc[loc.long()] = v.to(torch.bfloat16)
    return kc[:k.shape[0]].clone(), vc[:v.shape[0]].clone()


def main():
    L, M = 720, 14   # short-ish reuse shape (prefix>W so Wp=W=512, pad=(720-512)%32=16)
    N = L + M
    q = torch.randn(N, HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(N, HK, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(N, HK, D, device=DEV, dtype=torch.bfloat16)

    # (A) round-trip identity
    kr, vr = store_gather(k, v, N + 8)
    a_ok = torch.equal(kr, k) and torch.equal(vr, v)
    print(f"  [{'IDENTICAL' if a_ok else 'DIFFERS  '}] (A) store_kv->gather round-trip  "
          f"max|Δk|={(kr.float()-k.float()).abs().max():.3e}")

    # (B) serve sliding path: cold vs extend-over-ring-gathered-window
    out_cold = attn_hip.flash_prefill(q.contiguous(), k.contiguous(), v.contiguous(), SCALE, 1, W)[L:]
    Wp = min(L, W)
    pad = (L - Wp) % BC
    # gather window [L-Wp, L) from the RING (store round-trip), exactly like _gather_swa_windows
    kr_all, vr_all = store_gather(k[:L], v[:L], L + 8)
    k_win, v_win = kr_all[L - Wp:L], vr_all[L - Wp:L]
    front = pad + Wp
    zk = k_win.new_zeros((pad, HK, D)); zv = v_win.new_zeros((pad, HK, D))
    k_ext = torch.cat([zk, k_win, k[L:N]], 0).contiguous()
    v_ext = torch.cat([zv, v_win, v[L:N]], 0).contiguous()
    q_ext = torch.cat([q.new_zeros((front, HQ, D)), q[L:N]], 0).contiguous()
    out_ext = attn_hip.flash_prefill(q_ext, k_ext, v_ext, SCALE, 1, W)[front:]
    b_ok = torch.equal(out_cold, out_ext)
    print(f"  [{'IDENTICAL' if b_ok else 'DIFFERS  '}] (B) cold vs ring-gathered extend  "
          f"max|Δ|={(out_cold.float()-out_ext.float()).abs().max():.3e}")
    print("\nVERDICT:", "ring path byte-identical — divergence is elsewhere" if (a_ok and b_ok)
          else "ring round-trip / gather introduces the ULP (this is the serve divergence)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
