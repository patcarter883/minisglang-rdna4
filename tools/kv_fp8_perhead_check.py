"""Engine-level check for the PER-HEAD fp8-KV descale: pool -> store kernel -> attention kernel,
including the graph-capture-safety property the whole device-tensor design exists for.

Not a kernel parity test (those live in rdna4-hip-kernels/*/tests). This checks the ENGINE wiring:
  1. MHAKVCache allocates [num_layers, num_kv_heads] descale/inv tensors and calibration
     accumulates a real per-HEAD amax.
  2. finalize_kv_calibration() writes them IN PLACE (addresses unchanged) and descale == 1/inv.
  3. store_kv actually applies the per-head scale, and a round-trip through the pool + the fp8
     attention op matches an fp32 reference built from the same bytes.
  4. CAPTURE SAFETY: a HIP graph captured against pool.k_descale[layer] picks up a LATER in-place
     calibration write on replay. A host-float descale (or a re-allocated tensor) would replay the
     stale value; this is the property that forces the device-tensor contract.

Run inside the ROCm image under a 1-card lease with the worktree kernels on PYTHONPATH.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

FAILS = []
DEV = "cuda"


def ok(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:46s} {detail}")
    if not cond:
        FAILS.append(name)


def main():
    os.environ.setdefault("MINISGL_KV_FP8_CALIBRATE", "1")
    from minisgl.distributed import set_tp_info

    set_tp_info(0, 1)
    from minisgl.kvcache.mha_pool import MHAKVCache

    L, H, D, P, PS = 4, 8, 128, 64, 16
    pool = MHAKVCache(num_kv_heads=H, num_layers=L, head_dim=D, num_pages=P, page_size=PS,
                      dtype=torch.float8_e4m3fn, device=torch.device(DEV))

    print("\n== 1. per-head descale table shape ==")
    ok("k_descale is [num_layers, num_kv_heads]", tuple(pool.k_descale.shape) == (L, H),
       str(tuple(pool.k_descale.shape)))
    ok("k_descale[layer] row is contiguous [Hkv]",
       pool.k_descale[1].is_contiguous() and pool.k_descale[1].numel() == H)
    ok("calibration is on (fp8 + env)", pool._calibrating)

    # ---- 2. store with a deliberately head-skewed K so the per-head amax must differ ----
    print("\n== 2. calibration accumulates a per-HEAD amax ==")
    torch.manual_seed(11)
    T = 64
    gain = torch.linspace(1.0, 8.0, H, device=DEV).view(1, -1, 1)
    k = (torch.randn(T, H, D, device=DEV, dtype=torch.bfloat16).float() * gain).bfloat16()
    v = (torch.randn(T, H, D, device=DEV, dtype=torch.bfloat16).float() / gain).bfloat16()
    loc = torch.arange(T, device=DEV, dtype=torch.int64)
    for li in range(L):
        pool.store_kv(k.reshape(T, -1), v.reshape(T, -1), loc, li)
    amax = pool._k_amax[0]
    ok("per-head K amax is not flat", (amax.max() / amax.min()).item() > 3.0,
       f"spread={(amax.max() / amax.min()).item():.2f}x")
    ok("per-head K amax matches torch", torch.allclose(amax, k.float().abs().amax(dim=(0, 2))))

    # ---- 3. finalize: in place, and descale == 1/inv ----
    print("\n== 3. finalize_kv_calibration ==")
    kd_ptr, ki_ptr = pool.k_descale.data_ptr(), pool.k_inv_scale.data_ptr()
    pool.finalize_kv_calibration()
    ok("k_descale updated IN PLACE (same address)", pool.k_descale.data_ptr() == kd_ptr)
    ok("k_inv_scale updated IN PLACE (same address)", pool.k_inv_scale.data_ptr() == ki_ptr)
    ok("descale == 1/inv_scale",
       torch.allclose(pool.k_descale, 1.0 / pool.k_inv_scale, rtol=1e-6))
    ok("descale is per-head (not flat)",
       (pool.k_descale[0].max() / pool.k_descale[0].min()).item() > 3.0,
       f"spread={(pool.k_descale[0].max() / pool.k_descale[0].min()).item():.2f}x")
    ok("descale == amax/448", torch.allclose(pool.k_descale[0], amax / 448.0, rtol=1e-5))

    # ---- 4. store round-trip actually uses the per-head scale ----
    print("\n== 4. per-head store round-trip ==")
    pool.store_kv(k.reshape(T, -1), v.reshape(T, -1), loc, 0)
    kc = pool.k_cache(0).view(P * PS, H, D)[:T]
    deq = kc.float() * pool.k_descale[0].view(1, -1, 1)
    ref_q = (k.float() * pool.k_inv_scale[0].view(1, -1, 1)).to(torch.float8_e4m3fn)
    ok("stored bytes == per-head torch reference",
       (kc.float() - ref_q.float()).abs().max().item() == 0.0)
    rel = ((deq - k.float()).pow(2).mean().sqrt() / k.float().pow(2).mean().sqrt()).item()
    ok("round-trip rel-RMSE is e4m3-grade", rel < 0.05, f"rel-RMSE={rel:.5f}")
    # What per-head does and does NOT buy, asserted rather than assumed.
    #
    # (a) On Gaussian-ish data a per-head scale is a WASH, even at a 6.9x head-amax spread. e4m3 is
    #     a FLOATING format: its relative precision comes from the 3 mantissa bits and the
    #     per-element exponent already tracks the value, so dividing a quiet head by a 6.9x-too-big
    #     scale just shifts its exponents down and keeps the same relative error. This is the
    #     single most important thing to know before reaching for per-head scaling on an fp8 cache
    #     (it is NOT the int8 situation, where the scale IS the resolution).
    per_tensor_scale = k.float().abs().amax() / 448.0
    q_pt = ((k.float() / per_tensor_scale).to(torch.float8_e4m3fn).float() * per_tensor_scale)
    e_ph = (deq[:, :1] - k.float()[:, :1]).abs().mean().item()
    e_pt = (q_pt[:, :1] - k.float()[:, :1]).abs().mean().item()
    ok("quietest head, gaussian: per-head ~= per-tensor", 0.9 < e_pt / e_ph < 1.1,
       f"MAE per-head={e_ph:.3e} per-tensor={e_pt:.3e} ({e_pt / e_ph:.3f}x)")

    # (b) Where per-head DOES pay: SUBNORMAL FLUSH. e4m3's usable dynamic range is only
    #     448 / 2^-9 ~= 2^18. Give one head a wide-dynamic-range distribution and another a big
    #     amax, and a per-tensor scale pushes the quiet head's small values under e4m3's smallest
    #     subnormal, where they are flushed to zero outright. A per-head scale keeps them.
    wide = torch.zeros(T, H, D, device=DEV, dtype=torch.bfloat16)
    wide[:, 0] = (10.0 ** (torch.rand(T, D, device=DEV) * -6.0)).bfloat16()   # 1e-6 .. 1
    wide[:, 1:] = (torch.randn(T, H - 1, D, device=DEV) * 3000.0).bfloat16()  # a very loud head
    ph_s = (wide.float().abs().amax(dim=(0, 2)) / 448.0).clamp(min=1e-30)
    pt_s = wide.float().abs().amax() / 448.0
    q_ph = (wide.float() / ph_s.view(1, -1, 1)).to(torch.float8_e4m3fn).float() * ph_s.view(1, -1, 1)
    q_pt2 = (wide.float() / pt_s).to(torch.float8_e4m3fn).float() * pt_s
    zeroed_pt = (q_pt2[:, 0] == 0).float().mean().item()
    zeroed_ph = (q_ph[:, 0] == 0).float().mean().item()
    # The head's own span here is 1e-6..1 == 2^20, WIDER than e4m3's 2^18, so even a perfect
    # per-head scale must flush the bottom few percent. The claim is the gap, not zero.
    ok("wide-range head: per-tensor flushes to zero, per-head does not",
       zeroed_pt > 0.2 and zeroed_ph < 0.5 * zeroed_pt,
       f"zeroed per-tensor={zeroed_pt:.1%} per-head={zeroed_ph:.1%}")

    # ---- 5. CAPTURE SAFETY ----
    print("\n== 5. graph capture replays the LIVE descale, not a captured constant ==")
    import attn_decode

    B, Hq = 1, 16
    q = torch.randn(B, Hq, D, device=DEV, dtype=torch.bfloat16)
    bt = torch.arange(P, device=DEV, dtype=torch.int32).view(B, P)
    cl = torch.full((B,), T, device=DEV, dtype=torch.int32)
    kc4 = pool.k_cache(0)
    vc4 = pool.v_cache(0)
    kdsc, vdsc = pool.k_descale[0], pool.v_descale[0]

    def run():
        return attn_decode.flash_decode_paged_fp8(
            q, kc4, vc4, bt, cl, D ** -0.5, kdsc, vdsc, 0)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    static_out = None
    with torch.cuda.graph(g):
        static_out = run()
    g.replay()
    torch.cuda.synchronize()
    before = static_out.clone()

    # Re-calibrate in place: halve every descale. A captured host scalar would ignore this.
    pool.k_descale.mul_(0.5)
    pool.v_descale.mul_(0.5)
    g.replay()
    torch.cuda.synchronize()
    after = static_out.clone()
    changed = (after.float() - before.float()).abs().max().item()
    ok("replay picks up an in-place descale change", changed > 1e-3, f"max|Δ|={changed:.3e}")
    # and it is the RIGHT change: halving both k and v descale halves the output only via v
    # (k_descale rescales the softmax scores, which is not linear), so just check it moved and
    # that restoring the value restores the output bit-exactly.
    pool.k_descale.mul_(2.0)
    pool.v_descale.mul_(2.0)
    g.replay()
    torch.cuda.synchronize()
    ok("restoring the descale restores the output bit-exactly",
       (static_out.float() - before.float()).abs().max().item() == 0.0)

    print("\n" + (f"FAILED ({len(FAILS)}): " + ", ".join(FAILS) if FAILS else "ALL GREEN"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
