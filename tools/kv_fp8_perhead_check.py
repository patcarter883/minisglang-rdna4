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

    # (b) Where per-head DOES pay: SUBNORMAL FLUSH — measured THROUGH THE REAL STORE KERNEL, not
    #     a torch stand-in, because "per-head writing works" is a claim about the store path.
    #     e4m3's usable dynamic range is only 448 / 2^-10 ~= 2^19 (values under scale*2^-10 round to
    #     zero outright). Give one head a wide-dynamic-range distribution and another a loud amax,
    #     and a per-tensor scale pushes most of the quiet head under that floor. Per-head keeps it.
    print("\n== 4b. subnormal flush through the REAL store kernel (the per-head justification) ==")
    torch.manual_seed(23)
    wide = torch.zeros(T, H, D, device=DEV, dtype=torch.bfloat16)
    wide[:, 0] = (10.0 ** (torch.rand(T, D, device=DEV) * -6.0)).bfloat16()   # 1e-6 .. 1, span 2^20
    wide[:, 1:] = (torch.randn(T, H - 1, D, device=DEV) * 3000.0).bfloat16()  # a very loud head
    ph_s = (wide.float().abs().amax(dim=(0, 2)) / 448.0).clamp(min=1e-30)     # [H]  per-head
    pt_s = (wide.float().abs().amax() / 448.0).reshape(1)                     # [1]  per-tensor
    flat = wide.reshape(T, -1)
    res = {}
    for name, row, li in (("per-head", ph_s, 1), ("per-tensor", pt_s, 2)):
        pool.set_fp8_kv_scales(li, row, row)
        pool.store_kv(flat, flat, loc, li)
        cache = pool.k_cache(li).view(P * PS, H, D)[:T]
        # BIT-EXACT vs the torch reference that applies the SAME reciprocal, per head.
        inv = pool.k_inv_scale[li].view(1, -1, 1)
        ref = (wide.float() * inv).to(torch.float8_e4m3fn)
        d = (cache.float() - ref.float()).abs().max().item()
        ok(f"{name}: store == torch reference (bit-exact)", d == 0.0, f"max|Δ|={d:.3e}")
        deq = cache.float() * pool.k_descale[li].view(1, -1, 1)
        # PER-ELEMENT relative error, not rel-RMSE. rel-RMSE is dominated by the head's few LARGE
        # values, which both granularities represent fine; the damage is at the bottom of the range,
        # where a flushed element has relative error 1.0 and contributes almost nothing to an RMS.
        x0 = wide.float()[:, 0]
        res[name] = ((cache[:, 0].float() == 0).float().mean().item(),
                     ((deq[:, 0] - x0).abs() / x0.abs().clamp(min=1e-30)).mean().item())
    (z_ph, e_ph2), (z_pt, e_pt2) = res["per-head"], res["per-tensor"]
    # The head's own span (2^20) is WIDER than e4m3's 2^19, so even a perfect per-head scale must
    # flush the bottom few percent. The claim is the GAP, not zero.
    ok("wide head: per-tensor flushes >>, per-head does not",
       z_pt > 0.5 and z_ph < 0.15,
       f"zeroed per-tensor={z_pt:.1%} per-head={z_ph:.1%}")
    ok("wide head: per-head per-element error is far lower",
       e_ph2 < 0.25 * e_pt2,
       f"mean |Δ|/|x| per-head={e_ph2:.4f} per-tensor={e_pt2:.4f} ({e_pt2 / e_ph2:.1f}x)")

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

    # ---- 6. THE DEFAULT PATH IS UNPERTURBED ----
    # fp8-KV is opt-in. A bf16 pool must still round-trip BIT-EXACTLY through the same store kernel
    # (it passes no scale tensors at all, so the fp8 per-head plumbing must be genuinely inert).
    print("\n== 6. default bf16 KV store is still bit-exact ==")
    for dt in (torch.bfloat16, torch.float16):
        p2 = MHAKVCache(num_kv_heads=H, num_layers=2, head_dim=D, num_pages=P, page_size=PS,
                        dtype=dt, device=torch.device(DEV))
        ok(f"{str(dt).split('.')[-1]} pool is not fp8 (no scale plumbing)", not p2.kv_is_fp8)
        p2.store_kv(k.reshape(T, -1), v.reshape(T, -1), loc, 0)
        got = p2.k_cache(0).view(P * PS, H, D)[:T]
        d = (got.float() - k.to(dt).float()).abs().max().item()
        ok(f"{str(dt).split('.')[-1]} store round-trip max|Δ| == 0", d == 0.0, f"max|Δ|={d:.3e}")
        del p2

    # ---- 7. THE STORE INSIDE A CAPTURED GRAPH ----
    # store_kv runs inside the captured decode graph, which is the whole reason the reciprocals are
    # DEVICE tensors: a host scalar would freeze at capture. Capture store->decode as one graph,
    # then change the scale in place and check the replay both stores AND reads at the new value.
    print("\n== 7. fp8 store + decode captured and replayed as one graph ==")
    st_k = k.reshape(T, -1).clone()
    st_v = v.reshape(T, -1).clone()
    st_loc = loc.to(torch.int32)
    lid = 3

    def store_and_read():
        pool.store_kv(st_k, st_v, st_loc, lid)
        return attn_decode.flash_decode_paged_fp8(
            q, pool.k_cache(lid), pool.v_cache(lid), bt, cl, D ** -0.5,
            pool.k_descale[lid], pool.v_descale[lid], 0)

    scale_a = (k.float().abs().amax(dim=(0, 2)) / 448.0).clamp(min=1e-8)
    pool.set_fp8_kv_scales(lid, scale_a, scale_a)
    for _ in range(3):
        store_and_read()
    torch.cuda.synchronize()
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        out2 = store_and_read()
    g2.replay()
    torch.cuda.synchronize()
    eager_a = store_and_read()
    torch.cuda.synchronize()
    d = (out2.float() - eager_a.float()).abs().max().item()
    ok("replayed store+decode == eager store+decode", d == 0.0, f"max|Δ|={d:.3e}")
    cap_a = out2.clone()
    # Now move the scale IN PLACE. Both halves of the graph must follow it: the store re-quantizes
    # at the new reciprocal and the read undoes it, so the OUTPUT should come back close to where
    # it was (a consistent store/read pair is scale-invariant up to e4m3 rounding), while a graph
    # that had baked only ONE side would diverge wildly.
    # NOTE: move the scale UP, never down. scale_a is already amax/448, so shrinking it pushes the
    # stored value past e4m3fn's max, where it becomes NaN (the format has no inf) — that would test
    # saturation, not scale liveness.
    pool.set_fp8_kv_scales(lid, scale_a * 4.0, scale_a * 4.0)
    g2.replay()
    torch.cuda.synchronize()
    cap_b = out2.clone()
    rel = ((cap_b.float() - cap_a.float()).norm() / cap_a.float().norm()).item()
    ok("store+read stay CONSISTENT across an in-place scale change", rel < 0.05,
       f"rel|Δ|={rel:.4f} (store and read both followed the new scale)")
    # Control: break the pair — move ONLY the read side — and the same replay must diverge. This is
    # what proves the previous check was not vacuous.
    pool.k_descale[lid].mul_(4.0)
    pool.v_descale[lid].mul_(4.0)
    g2.replay()
    torch.cuda.synchronize()
    rel_bad = ((out2.float() - cap_b.float()).norm() / cap_b.float().norm()).item()
    ok("control: read-only scale change DOES diverge", rel_bad > 0.1, f"rel|Δ|={rel_bad:.4f}")

    # ---- 8. SCALE RESOLUTION: checkpoint -> pool (the thing that was never wired) ----
    # finalize_kv_calibration() had no caller in the engine, so every served scale was 1.0 and the
    # granularity was moot. This exercises the boot-time resolver end to end on a synthetic
    # checkpoint: the compressed-tensors kv_cache_scheme gate, the per-tensor -> per-head broadcast,
    # and the refusal to believe scales calibrated for a DIFFERENT grid.
    print("\n== 8. boot-time scale resolution from a checkpoint ==")
    import json
    import tempfile

    from safetensors.torch import save_file

    from minisgl.kvcache.fp8_scales import install_kv_fp8_scales, resolve_kv_fp8_scales

    NL = 4
    with tempfile.TemporaryDirectory() as td:
        def write_ckpt(scheme):
            cfg = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                   "num_hidden_layers": NL, "num_attention_heads": 16, "num_key_value_heads": H,
                   "hidden_size": 2048, "head_dim": D, "intermediate_size": 4096,
                   "vocab_size": 1000, "max_position_embeddings": 4096,
                   "quantization_config": {"quant_method": "compressed-tensors",
                                           "kv_cache_scheme": scheme}}
            json.dump(cfg, open(os.path.join(td, "config.json"), "w"))
            save_file({f"model.layers.{i}.self_attn.{c}_scale":
                       torch.tensor([0.01 * (i + 1) * (2.0 if c == "v" else 1.0)])
                       for i in range(NL) for c in "kv"},
                      os.path.join(td, "model.safetensors"))
            from minisgl.utils.hf import _load_hf_config
            _load_hf_config.cache_clear()   # @functools.cache on the path -> must reset per rewrite

        write_ckpt({"num_bits": 8, "type": "float", "strategy": "tensor", "symmetric": True})
        rs = resolve_kv_fp8_scales(td)
        ok("resolves the checkpoint's kv_cache_scheme scales", rs is not None and len(rs.scales) == NL,
           "" if rs is None else f"{len(rs.scales)} layers from {rs.source}")
        ok("checkpoint scales are per-TENSOR (numel 1)", rs is not None and not rs.per_head)

        from minisgl.models import ModelConfig
        from minisgl.utils import cached_load_hf_config as _clhc
        mc = ModelConfig.from_hf(_clhc(td))
        p8 = MHAKVCache(num_kv_heads=H, num_layers=NL, head_dim=D, num_pages=P, page_size=PS,
                        dtype=torch.float8_e4m3fn, device=torch.device(DEV))
        install_kv_fp8_scales(td, mc, p8, None)
        want = torch.tensor([0.01 * (i + 1) for i in range(NL)], device=DEV)
        ok("k_descale broadcast to every head, per layer",
           torch.allclose(p8.k_descale, want.view(-1, 1).expand(NL, H), rtol=1e-6),
           f"layer0={p8.k_descale[0, 0].item():.4g} layer3={p8.k_descale[3, 0].item():.4g}")
        ok("v_descale is the checkpoint's v_scale (not k's)",
           torch.allclose(p8.v_descale, (2 * want).view(-1, 1).expand(NL, H), rtol=1e-6))
        ok("k_inv_scale == 1/k_descale (store and read share the table)",
           torch.allclose(p8.k_inv_scale, 1.0 / p8.k_descale, rtol=1e-6))
        del p8

        # An int8 kv_cache_scheme calibrates amax/127, not amax/448 — believing it would store
        # everything at 28% of range. The resolver must REFUSE, not silently misapply.
        write_ckpt({"num_bits": 8, "type": "int", "strategy": "tensor", "symmetric": True})
        ok("REFUSES scales calibrated for a non-e4m3 grid", resolve_kv_fp8_scales(td) is None)
        # No scheme at all -> nothing, so the engine takes the warned identity fallback.
        write_ckpt(None)
        ok("no kv_cache_scheme -> no scales (warned identity fallback)",
           resolve_kv_fp8_scales(td) is None)

    print("\n" + (f"FAILED ({len(FAILS)}): " + ", ".join(FAILS) if FAILS else "ALL GREEN"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
