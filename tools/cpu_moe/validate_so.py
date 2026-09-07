"""Correctness gate for `libcpumoe.so`: the VNNI tiled core against a float64 oracle, on REAL
checkpoint bytes, with no GPU required.

WHAT THIS PROVES, AND WHY EACH PIECE IS NEEDED
  The serving path is  checkpoint -> loader split/fold -> post_load -> [CPU tier: pack + VNNI core].
  A layout error anywhere in that chain produces *plausible* numbers, not a crash: a wrong nibble
  order still decodes to a valid E2M1 magnitude, a wrong scale gather still multiplies by a real
  scale, and a dropped per-channel global just makes every weight uniformly small. So the test
  reconstructs the ENGINE's resident tensors with the engine's OWN code (`split_nvfp4_scale` /
  `fold_nvfp4_scale`, `convert_nvfp4_moe`), packs them with the `.so`, and compares against a
  float64 dequant of the SAME bytes.

TWO POLICIES, BECAUSE THE ENGINE HAS HELD TWO LAYOUTS
  --policy e4m3  `vnni_nvfp4_e4m3_g16`: the checkpoint's own 1-byte e4m3 block scale plus the
                 per-OUTPUT-CHANNEL f32 global (`_scales_op` + `_global_op`). THIS IS WHAT AN
                 NVFP4 SERVE USES. The oracle is the exact two-level product in float64, so this
                 measures the CPU tier against the checkpoint itself with no fold in between.
  --policy fp16  `vnni_nvfp4_fp16_g16`: the legacy single folded fp16 group scale. Kept as the A/B
                 comparand every RESULTS_*.txt number was measured against. Its oracle is built
                 from the FOLDED scales, which isolates what this library owns (tiling + int8
                 activations) from the fold's own fp16 rounding.

  THREE REFERENCE LEVELS, because they answer different questions:
    ACT_F64   — float64 activations. Bounds the LAYOUT: any tiling/gather/merge/global error shows
                here as a huge error. Expected ~8e-3 (the int8 activation cost) and nothing more.
    exact     — the same float64 math with the activation ALSO quantized to int8 the way the core
                does. Bounds the ARITHMETIC: this must agree to ~1e-6, and a miss means the core's
                bias correction or scale folding is wrong, not that int8 is lossy.
    no-global — (e4m3 only) the same oracle with the per-channel global DROPPED. Not a pass
                criterion; it is the number that says how load-bearing the global is, i.e. what a
                policy without a second scale level reading these bytes would have served. It must
                be enormous.

  Also reported, because the fast scale decode depends on it: the census of the e4m3 scale BYTES
  actually present. `e4m3x16_normpos_to_ps` is specialised to positive-normal e4m3 (0x08..0x7E) —
  three ops instead of nineteen — and the packer refuses outside that range. A checkpoint whose
  bytes leave it needs the general decode (`build_e4m3_lut`), not a widened guard.

--gpu additionally runs the SAME experts, the same routing and the same activations through
`kernels.w4a8_moe`, i.e. the device path this tier offloads FROM, and scores BOTH legs against the
one float64 oracle. That is what separates "the CPU tier is less accurate than the GPU" from "the
CPU tier is broken". It needs a GPU and therefore a lease.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import struct
import sys

import numpy as np
import torch

for _p in ("/engine/python", "/opt/minisgl/python"):
    if _p not in sys.path:
        sys.path.append(_p)
from minisgl.quant.nvfp4 import (  # noqa: E402
    convert_nvfp4_moe,
    fold_nvfp4_scale,
    split_nvfp4_scale,
)

E2M1 = np.array([0.0, .5, 1., 1.5, 2., 3., 4., 6., -0.0, -.5, -1., -1.5, -2., -3., -4., -6.],
                dtype=np.float64)

#: The checkpoint's spelling of the NVFP4 per-tensor global. It is a MULTIPLIER here (modelopt),
#: the reciprocal of what compressed-tensors ships, and the NAME is the only tell — so it is passed
#: to the engine's own `split`/`fold` rather than reimplemented (see quant/nvfp4.py).
GLOBAL_FIELD = "weight_scale_2"


def e4m3_lut() -> np.ndarray:
    """float8_e4m3fn (bias 7, no inf, 0x7F/0xFF = NaN) -> float64, 256 entries.

    The GENERAL decode, deliberately: it is the scalar twin of `WLoadVnniE4m3::scale_ref`, not of
    the specialised `e4m3x16_normpos_to_ps` the core's inner loop uses. An oracle that inherited the
    fast path's positive-normal precondition could not detect a violation of it.
    """
    out = np.empty(256, dtype=np.float64)
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        e, m = (b >> 3) & 0xF, b & 0x7
        if e == 0:
            v = m / 8.0 * 2.0 ** -6
        elif e == 15 and m == 7:
            v = 0.0  # NaN slot; never present in a real weight_scale
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        out[b] = s * v
    return out


E4M3 = e4m3_lut()


def read_shard(path, names):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        out = {}
        for k in names:
            m = hdr[k]
            fh.seek(base + m["data_offsets"][0])
            raw = fh.read(m["data_offsets"][1] - m["data_offsets"][0])
            dt = {"U8": np.uint8, "F32": np.float32, "F8_E4M3": np.uint8, "BF16": np.uint16}[
                m["dtype"]]
            a = np.frombuffer(raw, dtype=dt).reshape(m["shape"] if m["shape"] else ())
            if m["dtype"] == "F8_E4M3":
                a = torch.from_numpy(a.copy()).view(torch.float8_e4m3fn)
            else:
                a = torch.from_numpy(a.copy())
            out[k] = a
    return out


def unpack_codes(codes_u8):
    """(N, K/2) packed uint8 -> (N, K) uint8 codes, low nibble first."""
    n, kh = codes_u8.shape
    c = np.empty((n, kh * 2), dtype=np.uint8)
    c[:, 0::2] = codes_u8 & 0x0F
    c[:, 1::2] = codes_u8 >> 4
    return c


def dequant_fp16(codes_u8, scales_fp16):
    """(N, K/2) codes + (N, K/16) fp16 folded group scale -> (N, K) float64."""
    return E2M1[unpack_codes(codes_u8)] * np.repeat(scales_fp16.astype(np.float64), 16, axis=1)


def dequant_e4m3(codes_u8, block_u8, gvec, *, drop_global=False):
    """The EXACT two-level NVFP4 product in float64: E2M1[code] * e4m3(block) * global[n].

    `drop_global` is the negative control — what serving these bytes through a policy with no second
    level would compute. It is finite and wrong, which is the whole reason this file exists.
    """
    w = E2M1[unpack_codes(codes_u8)] * np.repeat(E4M3[block_u8], 16, axis=1)
    return w if drop_global else w * gvec.astype(np.float64)[:, None]


def quant_act_int8(x):
    """Exactly `moe_core.hpp::quantize_act_g16`: symmetric, per group of 16, amax/127, RNE."""
    g = x.reshape(-1, 16)
    amax = np.abs(g).max(axis=1)
    inv = np.where(amax > 0, 127.0 / np.maximum(amax, 1e-300), 0.0)
    q = np.rint(g * inv[:, None]).clip(-128, 127)
    return q, amax / 127.0


def rel_rms(got, ref):
    return float(np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="/model/layer-00000-experts-0000-0127.safetensors")
    ap.add_argument("--so", default="/engine/tools/cpu_moe/libcpumoe.so")
    ap.add_argument("--policy", choices=("e4m3", "fp16"), default="e4m3")
    ap.add_argument("--experts", type=int, default=16, help="how many to make resident")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tp", type=int, default=2, help="shard the intermediate dim like the serve")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpu", action="store_true",
                    help="also run kernels.w4a8_moe on the same experts (needs a GPU + a lease)")
    a = ap.parse_args()
    e4m3 = a.policy == "e4m3"

    pre = "model.language_model.layers.0.mlp.experts"
    E = a.experts
    names = []
    for e in range(E):
        for p in ("gate_proj", "up_proj", "down_proj"):
            for f in ("weight", "weight_scale", "weight_scale_2"):
                names.append(f"{pre}.{e}.{p}.{f}")
    t = read_shard(a.shard, names)

    H = t[f"{pre}.0.gate_proj.weight"].shape[1] * 2   # K of gate = hidden
    I_full = t[f"{pre}.0.gate_proj.weight"].shape[0]
    I = I_full // a.tp                                 # TP shards the intermediate dim
    print(f"[shapes] hidden={H} inter_full={I_full} inter_local={I} (tp={a.tp}) experts={E} "
          f"top_k={a.topk} policy={a.policy}")

    # ---- reconstruct the ENGINE's resident tensors, with the engine's own code ----------------
    # rank 0's shard: gate/up rows [0, I), down columns [0, I). The gate|up merge is `cat(dim=0)`
    # on BOTH the block scale and the global, exactly as `models/weight.py::_leaf` does it — which
    # is the whole reason the global has to be a per-output-channel vector rather than a scalar.
    sdt = torch.float8_e4m3fn if e4m3 else torch.float16
    w13_packed = torch.empty((E, 2 * I, H // 2), dtype=torch.uint8)
    w13_scale = torch.empty((E, 2 * I, H // 16), dtype=sdt)
    w13_glob = torch.empty((E, 2 * I), dtype=torch.float32)
    w2_packed = torch.empty((E, H, I // 2), dtype=torch.uint8)
    w2_scale = torch.empty((E, H, I // 16), dtype=sdt)
    w2_glob = torch.empty((E, H), dtype=torch.float32)

    def leaf(scale, glob):
        if e4m3:
            return split_nvfp4_scale(scale, glob, global_field=GLOBAL_FIELD)
        # The fold absorbs the global into the fp16 group scale, so policy E's second level is 1.
        return (fold_nvfp4_scale(scale, glob, global_field=GLOBAL_FIELD),
                torch.ones(scale.shape[0], dtype=torch.float32))

    for e in range(E):
        for j, p in enumerate(("gate_proj", "up_proj")):
            s, g = leaf(t[f"{pre}.{e}.{p}.weight_scale"][:I], t[f"{pre}.{e}.{p}.weight_scale_2"])
            w13_packed[e, j * I:(j + 1) * I] = t[f"{pre}.{e}.{p}.weight"][:I]
            w13_scale[e, j * I:(j + 1) * I] = s
            w13_glob[e, j * I:(j + 1) * I] = g
        s, g = leaf(t[f"{pre}.{e}.down_proj.weight_scale"][:, : I // 16],
                    t[f"{pre}.{e}.down_proj.weight_scale_2"])
        w2_packed[e] = t[f"{pre}.{e}.down_proj.weight"][:, : I // 2]
        w2_scale[e] = s
        w2_glob[e] = g

    # post_load: (E,N,K//8) int32 codes + (E,K//16,N) GROUP-MAJOR scales + (E,N) f32 global
    c13, c2 = convert_nvfp4_moe(w13_packed, w13_scale), convert_nvfp4_moe(w2_packed, w2_scale)
    w13_op, w2_op = c13["w_packed"].contiguous(), c2["w_packed"].contiguous()
    s13_op = c13["scales"].transpose(1, 2).contiguous()
    s2_op = c2["scales"].transpose(1, 2).contiguous()
    g13_op, g2_op = w13_glob.contiguous(), w2_glob.contiguous()
    print(f"[resident] w13 codes {tuple(w13_op.shape)} {w13_op.dtype} scales "
          f"{tuple(s13_op.shape)} {s13_op.dtype} global {tuple(g13_op.shape)} | "
          f"w2 codes {tuple(w2_op.shape)} scales {tuple(s2_op.shape)}")

    # The int32 view is byte-identical to the (N, K/2) uint8 packing — assert it rather than
    # assume it, because the whole pack entry point depends on it.
    assert torch.equal(w13_op.view(torch.uint8).reshape(E, 2 * I, H // 2), w13_packed), \
        "int32 op layout is NOT a byte-view of the uint8 nibble packing"
    print("[check] _w_op int32 view == weight_packed uint8 : PASS")

    # UNPACKED copies for the oracle, and (for --gpu) for the device leg: the pack below permutes
    # the resident tensors IN PLACE, so nothing that wants the row-major layout may run after it.
    ref13_codes, ref2_codes = w13_packed.numpy().copy(), w2_packed.numpy().copy()
    dev_copies = ((w13_op.clone(), s13_op.clone(), g13_op.clone(),
                   w2_op.clone(), s2_op.clone(), g2_op.clone()) if a.gpu else None)
    if e4m3:
        ref13_scale = w13_scale.view(torch.uint8).numpy().copy()
        ref2_scale = w2_scale.view(torch.uint8).numpy().copy()
        # THE FAST DECODE'S PRECONDITION, on the bytes actually served. The census that established
        # 0x08..0x7E covered layers 0-3 of one checkpoint; this asks the same question of the
        # tensors in front of us, and reports the answer whether it passes or not.
        allb = np.concatenate([ref13_scale.ravel(), ref2_scale.ravel()])
        n_bad = int(((allb < 0x08) | (allb > 0x7E)).sum())
        print(f"[e4m3 census] {allb.size:,} scale bytes, range 0x{int(allb.min()):02X}.."
              f"0x{int(allb.max()):02X}, outside positive-normal 0x08..0x7E: {n_bad}")
        gmax = float(g13_op.max())
        print(f"[global] w13 {float(g13_op.min()):.6e}..{gmax:.6e}  "
              f"w2 {float(g2_op.min()):.6e}..{float(g2_op.max()):.6e}  (a MULTIPLIER; dropping it "
              f"is a ~{1 / gmax:.3g}x error per weight)")
    else:
        ref13_scale, ref2_scale = w13_scale.numpy().copy(), w2_scale.numpy().copy()

    # ---- pack in place ------------------------------------------------------------------------
    lib = ctypes.CDLL(a.so)
    vp, ci = ctypes.c_void_p, ctypes.c_int
    lib.cpu_moe_policies.restype = ctypes.c_char_p
    print(f"[so] {a.so} advertises: {lib.cpu_moe_policies().decode()}")
    pack = lib.cpu_moe_pack_e4m3 if e4m3 else lib.cpu_moe_pack_fp16
    pack.restype, pack.argtypes = ci, [vp, vp, ci, ci, vp]
    lib.cpu_moe_open2.restype, lib.cpu_moe_open2.argtypes = vp, [ci] * 4 + [vp, ci, ci]
    lib.cpu_moe_run2.restype, lib.cpu_moe_run2.argtypes = ci, [vp] * 10 + [ci, ci, vp]
    lib.cpu_moe_policy_name.restype, lib.cpu_moe_policy_name.argtypes = ctypes.c_char_p, [vp]
    lib.cpu_moe_calls.restype, lib.cpu_moe_calls.argtypes = ctypes.c_longlong, [vp]
    lib.cpu_moe_close.argtypes = [vp]

    scratch = np.empty(max(2 * I * H // 2, H * I // 2, 2 * I * H // 16 * 2), dtype=np.uint8)
    sp = scratch.ctypes.data_as(vp)
    for e in range(E):
        for op, sop, n, k in ((w13_op, s13_op, 2 * I, H), (w2_op, s2_op, H, I)):
            rc = pack(vp(op[e].data_ptr()), vp(sop[e].data_ptr()), n, k, sp)
            assert rc == 0, (f"pack(N={n},K={k}) rc={rc}" + (
                f" — e4m3 scale byte 0x{-rc - 1:02X} outside 0x08..0x7E" if rc < 0 else ""))
    print(f"[pack] {E} experts tiled in place (zero extra resident bytes)")

    cpus = (ci * a.threads)(*range(2, 2 + a.threads))
    h = lib.cpu_moe_open2(H, I, a.topk, a.threads, cpus, a.threads, 1 if e4m3 else 0)
    assert h, "cpu_moe_open2 returned NULL"
    print(f"[open] handle policy = {lib.cpu_moe_policy_name(vp(h)).decode()}")

    # ---- drive it ------------------------------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    M = a.tokens
    x = (rng.standard_normal((M, H)) * 0.02).astype(np.float32)
    ids = np.stack([rng.choice(E, size=a.topk, replace=False) for _ in range(M)]).astype(np.int32)
    rw = rng.random((M, a.topk)).astype(np.float32)
    rw /= rw.sum(axis=1, keepdims=True)
    out = np.zeros((M, H), dtype=np.float32)

    gp = (lambda tt: vp(tt.data_ptr())) if e4m3 else (lambda tt: vp(None))
    rc = lib.cpu_moe_run2(
        vp(h),
        vp(w13_op.data_ptr()), vp(s13_op.data_ptr()), gp(g13_op),
        vp(w2_op.data_ptr()), vp(s2_op.data_ptr()), gp(g2_op),
        x.ctypes.data_as(vp), ids.ctypes.data_as(vp), rw.ctypes.data_as(vp),
        M, a.topk, out.ctypes.data_as(vp))
    assert rc == 0, f"cpu_moe_run2 rc={rc}"
    print(f"[counter] cpu_moe_calls={lib.cpu_moe_calls(vp(h))}")

    # ---- oracles -------------------------------------------------------------------------------
    def build(codes, scales, glob, drop_global=False):
        if e4m3:
            return np.stack([dequant_e4m3(codes[e], scales[e], glob[e].numpy(),
                                          drop_global=drop_global) for e in range(E)])
        return np.stack([dequant_fp16(codes[e], scales[e]) for e in range(E)])

    W13, W2 = build(ref13_codes, ref13_scale, g13_op), build(ref2_codes, ref2_scale, g2_op)
    print(f"[oracle] |W13| mean {np.abs(W13).mean():.6e} max {np.abs(W13).max():.6e}")

    def oracle(xrow, sel, w, act_int8: bool, A=None, B=None):
        A, B = (W13 if A is None else A), (W2 if B is None else B)
        y = np.zeros(H, dtype=np.float64)
        if act_int8:
            q, sc = quant_act_int8(xrow.astype(np.float64))
            xu = (q * sc[:, None]).reshape(-1)
        else:
            xu = xrow.astype(np.float64)
        for e, ww in zip(sel, w):
            gu = A[e] @ xu
            g, u = gu[:I], gu[I:]
            hact = g / (1.0 + np.exp(-g)) * u
            if act_int8:
                qh, sch = quant_act_int8(hact)
                hact = (qh * sch[:, None]).reshape(-1)
            y += ww * (B[e] @ hact)
        return y

    ref_f64 = [oracle(x[m], ids[m], rw[m], False) for m in range(M)]
    ref_i8 = [oracle(x[m], ids[m], rw[m], True) for m in range(M)]
    r_f64 = [rel_rms(out[m].astype(np.float64), ref_f64[m]) for m in range(M)]
    r_i8 = [rel_rms(out[m].astype(np.float64), ref_i8[m]) for m in range(M)]
    print(f"\n[rel_rms vs float64-activation oracle ] {np.mean(r_f64):.6e}  "
          f"(per token: {', '.join(f'{v:.3e}' for v in r_f64)})")
    print(f"[rel_rms vs int8-activation   oracle ] {np.mean(r_i8):.6e}  "
          f"(per token: {', '.join(f'{v:.3e}' for v in r_i8)})")

    ok_layout = np.mean(r_f64) < 3e-2
    ok_exact = np.mean(r_i8) < 5e-6
    print(f"\nLAYOUT  (< 3e-2, the int8 activation cost) : {'PASS' if ok_layout else 'FAIL'}")
    print(f"EXACT   (< 5e-6 vs its own int8 twin)      : {'PASS' if ok_exact else 'FAIL'}")

    if e4m3:
        # NEGATIVE CONTROL, not a pass criterion: how wrong is a policy with no second level? This
        # is the error `cpu_native.pack_layer`'s refusal was protecting against, measured rather
        # than asserted.
        Wn13 = build(ref13_codes, ref13_scale, g13_op, True)
        Wn2 = build(ref2_codes, ref2_scale, g2_op, True)
        r_ng = np.mean([rel_rms(oracle(x[m], ids[m], rw[m], False, Wn13, Wn2), ref_f64[m])
                        for m in range(M)])
        print(f"CONTROL global DROPPED -> rel_rms {r_ng:.6e}  (must be enormous: this is what a "
              f"policy without the second scale level would have served)")

    ok_gpu = True
    if a.gpu:
        ok_gpu = gpu_leg(dev_copies, I, H, x, ids, rw, out, ref_f64, e4m3)
    else:
        print("\n[gpu] skipped (--gpu not given): the DEVICE comparand is not measured here.")

    lib.cpu_moe_close(vp(h))
    return 0 if (ok_layout and ok_exact and ok_gpu) else 1


def gpu_leg(dev_copies, I, H, x, ids, rw, cpu_out, ref_f64, e4m3):
    """The SAME experts through `kernels.w4a8_moe` — the device path this tier offloads FROM.

    Not a second oracle: the point is that BOTH legs are scored against the ONE float64 oracle, so
    "the CPU tier is less accurate than the GPU" and "the CPU tier is broken" stop being the same
    observation. The GPU path quantizes activations to fp8 per token; the CPU path to int8 per group
    of 16. Neither is the reference.

    It runs on tensors CLONED BEFORE the pack, because the pack permutes the resident bytes in
    place and the kernel reads row-major.
    """
    from minisgl.distributed.info import set_tp_info, try_get_tp_info
    from minisgl.quant import kernels

    # The `engaged()` dispatch ledger logs rank-0-only and therefore needs TP coordinates. A probe
    # is not a serve, so nothing has set them; TP=1 is the truth for this single-process leg (the
    # SHARDING is already baked into the tensors above by `--tp`).
    if try_get_tp_info() is None:
        set_tp_info(0, 1)

    if not e4m3:
        print("\n[gpu] skipped: the device NVFP4 path consumes the two-level scale; the fp16-fold "
              "policy has no live device comparand.")
        return True
    w13_op, s13_op, g13_op, w2_op, s2_op, g2_op = dev_copies
    dev = torch.device("cuda")
    M, topk = ids.shape
    y = kernels.w4a8_moe(
        torch.from_numpy(x).to(dev).to(torch.bfloat16),
        w13_op.to(dev), s13_op.to(dev), g13_op.contiguous().view(torch.int32).to(dev),
        w2_op.to(dev), s2_op.to(dev), g2_op.contiguous().view(torch.int32).to(dev),
        None, topk, False,
        topk_weights=torch.from_numpy(rw).to(dev),
        topk_ids=torch.from_numpy(ids).to(dev).to(torch.int32),
        weight_is_e2m1=True, activation="silu",
    )
    g = y.to(torch.float32).cpu().numpy().astype(np.float64)
    r_gpu = np.mean([rel_rms(g[m], ref_f64[m]) for m in range(M)])
    r_cpu = np.mean([rel_rms(cpu_out[m].astype(np.float64), ref_f64[m]) for m in range(M)])
    r_x = np.mean([rel_rms(cpu_out[m].astype(np.float64), g[m]) for m in range(M)])
    print(f"\n[gpu] rel_rms(GPU w4a8_moe, float64 oracle) = {r_gpu:.6e}")
    print(f"[gpu] rel_rms(CPU tier    , float64 oracle) = {r_cpu:.6e}")
    print(f"[gpu] rel_rms(CPU tier    , GPU w4a8_moe  ) = {r_x:.6e}")
    # The CPU tier must be in the same league as the thing it replaces, not merely finite. A tier
    # an order of magnitude worse than the device path is a defect even if it "looks fine".
    ok = bool(r_cpu < 10 * max(r_gpu, 1e-9))
    print(f"GPU-RELATIVE (CPU within 10x of the device path) : {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(main())
