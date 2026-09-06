"""Correctness gate for `libcpumoe.so`: the VNNI tiled core against a float64 oracle, on REAL
checkpoint bytes, with no GPU.

WHAT THIS PROVES, AND WHY EACH PIECE IS NEEDED
  The serving path is  checkpoint -> loader fold -> post_load -> [CPU tier: pack + VNNI core].
  A layout error anywhere in that chain produces *plausible* numbers, not a crash: a wrong nibble
  order still decodes to a valid E2M1 magnitude, a wrong scale gather still multiplies by a real
  scale. So the test reconstructs the ENGINE's resident tensors with the engine's OWN code
  (`fold_nvfp4_scale`, `convert_nvfp4_moe`), packs them with the `.so`, and compares against a
  float64 dequant of the SAME folded scales.

  The oracle is deliberately built from the FOLDED fp16 scales, not from the raw e4m3+global. That
  isolates what this library is responsible for (tiling + int8 activations) from what the loader
  already owns (the fold's own fp16 rounding, measured elsewhere at 3.577e-04).

  Two reference levels, because they answer different questions:
    ACT_F64  — float64 activations. Bounds the LAYOUT: any tiling/gather/merge error shows here as
               a huge error. Expected ~8e-3 (the int8 activation cost) and nothing more.
    exact    — the same float64 math with the activation ALSO quantized to int8 the way the core
               does. Bounds the ARITHMETIC: this must agree to ~1e-6, and a miss means the core's
               bias correction or scale folding is wrong, not that int8 is lossy.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "/engine/python")
from minisgl.quant.nvfp4 import convert_nvfp4_moe, fold_nvfp4_scale  # noqa: E402

E2M1 = np.array([0.0, .5, 1., 1.5, 2., 3., 4., 6., -0.0, -.5, -1., -1.5, -2., -3., -4., -6.],
                dtype=np.float64)


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


def dequant_f64(codes_u8, scales_fp16):
    """(N, K/2) packed uint8 + (N, K/16) fp16 -> (N, K) float64. The oracle's weight matrix."""
    n, kh = codes_u8.shape
    k = kh * 2
    c = np.empty((n, k), dtype=np.uint8)
    c[:, 0::2] = codes_u8 & 0x0F
    c[:, 1::2] = codes_u8 >> 4
    w = E2M1[c]
    s = scales_fp16.astype(np.float64)
    return w * np.repeat(s, 16, axis=1)


def quant_act_int8(x):
    """Exactly `moe_core.hpp::quantize_act_g16`: symmetric, per group of 16, amax/127, RNE."""
    g = x.reshape(-1, 16)
    amax = np.abs(g).max(axis=1)
    inv = np.where(amax > 0, 127.0 / np.maximum(amax, 1e-300), 0.0)
    q = np.rint(g * inv[:, None]).clip(-128, 127)
    sc = amax / 127.0
    return q, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard",
                    default="/model/layer-00000-experts-0000-0127.safetensors")
    ap.add_argument("--so", default="/engine/tools/cpu_moe/libcpumoe.so")
    ap.add_argument("--experts", type=int, default=16, help="how many to make resident")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tp", type=int, default=2, help="shard the intermediate dim like the serve")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

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
          f"top_k={a.topk}")

    # ---- reconstruct the ENGINE's resident tensors, with the engine's own code ----------------
    # rank 0's shard: gate/up rows [0, I), down columns [0, I).
    w13_packed = torch.empty((E, 2 * I, H // 2), dtype=torch.uint8)
    w13_scale = torch.empty((E, 2 * I, H // 16), dtype=torch.float16)
    w2_packed = torch.empty((E, H, I // 2), dtype=torch.uint8)
    w2_scale = torch.empty((E, H, I // 16), dtype=torch.float16)
    for e in range(E):
        for j, p in enumerate(("gate_proj", "up_proj")):
            w = t[f"{pre}.{e}.{p}.weight"][:I]
            s = fold_nvfp4_scale(t[f"{pre}.{e}.{p}.weight_scale"][:I],
                                 t[f"{pre}.{e}.{p}.weight_scale_2"],
                                 global_field="weight_scale_2")
            w13_packed[e, j * I:(j + 1) * I] = w
            w13_scale[e, j * I:(j + 1) * I] = s
        w = t[f"{pre}.{e}.down_proj.weight"][:, : I // 2]
        s = fold_nvfp4_scale(t[f"{pre}.{e}.down_proj.weight_scale"][:, : I // 16],
                             t[f"{pre}.{e}.down_proj.weight_scale_2"],
                             global_field="weight_scale_2")
        w2_packed[e] = w
        w2_scale[e] = s

    # post_load: (E,N,K//8) int32 codes + (E,K//16,N) fp16 GROUP-MAJOR scales
    c13 = convert_nvfp4_moe(w13_packed, w13_scale)
    c2 = convert_nvfp4_moe(w2_packed, w2_scale)
    w13_op = c13["w_packed"].contiguous()
    s13_op = c13["scales"].transpose(1, 2).contiguous()
    w2_op = c2["w_packed"].contiguous()
    s2_op = c2["scales"].transpose(1, 2).contiguous()
    print(f"[resident] w13 codes {tuple(w13_op.shape)} {w13_op.dtype} scales "
          f"{tuple(s13_op.shape)} | w2 codes {tuple(w2_op.shape)} scales {tuple(s2_op.shape)}")

    # The int32 view is byte-identical to the (N, K/2) uint8 packing — assert it rather than
    # assume it, because the whole pack entry point depends on it.
    assert torch.equal(w13_op.view(torch.uint8).reshape(E, 2 * I, H // 2), w13_packed), \
        "int32 op layout is NOT a byte-view of the uint8 nibble packing"
    print("[check] _w_op int32 view == weight_packed uint8 : PASS")

    # ---- pack in place ------------------------------------------------------------------------
    lib = ctypes.CDLL(a.so)
    lib.cpu_moe_pack_fp16.restype = ctypes.c_int
    lib.cpu_moe_pack_fp16.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_void_p]
    lib.cpu_moe_open.restype = ctypes.c_void_p
    lib.cpu_moe_open.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_int]
    lib.cpu_moe_run.restype = ctypes.c_int
    lib.cpu_moe_run.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_void_p] * 3 + \
                               [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.cpu_moe_calls.restype = ctypes.c_longlong
    lib.cpu_moe_calls.argtypes = [ctypes.c_void_p]

    # keep an UNPACKED copy for the oracle before the in-place permutation destroys the layout
    ref13_codes = w13_packed.numpy().copy()
    ref13_scale = w13_scale.numpy().copy()
    ref2_codes = w2_packed.numpy().copy()
    ref2_scale = w2_scale.numpy().copy()

    scratch = np.empty(max(2 * I * H // 2, H * I // 2, 2 * I * H // 16 * 2), dtype=np.uint8)
    sp = scratch.ctypes.data_as(ctypes.c_void_p)
    for e in range(E):
        rc = lib.cpu_moe_pack_fp16(ctypes.c_void_p(w13_op[e].data_ptr()),
                                   ctypes.c_void_p(s13_op[e].data_ptr()), 2 * I, H, sp)
        assert rc == 0, rc
        rc = lib.cpu_moe_pack_fp16(ctypes.c_void_p(w2_op[e].data_ptr()),
                                   ctypes.c_void_p(s2_op[e].data_ptr()), H, I, sp)
        assert rc == 0, rc
    print(f"[pack] {E} experts tiled in place (zero extra resident bytes)")

    cpus = (ctypes.c_int * a.threads)(*range(2, 2 + a.threads))
    h = lib.cpu_moe_open(H, I, a.topk, a.threads, cpus, a.threads)
    assert h, "cpu_moe_open returned NULL"

    # ---- drive it ------------------------------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    M = a.tokens
    x = (rng.standard_normal((M, H)) * 0.02).astype(np.float32)
    ids = np.stack([rng.choice(E, size=a.topk, replace=False) for _ in range(M)]).astype(np.int32)
    rw = rng.random((M, a.topk)).astype(np.float32)
    rw /= rw.sum(axis=1, keepdims=True)
    out = np.zeros((M, H), dtype=np.float32)

    rc = lib.cpu_moe_run(
        ctypes.c_void_p(h),
        ctypes.c_void_p(w13_op.data_ptr()), ctypes.c_void_p(s13_op.data_ptr()),
        ctypes.c_void_p(w2_op.data_ptr()), ctypes.c_void_p(s2_op.data_ptr()),
        x.ctypes.data_as(ctypes.c_void_p), ids.ctypes.data_as(ctypes.c_void_p),
        rw.ctypes.data_as(ctypes.c_void_p), M, a.topk, out.ctypes.data_as(ctypes.c_void_p))
    assert rc == 0, f"cpu_moe_run rc={rc}"
    print(f"[counter] cpu_moe_calls={lib.cpu_moe_calls(ctypes.c_void_p(h))}")

    # ---- oracles -------------------------------------------------------------------------------
    W13 = np.stack([dequant_f64(ref13_codes[e], ref13_scale[e]) for e in range(E)])
    W2 = np.stack([dequant_f64(ref2_codes[e], ref2_scale[e]) for e in range(E)])

    def oracle(xrow, sel, w, act_int8: bool):
        y = np.zeros(H, dtype=np.float64)
        if act_int8:
            q, sc = quant_act_int8(xrow.astype(np.float64))
            xu = (q * sc[:, None]).reshape(-1)
        else:
            xu = xrow.astype(np.float64)
        for e, ww in zip(sel, w):
            gu = W13[e] @ xu
            g, u = gu[:I], gu[I:]
            hact = g / (1.0 + np.exp(-g)) * u
            if act_int8:
                qh, sch = quant_act_int8(hact)
                hact = (qh * sch[:, None]).reshape(-1)
            y += ww * (W2[e] @ hact)
        return y

    def rel_rms(got, ref):
        return float(np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))

    r_f64 = [rel_rms(out[m].astype(np.float64), oracle(x[m], ids[m], rw[m], False))
             for m in range(M)]
    r_i8 = [rel_rms(out[m].astype(np.float64), oracle(x[m], ids[m], rw[m], True))
            for m in range(M)]
    print(f"\n[rel_rms vs float64-activation oracle ] {np.mean(r_f64):.6e}  "
          f"(per token: {', '.join(f'{v:.3e}' for v in r_f64)})")
    print(f"[rel_rms vs int8-activation   oracle ] {np.mean(r_i8):.6e}  "
          f"(per token: {', '.join(f'{v:.3e}' for v in r_i8)})")

    ok_layout = np.mean(r_f64) < 3e-2
    ok_exact = np.mean(r_i8) < 5e-6
    print(f"\nLAYOUT  (< 3e-2, the int8 activation cost) : {'PASS' if ok_layout else 'FAIL'}")
    print(f"EXACT   (< 5e-6 vs its own int8 twin)      : {'PASS' if ok_exact else 'FAIL'}")
    lib.cpu_moe_close(ctypes.c_void_p(h))
    return 0 if (ok_layout and ok_exact) else 1


if __name__ == "__main__":
    sys.exit(main())
