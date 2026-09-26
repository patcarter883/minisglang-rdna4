"""KV-cache format fidelity on REAL attention tensors, against the production per-head fp8 cache.

Consumes the Q/K/V capture from tools/kvarn_kv_capture.py (archive/kvarn-eval) and scores each
format by what attention does with it: score cosine (q.K), attention-output cosine (softmax . V) and
K/V relative error, per layer and rank. Host-side, CPU.

Formats (all quantize each token's per-head vector along head_dim, as a cache would store it):
    fp8        per-head calibrated e4m3 (the sidecar scales) — the bar
    mxfp4      e2m1, one power-of-two (e8m0) scale per 32 elements (OCP MX)       4.25 bit
    nvfp4      e2m1, e4m3 scale per 16 elements x fp32 per-head scale              4.5 bit
    had_nvfp4  randomized Hadamard rotation, then nvfp4                            4.5 bit
    tq_mse_bN  TurboQuant-MSE: per-vector fp16 norm, randomized Hadamard, N-bit Lloyd-Max codebook
               for the rotated coordinates                                           N + 16/D bit
    tq_prod_b4 TurboQuant-prod on K (3-bit MSE + 1-bit QJL sign sketch of the residual, unbiased
               inner products), TurboQuant-MSE 4-bit on V                            4 + 32/D bit (K)

    python tools/kv_quant_fidelity_eval.py --capture-dir /home/pat/fixtures/minisgl-kv-calib/kvarn_capture \\
        --sidecar kv_scales_qwen35b_tp2.safetensors --report kv_quant_fidelity_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 value, saturating at 6 (magnitudes; sign kept)."""
    a = x.abs().clamp(max=6.0)
    idx = (a.unsqueeze(-1) - E2M1).abs().argmin(-1)
    return E2M1[idx] * x.sign()


def mxfp4(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    t, d = x.shape
    b = x.float().view(t, d // block, block)
    amax = b.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
    scale = torch.exp2(torch.floor(torch.log2(amax)) - 2)          # e8m0; e2m1 emax = 2
    return (e2m1(b / scale) * scale).view(t, d)


def nvfp4(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    t, d = x.shape
    xf = x.float()
    g = (xf.abs().max() / (448.0 * 6.0)).clamp_min(1e-12)           # fp32 per-head scale
    b = xf.view(t, d // block, block)
    s = (b.abs().amax(-1, keepdim=True) / (6.0 * g)).to(torch.float8_e4m3fn).float()
    s = torch.where(s > 0, s, torch.ones_like(s))
    return (e2m1(b / (s * g)) * s * g).view(t, d)


def hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def lloyd_max_gaussian(bits: int) -> torch.Tensor:
    """Optimal (MSE) scalar codebook for N(0, 1)."""
    xs = torch.linspace(-8, 8, 400001, dtype=torch.float64)
    w = torch.exp(-xs * xs / 2)
    k = 2 ** bits
    c = torch.linspace(-2.5, 2.5, k, dtype=torch.float64)
    for _ in range(300):
        edges = (c[1:] + c[:-1]) / 2
        idx = torch.bucketize(xs, edges)
        num = torch.zeros(k, dtype=torch.float64).index_add_(0, idx, w * xs)
        den = torch.zeros(k, dtype=torch.float64).index_add_(0, idx, w)
        c = num / den
    return c.float()


class TurboQuant:
    """Randomized-Hadamard rotation shared by every token of a head; per-vector norm."""

    def __init__(self, d: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.d = d
        self.signs = torch.randint(0, 2, (d,), generator=g).float() * 2 - 1
        self.h = hadamard(d)
        self.s = torch.randn(d, d, generator=g)                        # QJL projection
        self.books = {b: lloyd_max_gaussian(b) / math.sqrt(d) for b in (2, 3, 4)}

    def _rot(self, u):
        return (u * self.signs) @ self.h.t()

    def _unrot(self, y):
        return (y @ self.h) * self.signs

    def _q(self, y, bits):
        book = self.books[bits]
        return book[(y.unsqueeze(-1) - book).abs().argmin(-1)]

    def mse(self, x, bits):
        xf = x.float()
        n = xf.norm(dim=-1, keepdim=True).half().float().clamp_min(1e-12)
        y = self._rot(xf / n)
        return self._unrot(self._q(y, bits)) * n

    def prod(self, x, bits):
        xf = x.float()
        n = xf.norm(dim=-1, keepdim=True).half().float().clamp_min(1e-12)
        y = self._rot(xf / n)
        yq = self._q(y, bits - 1)
        r = y - yq
        gamma = r.norm(dim=-1, keepdim=True).half().float()
        z = torch.sign(r @ self.s.t())
        est = yq + gamma * math.sqrt(math.pi / 2) / self.d * (z @ self.s)
        return self._unrot(est) * n


def fp8(x, scale):
    return (x.float() / scale).to(torch.float8_e4m3fn).float() * scale


def metrics(q, k, v, kh, vh, gr):
    nq, hq, d = q.shape
    sm = 1.0 / math.sqrt(d)
    sc, oc = [], []
    cos = torch.nn.functional.cosine_similarity
    for h in range(hq):
        hk = h // gr
        qs = q[:, h].float()
        rs, hs = qs @ k[:, hk].float().t() * sm, qs @ kh[:, hk].t() * sm
        sc.append(cos(rs.flatten(), hs.flatten(), 0).item())
        ro, ho = torch.softmax(rs, -1) @ v[:, hk].float(), torch.softmax(hs, -1) @ vh[:, hk]
        oc.append(cos(ro.flatten(), ho.flatten(), 0).item())
    return sum(sc) / len(sc), sum(oc) / len(oc)


def rel(x, xh):
    return ((x.float() - xh).norm() / x.float().norm().clamp_min(1e-12)).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--max-kv", type=int, default=8192)
    ap.add_argument("--max-q", type=int, default=256)
    args = ap.parse_args()
    from safetensors.torch import load_file

    scales = load_file(args.sidecar)
    torch.manual_seed(0)
    names = ["fp8", "mxfp4", "nvfp4", "had_nvfp4", "tq_mse_b4", "tq_prod_b4", "tq_mse_b3"]
    agg = {n: [] for n in names}
    report = {"layers": []}
    D = None
    for rf in sorted(f for f in os.listdir(args.capture_dir) if f.startswith("kvarn_capture_rank")):
        blob = torch.load(os.path.join(args.capture_dir, rf), map_location="cpu", weights_only=False)
        meta = blob["meta"]
        rank, D = meta["rank"], meta["head_dim"]
        for lid, t in sorted(blob["layers"].items()):
            q = t["q"][: args.max_q]
            hk = t["k"].shape[-1] // D
            k = t["k"][: args.max_kv].view(-1, hk, D)
            v = t["v"][: args.max_kv].view(-1, hk, D)
            gr = q.shape[1] // hk
            gl = meta["compact_to_global"][int(lid)]
            krow = scales[f"model.layers.{gl}.self_attn.k_scale"].float()
            vrow = scales[f"model.layers.{gl}.self_attn.v_scale"].float()
            ks = krow[rank * hk:(rank + 1) * hk] if krow.numel() > hk else krow.expand(hk)
            vs = vrow[rank * hk:(rank + 1) * hk] if vrow.numel() > hk else vrow.expand(hk)
            row = {"rank": rank, "layer": int(lid)}
            for n in names:
                kh = torch.empty(k.shape)
                vh = torch.empty(v.shape)
                for h in range(hk):
                    tq = TurboQuant(D, seed=1000 * gl + 10 * h + rank)
                    kk, vv = k[:, h], v[:, h]
                    if n == "fp8":
                        kh[:, h], vh[:, h] = fp8(kk, float(ks[h])), fp8(vv, float(vs[h]))
                    elif n == "mxfp4":
                        kh[:, h], vh[:, h] = mxfp4(kk), mxfp4(vv)
                    elif n == "nvfp4":
                        kh[:, h], vh[:, h] = nvfp4(kk), nvfp4(vv)
                    elif n == "had_nvfp4":
                        rk, rv = tq._rot(kk.float()), tq._rot(vv.float())
                        kh[:, h], vh[:, h] = tq._unrot(nvfp4(rk)), tq._unrot(nvfp4(rv))
                    elif n == "tq_mse_b4":
                        kh[:, h], vh[:, h] = tq.mse(kk, 4), tq.mse(vv, 4)
                    elif n == "tq_prod_b4":
                        kh[:, h], vh[:, h] = tq.prod(kk, 4), tq.mse(vv, 4)
                    elif n == "tq_mse_b3":
                        kh[:, h], vh[:, h] = tq.mse(kk, 3), tq.mse(vv, 3)
                s_cos, o_cos = metrics(q, k, v, kh, vh, gr)
                row[n] = {"score_cos": s_cos, "out_cos": o_cos, "k_rel": rel(k, kh), "v_rel": rel(v, vh)}
                agg[n].append(row[n])
            report["layers"].append(row)
            print(f"r{rank} L{lid:>2} " + " ".join(f"{n}={row[n]['out_cos']:.5f}" for n in names), flush=True)
    bits = {"fp8": 8.0, "mxfp4": 4.25, "nvfp4": 4.5, "had_nvfp4": 4.5, "tq_mse_b4": 4 + 16 / D,
            "tq_prod_b4": 4 + 24 / D, "tq_mse_b3": 3 + 16 / D}
    summary = {}
    print("\n format       bits  x-cap vs fp8  score_cos mean [worst]   out_cos mean [worst]    1-out_cos vs fp8   k_rel   v_rel")
    base = None
    for n in names:
        rows = agg[n]
        m = lambda f: sum(r[f] for r in rows) / len(rows)
        w = lambda f: min(r[f] for r in rows)
        summary[n] = {"bits": round(bits[n], 3), "capacity_vs_fp8": round(8.0 / bits[n], 2),
                      "score_cos_mean": m("score_cos"), "score_cos_worst": w("score_cos"),
                      "out_cos_mean": m("out_cos"), "out_cos_worst": w("out_cos"),
                      "k_rel_mean": m("k_rel"), "v_rel_mean": m("v_rel")}
        err = 1 - summary[n]["out_cos_mean"]
        base = base or err
        print(f" {n:11s} {bits[n]:5.2f}   {8.0 / bits[n]:5.2f}x      {m('score_cos'):.5f} [{w('score_cos'):.5f}]"
              f"      {m('out_cos'):.5f} [{w('out_cos'):.5f}]      {err / base:6.1f}x        {m('k_rel'):.4f}  {m('v_rel'):.4f}")
    report["summary"] = summary
    if args.report:
        json.dump(report, open(args.report, "w"), indent=1)
        print("wrote", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
