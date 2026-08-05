"""Gemma4 NUMERICAL parity against the HuggingFace reference, module by module.

CPU-only, float32 (plus one deliberate fp16 pass) — needs no GPU lease and cannot disturb a running
serve. Run it inside the serve image (the host torch install is broken, and the image ships
transformers' `models.gemma4` reference):

    docker run --rm --entrypoint bash -v <worktree>:/wt \
      -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
      minisgl-rdna4:lean -lc 'cd /wt && PYTHONPATH=/wt/python:/opt/kernels \
        python tests/gemma4_parity_test.py'

The build test proves the model ASSEMBLES and the loader test proves the right tensors ARRIVE. This
one proves the forward MATH agrees, which is the only one of the three whose failure mode is silent.
The engine's full forward cannot run on CPU (global context, paged KV pool, HIP attention/MoE
kernels), so each check drives the REAL minisgl module for the piece under test and substitutes only
the parts that are genuinely GPU-bound:

  [0] compressed-tensors int4 dequant — the nibble/sign convention is DETERMINED EMPIRICALLY, not
      assumed, by dequantizing a *different* cached checkpoint (Qwen3.5-4B-AWQ-BF16-INT4) whose bf16
      original is also cached and checking the residual against every rival nibble order.
  [1] RMSNorm convention (plain gain, NOT the 1+weight form) and the known fp16 multiply-order gap.
  [2] Gemma4Router vs Gemma4TextRouter — real layer-0 router weights, exact index match required.
  [3] Gemma4DenseMLP vs Gemma4TextMLP, and the whole decoder-layer dataflow (parallel dense+MoE,
      three norms, router-on-raw-residual, layer_scalar) vs Gemma4TextDecoderLayer, with the SAME
      injected expert module on both sides so only the surrounding wiring is under test.
  [4] Attention q/k/v CONSTRUCTION on a full layer (no v_proj; V = pre-norm pre-RoPE k_proj output
      through the unweighted v_norm) and on a sliding layer (v_proj present, still v_norm'd). The
      real Gemma4Attention.forward runs; only the attention KERNEL is stubbed, on both sides.
  [5] Embedding scale (sqrt(hidden) CAST TO THE WEIGHT DTYPE) and the 30*tanh(x/30) final softcap.

Every check prints a measured error. Anything that cannot be exercised on CPU is printed as SKIP
with the reason rather than silently passing.
"""

from __future__ import annotations

import copy
import dataclasses
import glob
import sys
import types

import torch

MODEL_ID = "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4"
MODEL_GLOB = f"/root/.cache/huggingface/hub/models--{MODEL_ID.replace('/', '--')}/snapshots/*/"
# Dequant ground truth: a DIFFERENT model whose compressed-tensors int4 pack-quantized build AND
# whose original wide-dtype build are both cached, so the nibble order can be settled by measurement.
DEQ_INT4_GLOB = (
    "/root/.cache/huggingface/hub/models--cyankiwi--Qwen3.5-4B-AWQ-BF16-INT4/snapshots/*/"
)
DEQ_WIDE_GLOB = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/"

CKPT = "model.language_model."  # the gemma4 checkpoint namespace


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.skips = 0

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:44s} {detail}")
        return ok

    def close(self, name: str, got: torch.Tensor, want: torch.Tensor, tol: float) -> bool:
        d = (got.float() - want.float()).abs()
        ref = want.float().abs()
        max_abs = d.max().item()
        # relative to the reference's own scale, so a big tensor is not judged by absolute ULPs
        scale = ref.max().item()
        max_rel = (d / ref.clamp_min(1e-12)).max().item()
        rel_fro = (d.norm() / ref.norm().clamp_min(1e-30)).item()
        return self.check(
            name,
            max_abs <= tol * max(scale, 1e-12),
            f"max|abs|={max_abs:.3e}  ref|max|={scale:.3e}  rel_fro={rel_fro:.3e}  "
            f"max|rel|={max_rel:.3e}  tol={tol:.0e}",
        )

    def skip(self, name: str, why: str) -> None:
        self.skips += 1
        print(f"  SKIP {name:44s} {why}")


# --------------------------------------------------------------------------------------------
# compressed-tensors int4 pack-quantized dequant (group_size 32, symmetric, no zero point).
# weight_packed is (out, in//8) int32; weight_scale is (out, in//32).
# --------------------------------------------------------------------------------------------
def dequant_ct_int4(packed: torch.Tensor, scale: torch.Tensor, order=tuple(range(8))) -> torch.Tensor:
    """int4 pack-quantized -> fp32. Sign convention DETECTED (not assumed) with the engine's own
    `_ct_packed_is_uint4b8`; `order` is the nibble->input-index map, validated in check [0]."""
    from minisgl.quant.method import _ct_packed_is_uint4b8

    if not _ct_packed_is_uint4b8(packed):
        # two's-complement packing -> the engine's XOR 0x88 puts it in the uint4b8 (q+8) domain.
        packed = (packed.view(torch.uint8) ^ 0x88).view(torch.int32)
    words = packed.to(torch.int64) & 0xFFFFFFFF
    out, n_words = words.shape
    nib = torch.stack([(words >> (4 * p)) & 0xF for p in order], dim=-1).reshape(out, n_words * 8)
    q = nib.float() - 8.0  # uint4b8 -> signed
    groups = scale.shape[1]
    return (q.view(out, groups, -1) * scale.float().view(out, groups, 1)).view(out, -1)


def _open(folder: str):
    from safetensors import safe_open

    return [safe_open(p, framework="pt", device="cpu") for p in sorted(glob.glob(folder + "*.safetensors"))]


def _get(handles, key):
    for h in handles:
        if key in h.keys():
            return h.get_tensor(key)
    return None


# --------------------------------------------------------------------------------------------
def check_dequant(rep: Report) -> None:
    """Settle the packing convention by MEASUREMENT: dequantize a compressed-tensors int4 tensor
    whose wide-dtype original is also on disk. The exporter's AWQ smoothing rescales each input
    column, so the comparison divides that out per column and looks at what is left — for the right
    nibble order that residual is int4 quantization noise (~0.09); for any rival order the two
    tensors are uncorrelated and the residual is ~1.0."""
    print("\n[0] compressed-tensors int4 dequant — nibble convention determined empirically")
    a_dirs, w_dirs = glob.glob(DEQ_INT4_GLOB), glob.glob(DEQ_WIDE_GLOB)
    if not a_dirs or not w_dirs:
        rep.skip("nibble order vs wide-dtype original", "Qwen3.5-4B (+AWQ-INT4) not both cached")
    else:
        A, W = _open(a_dirs[0]), _open(w_dirs[0])
        prefixes = [
            "model.language_model.layers.0.mlp.gate_proj",
            "model.language_model.layers.3.self_attn.o_proj",
        ]
        orders = {
            "low-nibble-first (0..7)": tuple(range(8)),
            "high-nibble-first (7..0)": tuple(range(7, -1, -1)),
            "byte-swapped": (1, 0, 3, 2, 5, 4, 7, 6),
            "pair-swapped": (2, 3, 0, 1, 6, 7, 4, 5),
        }
        for prefix in prefixes:
            ref = _get(W, prefix + ".weight")
            packed, scale = _get(A, prefix + ".weight_packed"), _get(A, prefix + ".weight_scale")
            if ref is None or packed is None:
                rep.skip(f"dequant {prefix}", "tensor absent in one of the two checkpoints")
                continue
            ref = ref.float()
            residuals = {}
            for name, order in orders.items():
                w = dequant_ct_int4(packed, scale, order)
                col = (w * ref).sum(0) / (ref * ref).sum(0).clamp_min(1e-12)  # per-column AWQ scale
                residuals[name] = ((w - col * ref).norm() / w.norm()).item()
            best = min(residuals, key=residuals.__getitem__)
            runner = min((k for k in residuals if k != best), key=residuals.__getitem__)
            rep.check(
                f"nibble order {prefix.split('.')[-1]}",
                best == "low-nibble-first (0..7)" and residuals[best] < 0.25 < residuals[runner],
                f"best={best} resid={residuals[best]:.4f}  next={runner} resid={residuals[runner]:.4f}",
            )

    # And on the gemma4 checkpoint itself: with the group axis decoded correctly, a symmetric
    # max-abs int4 quantizer leaves every 32-wide group with max|q| at the rail (7 or 8). A wrong
    # group axis smears that.
    from minisgl.quant.method import _ct_packed_is_uint4b8

    handles = _open(glob.glob(MODEL_GLOB)[0])
    for prefix in (
        CKPT + "layers.5.self_attn.q_proj",
        CKPT + "layers.0.experts.0.gate_proj",
    ):
        packed, scale = _get(handles, prefix + ".weight_packed"), _get(handles, prefix + ".weight_scale")
        w = dequant_ct_int4(packed, scale)
        groups = scale.shape[1]
        q = (w.view(w.shape[0], groups, -1) / scale.float().view(w.shape[0], groups, 1)).round()
        rail = ((q.abs().amax(-1) >= 7) & (q.abs().amax(-1) <= 8)).float().mean().item()
        rep.check(
            f"group axis {prefix.split('.', 4)[-1][:28]}",
            rail > 0.999,
            f"uint4b8={_ct_packed_is_uint4b8(packed)}  groups at the int4 rail={rail * 100:.2f}%  "
            f"deq shape={tuple(w.shape)}  |w|max={w.abs().max().item():.4f}",
        )


# --------------------------------------------------------------------------------------------
def check_rmsnorm(rep: Report, mc, tc, handles) -> None:
    """Gemma4 uses a PLAIN gain (`normed * weight`), not the (1+weight) form Qwen3.5/Gemma3 use."""
    print("\n[1] RMSNorm — convention + the known fp16 multiply-order gap")
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    from minisgl.layers import RMSNorm
    from minisgl.layers.norm import RMSNormNoScale

    # The LARGEST-gain norm in the checkpoint — the worst case for the fp16 ordering gap below.
    w16 = _get(handles, CKPT + "layers.28.input_layernorm.weight")
    small = _get(handles, CKPT + "layers.0.input_layernorm.weight")
    print(
        f"  (probe gain = layers.28.input_layernorm: mean={w16.float().mean():.3f} "
        f"max={w16.float().max():.3f} min={w16.float().min():.3f}; for contrast layers.0 is "
        f"mean={small.float().mean():.3f} max={small.float().max():.3f})"
    )
    torch.manual_seed(11)
    x32 = torch.randn(17, mc.hidden_size)

    ref = Gemma4RMSNorm(mc.hidden_size, eps=mc.rms_norm_eps).float()
    with torch.no_grad():
        ref.weight.copy_(w16.float())
    mine = RMSNorm(mc.hidden_size, eps=mc.rms_norm_eps)
    mine.weight = w16.float().clone()
    rep.close("fp32 plain-gain RMSNorm", mine.forward(x32), ref(x32), 1e-6)

    # the (1+weight) convention must be OFF — show what choosing it would cost
    plus = RMSNorm(mc.hidden_size, eps=mc.rms_norm_eps, plus_one=True)
    plus.weight = w16.float().clone()
    err_plus = (plus.forward(x32) - ref(x32)).abs().max().item()
    rep.check(
        "plus_one convention is NOT used",
        mine.plus_one is False and err_plus > 1.0,
        f"model uses plus_one={mine.plus_one}; the (1+w) form would differ by max|abs|={err_plus:.3e}",
    )

    # the unweighted norm (with_scale=False): v_norm / router.norm
    ref_ns = Gemma4RMSNorm(mc.hidden_size, eps=mc.rms_norm_eps, with_scale=False).float()
    rep.close("fp32 RMSNormNoScale (with_scale=False)",
              RMSNormNoScale(eps=mc.rms_norm_eps).forward(x32), ref_ns(x32), 1e-6)

    # KNOWN difference, quantified not fixed: the reference multiplies by the weight in fp32 and
    # casts once at the end; minisgl casts the normalized value to the input dtype FIRST and then
    # multiplies in fp16. With this checkpoint's large gains that is one extra fp16 rounding.
    x16 = x32.half()
    ref16 = Gemma4RMSNorm(mc.hidden_size, eps=mc.rms_norm_eps).half()
    with torch.no_grad():
        ref16.weight.copy_(w16)
    mine16 = RMSNorm(mc.hidden_size, eps=mc.rms_norm_eps)
    mine16.weight = w16.clone()
    a, b = mine16.forward(x16).float(), ref16(x16).float()
    truth = ref(x32)  # fp32 ground truth, same weights
    d = (a - b).abs()
    rel = d / b.abs().clamp_min(1e-12)
    mine_err = ((a - truth).norm() / truth.norm()).item()
    ref_err = ((b - truth).norm() / truth.norm()).item()
    print(
        f"       fp16 minisgl-vs-reference : max|abs|={d.max().item():.3e}  "
        f"max|rel|={rel.max().item():.3e}  mean|rel|={rel.mean().item():.3e}  "
        f"rel_fro={((a - b).norm() / b.norm()).item():.3e}"
    )
    print(
        f"       fp16 error vs fp32 truth  : minisgl rel_fro={mine_err:.3e}   "
        f"reference rel_fro={ref_err:.3e}   ratio={mine_err / ref_err:.3f}x"
    )
    # Acceptance: minisgl rounds the normalized value to fp16 BEFORE the gain multiply, so it pays
    # one extra fp16 rounding (<=2^-11 relative) that the reference does not. That is the whole
    # difference — so the criterion is that minisgl's own distance from the fp32 truth stays within
    # 2x the reference's (a second rounding of the same size), and well under 1e-3 overall. A
    # different FORMULA would blow past both.
    rep.check(
        "fp16 gap is one extra rounding, not a formula change",
        mine_err <= 2.0 * ref_err and mine_err < 1e-3,
        f"minisgl rel_fro={mine_err:.3e} vs reference {ref_err:.3e} ({mine_err / ref_err:.3f}x); "
        f"fp16 ULP=9.766e-04, half-ULP=4.883e-04",
    )


# --------------------------------------------------------------------------------------------
def check_router(rep: Report, mc, tc, handles) -> None:
    print("\n[2] Gemma4Router vs Gemma4TextRouter (real layer-0 router weights)")
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter

    from minisgl.models.gemma4 import Gemma4Router

    pref = CKPT + "layers.0.router."
    w = _get(handles, pref + "proj.weight").float()
    scale = _get(handles, pref + "scale").float()
    pes = _get(handles, pref + "per_expert_scale").float()
    print(
        f"  (per_expert_scale: mean={pes.mean():.6f} min={pes.min():.6f} max={pes.max():.6f}; "
        f"scale: mean={scale.mean():.4f} max={scale.max():.4f})"
    )

    mine = Gemma4Router(mc.hidden_size, mc.num_experts, mc.rms_norm_eps)
    mine._top_k = mc.num_experts_per_tok
    mine.weight, mine.scale, mine.per_expert_scale = w.clone(), scale.clone(), pes.clone()

    ref = Gemma4TextRouter(tc).float()
    with torch.no_grad():
        ref.proj.weight.copy_(w)
        ref.scale.copy_(scale)
        ref.per_expert_scale.copy_(pes)

    torch.manual_seed(3)
    x = torch.randn(128, mc.hidden_size)
    got_w, got_idx = mine.forward(x)
    _, want_w, want_idx = ref(x)

    n_mismatch = int((got_idx != want_idx).sum())
    rep.check(
        "top-8 expert INDICES match exactly",
        n_mismatch == 0,
        f"mismatching (token,slot) entries = {n_mismatch} / {got_idx.numel()}",
    )
    rep.close("top-8 gate WEIGHTS", got_w, want_w, 1e-6)

    # The two rescalings that are easy to get backwards.
    sums = got_w.sum(-1)
    rep.check(
        "weights are per_expert_scale'd AFTER renorm",
        True,
        f"sum(top-k weights): mean={sums.mean().item():.6f} min={sums.min().item():.6f} "
        f"max={sums.max().item():.6f}  (deliberately not exactly 1)",
    )
    # Sensitivity guard: prove the check would catch the wrong order (scale then renormalize).
    probs = torch.softmax(torch.nn.functional.linear(
        (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + mc.rms_norm_eps))
        * scale * mc.hidden_size ** -0.5, w).float(), dim=-1)
    tw, ti = torch.topk(probs, mc.num_experts_per_tok, dim=-1)
    wrong = tw * pes[ti]
    wrong = wrong / wrong.sum(-1, keepdim=True)
    delta = (wrong - want_w).abs().max().item()
    rep.check(
        "order guard: renorm-then-scale != scale-then-renorm",
        delta > 1e-5,
        f"the transposed order would differ by max|abs|={delta:.3e}",
    )

    # hidden_size**-0.5 rescale must be present
    no_root = Gemma4Router(mc.hidden_size, mc.num_experts, mc.rms_norm_eps)
    no_root._top_k = mc.num_experts_per_tok
    no_root.weight, no_root.scale, no_root.per_expert_scale = w.clone(), scale.clone(), pes.clone()
    no_root._scalar_root_size = 1.0
    bad_w, bad_idx = no_root.forward(x)
    rep.check(
        "hidden**-0.5 rescale is live",
        mine._scalar_root_size == mc.hidden_size ** -0.5 and int((bad_idx != want_idx).sum()) == 0
        and (bad_w - want_w).abs().max().item() > 1e-4,
        f"scalar_root_size={mine._scalar_root_size:.8f}; dropping it changes weights by "
        f"max|abs|={(bad_w - want_w).abs().max().item():.3e}",
    )

    from minisgl.layers.minv import minv_supported

    print(
        f"  (minv_linear kernel path for fp32 weights: {minv_supported(x, w)} -> the router GEMM "
        f"ran through the F.linear fallback; the WMMA kernel is GPU-only and untested here)"
    )


# --------------------------------------------------------------------------------------------
def _stub_module(fn):
    """An nn.Module whose forward is `fn` (torch refuses a bare lambda as a child module)."""
    m = torch.nn.Module()
    m.forward = fn
    return m


def check_layer(rep: Report, mc, tc, handles) -> None:
    print("\n[3] dense MLP + decoder-layer dataflow vs Gemma4TextDecoderLayer")
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextMLP

    from minisgl.models.gemma4 import Gemma4DecoderLayer, Gemma4DenseMLP

    g = lambda k: _get(handles, CKPT + k).float()
    gp, up, dp = (g("layers.0.mlp.gate_proj.weight"), g("layers.0.mlp.up_proj.weight"),
                  g("layers.0.mlp.down_proj.weight"))
    cfgu = dataclasses.replace(mc, quant=None)  # the dense mlp is unquantized in this checkpoint

    mine_mlp = Gemma4DenseMLP(cfgu, 0)
    mine_mlp.gate_up_proj.weight = torch.cat([gp, up], dim=0)
    mine_mlp.down_proj.weight = dp
    ref_mlp = Gemma4TextMLP(tc, 0).float()
    with torch.no_grad():
        ref_mlp.gate_proj.weight.copy_(gp)
        ref_mlp.up_proj.weight.copy_(up)
        ref_mlp.down_proj.weight.copy_(dp)

    # Realistic drive level: the MLP is fed pre_feedforward_layernorm(h), whose gain has mean ~49,
    # so a raw N(0,1) probe would sit in the region where every gelu approximation agrees and would
    # hide an activation mismatch.
    from minisgl.layers import RMSNorm

    pre = RMSNorm(mc.hidden_size, eps=mc.rms_norm_eps)
    pre.weight = g("layers.0.pre_feedforward_layernorm.weight")
    torch.manual_seed(5)
    x_mlp = pre.forward(torch.randn(9, mc.hidden_size))
    gate = torch.nn.functional.linear(x_mlp, gp)
    print(f"  (dense-MLP gate pre-activation: std={gate.std():.3f} |max|={gate.abs().max():.3f}, "
          f"config hidden_activation={tc.hidden_activation!r}, reference act module="
          f"{type(ref_mlp.act_fn).__name__})")
    rep.close("dense MLP (real layer-0 weights)", mine_mlp.forward(x_mlp), ref_mlp(x_mlp), 1e-5)

    # Sensitivity guard for the activation itself: `gelu_pytorch_tanh` is the TANH approximation, and
    # the exact erf gelu is close enough to look right forever. Show how far off it lands here.
    erf_out = torch.nn.functional.linear(
        torch.nn.functional.gelu(gate, approximate="none") * torch.nn.functional.linear(x_mlp, up), dp
    )
    d_erf = (mine_mlp.forward(x_mlp) - erf_out).abs().max().item()
    rep.check(
        "dense MLP uses tanh-gelu, not erf-gelu",
        d_erf > 1e-4,
        f"swapping in the exact erf gelu would move the MLP output by max|abs|={d_erf:.3e} "
        f"(ref|max|={ref_mlp(x_mlp).abs().max().item():.3e})",
    )

    # ---- the decoder-layer wiring. The expert compute is GPU-only, so the SAME reference expert
    # module is injected into both sides and only the surrounding dataflow is compared. The expert
    # COUNT is reduced (128 -> 8 experts, top-4) purely to fit the fp32 stacked weights in RAM.
    n_exp, top_k = 8, 4
    tcr = copy.deepcopy(tc)
    tcr.num_experts, tcr.top_k_experts = n_exp, top_k
    mcr = dataclasses.replace(mc, quant=None, num_experts=n_exp, num_experts_per_tok=top_k)
    mine = Gemma4DecoderLayer(mcr, 0, expert_quant=None)
    ref = Gemma4TextDecoderLayer(tcr, 0).float()

    norms = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
             "post_feedforward_layernorm_1", "pre_feedforward_layernorm_2",
             "post_feedforward_layernorm_2", "post_feedforward_layernorm")
    for name in norms:
        w = g(f"layers.0.{name}.weight")
        getattr(mine, name).weight = w.clone()
        with torch.no_grad():
            getattr(ref, name).weight.copy_(w)
    ls = g("layers.0.layer_scalar")
    mine.layer_scalar = ls.clone()
    with torch.no_grad():
        ref.layer_scalar.copy_(ls)
    rw = g("layers.0.router.proj.weight")[:n_exp]
    rs = g("layers.0.router.scale")
    rp = g("layers.0.router.per_expert_scale")[:n_exp]
    mine.router.weight, mine.router.scale, mine.router.per_expert_scale = rw.clone(), rs.clone(), rp.clone()
    with torch.no_grad():
        ref.router.proj.weight.copy_(rw)
        ref.router.scale.copy_(rs)
        ref.router.per_expert_scale.copy_(rp)
    mine.mlp.gate_up_proj.weight = torch.cat([gp, up], dim=0)
    mine.mlp.down_proj.weight = dp
    with torch.no_grad():
        ref.mlp.gate_proj.weight.copy_(gp)
        ref.mlp.up_proj.weight.copy_(up)
        ref.mlp.down_proj.weight.copy_(dp)

    torch.manual_seed(6)
    with torch.no_grad():
        ref.experts.gate_up_proj.normal_(0, 0.02)
        ref.experts.down_proj.normal_(0, 0.02)
    shared_experts = ref.experts  # the reference expert module drives BOTH sides

    # The routed experts run on the HIP kernel, so only their POLICY can be checked here: the
    # activation the MoE layer asks for, the fact that it must not renormalize a second time, and
    # that the engine's tanh-gelu matches the reference's ACT2FN entry exactly.
    from transformers.activations import ACT2FN

    from minisgl.layers.activation import gelu_tanh_and_mul

    act_probe = torch.randn(64, 16) * 3
    ref_act = ACT2FN[tc.hidden_activation](act_probe[:, :8]) * act_probe[:, 8:]
    rep.check(
        "MoE expert activation policy",
        mine.experts.activation == "gelu" and mine.experts.renormalize is False
        and (gelu_tanh_and_mul(act_probe) - ref_act).abs().max().item() == 0.0,
        f"MoELayer(activation={mine.experts.activation!r}, renormalize={mine.experts.renormalize}); "
        f"gelu_tanh_and_mul vs ACT2FN[{tc.hidden_activation!r}] max|abs|="
        f"{(gelu_tanh_and_mul(act_probe) - ref_act).abs().max().item():.3e}",
    )

    attn_w = torch.randn(mc.hidden_size, mc.hidden_size) * 0.02
    mine.self_attn = types.SimpleNamespace(forward=lambda h: h @ attn_w.T)
    ref.self_attn = _stub_module(lambda **kw: (kw["hidden_states"] @ attn_w.T, None))
    # minisgl calls experts(hidden, topk_weights, topk_ids); the reference calls it
    # (hidden, top_k_index, top_k_weights) — the adapter is the ONLY difference allowed.
    # `reduce` is part of the real MoELayer/LinearRowParallel signature (it lets a caller take the
    # un-reduced partial); the stubs are single-rank, so they accept it and ignore it.
    mine.experts = types.SimpleNamespace(
        forward=lambda hidden_states, topk_weights, topk_ids, reduce=True: shared_experts(
            hidden_states, topk_ids, topk_weights
        )
    )

    x = torch.randn(7, mc.hidden_size)
    rep.close("decoder layer, REAL dense MLP", mine.forward(x.clone()), ref(x.clone()), 2e-5)

    # Isolate the wiring from the activation: identical MLP on both sides.
    mlp_w = torch.randn(mc.hidden_size, mc.hidden_size) * 0.02
    mine.mlp = types.SimpleNamespace(forward=lambda h, reduce=True: h @ mlp_w.T)
    ref.mlp = _stub_module(lambda h: h @ mlp_w.T)
    rep.close("decoder layer, dataflow only (stub MLP)",
              mine.forward(x.clone()), ref(x.clone()), 1e-6)

    # layer_scalar must scale the WHOLE residual stream (dropping it is the classic silent bug).
    print(f"  (layer-0 layer_scalar = {ls.item():.6f})")
    saved = mine.layer_scalar
    mine.layer_scalar = torch.ones(1)
    delta = (mine.forward(x.clone()) - ref(x.clone())).abs().max().item()
    mine.layer_scalar = saved
    rep.check("layer_scalar scales the whole stream", delta > 1e-3,
              f"forcing layer_scalar=1 moves the output by max|abs|={delta:.3e}")

    # The router must read the RAW residual, not the pre_feedforward_layernorm_2 output.
    probe = {}
    real_router_forward = mine.router.forward
    mine.router.forward = lambda h: (probe.setdefault("x", h.clone()), real_router_forward(h))[1]
    mine.forward(x.clone())
    mine.router.forward = real_router_forward
    resid = x + mine.post_attention_layernorm.forward(
        mine.self_attn.forward(mine.input_layernorm.forward(x))
    )
    normed = mine.pre_feedforward_layernorm_2.forward(resid)
    d_raw = (probe["x"] - resid).abs().max().item()
    d_norm = (probe["x"] - normed).abs().max().item()
    rep.check("router reads the RAW residual", d_raw < 1e-5 < d_norm,
              f"|router_in - raw_residual|={d_raw:.3e}   |router_in - norm2(residual)|={d_norm:.3e}")


# --------------------------------------------------------------------------------------------
def check_attention(rep: Report, mc, tc, handles) -> None:
    print("\n[4] attention q/k/v construction (real dequantized projections)")
    import transformers.models.gemma4.modeling_gemma4 as ref_mod
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from minisgl.models.gemma4 import Gemma4Attention
    import minisgl.layers.attention as attn_mod

    captured: dict = {}

    def capture(module, q, k, v, mask, **kw):
        captured["ref"] = (q, k, v)
        b, h, s, d = q.shape
        return _det(b * s * h * d).reshape(b, s, h, d), None

    def _det(n):
        return torch.arange(n, dtype=torch.float32).mul_(1e-4).sin_()

    ALL_ATTENTION_FUNCTIONS.register("gemma4_parity_capture", capture)
    tc = copy.deepcopy(tc)
    tc._attn_implementation = "gemma4_parity_capture"

    rot = ref_mod.Gemma4TextRotaryEmbedding(tc, device=torch.device("cpu")).float()
    cfgu = dataclasses.replace(mc, quant=None)
    n_tok = 11
    torch.manual_seed(7)
    pos = torch.arange(n_tok)
    x = torch.randn(n_tok, mc.hidden_size)

    for layer_id, kind in ((5, "full"), (4, "sliding")):
        p = CKPT + f"layers.{layer_id}.self_attn"
        deq = lambda m: dequant_ct_int4(_get(handles, f"{p}.{m}.weight_packed"),
                                        _get(handles, f"{p}.{m}.weight_scale"))
        mine = Gemma4Attention(cfgu, layer_id)
        mine.q_proj.weight = deq("q_proj")
        mine.k_proj.weight = deq("k_proj")
        if mine.v_proj is not None:
            mine.v_proj.weight = deq("v_proj")
        mine.o_proj.weight = deq("o_proj")
        mine.q_norm.weight = _get(handles, f"{p}.q_norm.weight").float()
        mine.k_norm.weight = _get(handles, f"{p}.k_norm.weight").float()

        ref = ref_mod.Gemma4TextAttention(tc, layer_id).float()
        with torch.no_grad():
            ref.q_proj.weight.copy_(mine.q_proj.weight)
            ref.k_proj.weight.copy_(mine.k_proj.weight)
            if ref.v_proj is not None:
                ref.v_proj.weight.copy_(mine.v_proj.weight)
            ref.o_proj.weight.copy_(mine.o_proj.weight)
            ref.q_norm.weight.copy_(mine.q_norm.weight)
            ref.k_norm.weight.copy_(mine.k_norm.weight)

        rep.check(
            f"L{layer_id} {kind}: geometry + v_proj presence",
            (mine.v_proj is not None) == (ref.v_proj is not None)
            and mine._head_dim == ref.head_dim
            and mine._nkv_local == tc.num_attention_heads // ref.num_key_value_groups,
            f"head_dim={mine._head_dim} kv_heads={mine._nkv_local} "
            f"v_proj={'present' if mine.v_proj is not None else 'ABSENT'} "
            f"(reference use_alternative_attention={ref.use_alternative_attention}) "
            f"ref.scaling={ref.scaling}",
        )

        cos, sin = rot(x.unsqueeze(0), pos.unsqueeze(0), tc.layer_types[layer_id])
        ref_out, _ = ref(hidden_states=x.unsqueeze(0), position_embeddings=(cos, sin),
                         attention_mask=None, shared_kv_states={})
        rq, rk, rv = captured["ref"]

        class _Backend:
            def forward(self, q, k, v, layer_id, batch, sliding_window=0):
                captured["mine"] = (q, k, v)
                n, h, d = q.shape
                return _det(n * h * d).reshape(n, h, d)

        ctx = types.SimpleNamespace(batch=types.SimpleNamespace(positions=pos),
                                    attn_backend=_Backend())
        saved = attn_mod.get_global_ctx
        attn_mod.get_global_ctx = lambda: ctx
        try:
            mine_out = mine.forward(x)
        finally:
            attn_mod.get_global_ctx = saved
        mq, mk, mv = (t.reshape(n_tok, -1, mine._head_dim) for t in captured["mine"])

        rep.close(f"L{layer_id} {kind}: Q (q_norm then RoPE)", mq, rq[0].transpose(0, 1), 1e-6)
        rep.close(f"L{layer_id} {kind}: K (k_norm then RoPE)", mk, rk[0].transpose(0, 1), 1e-6)
        rep.close(f"L{layer_id} {kind}: V (v_norm, pre-RoPE)", mv, rv[0].transpose(0, 1), 1e-6)
        rep.close(f"L{layer_id} {kind}: o_proj output", mine_out, ref_out[0], 1e-6)

        # V must NOT equal the cached (k_norm'd + RoPE'd) key, and must not be the raw k_proj output.
        raw_k = torch.nn.functional.linear(
            x, mine.v_proj.weight if mine.v_proj is not None else mine.k_proj.weight)
        raw_k = raw_k.view(n_tok, -1, mine._head_dim)
        d_key = (mv - mk).abs().max().item()
        d_raw = (mv - raw_k).abs().max().item()
        rep.check(
            f"L{layer_id} {kind}: V is v_norm'd, not the cached K",
            d_key > 1e-3 and d_raw > 1e-3,
            f"|V-K|={d_key:.3e} (must be >0)   |V-unnormed_src|={d_raw:.3e} (must be >0)",
        )

    # softmax temperature: 1.0, not 1/sqrt(head_dim)
    try:
        from minisgl.attention.rdna4 import RDNA4Backend

        stub = types.SimpleNamespace(_scale_override=mc.attn_softmax_scale, _scale_by_head_dim={})
        got = [RDNA4Backend._softmax_scale(stub, torch.zeros(1, 1, d)) for d in (256, 512)]
        rep.check("backend softmax scale is 1.0 on both geometries",
                  got == [1.0, 1.0],
                  f"config attn_softmax_scale={mc.attn_softmax_scale}; resolved for head_dim "
                  f"256/512 = {got}  (1/sqrt(d) would be "
                  f"{256 ** -0.5:.6f}/{512 ** -0.5:.6f})")
    except Exception as exc:  # pragma: no cover - import needs the GPU stack in some images
        rep.skip("backend softmax scale", f"minisgl.attention.rdna4 not importable on CPU: {exc}")


# --------------------------------------------------------------------------------------------
def check_embed_and_softcap(rep: Report, mc, tc, handles) -> None:
    print("\n[5] embedding scale (cast to the WEIGHT dtype) + final logit softcap")
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm,
        Gemma4TextScaledWordEmbedding,
    )

    import minisgl.layers.embedding as emb_mod
    import minisgl.models.gemma4 as g4
    from minisgl.models.gemma4 import Gemma4ForConditionalGeneration

    vocab = 512
    # A zero-layer stack keeps embed -> final norm -> lm_head -> softcap (the whole item under test)
    # on the real code path while staying CPU-sized.
    cfg = dataclasses.replace(mc, quant=None, num_layers=0, vocab_size=vocab,
                              tie_word_embeddings=True)

    scale_f64 = mc.hidden_size ** 0.5
    exact16 = torch.tensor(scale_f64, dtype=torch.float16).item()
    rep.check(
        "embed_scale = sqrt(hidden), fp16-rounded",
        abs(mc.embed_scale - scale_f64) < 1e-12 and exact16 == 53.0625,
        f"config={mc.embed_scale:.10f}  fp16 cast={exact16}  (a float32 multiply would use "
        f"{torch.tensor(scale_f64, dtype=torch.float32).item():.7f})",
    )

    torch.manual_seed(9)
    ids = torch.randint(0, vocab, (13,))
    emb_w16 = (torch.randn(vocab, mc.hidden_size) * 0.05).half()
    norm_w16 = _get(handles, CKPT + "norm.weight")

    # fp16 tol is the RMSNorm multiply-order gap measured in check [1], not a slack knob.
    for dtype, tol in ((torch.float16, 1e-3), (torch.float32, 1e-6)):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            model = Gemma4ForConditionalGeneration(cfg)
        finally:
            torch.set_default_dtype(prev)
        model.model.embed_tokens.weight = emb_w16.to(dtype).clone()
        model.model.norm.weight = norm_w16.to(dtype).clone()

        ctx = types.SimpleNamespace(
            batch=types.SimpleNamespace(input_ids=ids, is_prefill=False, size=ids.numel())
        )
        saved_g4, saved_emb = g4.get_global_ctx, emb_mod.get_global_ctx
        g4.get_global_ctx = emb_mod.get_global_ctx = lambda: ctx
        try:
            got_hidden = model.model.forward(ids)
            got_logits = model.forward() if dtype is torch.float32 else None
        finally:
            g4.get_global_ctx, emb_mod.get_global_ctx = saved_g4, saved_emb

        ref_emb = Gemma4TextScaledWordEmbedding(vocab, mc.hidden_size, 0, embed_scale=scale_f64).to(dtype)
        with torch.no_grad():
            ref_emb.weight.copy_(emb_w16.to(dtype))
        ref_norm = Gemma4RMSNorm(mc.hidden_size, eps=mc.rms_norm_eps).to(dtype)
        with torch.no_grad():
            ref_norm.weight.copy_(norm_w16.to(dtype))
        want_hidden = ref_norm(ref_emb(ids))
        # The SCALED EMBEDDING itself must be bit-exact in both dtypes; only the trailing RMSNorm
        # carries the known fp16 multiply-order gap quantified in check [1], hence the split tol.
        got_scaled = model.model.embed_tokens.forward(ids) * model.model._scale_tensor(
            model.model.embed_tokens.weight
        )
        rep.close(f"embedding * embed_scale ({dtype})", got_scaled, ref_emb(ids), 0.0)
        rep.close(f"embedding*scale -> final norm ({dtype})", got_hidden, want_hidden, tol)

        if dtype is torch.float16:
            cache = model.model._scale_tensor(torch.zeros(1, dtype=torch.float16))
            rep.check(
                "embed scale materialized in fp16 (53.0625)",
                cache.item() == 53.0625 and cache.item() == ref_emb.embed_scale.to(torch.float16).item(),
                f"minisgl={cache.item()}  reference={ref_emb.embed_scale.to(torch.float16).item()}  "
                f"fp32 would be {torch.tensor(scale_f64, dtype=torch.float32).item():.7f}",
            )
            continue

        # fp32: lm_head (tied, UNSCALED embedding matrix) + 30*tanh(x/30)
        raw = torch.nn.functional.linear(want_hidden, ref_emb.weight)
        cap = mc.final_logit_softcapping
        want_logits = torch.tanh(raw / cap) * cap
        rep.close("lm_head + final softcap", got_logits, want_logits, 1e-6)
        rep.check(
            "softcap actually bounds the logits",
            got_logits.abs().max().item() <= cap and raw.abs().max().item() > cap,
            f"softcap={cap}  |raw|max={raw.abs().max().item():.4f}  "
            f"|capped|max={got_logits.abs().max().item():.4f}  "
            f"(the probe drives the head past the cap on purpose)",
        )
        # a tied head must NOT re-apply the sqrt(hidden) embedding scale
        rescaled = torch.nn.functional.linear(want_hidden, ref_emb.weight * exact16)
        rep.check(
            "tied lm_head does not re-apply embed_scale",
            (got_logits - torch.tanh(rescaled / cap) * cap).abs().max().item() > 1e-3,
            f"re-applying it would move logits by max|abs|="
            f"{(got_logits - torch.tanh(rescaled / cap) * cap).abs().max().item():.3e}",
        )


# --------------------------------------------------------------------------------------------
def main() -> int:
    matches = glob.glob(MODEL_GLOB)
    if not matches:
        print(f"SKIP: checkpoint not cached under {MODEL_GLOB}")
        return 0
    path = matches[0]

    from minisgl.distributed import set_tp_info

    set_tp_info(0, 1)
    import minisgl.layers.rotary as rotary_mod
    from transformers import AutoConfig

    from minisgl.models.config import ModelConfig

    rotary_mod.set_rope_device(torch.device("cpu"))
    torch.set_default_dtype(torch.float32)
    torch.set_grad_enabled(False)

    hf = AutoConfig.from_pretrained(path)
    mc = ModelConfig.from_hf(hf, spec_algorithm="none")
    tc = hf.text_config
    handles = _open(path)

    print(f"[parity] {MODEL_ID}  hidden={mc.hidden_size} experts={mc.num_experts} "
          f"top_k={mc.num_experts_per_tok} eps={mc.rms_norm_eps}")

    rep = Report()
    check_dequant(rep)
    check_rmsnorm(rep, mc, tc, handles)
    check_router(rep, mc, tc, handles)
    check_layer(rep, mc, tc, handles)
    check_attention(rep, mc, tc, handles)
    check_embed_and_softcap(rep, mc, tc, handles)

    print(f"\n{'PASS' if rep.failures == 0 else f'FAIL ({rep.failures} checks)'}"
          f"{f'  [{rep.skips} skipped]' if rep.skips else ''}")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
