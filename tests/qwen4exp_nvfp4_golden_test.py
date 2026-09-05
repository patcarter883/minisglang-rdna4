"""NVFP4 two-level scale vs an fp64 GOLDEN, on the REAL `RadixArk/Qwen3.8-Flash-Next-NVFP4` bytes.

`tests/core/test_nvfp4_two_level_scale.py` proves the PLUMBING (shapes, dtypes, bytes, merge, stack,
arena rows) on synthetic leaves. This proves the NUMBER, on real checkpoint bytes, and it is the only
place the accuracy claim behind the whole change is actually measured:

    the e4m3 block scale + per-output-channel f32 global is CLOSER TO THE TRUTH than the
    fp16 fold it replaces -- by ~4 orders of magnitude, on every expert tensor tested.

WHY THE GATE CANNOT BE "BIT-EXACT VS THE SHIPPED PATH". The normal parity gate for a refactor is
"identical to what shipped". That gate is WRONG here and would reject the change on purpose: the new
path differs from the fp16 fold *by being right*. `fold_nvfp4_scale` rounds `e4m3_block * global` --
a ~30-bit product -- into an 11-bit fp16 significand, so it is lossy on every single group of every
single expert. So the gate is stated against a THIRD thing that neither arm produced:

    a float64 dequant built from the RAW safetensors BYTES,
    `E2M1_LUT[nibble] * f64(e4m3_byte) * f64(global)`,

and the assertion is `relerr(split) << relerr(fold)`, both measured against it.

THE GOLDEN IS INDEPENDENT ON PURPOSE. Its E2M1 codebook and its 256-entry e4m3 byte table are built
here from the OCP spec by bit arithmetic -- not read from `mxfp4.FP4_E2M1_LUT`, not obtained by
`.view(torch.float8_e4m3fn).to(torch.float64)`. A golden that shared a decode table with the code
under test could not catch a wrong table. Both tables ARE then cross-checked against the repo LUT and
against torch's own float8 bitcast (section 0) so that a typo in the spec transcription is loud
rather than silently generous.

A GREEN RUN THAT PROVES NOTHING IS THE FAILURE MODE THIS GUARDS. Section 3 therefore MUTATES the
split path in the five ways it can realistically be got wrong -- global inverted, global dropped,
global value-converted, per-expert offset dropped, container-wide scalar -- and asserts the gate
CATCHES each one (mutant error must exceed even the fp16 fold's). A gate that cannot fail is not a
gate.

CPU ONLY. No GPU, no lease -- pure f64 host arithmetic over mmap'd bytes. Run in the serve image:

    docker run --rm -v <worktree>:/engine -v /home/pat/.cache/hf-q4e:/model:ro \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_nvfp4_golden_test.py /model'

WHAT THIS DOES NOT PROVE. It measures the HOST reference (`dequantize_nvfp4_split`), which is the
documented mirror of the kernel's `E4m3GroupScaleGlobal` policy. That the compiled kernel agrees with
its mirror is an ON-DEVICE op-parity test and is not in scope here -- see the GPU follow-ups in the
branch's commit message.
"""

from __future__ import annotations

import glob
import os
import sys

import torch
from safetensors import safe_open

DEFAULT_MODEL = "/model"
G = 16  # NVFP4 block-scale group

#: (layer, expert) pairs the elementwise/GEMV sections sweep. Layers 0/23/47 are the first, middle
#: and last decoder layers; experts 0/7/300 span two different 128-expert shards so a per-shard
#: global is exercised, not just expert 0's.
CASES = [(l, e) for l in (0, 23, 47) for e in (0, 7, 300)]
PROJS = ("gate_proj", "up_proj", "down_proj")
#: Layers whose FULL 512-expert global vectors are read for the merge/stack granularity census.
CENSUS_LAYERS = (0, 23, 47)

#: Gate thresholds. Set from measured margins (fold 2.7e-04..4.4e-04, split 3.6e-08..5.5e-08, ratio
#: 5.6e3..1.2e4), left ~50x loose so this fails on a REGRESSION and not on f32 noise.
FOLD_IS_LOSSY_ABOVE = 1e-4
SPLIT_IS_EXACT_BELOW = 1e-6
MIN_ACCURACY_RATIO = 100.0

_failures = 0


def check_true(name: str, cond: bool, detail: str = "") -> None:
    """A check whose evidence is a MEASUREMENT -- the detail carries the numbers, so a pass is
    auditable and not just a green tick."""
    global _failures
    _failures += not cond
    print(f"  {'ok  ' if cond else 'FAIL'} {name:52s} {detail}")


# ---------------------------------------------------------------------------------------------
# 0. The golden's decode tables, from the OCP spec. Not imported from the code under test.
# ---------------------------------------------------------------------------------------------


def e4m3_byte_table() -> torch.Tensor:
    """f64 value of all 256 OCP `float8_e4m3fn` bytes: sign(1) exp(4, bias 7) mantissa(3).

    `fn` = Finite + NaN: there is no infinity, and the only NaN is S.1111.111, which lets the top
    exponent carry real values up to 448 (=1.75*2^8). That is exactly why this checkpoint's
    `down_proj` block scales sit at byte 126 on every expert -- they SATURATE the format -- and why
    misreading those bytes as small integers would look plausible right up until it did not.
    """
    vals = []
    for b in range(256):
        sign = -1.0 if (b >> 7) else 1.0
        exp = (b >> 3) & 0xF
        man = b & 0x7
        if exp == 0xF and man == 0x7:
            vals.append(float("nan"))
        elif exp == 0:
            vals.append(sign * (man / 8.0) * (2.0**-6))  # subnormal: 2^(1-bias)
        else:
            vals.append(sign * (1.0 + man / 8.0) * (2.0 ** (exp - 7)))
    return torch.tensor(vals, dtype=torch.float64)


def e2m1_code_table() -> torch.Tensor:
    """f64 value of all 16 OCP E2M1 codes: sign(1) exp(2, bias 1) mantissa(1) -> +-{0,.5,1,1.5,2,3,4,6}."""
    vals = []
    for c in range(16):
        sign = -1.0 if (c >> 3) else 1.0
        exp = (c >> 1) & 0x3
        man = c & 0x1
        vals.append(sign * ((man / 2.0) if exp == 0 else (1.0 + man / 2.0) * (2.0 ** (exp - 1))))
    return torch.tensor(vals, dtype=torch.float64)


E4M3_F64 = e4m3_byte_table()
E2M1_F64 = e2m1_code_table()


def golden_f64(packed_u8: torch.Tensor, scale_bytes: torch.Tensor, global_mul: float) -> torch.Tensor:
    """(N, K) f64 truth: `E2M1[nibble] * f64(e4m3_byte) * f64(global)`, straight off the raw bytes.

    Nibble order is LOW-first, matching `mxfp4.unpack_e2m1_nibbles`; getting it backwards would
    transpose each weight pair and is the reason section 0 also checks this against the repo unpacker
    on real bytes rather than trusting the comment.
    """
    n, k_half = packed_u8.shape
    lo = (packed_u8 & 0xF).to(torch.int64)
    hi = (packed_u8 >> 4).to(torch.int64)
    codes = torch.stack([lo, hi], dim=-1).reshape(n, k_half * 2)
    block = E4M3_F64[scale_bytes.to(torch.int64)].repeat_interleave(G, dim=-1)
    return E2M1_F64[codes] * block * torch.tensor(global_mul, dtype=torch.float64)


def relerr(w: torch.Tensor, gold: torch.Tensor) -> "tuple[float, float]":
    """(max, mean) relative error against the golden, over its NONZERO entries.

    Zeros are excluded because E2M1 code 0 is exactly 0.0 on every path -- including them would
    divide by zero, and counting them as exact would dilute the mean with ~1/8 free hits.
    """
    nz = gold != 0
    r = (w.to(torch.float64) - gold).abs() / gold.abs().clamp_min(1e-300)
    return float(r[nz].max()), float(r[nz].mean())


# ---------------------------------------------------------------------------------------------
# Checkpoint access
# ---------------------------------------------------------------------------------------------


def shard_for(model: str, layer: int, expert: int) -> str:
    lo = (expert // 128) * 128
    return os.path.join(model, f"layer-{layer:05d}-experts-{lo:04d}-{lo + 127:04d}.safetensors")


def load_leaf(model: str, layer: int, expert: int, proj: str):
    """(packed u8, block-scale e4m3, global f32) exactly as the checkpoint stores them."""
    p = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
    with safe_open(shard_for(model, layer, expert), "pt") as f:
        return (
            f.get_tensor(f"{p}.weight"),
            f.get_tensor(f"{p}.weight_scale"),
            f.get_tensor(f"{p}.weight_scale_2"),
        )


def _bf16_shared_experts(model: str) -> dict:
    """{(layer, proj): bf16 weight} for each tested layer's `mlp.shared_expert`.

    THE ANCHOR THE QUANTIZER NEVER TOUCHED. `config.json`'s quantization `ignore` list carries
    `*.mlp.shared_expert.*`, so these ship as plain bf16 in the `model-bf16-*` shards. A routed expert
    and its layer's shared expert are the same kind of matrix at the same width, which is what makes
    "within a small factor of each other" a meaningful physical statement about the DEQUANT and not a
    restatement of the format.
    """
    want = {
        f"model.language_model.layers.{layer}.mlp.shared_expert.{proj}.weight": (layer, proj)
        for layer, _ in CASES
        for proj in PROJS
    }
    out = {}
    for shard in sorted(glob.glob(os.path.join(model, "model-bf16-*.safetensors"))):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                if key in want:
                    out[want[key]] = f.get_tensor(key).to(torch.float32)
    missing = set(want.values()) - set(out)
    if missing:
        raise RuntimeError(f"bf16 shared-expert anchor missing for {sorted(missing)}")
    return out


def arms(nvfp4, packed, sc, g2):
    """Both served arms from ONE checkpoint (block, global) pair, plus the golden they approximate.

    Built through the REAL entry points -- `fold_nvfp4_scale`, `split_nvfp4_scale`,
    `convert_nvfp4_weight` -- so this measures the shipped converters and not a restatement of them.
    """
    mul = float(nvfp4.nvfp4_global_multiplier(g2, global_field="weight_scale_2"))
    gold = golden_f64(packed, sc.view(torch.uint8), mul)

    folded = nvfp4.fold_nvfp4_scale(sc, g2, global_field="weight_scale_2")
    cf = nvfp4.convert_nvfp4_weight(packed, folded)
    w_fold = nvfp4.dequantize_nvfp4_folded(cf["w_packed"], cf["scales"])

    block, gvec = nvfp4.split_nvfp4_scale(sc, g2, global_field="weight_scale_2")
    cs = nvfp4.convert_nvfp4_weight(packed, block)
    w_split = nvfp4.dequantize_nvfp4_split(cs["w_packed"], cs["scales"], gvec)
    return gold, w_fold, w_split, cs, gvec, mul


# ---------------------------------------------------------------------------------------------
# 0. Is the golden itself trustworthy?
# ---------------------------------------------------------------------------------------------


def section_golden_selfcheck(model: str, nvfp4) -> None:
    print("\n[0] the golden's own decode tables, cross-checked against torch and the repo LUT")
    from minisgl.quant.mxfp4 import FP4_E2M1_LUT

    check_true(
        "E2M1 spec table == repo FP4_E2M1_LUT",
        torch.equal(E2M1_F64, torch.tensor(FP4_E2M1_LUT, dtype=torch.float64)),
        "16/16 codes",
    )
    torch_tab = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float64)
    finite = ~torch.isnan(torch_tab)
    check_true(
        "e4m3 spec table == torch float8_e4m3fn bitcast",
        torch.equal(torch_tab[finite], E4M3_F64[finite])
        and torch.equal(torch.isnan(torch_tab), torch.isnan(E4M3_F64)),
        f"{int(finite.sum())}/256 finite bytes agree, NaN masks identical",
    )
    check_true("e4m3 saturates at 448 (byte 126)", float(E4M3_F64[126]) == 448.0, "= 1.75 * 2^8")

    # Nibble ORDER, checked on real bytes against the repo unpacker rather than on a comment.
    packed, _, _ = load_leaf(model, 0, 0, "gate_proj")
    mine = torch.stack(
        [(packed & 0xF).to(torch.int64), (packed >> 4).to(torch.int64)], dim=-1
    ).reshape(packed.shape[0], packed.shape[1] * 2)
    check_true(
        "golden nibble order == mxfp4.unpack_e2m1_nibbles",
        torch.equal(mine, nvfp4.unpack_e2m1_nibbles(packed).to(torch.int64)),
        f"{mine.numel()} codes on real layer-0 bytes",
    )


# ---------------------------------------------------------------------------------------------
# 1. THE CLAIM: elementwise, on every tested expert tensor.
# ---------------------------------------------------------------------------------------------


def section_elementwise(model: str, nvfp4) -> None:
    print("\n[1] elementwise relative error vs the f64 golden (the fold is LOSSY, the split is not)")
    print(f"      {'case':22s} {'rms|W|':>8s} {'fold max':>10s} {'fold mean':>10s} {'split max':>10s} {'ratio':>9s}")
    worst_split, best_ratio = 0.0, float("inf")
    for layer, expert in CASES:
        for proj in PROJS:
            packed, sc, g2 = load_leaf(model, layer, expert, proj)
            gold, w_fold, w_split, _, _, _ = arms(nvfp4, packed, sc, g2)
            fm, fa = relerr(w_fold, gold)
            sm, _ = relerr(w_split, gold)
            ratio = fm / max(sm, 1e-300)
            worst_split = max(worst_split, sm)
            best_ratio = min(best_ratio, ratio)
            print(
                f"      L{layer:02d} e{expert:<4d} {proj:10s} {float(gold.pow(2).mean().sqrt()):8.5f} "
                f"{fm:10.3e} {fa:10.3e} {sm:10.3e} {ratio:9.1f}"
            )
            if fm <= FOLD_IS_LOSSY_ABOVE or sm >= SPLIT_IS_EXACT_BELOW or ratio <= MIN_ACCURACY_RATIO:
                check_true(f"L{layer} e{expert} {proj}", False, f"fold={fm:.3e} split={sm:.3e}")
    check_true(
        "the split is CLOSER TO THE GOLDEN than the fp16 fold",
        best_ratio > MIN_ACCURACY_RATIO,
        f"on all {len(CASES) * len(PROJS)} tensors; worst margin {best_ratio:.0f}x (gate >{MIN_ACCURACY_RATIO:.0f}x)",
    )
    check_true(
        "the split is exact to f32 round-off, not to zero",
        worst_split < SPLIT_IS_EXACT_BELOW,
        f"worst {worst_split:.3e} ~ 2^-24.5; the f32 product of a 6-bit E2M1*e4m3 "
        f"significand and a 24-bit global needs ~30 bits, so ONE rounding survives",
    )


# ---------------------------------------------------------------------------------------------
# 1b. THE BLIND SPOT OF EVERY RELATIVE GATE, and the two absolute checks that close it.
# ---------------------------------------------------------------------------------------------


def section_absolute_anchor(model: str, nvfp4) -> None:
    """A relative gate CANNOT catch a globally-inverted convention. These two checks can.

    Section 1 compares both arms against a golden that reads the direction out of the SAME
    `NVFP4_GLOBAL_SCALE_IS_RECIPROCAL` table the shipped path does. Flip that table and the golden
    flips with it: every relative error stays at 4e-08, the gate goes green, and the engine serves
    weights that are 2.3e+07 times too large. That is not hypothetical -- this checkpoint's
    `weight_scale_2` really is the reciprocal of what the format docs describe, and `config.py:385`
    already ignores the name, so the wrong reading is one edit away and is FINITE rather than NaN.

    So the direction is pinned to two things OUTSIDE that table:

    (1) THE PRODUCER'S CONSTRUCTION, as an exact bound. modelopt builds
        `weight_scale_2 = amax(|W|) / (FP4_E2M1_MAX * FP8_E4M3_MAX)`, so the dequantized amax must
        satisfy `amax(|W|) <= 2688 * weight_scale_2_RAW` -- stated against the raw checkpoint scalar,
        not the normalised multiplier, which is what makes it a direction test. Measured: the ratio
        is an exact integer `6 * max_block_scale` on every tensor, and exactly 2688 on every
        `down_proj` (whose block scales saturate the e4m3 format at byte 126 = 448). Inverting the
        direction sends this ratio to ~2e+10.

    (2) A PHYSICAL ANCHOR the quantizer never touched: the SAME layer's `mlp.shared_expert`, which
        this checkpoint ships UNQUANTIZED in bf16 (it is on the `ignore` list). A routed expert and a
        shared expert of the same layer are the same kind of matrix, so their mean magnitudes must be
        within a small factor. Measured 1.41x..1.69x; inversion makes it ~3e+07.
    """
    print("\n[1b] absolute anchors -- the direction cannot be checked by a RELATIVE metric")
    bf16 = _bf16_shared_experts(model)
    print(f"      {'case':22s} {'mean|W|':>9s} {'amax/raw g':>11s} {'bound':>7s} {'vs bf16 shared':>15s}")
    worst_ratio, n_saturated, exact_all = 0.0, 0, True
    lo_bf16, hi_bf16 = float("inf"), 0.0
    for layer, expert in CASES:
        for proj in PROJS:
            packed, sc, g2 = load_leaf(model, layer, expert, proj)
            mul = float(nvfp4.nvfp4_global_multiplier(g2, global_field="weight_scale_2"))
            gold = golden_f64(packed, sc.view(torch.uint8), mul)
            amax = float(gold.abs().max())
            # (1) against the RAW stored scalar -- this is what makes it a DIRECTION test.
            ratio = amax / float(g2)
            # The exact form: amax == FP4_E2M1_MAX * max_block_scale * global.
            bs_max = float(E4M3_F64[int(sc.view(torch.uint8).max())])
            exact = abs(amax - 6.0 * bs_max * mul) <= 1e-12 * amax
            rel_to_bf16 = float(gold.abs().mean()) / float(bf16[(layer, proj)].abs().mean())
            worst_ratio = max(worst_ratio, ratio)
            n_saturated += abs(ratio - 2688.0) < 1e-6
            exact_all &= exact
            lo_bf16, hi_bf16 = min(lo_bf16, rel_to_bf16), max(hi_bf16, rel_to_bf16)
            print(
                f"      L{layer:02d} e{expert:<4d} {proj:10s} {float(gold.abs().mean()):9.6f} "
                f"{ratio:11.2f} {'<=2688' if ratio <= 2688.0 * (1 + 1e-9) else 'VIOLATED':>7s} "
                f"{rel_to_bf16:14.3f}x"
            )
    check_true(
        "amax(|W|) <= 2688 * RAW weight_scale_2",
        worst_ratio <= 2688.0 * (1.0 + 1e-9),
        f"worst {worst_ratio:.2f} of 2688 (=6*448) over {len(CASES) * len(PROJS)} tensors; "
        f"inverting the direction sends this to ~2e+10",
    )
    check_true(
        "amax == FP4_E2M1_MAX * max_block_scale * global, exactly",
        exact_all and n_saturated > 0,
        f"exact on all tensors; {n_saturated} of them SATURATE e4m3 (ratio == 2688 to the bit), "
        f"which is only possible if all three scale levels decode correctly",
    )
    check_true(
        "magnitude matches the layer's UNQUANTIZED bf16 shared expert",
        0.5 <= lo_bf16 and hi_bf16 <= 5.0,
        f"mean|W| is {lo_bf16:.2f}x..{hi_bf16:.2f}x the bf16 anchor (band 0.5..5.0); "
        f"inversion makes it ~3e+07x",
    )


# ---------------------------------------------------------------------------------------------
# 2. Through a GEMV -- reported under BOTH metrics, because they disagree by 3 orders of magnitude.
# ---------------------------------------------------------------------------------------------


def section_gemv(model: str, nvfp4) -> None:
    """The dot product is where a scale error becomes a logit error, so it is measured, not assumed.

    TWO METRICS, DELIBERATELY. The fp16 fold's error is SYSTEMATIC -- all 16 weights of a group share
    one rounded scale -- so it does not average away over a reduction; under a norm metric it carries
    through essentially undiluted (~2e-04..3e-04). The POINTWISE relative error is far larger and far
    noisier because a random dot product occasionally lands near zero: over 32 seeds on a 4-row GEMV
    it spans 3.6e-04..1.7e-01 for the same tensor. The published 1.2e-03..3.2e-03 figure is a draw
    from that heavy tail, so the GATE is on the stable norm metric and the pointwise number is
    printed for continuity only.
    """
    print("\n[2] through a GEMV -- gated on the norm metric, pointwise printed for continuity")
    print(f"      {'case':22s} {'fold relL2':>11s} {'split relL2':>11s} {'ratio':>8s} {'fold ptwise':>12s}")
    worst = float("inf")
    for layer, expert in CASES[:5]:
        for proj in PROJS:
            packed, sc, g2 = load_leaf(model, layer, expert, proj)
            gold, w_fold, w_split, _, _, _ = arms(nvfp4, packed, sc, g2)
            gen = torch.Generator().manual_seed(1234)
            x = torch.randn((4, gold.shape[1]), generator=gen, dtype=torch.float64)
            yg = x @ gold.T
            yf = (x.to(torch.float32) @ w_fold.T).to(torch.float64)
            ys = (x.to(torch.float32) @ w_split.T).to(torch.float64)
            l2 = lambda y: float(((y - yg).norm(dim=1) / yg.norm(dim=1)).max())  # noqa: E731
            f2, s2 = l2(yf), l2(ys)
            pt = float(((yf - yg).abs() / yg.abs().clamp_min(1e-300)).max())
            worst = min(worst, f2 / max(s2, 1e-300))
            print(f"      L{layer:02d} e{expert:<4d} {proj:10s} {f2:11.3e} {s2:11.3e} {f2/max(s2,1e-300):8.1f} {pt:12.3e}")
    check_true(
        "the split's GEMV is closer to the golden than the fold's",
        worst > MIN_ACCURACY_RATIO,
        f"worst margin {worst:.0f}x on per-row relative L2 (gate >{MIN_ACCURACY_RATIO:.0f}x)",
    )


# ---------------------------------------------------------------------------------------------
# 3. CAN THE GATE FAIL? Five real ways to get the split wrong, each must be caught.
# ---------------------------------------------------------------------------------------------


def section_mutants(model: str, nvfp4) -> None:
    """Every one of these loads, runs, and returns finite plausible numbers. That is the point.

    Each mutant must land ABOVE the fp16 fold's own error -- i.e. the gate in section 1 would not
    merely notice it, it would rank the "improvement" as worse than what it replaced.
    """
    print("\n[3] mutation sensitivity -- each must exceed even the fp16 fold's error")
    packed, sc, g2 = load_leaf(model, 0, 0, "gate_proj")
    gold, w_fold, w_split, cs, gvec, _ = arms(nvfp4, packed, sc, g2)
    fold_err = relerr(w_fold, gold)[0]
    base = relerr(w_split, gold)[0]
    print(f"      reference: correct split {base:.3e}, fp16 fold {fold_err:.3e}")

    def mutant(name: str, w: torch.Tensor, why: str) -> None:
        """Gated on the FIXED thresholds section 1 uses, not on the live fold value.

        Coupling this to `fold_err` would make the mutants' verdict depend on the health of the arm
        being replaced -- and once the dense path is converted and `fold_nvfp4_scale` is deleted,
        that reference disappears entirely. The claim being asserted is absolute: every one of these
        lands above the "visibly lossy" line that section 1 rejects on.
        """
        err = relerr(w, gold)[0]
        check_true(
            name,
            err > FOLD_IS_LOSSY_ABOVE and err > MIN_ACCURACY_RATIO * base,
            f"{err:.3e} (fold {fold_err:.3e}, correct {base:.3e}) -- {why}",
        )

    # (a) The FORMAT TRAP: this checkpoint's `weight_scale_2` is a MULTIPLIER, but the documented
    #     compressed-tensors convention is a DIVISOR. Getting it backwards is finite, not NaN.
    mutant(
        "direction inverted (divide, not multiply)",
        nvfp4.dequantize_nvfp4_split(cs["w_packed"], cs["scales"], 1.0 / gvec),
        "the documented sign is the WRONG one for this producer",
    )
    # (b) The global silently lost between loader and kernel (e.g. the `w_zeros` slot never wired).
    mutant(
        "global dropped entirely",
        nvfp4.dequantize_nvfp4_split(cs["w_packed"], cs["scales"], torch.ones_like(gvec)),
        "an unwired w_zeros slot reads as no global at all",
    )
    # (c) e4m3 bytes VALUE-converted instead of bitcast -- byte 126 becomes 126.0, saturating to 448.
    mutant(
        "block scale value-converted, not bitcast",
        nvfp4.dequantize_nvfp4_split(
            cs["w_packed"], sc.view(torch.uint8).to(torch.float32).to(torch.float8_e4m3fn), gvec
        ),
        "`.to(f8)` on raw bytes instead of `.view(f8)`",
    )
    # (d) GRANULARITY, output-channel axis: one scalar for a whole container instead of the N-vector.
    mutant(
        "global collapsed to a container-wide scalar",
        nvfp4.dequantize_nvfp4_split(
            cs["w_packed"], cs["scales"], torch.full_like(gvec, float(gvec[0]) * 0.5)
        ),
        "a scalar cannot describe a merged gate|up range",
    )

    # (e) GRANULARITY, EXPERT axis -- the one the real bytes can prove, because `down_proj` carries
    #     132..281 DISTINCT per-expert globals per layer. This is the `wz_expert` offset bug: the
    #     epilogue reads the global without the per-expert stride and serves expert 0's to all 512.
    print("      [3e] expert-stack granularity on down_proj (the wz_expert offset)")
    n_experts = 8
    golds, corrects, wrongs = [], [], []
    g0 = None
    for e in range(n_experts):
        p, s, g = load_leaf(model, 0, e, "down_proj")
        gd, _, ws, c, gv, _ = arms(nvfp4, p, s, g)
        g0 = gv if g0 is None else g0
        golds.append(gd)
        corrects.append(ws)
        wrongs.append(nvfp4.dequantize_nvfp4_split(c["w_packed"], c["scales"], g0))
    gs, cor, wr = torch.stack(golds), torch.stack(corrects), torch.stack(wrongs)
    ec, ew = relerr(cor, gs)[0], relerr(wr, gs)[0]
    check_true(
        "per-expert global offset is load-bearing",
        ew > FOLD_IS_LOSSY_ABOVE and ec < SPLIT_IS_EXACT_BELOW,
        f"correct {ec:.3e}, expert-0-global-for-all {ew:.3e} over {n_experts} experts",
    )


# ---------------------------------------------------------------------------------------------
# 4. The MERGE CONSTRAINT, on all 512 experts rather than on the brief's word.
# ---------------------------------------------------------------------------------------------


def section_merge_census(model: str, nvfp4) -> None:
    """The fold ran at the LEAF so no per-tensor scalar had to survive the gate|up merge or the
    expert stack. The split discharges that constraint BY SHAPE instead -- the global is a
    per-OUTPUT-CHANNEL vector, and `torch.cat(dim=0)` carries it with no special case -- but only if
    the global really is constant on each contiguous output-channel range. That is a property of THIS
    PRODUCER, so it is counted here on all 512 experts of three layers rather than assumed.
    """
    print("\n[4] weight_scale_2 census over ALL 512 experts (is the N-vector the right contract?)")
    for layer in CENSUS_LAYERS:
        gate, up, down = [], [], []
        for lo in (0, 128, 256, 384):
            with safe_open(shard_for(model, layer, lo), "pt") as f:
                for e in range(lo, lo + 128):
                    p = f"model.language_model.layers.{layer}.mlp.experts.{e}"
                    gate.append(float(f.get_tensor(f"{p}.gate_proj.weight_scale_2")))
                    up.append(float(f.get_tensor(f"{p}.up_proj.weight_scale_2")))
                    down.append(float(f.get_tensor(f"{p}.down_proj.weight_scale_2")))
        same = sum(a == b for a, b in zip(gate, up))
        check_true(
            f"L{layer:02d} gate/up share one global",
            same == 512,
            f"{same}/512 experts | distinct gate={len(set(gate))} up={len(set(up))} "
            f"down={len(set(down))} (down range {min(down):.3e}..{max(down):.3e})",
        )
        check_true(
            f"L{layer:02d} down_proj global is genuinely per-expert",
            len(set(down)) > 100,
            f"{len(set(down))} distinct values -- so a dropped per-expert offset IS measurable",
        )

    # The merged vector itself: gate|up concatenated is constant on each contiguous range, which is
    # exactly what an N-vector expresses and a scalar cannot.
    print("      [4b] the merged gate|up global, built from the real leaves")
    for layer, expert in ((0, 0), (23, 300), (47, 7)):
        parts = []
        for proj in ("gate_proj", "up_proj"):
            _, sc, g2 = load_leaf(model, layer, expert, proj)
            _, gvec = nvfp4.split_nvfp4_scale(sc, g2, global_field="weight_scale_2")
            parts.append(gvec)
        inter = parts[0].shape[0]
        merged = torch.cat(parts, dim=0)  # the loader's own gate|up merge
        ok = (
            merged.shape == (2 * inter,)
            and merged.dtype == torch.float32
            and bool((merged[:inter] == parts[0][0]).all())
            and bool((merged[inter:] == parts[1][0]).all())
        )
        check_true(
            f"L{layer:02d} e{expert} merged global is piecewise-constant",
            ok,
            f"shape {tuple(merged.shape)}, gate={float(parts[0][0]):.4e} up={float(parts[1][0]):.4e}",
        )
    print(
        "      NOTE: gate and up carry EQUAL globals on 1536/1536 experts here, so a merge that\n"
        "      dropped up's global would be invisible TO A VALUE CHECK on these bytes. That case is\n"
        "      covered by tests/core/test_nvfp4_two_level_scale.py, which drives the loader's own\n"
        "      _gate_up_merge with DIFFERENT globals. The granularity failure the real bytes CAN\n"
        "      prove is the per-expert one, and section [3e] does."
    )


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    if not os.path.isfile(os.path.join(model, "config.json")):
        print(f"SKIP: no config.json under {model}")
        return 0
    from minisgl.quant import nvfp4

    print(f"[nvfp4-golden] fp64 golden over raw bytes at {model}")
    section_golden_selfcheck(model, nvfp4)
    section_elementwise(model, nvfp4)
    section_absolute_anchor(model, nvfp4)
    section_gemv(model, nvfp4)
    section_mutants(model, nvfp4)
    section_merge_census(model, nvfp4)
    print(f"\n{'FAILURES: %d' % _failures if _failures else 'ALL CHECKS PASSED'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
