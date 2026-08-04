"""The gate|up merge ORDER inside the Gemma4 MoE w13 container, proven against the checkpoint.

CPU-only (streams the checkpoint to CPU, never touches a GPU) — no lease, cannot disturb a serve:

    docker run --rm --entrypoint bash -v <worktree>:/wt \
      -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
      minisgl-rdna4:lean -lc 'cd /wt && PYTHONPATH=/wt/python:/wt/tests:/opt/kernels \
        python tests/gemma4_expert_merge_test.py [tp_size] [rank]'

Why this exists. The loader folds 128 per-expert `experts.<e>.{gate,up}_proj.weight_packed` into ONE
`experts.gate_up_proj.weight_packed` of shape (E, 2*inter, K//8) by `torch.cat([gate, up], dim=0)`
then `torch.stack` over E. Everything downstream splits that container back apart positionally —
the MoE gemm1 epilogue applies `gelu_tanh_and_mul`, i.e. `act(y[:, :inter]) * y[:, inter:]`, and the
HuggingFace reference does the same thing with `linear(x, gate_up_proj[e]).chunk(2, dim=-1)`. NOTHING
in the existing tests pins the two together: `gemma4_loader_test.py` checks key sets, shapes and
dtypes, all of which are IDENTICAL under a swap, and `gemma4_parity_test.py` compares the routed
experts only at the POLICY level (which activation, no second renormalize) because the expert GEMM
is GPU-only. A swapped merge computes `act(up) * gate` — no crash, no shape error, no failing test,
just a quietly wrong model. That is the exact silent-failure class the parity test was written for,
with a hole where the experts are.

What is proven here, and what is NOT:

  PROVEN  the loader's plumbing — the tensor the checkpoint names `gate_proj` lands in rows
          [0:inter] of w13 and `up_proj` in [inter:2*inter], bit-for-bit, for EVERY expert of
          multiple layers, in the packed weights AND their group scales; plus the discrimination
          guard that the two halves genuinely differ (so the bit-identity is not vacuous), and the
          consumer convention (`gelu_tanh_and_mul` / HF `.chunk(2, dim=-1)`) that reads the first
          half as the gated one.
  NOT     that the EXPORTER named them correctly. `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` ships
          per-expert Linears, while transformers' `Gemma4TextExperts` holds ONE fused parameter and
          carries no per-expert conversion mapping — so the split came from the quantizer, and no
          wide-dtype gemma-4 is cached to settle it by measurement the way check [0] of the parity
          test settles the nibble order. A weight-statistic oracle was tried and REJECTED: on the
          unquantized dense MLP (whose names are fixed by Google's own Gemma4TextMLP) gate is
          heavier-tailed than up in 29/30 layers, but that signature does not transfer to the
          int4 experts (3/18), so it cannot arbitrate. Coherent generation covers this residue.
"""

from __future__ import annotations

import glob
import os
import sys

import torch

MODEL_ID = "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4"
MODEL_GLOB = f"/root/.cache/huggingface/hub/models--{MODEL_ID.replace('/', '--')}/snapshots/*/"

CKPT = "model.language_model."  # the gemma4 checkpoint namespace
# Both ends of the stack: layer 0 (sliding) and layer 29 (full attention, last layer). The merge is
# layer-independent, but a buffer-reuse or streaming-order bug would not be.
PROBE_LAYERS = (0, 29)
# Experts dequantized for the numeric checks. First, last, and two interior ids, so an off-by-one in
# the expert stack (which would shift the halves for every id but one) cannot hide.
PROBE_EXPERTS = (0, 1, 63, 127)


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.skips = 0

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:46s} {detail}")
        return ok

    def skip(self, name: str, why: str) -> None:
        self.skips += 1
        print(f"  SKIP {name:46s} {why}")


def _open(folder: str):
    from safetensors import safe_open

    return [
        safe_open(p, framework="pt", device="cpu")
        for p in sorted(glob.glob(folder + "*.safetensors"))
    ]


def _get(handles, key):
    for h in handles:
        if key in h.keys():
            return h.get_tensor(key)
    return None


def _collect_from_loader(tp_size: int, rank: int, wanted: "set[str]") -> "dict[str, torch.Tensor]":
    """Drive the REAL streaming loader and keep only the w13 containers under test.

    Deliberately `load_weight`, not a re-implementation: the merge order is a property of that code
    path (gate/up merge -> per-expert stack -> TP shard), so a test that rebuilt the container itself
    would be checking its own arithmetic. Every other tensor is dropped as it streams, and the scan
    stops as soon as the probe layers have arrived, so peak memory is one container plus the
    loader's own merge buffer."""
    from minisgl.distributed import set_tp_info

    set_tp_info(rank, tp_size)

    import minisgl.layers.rotary as rotary_mod
    from minisgl.models import load_weight

    rotary_mod.set_rope_device(torch.device("cpu"))

    got: "dict[str, torch.Tensor]" = {}
    stream = load_weight(MODEL_ID, torch.device("cpu"), spec_algorithm="none")
    for name, tensor in stream:
        if name in wanted:
            got[name] = tensor
            if len(got) == len(wanted):
                break
        del tensor
    stream.close()
    return got


def main() -> int:
    tp_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    matches = glob.glob(MODEL_GLOB)
    if not matches:
        print(f"SKIP: checkpoint not cached under {MODEL_GLOB}")
        return 0
    path = matches[0]

    from transformers import AutoConfig

    from minisgl.models.config import ModelConfig

    torch.set_default_dtype(torch.float32)
    torch.set_grad_enabled(False)
    hf_cfg = AutoConfig.from_pretrained(path)
    mc = ModelConfig.from_hf(hf_cfg, spec_algorithm="none")
    handles = _open(path)

    rep = Report()
    # inter is the PER-RANK intermediate: plain TP splits gate and up on their output dim BEFORE the
    # merge, so each rank's container is (E, 2*inter/tp, K//8) and the halves are the sharded ones.
    inter = mc.moe_intermediate_size // tp_size
    print(
        f"[expert-merge] {MODEL_ID}  E={mc.num_experts} moe_inter={mc.moe_intermediate_size} "
        f"hidden={mc.hidden_size}  tp={tp_size} rank={rank} -> per-rank inter={inter}"
    )

    fields = ("weight_packed", "weight_scale")
    wanted = {
        f"model.layers.{L}.experts.gate_up_proj.{f}" for L in PROBE_LAYERS for f in fields
    }
    print(f"\n[A] streaming the real loader for {sorted(wanted)}")
    merged = _collect_from_loader(tp_size, rank, wanted)
    rep.check(
        "loader emitted every probed w13 container",
        set(merged) == wanted,
        f"got {len(merged)}/{len(wanted)}: missing={sorted(wanted - set(merged))}",
    )
    if set(merged) != wanted:
        print("\nFAIL (loader did not emit the containers under test)")
        return 1

    # ---------------------------------------------------------------------------------------
    print("\n[B] container geometry (the split point the kernel and the HF reference both assume)")
    for key in sorted(merged):
        t = merged[key]
        packed = key.endswith("weight_packed")
        want = (
            mc.num_experts,
            2 * inter,
            mc.hidden_size // 8 if packed else mc.hidden_size // 32,
        )
        rep.check(
            key.split("layers.")[1],
            tuple(t.shape) == want and t.dtype == (torch.int32 if packed else torch.float16),
            f"shape={tuple(t.shape)} want={want} dtype={t.dtype}",
        )

    # ---------------------------------------------------------------------------------------
    # The whole question, at the bit level: is w13[e, :inter] the tensor the checkpoint calls
    # gate_proj, and w13[e, inter:] the one it calls up_proj — for every expert. int32 packed words
    # and fp16 scales are compared with exact equality; there is no tolerance to hide behind.
    print(f"\n[C] per-expert bit-identity of both halves, all {mc.num_experts} experts")
    for L in PROBE_LAYERS:
        for field in fields:
            w13 = merged[f"model.layers.{L}.experts.gate_up_proj.{field}"]
            bad_gate = bad_up = 0
            swapped_gate = swapped_up = 0
            elems = 0
            for e in range(mc.num_experts):
                p = f"{CKPT}layers.{L}.experts.{e}."
                g = _get(handles, f"{p}gate_proj.{field}")
                u = _get(handles, f"{p}up_proj.{field}")
                if tp_size > 1:  # the loader shards the output dim before the merge
                    g = g.chunk(tp_size, dim=0)[rank]
                    u = u.chunk(tp_size, dim=0)[rank]
                lo, hi = w13[e, :inter], w13[e, inter:]
                bad_gate += int((lo != g).sum())
                bad_up += int((hi != u).sum())
                # ...and the same comparison under the swapped assignment. This is the guard that
                # makes the two counts above mean something: if gate and up happened to be equal,
                # zero mismatches would prove nothing.
                swapped_gate += int((lo != u).sum())
                swapped_up += int((hi != g).sum())
                elems += g.numel()
            rep.check(
                f"L{L} {field}: w13[:{inter}]==gate, w13[{inter}:]==up",
                bad_gate == 0 and bad_up == 0,
                f"mismatching elements gate={bad_gate}/{elems} up={bad_up}/{elems}",
            )
            rep.check(
                f"L{L} {field}: the halves are distinguishable",
                swapped_gate > 0.5 * elems and swapped_up > 0.5 * elems,
                f"under the SWAPPED assignment {100 * swapped_gate / elems:.2f}% / "
                f"{100 * swapped_up / elems:.2f}% of elements disagree — so the identity above is "
                f"a real constraint, not two equal tensors",
            )

    # ---------------------------------------------------------------------------------------
    # Bit-identity of the packed words settles the plumbing; this settles what the plumbing MEANS,
    # by running the dequant the kernel's decode implements and then the gated FFN itself.
    print("\n[D] dequantized expert FFN — correct order vs the swap that would not crash")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gemma4_parity_test import dequant_ct_int4  # nibble convention settled by measurement there

    from minisgl.layers.activation import gelu_tanh_and_mul

    torch.manual_seed(17)
    x = torch.randn(8, mc.hidden_size)
    for L in PROBE_LAYERS:
        w13p = merged[f"model.layers.{L}.experts.gate_up_proj.weight_packed"]
        w13s = merged[f"model.layers.{L}.experts.gate_up_proj.weight_scale"]
        for e in PROBE_EXPERTS:
            p = f"{CKPT}layers.{L}.experts.{e}."
            sl = slice(None) if tp_size == 1 else slice(rank * inter, (rank + 1) * inter)
            deq = lambda m: dequant_ct_int4(
                _get(handles, f"{p}{m}.weight_packed")[sl],
                _get(handles, f"{p}{m}.weight_scale")[sl],
            )
            gate, up = deq("gate_proj"), deq("up_proj")
            # ...and the same two halves as the kernel gets them: out of the merged container.
            stacked = dequant_ct_int4(w13p[e], w13s[e])
            d_gate = (stacked[:inter] - gate).abs().max().item()
            d_up = (stacked[inter:] - up).abs().max().item()
            rep.check(
                f"L{L} e{e}: dequantized halves",
                d_gate == 0.0 and d_up == 0.0,
                f"max|abs| gate={d_gate:.3e} up={d_up:.3e}  |gate|max={gate.abs().max():.4f} "
                f"|up|max={up.abs().max():.4f}",
            )
            if e != PROBE_EXPERTS[0]:
                continue
            # What a swap would actually cost. `down` is row-parallel (input dim), so at tp>1 it is
            # sliced on dim 1 to match the sharded intermediate.
            down_full = dequant_ct_int4(
                _get(handles, f"{p}down_proj.weight_packed"),
                _get(handles, f"{p}down_proj.weight_scale"),
            )
            down = down_full if tp_size == 1 else down_full[:, sl]
            pre = torch.cat([x @ gate.T, x @ up.T], dim=-1)
            right = gelu_tanh_and_mul(pre) @ down.T
            wrong = gelu_tanh_and_mul(torch.cat([pre[:, inter:], pre[:, :inter]], -1)) @ down.T
            cos = torch.nn.functional.cosine_similarity(
                right.flatten(), wrong.flatten(), dim=0
            ).item()
            rep.check(
                f"L{L} e{e}: act(up)*gate is WRONG but plausible",
                (right - wrong).norm().item() > 0.1 * right.norm().item(),
                f"rel_fro={((right - wrong).norm() / right.norm()).item():.4f}  cos={cos:.4f}  "
                f"|right|max={right.abs().max():.4f} |wrong|max={wrong.abs().max():.4f} — no NaN, "
                f"no crash, same shape: it would only show up as quality",
            )

    # ---------------------------------------------------------------------------------------
    # Both consumers of the container split it positionally; pin that the FIRST half is the gated
    # one on both sides, so [C] is anchored to what actually runs.
    print("\n[E] consumer convention: the FIRST half is the gated one")
    torch.manual_seed(23)
    probe = torch.randn(16, 2 * 32) * 3
    ours = gelu_tanh_and_mul(probe)
    try:
        from transformers.activations import ACT2FN
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

        hf_gate, hf_up = probe.chunk(2, dim=-1)  # exactly Gemma4TextExperts.forward
        hf = ACT2FN["gelu_pytorch_tanh"](hf_gate) * hf_up
        # The LAYOUT must match too, or "first half" would mean different things on the two sides.
        # HF holds (E, 2*inter, hidden) and chunks the linear OUTPUT, which is this container's row
        # split — read off the real module rather than asserted from the source.
        with torch.device("meta"):
            hf_shape = tuple(Gemma4TextExperts(hf_cfg.text_config).gate_up_proj.shape)
        rep.check(
            "gelu_tanh_and_mul == HF chunk(2,-1) -> act(gate)*up",
            (ours - hf).abs().max().item() == 0.0
            and hf_shape == (mc.num_experts, 2 * mc.moe_intermediate_size, mc.hidden_size),
            f"max|abs|={(ours - hf).abs().max().item():.3e}  HF gate_up_proj{hf_shape} vs this "
            f"container ({mc.num_experts}, {2 * inter}, {mc.hidden_size} packed) at tp={tp_size}",
        )
        swapped = ACT2FN["gelu_pytorch_tanh"](hf_up) * hf_gate
        rep.check(
            "the two orders are not accidentally equal",
            (ours - swapped).abs().max().item() > 1.0,
            f"act(up)*gate differs from act(gate)*up by max|abs|="
            f"{(ours - swapped).abs().max().item():.3e} on the same probe",
        )
    except ImportError as exc:  # pragma: no cover - image without the gemma4 reference
        rep.skip("HF Gemma4TextExperts convention", f"reference not importable: {exc}")

    print(
        f"\n{'PASS' if rep.failures == 0 else f'FAIL ({rep.failures} checks)'}"
        f"{f'  [{rep.skips} skipped]' if rep.skips else ''}"
    )
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
