"""The qwen4_exp PLE block RUNS, wired the way the decoder layer wires it, on REAL weights.

`qwen4exp_ple_test.py` pins the block's numerics against a transcription of the reference and drives
`PLERuntime` over a synthetic table; it needs pytest, which is not in the serve image. This is the
complementary end-to-end check, as a plain script, over the pieces that only exist for real:

    the real layer-1 `ple.*` tensors  ->  Qwen4ExpPLE
    the real 51.2 GB NVMe n-gram table -> ShardedRowTable -> PLEEmbeddingSource -> PLERuntime
    a real `Context` with `.ple` set    -> Qwen4ExpPLE.forward(hidden_wide)

That last arrow is the seam nothing else exercises: `forward` reads the staged batch out of
`get_global_ctx().ple` — the same way a GDN layer reads `ctx.gdn_state` — and refuses if none was
staged. Refusing is the right behaviour and is tested elsewhere; what is NOT tested elsewhere is
that the wired path actually produces a finite delta from real weights and real table rows.

Three things are asserted, in increasing strength:

  1. the block runs and its output is finite, non-zero and the right (wide) shape;
  2. the n-gram embeddings that reach it are real table rows, not zeros — a silently-empty gather
     would still produce a finite output (the value path would just be the bias-free zero) and would
     look completely normal;
  3. **six single-token DECODE steps equal one six-token PREFILL of the same tokens.** This is the
     state test, and it is the one that catches everything subtle: a conv window kept at the wrong
     end, a state written before it is read, a token history advanced at the wrong moment, or the
     dilation-9 window mis-sized to the GDN-style k-1 = 3. On real weights, not a fake table.

CPU only — no GPU. The table is mmapped, so the 51.2 GB is page cache, not RSS. Run:

    docker run --rm -v <worktree>:/engine -v <ckpt>:/model:ro \
      -v <ple>:/ple:ro --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_ple_wired_test.py'
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch

MODEL = os.environ.get("Q4E_MODEL", "/model")
PLE_DIR = os.environ.get("Q4E_PLE", "/ple")

_failures = 0


def check(name: str, got, want) -> None:
    global _failures
    ok = got == want
    _failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name:52s} got={got!r:<24} want={want!r}")


def check_true(name: str, cond, detail="") -> None:
    global _failures
    _failures += not cond
    print(f"  {'ok  ' if cond else 'FAIL'} {name:52s} {detail}")


def main() -> int:
    if not os.path.isfile(os.path.join(MODEL, "config.json")):
        print(f"SKIP: no config.json under {MODEL}")
        return 0
    ple_files = sorted(glob.glob(f"{PLE_DIR}/model-plefp8-*.safetensors"))
    meta_files = sorted(glob.glob(f"{PLE_DIR}/model-bf16-*.safetensors"))
    if not ple_files:
        print(f"SKIP: no model-plefp8-*.safetensors under {PLE_DIR}")
        return 0

    import safetensors

    from minisgl import core
    from minisgl.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(0, 1)

    from minisgl.models.config import ModelConfig
    from minisgl.models.qwen4exp import Qwen4ExpPLE
    from minisgl.models.weight import qwen4_exp_remap
    from minisgl.ple import PLEEmbeddingSource, PLERuntime, Qwen4ExpNGramHasher
    from minisgl.utils import cached_load_hf_config

    dev = torch.device("cpu")
    # fp32, not bf16: this is a numerical-equivalence test (decode chain vs prefill) and bf16 would
    # make the tolerance the subject rather than the state handling. The block is dtype-agnostic.
    torch.set_default_dtype(torch.float32)
    mc = ModelConfig.from_hf(cached_load_hf_config(MODEL), spec_algorithm="none")
    print(f"[config] ple_layer_ids={mc.ple_layer_ids} wide={mc.hc_hidden_size} "
          f"embed={mc.ple_embed_dim} ngram={mc.ngram_size} k={mc.ple_conv_kernel_size}")

    # ---- the real layer-1 PLE tensors, through the SAME remap the loader uses ----
    ple = Qwen4ExpPLE(mc)
    want = set(ple.state_dict())
    got: "dict[str, torch.Tensor]" = {}
    for f_path in sorted(glob.glob(f"{MODEL}/model-bf16-*.safetensors")):
        with safetensors.safe_open(f_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if ".ple." not in name:
                    continue
                plan = qwen4_exp_remap(name)
                if plan[0] != "direct":
                    continue
                leaf = plan[1].split(".ple.", 1)[1]
                if leaf in want:
                    t = f.get_tensor(name)
                    got[leaf] = t if t.dtype == torch.int64 else t.to(torch.float32)
    print(f"\n[1] the real layer-1 ple.* tensors load into the block")
    check("tensors found", sorted(got), sorted(want))
    ple.load_state_dict(dict(got))
    check_true(
        "key_proj is real (non-zero, finite)",
        bool(ple.key_proj.weight.abs().sum() > 0 and ple.key_proj.weight.isfinite().all()),
        f"|W|max={ple.key_proj.weight.abs().max():.4f}",
    )

    # ---- the real table + a real runtime ----
    # `from_checkpoint_multipliers` prefers the checkpoint's own `layer_multipliers` AND checks it
    # against the derivation from (vocab_size, ngram_size, layer index, seed). A disagreement means
    # every row id is wrong, which has no other symptom — a wrong multiplier reads a REAL embedding
    # from the wrong row, for every token, and errors nowhere.
    hasher = Qwen4ExpNGramHasher.from_checkpoint_multipliers(
        ngram_size=mc.ngram_size,
        heads_per_ngram=mc.heads_per_ngram,
        eos_token_id=mc.ngram_eos_token_id,
        checkpoint_multipliers=ple.ple_embedding.layer_multipliers.cpu().numpy(),
        vocab_size=mc.vocab_size,
        ple_layer_index=0,
        seed=mc.ngram_seed,
    )
    src = PLEEmbeddingSource(
        ple_files=ple_files,
        meta_files=meta_files,
        hasher=hasher,
        embed_dim=mc.ple_embed_dim,
        max_tokens=64,
        device=dev,
        dtype=torch.float32,
    )
    state = ple.make_state_cache(
        num_slots=4, eos_token_id=mc.ngram_eos_token_id, device=dev, dtype=torch.float32
    )
    rt = PLERuntime(source=src, state=state, max_seqs=4, device=dev)

    # A real `Context`, with `.ple` set exactly as the Engine would set it. This is the seam.
    saved_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    ctx = core.Context(page_size=1)
    ctx.ple = rt
    core.set_global_ctx(ctx)

    try:
        rng = np.random.default_rng(20260903)
        toks = rng.integers(0, mc.vocab_size, size=6, dtype=np.int64)
        hidden = torch.randn(6, mc.hc_hidden_size, dtype=torch.float32)

        print("\n[2] Qwen4ExpPLE.forward runs through ctx.ple on real weights + the real table")
        batch = rt.prepare([1], [toks])
        emb = batch.embeddings
        check("staged embedding shape", tuple(emb.shape), (6, mc.ple_embed_dim))
        # A gather that silently returned zeros still yields a finite output — the value path just
        # collapses — and looks entirely normal downstream. So the ROWS are checked, not the output.
        check_true(
            "gathered rows are real table data, not zeros",
            bool(emb.abs().sum() > 0 and emb.isfinite().all()),
            f"|e|mean={emb.abs().mean():.5f} max={emb.abs().max():.5f} "
            f"nonzero_rows={int((emb.abs().sum(-1) > 0).sum())}/6",
        )
        out = ple.forward(hidden)
        check("output is the WIDE delta", tuple(out.shape), (6, mc.hc_hidden_size))
        check_true(
            "output finite and non-zero",
            bool(out.isfinite().all() and out.abs().sum() > 0),
            f"|out|mean={out.abs().mean():.5f} max={out.abs().max():.5f}",
        )
        rt.commit([1], [toks])
        prefill_out = out.clone()

        print("\n[3] six DECODE steps == one six-token PREFILL (the recurrent-state test)")
        # Slot 2 is a fresh sequence with the same tokens, fed one at a time. Same math, same table
        # rows, different code path: the static index_select/index_copy_ conv instead of F.conv1d.
        rows = []
        for i, tok in enumerate(toks):
            one = np.array([tok], dtype=np.int64)
            rt.prepare([2], [one])
            rows.append(ple.forward(hidden[i : i + 1]))
            rt.commit([2], [one])
        decode_out = torch.cat(rows, dim=0)
        delta = (decode_out - prefill_out).abs().max().item()
        check("decode output shape", tuple(decode_out.shape), tuple(prefill_out.shape))
        check_true(
            "max|decode - prefill| within fp32 accumulation noise",
            delta < 2e-4,
            f"max|Δ|={delta:.3e} (prefill |out|max={prefill_out.abs().max():.4f})",
        )

        print("\n[4] the refusal is still live: no staged batch -> structural error")
        rt.batch = None
        try:
            ple.forward(hidden)
            check_true("PLE.forward with no staged batch refuses", False)
        except RuntimeError as e:
            check_true("PLE.forward with no staged batch refuses", True, str(e).split(".")[0][:70])
    finally:
        core._GLOBAL_CTX = saved_ctx
        src.close()

    print("\n" + ("PASS" if not _failures else f"FAIL ({_failures} checks)"))
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
