"""MXFP4-FP8-GPTQ Qwen3.8-Flash-Next: the remap, the chunk plan, and the TP SHARD RULES.

`qwen4exp_remap_test.py` covers the base NVFP4 checkpoint. This covers the repacked one
(`tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ`), which differs in four structural ways, each of
which the loader gets wrong in a DIFFERENT and mostly silent manner if unhandled:

  * routed experts ship PRE-STACKED and gate/up-FUSED ([E, N, K']), so every TP axis is one higher
    than the per-expert 2-D rules;
  * attention and GDN projections ship blockwise fp8 with a (N/128, K/128) `weight_scale_inv`;
  * the shared expert is MXFP4 (the base checkpoint leaves it bf16);
  * a routing-profile sidecar ships alongside the weights.

WHY THE SHARD ASSERTIONS LOOK THE WAY THEY DO. A wrong shard axis does not raise and does not
produce NaN -- each rank quietly computes over the wrong columns. Against a `randn` fixture, "rank
0 got a tensor of the right shape" is satisfied by several wrong answers at once, so every fixture
here is INDEX-ENCODED: element [e, n, k] holds e*10000 + n*100 + k, which makes the identity of
every element recoverable and a wrong axis, a wrong half, or a transposed split provable rather
than merely unlikely. The expected tensors are built by independent indexing, not by calling the
same expression the implementation uses.

Run CPU-only in the serve image:

    docker run --rm -v <worktree>:/engine --entrypoint bash minisgl-rdna4:lean -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_mxfp4_remap_test.py'
"""

from __future__ import annotations

import collections
import gzip
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "qwen4exp")
KEYS_GZ = os.path.join(FIXTURES, "ckpt_keys_mxfp4.txt.gz")

_failures = 0


def check(name: str, got, want) -> None:
    global _failures
    ok = torch.equal(got, want) if isinstance(want, torch.Tensor) else got == want
    if not ok:
        _failures += 1
        if isinstance(want, torch.Tensor):
            print(f"  FAIL {name}\n       got  shape {tuple(got.shape)}\n       want shape {tuple(want.shape)}")
            if got.shape == want.shape:
                bad = (got != want).nonzero()[:4].tolist()
                print(f"       first differing indices {bad}")
        else:
            print(f"  FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok   {name}")


def raises(name: str, fn) -> None:
    global _failures
    try:
        fn()
    except Exception as e:
        print(f"  ok   {name} -> {type(e).__name__}")
        return
    _failures += 1
    print(f"  FAIL {name}: returned instead of raising")


def load_fixture() -> "list[tuple[str, str, tuple, str]]":
    rows = []
    with gzip.open(KEYS_GZ, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, dtype, shape, fn = line.rstrip("\n").split("\t")
            rows.append((name, dtype, tuple(int(x) for x in shape.split(",") if x), fn))
    return rows


def _config():
    """Only the attributes `_shard_qwen4_exp` actually reads, with this checkpoint's real values."""
    quant = types.SimpleNamespace(block_structure=[128, 128])
    return types.SimpleNamespace(
        linear_key_head_dim=128, linear_num_key_heads=16,
        linear_value_head_dim=128, linear_num_value_heads=48,
        quant=quant,
    )


def indexed(*shape: int) -> torch.Tensor:
    """A tensor whose every element names its own position: [e,n,k] -> e*10000 + n*100 + k.

    Deliberately not `randn`: a wrong shard axis produces a right-SHAPED tensor, and against random
    data several wrong answers are indistinguishable from the right one.
    """
    t = torch.zeros(*shape, dtype=torch.int64)
    if len(shape) == 3:
        for e in range(shape[0]):
            for n in range(shape[1]):
                for k in range(shape[2]):
                    t[e, n, k] = e * 10000 + n * 100 + k
    elif len(shape) == 2:
        for n in range(shape[0]):
            for k in range(shape[1]):
                t[n, k] = n * 100 + k
    else:
        raise AssertionError("2-D or 3-D only")
    return t


def main() -> int:
    from minisgl.models.weight import (
        _QWEN4EXP_NATIVE_OK,
        _Q4_SHARED_DEQ,
        _q4_is_stacked_expert_key,
        _q4_stacked_expert_layer,
        _Q4_STACKED_LAYERS_PER_CHUNK,
        _shard_qwen4_exp,
        qwen4_exp_remap,
    )
    import re

    rows = load_fixture()
    print(f"\n== fixture: {len(rows)} tensors from the MXFP4-FP8-GPTQ headers")

    # ---- 1. every key gets a plan, and the skip ledger is stated ---------------------------
    print("\n== every checkpoint key resolves")
    nvfp4_modules: "frozenset[str]" = frozenset()  # this checkpoint is MXFP4, not NVFP4
    reasons: collections.Counter = collections.Counter()
    unrecognised = []
    for name, _dt, _sh, _fn in rows:
        base, _, field = name.rpartition(".")
        emitted = [base + ".weight"] if (_Q4_SHARED_DEQ in name
                                         and field in ("weight_packed", "weight_scale")) else [name]
        for nk in emitted:
            try:
                plan = qwen4_exp_remap(nk, nvfp4_modules=nvfp4_modules)
            except Exception as e:
                unrecognised.append((name, f"{type(e).__name__}: {e}"))
                continue
            reasons[plan[1] if plan[0] == "skip" else plan[0]] += 1
    check("unrecognised keys", len(unrecognised), 0)
    for k, v in unrecognised[:5]:
        print(f"       {k}: {v}")
    print("       ledger: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    # The MTP head must be skipped WHOLESALE -- its per-expert fp8 blockwise tensors and blockwise
    # shared expert are forms the dequant fold does not handle, and they are fine ONLY because
    # nothing reaches it. If an mtp key ever stops being skipped this assertion is the alarm.
    mtp = [n for n, _d, _s, _f in rows if n.startswith("mtp.")]
    mtp_skipped = sum(1 for n in mtp if qwen4_exp_remap(n, nvfp4_modules=nvfp4_modules)[0] == "skip")
    check("every mtp.* key skipped", mtp_skipped, len(mtp))
    print(f"       ({len(mtp)} mtp tensors, incl. per-expert fp8 blockwise + a blockwise shared expert)")

    # ---- 2. the shared expert is admitted as bf16 `.weight` ONLY ---------------------------
    print("\n== shared expert: folded to bf16, packed leaves NOT admitted")
    pats = [re.compile(p) for p in _QWEN4EXP_NATIVE_OK]
    def admitted(k): return any(p.match(k) for p in pats)
    check("shared_expert.gate_proj.weight admitted",
          admitted("model.layers.0.mlp.shared_expert.gate_proj.weight"), True)
    for leaf in ("weight_packed", "weight_scale"):
        check(f"shared_expert.gate_proj.{leaf} NOT admitted",
              admitted(f"model.layers.0.mlp.shared_expert.gate_proj.{leaf}"), False)
    n_shared_packed = sum(1 for n, _d, _s, _f in rows
                          if _Q4_SHARED_DEQ in n and not n.startswith("mtp.")
                          and n.endswith(("weight_packed", "weight_scale")))
    check("backbone shared-expert packed leaves in the checkpoint", n_shared_packed, 48 * 3 * 2)

    # ---- 3. stacked expert TP shard rules --------------------------------------------------
    print("\n== pre-stacked expert shard rules (index-encoded, so a wrong axis is provable)")
    cfg, E, H, I = _config(), 3, 8, 4      # 3 experts, hidden 8, intermediate 4
    # gate_up: [E, 2*I, H] -- output axis is dim 1 and is [gate | up] CONCATENATED.
    gu = indexed(E, 2 * I, H)
    for r in (0, 1):
        got = _shard_qwen4_exp("model.layers.0.mlp.experts.gate_up_proj_packed", gu, r, 2, cfg)
        # Independent expectation: take rank r's half of gate, then rank r's half of up.
        half = I // 2
        want = torch.cat([gu[:, r * half:(r + 1) * half, :],
                          gu[:, I + r * half: I + (r + 1) * half, :]], dim=1)
        check(f"gate_up rank {r} splits each half on dim 1", got, want)
    # The hazard the per-half split exists for: a plain chunk gives rank 0 ALL of gate.
    naive = gu.chunk(2, dim=1)[0]
    got0 = _shard_qwen4_exp("model.layers.0.mlp.experts.gate_up_proj_packed", gu, 0, 2, cfg)
    check("gate_up rank 0 is NOT a plain chunk(dim=1) (which would be all-gate)",
          torch.equal(got0, naive), False)
    # down_proj: [E, H, I] -- INPUT axis is dim 2; dim 1 is hidden and must stay whole.
    dn = indexed(E, H, I)
    for r in (0, 1):
        got = _shard_qwen4_exp("model.layers.0.mlp.experts.down_proj_packed", dn, r, 2, cfg)
        want = dn[:, :, r * (I // 2):(r + 1) * (I // 2)]
        check(f"down_proj rank {r} splits dim 2 (input), hidden whole", got, want)
        check(f"down_proj rank {r} keeps all {H} hidden rows", got.shape[1], H)
    raises("gate_up with an odd output axis raises",
           lambda: _shard_qwen4_exp("model.layers.0.mlp.experts.gate_up_proj_packed",
                                    indexed(E, 5, H), 0, 2, cfg))
    raises("down_proj with an indivisible input axis raises",
           lambda: _shard_qwen4_exp("model.layers.0.mlp.experts.down_proj_packed",
                                    indexed(E, H, 3), 0, 2, cfg))

    # ---- 4. blockwise-fp8 `weight_scale_inv` shard rules -----------------------------------
    print("\n== weight_scale_inv shard rules (block units, same axis as the weight)")
    # in_proj_qkv: rows are [q | k | v] head blocks in 128-row scale-block units.
    # key_dim = 128*16 = 2048 -> 16 blocks; value_dim = 128*48 = 6144 -> 48 blocks. Total 80.
    sc = indexed(16 + 16 + 48, 20)
    for r in (0, 1):
        got = _shard_qwen4_exp("model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv",
                               sc, r, 2, cfg)
        # Independent expectation: rank r's half of EACH of the three blocks, in order.
        want = torch.cat([sc[r * 8:(r + 1) * 8],                     # q: 16 blocks / 2
                          sc[16 + r * 8: 16 + (r + 1) * 8],          # k: 16 blocks / 2
                          sc[32 + r * 24: 32 + (r + 1) * 24]], 0)    # v: 48 blocks / 2
        check(f"in_proj_qkv scale rank {r} splits q/k/v blocks separately", got, want)
    naive_q = sc.chunk(2, dim=0)[0]
    check("in_proj_qkv scale rank 0 is NOT a plain chunk(dim=0)",
          torch.equal(_shard_qwen4_exp("model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv",
                                       sc, 0, 2, cfg), naive_q), False)
    col = indexed(96, 20)
    check("q_proj scale is column-parallel (dim 0)",
          _shard_qwen4_exp("model.layers.0.self_attn.q_proj.weight_scale_inv", col, 1, 2, cfg),
          col.chunk(2, dim=0)[1])
    row = indexed(20, 48)
    check("o_proj scale is row-parallel (dim 1)",
          _shard_qwen4_exp("model.layers.0.self_attn.o_proj.weight_scale_inv", row, 1, 2, cfg),
          row.chunk(2, dim=1)[1])
    check("out_proj scale is row-parallel (dim 1)",
          _shard_qwen4_exp("model.layers.0.linear_attn.out_proj.weight_scale_inv", row, 1, 2, cfg),
          row.chunk(2, dim=1)[1])
    # A module with no rule must RAISE, never replicate: replication mis-scales every row outside
    # rank 0's shard, which is silent and fluent.
    raises("weight_scale_inv on an unruled module raises",
           lambda: _shard_qwen4_exp("model.layers.0.mlp.some_new_proj.weight_scale_inv",
                                    row, 0, 2, cfg))

    # ---- 5. the chunk plan partitions the native keys --------------------------------------
    print("\n== chunk plan partitions the native keys")
    native = set()
    for name, _d, _s, _f in rows:
        base, _, field = name.rpartition(".")
        emitted = [base + ".weight"] if (_Q4_SHARED_DEQ in name
                                         and field in ("weight_packed", "weight_scale")) else [name]
        for nk in emitted:
            plan = qwen4_exp_remap(nk, nvfp4_modules=nvfp4_modules)
            if plan[0] != "skip" and len(plan) > 1 and isinstance(plan[1], str):
                native.add(plan[1])
    num_layers = 48
    step = _Q4_STACKED_LAYERS_PER_CHUNK
    filters = [lambda k: not _q4_is_stacked_expert_key(k)]
    for i in range(0, num_layers, step):
        want_l = frozenset(range(i, min(i + step, num_layers)))
        filters.append(lambda k, _w=want_l: (l := _q4_stacked_expert_layer(k)) is not None and l in _w)
    counts = collections.Counter(sum(1 for f in filters if f(k)) for k in native)
    check("every native key claimed by exactly one chunk", dict(counts), {1: len(native)})
    # NON-VACUITY: the body filter is the negation of the expert one, so it would claim everything
    # and this check would pass while proving nothing if no key were a stacked-expert key at all.
    claimed = sum(1 for k in native if _q4_is_stacked_expert_key(k))
    check("expert chunks claim 4 leaves x 48 layers", claimed, 4 * num_layers)

    print(f"\n{'FAILURES: %d' % _failures if _failures else 'all checks passed'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
