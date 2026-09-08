"""GPU-ONLY: the granule descriptor over the REAL register-direct repacks.

Every other granule test runs on CPU. These four cannot: the `_w_rep` buffers are built by the
compiled `fp8_wmma` repack kernels (`repack_int4_to_w_rep_moe`,
`mxfp4_to_w_rep_moe`, `repack_fp8_to_w_rep_moe`), which need a device. They close the one gap in
`test_granule_spec.py`, whose `test_regdirect_shapes` SYNTHESIZES the post-regdirect buffer set
rather than producing it.

Marked `gpu`, therefore EXCLUDED from a default `pytest` run (`addopts = -m "not gpu"`). The two
gfx1201 cards are shared across every repo on this box; run these deliberately and SERIALLY, under
the booking protocol in CLAUDE.md:

    gpu-lease -n 1 -- docker run --rm --device /dev/kfd --device /dev/dri --group-add video \\
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \\
      --ipc host --shm-size 16gb \\
      -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \\
      -v <worktree>:/engine --entrypoint bash minisgl-rdna4:lean \\
      -lc 'PYTHONPATH=/engine/python:/opt/kernels pytest -m gpu /engine/tests/weights/'

These assert only the DESCRIPTOR (which buffers survive, that dim 0 is E, that scales travel with
weights) — not kernel numerics, which the format's own parity tests own.
"""

from __future__ import annotations

import pytest
import torch
from minisgl.layers import moe as moe_mod
from minisgl.layers.moe import (
    _GroupedCompressedTensorsExperts,
    _GroupedFP8Experts,
    _GroupedMxFp4Experts,
)
from minisgl.quant.config import QuantConfig
from minisgl.weights.granule import spec_for_container

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="regdirect repacks need a device"),
]

E, N, K = 8, 128, 512


def _rand(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    if dtype in (torch.int32, torch.uint8):
        hi = 2**31 - 1 if dtype == torch.int32 else 255
        t = torch.randint(0, hi, shape, generator=g, dtype=torch.int64).to(dtype)
    elif dtype == torch.float8_e4m3fn:
        t = torch.randn(shape, generator=g).to(dtype)
    else:
        t = torch.randn(shape, generator=g).to(dtype)
    return t.cuda()


def _fill_on_device(container, seed=0):
    for i, (name, v) in enumerate(list(vars(container).items())):
        if isinstance(v, torch.Tensor):
            setattr(container, name, _rand(tuple(v.shape), v.dtype, seed + i))
    container.post_load()
    return container


def _assert_granule_is_sane(spec, expect_present, expect_absent):
    names = {n for c in spec.components for n in c.names}
    for want in expect_present:
        assert want in names, f"{want} missing from the granule; components={sorted(names)}"
    for gone in expect_absent:
        assert gone not in names, f"{gone} should have been dropped by the regdirect repack"
    for c in spec.components:
        assert c.stacked_shape[0] == E
    assert spec.granule_bytes > 0


def test_ct_int4_w4a16_regdirect(monkeypatch):
    monkeypatch.setattr(moe_mod.kernels, "MOE_W4A16", "1", raising=False)
    q = QuantConfig(method="compressed-tensors", bits=4, group_size=32, sym=False)
    c = _fill_on_device(_GroupedCompressedTensorsExperts(E, N, K, q))
    _assert_granule_is_sane(
        spec_for_container(c, E), ["_w_rep", "_scales_op", "_zeros_op"], ["_w_op", "weight_packed"]
    )


def test_mxfp4_regdirect(monkeypatch):
    monkeypatch.setattr(moe_mod.kernels, "MOE_MXFP4_REGDIRECT", True, raising=False)
    q = QuantConfig(method="compressed-tensors", bits=4, group_size=32, sym=True,
                    weight_type="float", ct_format="mxfp4-pack-quantized")
    c = _fill_on_device(_GroupedMxFp4Experts(E, N, K, q))
    _assert_granule_is_sane(
        spec_for_container(c, E), ["_w_rep", "_scales_rd"], ["weight_packed", "weight_scale"]
    )


def test_fp8_w8a8_regdirect(monkeypatch):
    monkeypatch.setattr(moe_mod.kernels, "MOE_W8A8_REGDIRECT", True, raising=False)
    monkeypatch.delenv("MINISGL_ZAYA_OLDMOE", raising=False)
    monkeypatch.delenv("MINISGL_ZAYA_W8A16", raising=False)
    c = _fill_on_device(_GroupedFP8Experts(E, N, K))
    _assert_granule_is_sane(spec_for_container(c, E), ["_w_rep", "_scales_op"], ["weight", "_w_op"])
    assert c.offload_refusal() is None
