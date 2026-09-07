"""`weights/cpu_native.NativeVnniBackend` — the BINDING between the engine's tensors and the `.so`.

WHY THIS FILE EXISTS. `tools/cpu_moe/validate_so.py` proves the C core is numerically right and
`test_cpu_moe_reference.py` proves the float64 reference backend is, but NOTHING covered the class
the serve actually instantiates: the ctypes signatures, the policy selection, the in-place pack over
the engine's `(E, N, K//8)` int32 / group-major-scale layout, and the pointer plumbing in `compute`.
That gap cost a full 6-minute TP=2 checkpoint load to surface a one-line `AttributeError` in
`__init__` — a class of bug this file catches in about a second, with no GPU and no checkpoint.

The oracle is a float64 dequant of the SAME synthetic bytes, so a layout error (wrong nibble order,
wrong scale gather, a global applied to the wrong channels) shows up as a large error rather than as
plausible output. Synthetic weights are the right choice HERE precisely because the checkpoint-bytes
question is already answered elsewhere: what is under test is the plumbing.

SKIPS, LOUDLY, if `libcpumoe.so` is absent — build it with `make -C tools/cpu_moe`.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from minisgl.weights.cpu_tier import CpuTierError  # noqa: E402

_cn = pytest.importorskip("minisgl.weights.cpu_native")

E2M1 = np.array([0.0, .5, 1., 1.5, 2., 3., 4., 6., -0.0, -.5, -1., -1.5, -2., -3., -4., -6.],
                dtype=np.float64)

H, I, E, TOPK = 64, 32, 4, 2


def _so_or_skip() -> str:
    try:
        return _cn.find_library()
    except CpuTierError as e:
        pytest.skip(f"libcpumoe.so not built: {e}")


def _e4m3_lut() -> np.ndarray:
    out = np.empty(256, dtype=np.float64)
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        e, m = (b >> 3) & 0xF, b & 0x7
        if e == 0:
            v = m / 8.0 * 2.0 ** -6
        elif e == 15 and m == 7:
            v = 0.0
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        out[b] = s * v
    return out


E4M3 = _e4m3_lut()


class _FakeExperts:
    """The three attributes `pack_layer` reads off a `_GroupedNvFp4Experts`, and nothing else.

    Deliberately not the real container: building one needs a checkpoint and a `QuantConfig`, and
    what is under test is the BINDING's reading of the layout, not the container's construction.
    The layout claims are asserted here so this stays honest if `post_load` ever changes:
        _w_op      (E, N, K//8) int32   — a byte view of (N, K/2) packed E2M1 nibbles
        _scales_op (E, K//g, N)         — GROUP-MAJOR
        _global_op (E, N) int32         — f32 BITCAST (it rides the op's `w_zeros` pointer slot)
    """

    def __init__(self, n: int, k: int, *, e4m3: bool, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.n, self.k = n, k
        self.codes = rng.integers(0, 256, size=(E, n, k // 2), dtype=np.uint8)
        self._w_op = torch.from_numpy(self.codes.copy()).view(torch.int32).reshape(E, n, k // 8)
        if e4m3:
            # POSITIVE-NORMAL e4m3 only (0x08..0x7E) — the range the core's fast decode is
            # specialised to and the packer refuses outside of. The real checkpoint occupies
            # 0x4A..0x7E; drawing wider here would test the guard, not the arithmetic.
            self.block = rng.integers(0x40, 0x7F, size=(E, n, k // 16), dtype=np.uint8)
            self._scales_op = (torch.from_numpy(self.block.copy())
                               .view(torch.float8_e4m3fn).transpose(1, 2).contiguous())
            # Per-OUTPUT-CHANNEL and genuinely varying: a constant vector would pass even if the
            # core applied channel 0's multiplier to every row.
            self.gvec = (rng.random((E, n)) * 1e-3 + 1e-5).astype(np.float32)
            self._global_op = torch.from_numpy(self.gvec.copy()).contiguous().view(torch.int32)
        else:
            self.fp16 = (rng.random((E, n, k // 16)) * 0.02 + 1e-3).astype(np.float16)
            self._scales_op = torch.from_numpy(self.fp16.copy()).transpose(1, 2).contiguous()
            self.gvec = np.ones((E, n), dtype=np.float32)
        self.e4m3 = e4m3

    def dequant_f64(self) -> np.ndarray:
        """(E, N, K) float64 — the same bytes, decoded independently of the core."""
        c = np.empty((E, self.n, self.k), dtype=np.uint8)
        c[:, :, 0::2] = self.codes & 0x0F
        c[:, :, 1::2] = self.codes >> 4
        w = E2M1[c]
        if self.e4m3:
            s = np.repeat(E4M3[self.block], 16, axis=2)
            return w * s * self.gvec.astype(np.float64)[:, :, None]
        return w * np.repeat(self.fp16.astype(np.float64), 16, axis=2)


def _quant_act_int8(x):
    g = x.reshape(-1, 16)
    amax = np.abs(g).max(axis=1)
    inv = np.where(amax > 0, 127.0 / np.maximum(amax, 1e-300), 0.0)
    return (np.rint(g * inv[:, None]).clip(-128, 127) * (amax / 127.0)[:, None]).reshape(-1)


def _oracle(W13, W2, x, ids, rw):
    y = np.zeros(H, dtype=np.float64)
    xu = _quant_act_int8(x.astype(np.float64))
    for e, ww in zip(ids, rw):
        gu = W13[e] @ xu
        g, u = gu[:I], gu[I:]
        y += ww * (W2[e] @ _quant_act_int8(g / (1.0 + np.exp(-g)) * u))
    return y


def _backend(dtype):
    return _cn.NativeVnniBackend(H, I, TOPK, 1, [1], so_path=_so_or_skip(), scales_dtype=dtype)


def _run(e4m3: bool):
    dt = torch.float8_e4m3fn if e4m3 else torch.float16
    w13 = _FakeExperts(2 * I, H, e4m3=e4m3, seed=11)
    w2 = _FakeExperts(H, I, e4m3=e4m3, seed=22)
    W13, W2 = w13.dequant_f64(), w2.dequant_f64()   # BEFORE the pack permutes the bytes in place
    b = _backend(dt)
    try:
        b.pack_layer(0, w13, w2, E)
        assert b.num_layers == 1
        rng = np.random.default_rng(7)
        x = (rng.standard_normal(H) * 0.02).astype(np.float32)
        ids = rng.choice(E, size=TOPK, replace=False).astype(np.int32)
        rw = rng.random(TOPK).astype(np.float32)
        rw /= rw.sum()
        out = b.compute(torch.from_numpy(x).reshape(1, H),
                        torch.from_numpy(ids).reshape(1, TOPK),
                        torch.from_numpy(rw).reshape(1, TOPK), layer=0)
        got = out.numpy().reshape(-1).astype(np.float64)
        ref = _oracle(W13, W2, x, ids, rw)
        rel = float(np.sqrt(np.mean((got - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))
        return b, rel
    finally:
        pass


class TestNativeVnniBackend:
    def test_e4m3_policy_matches_the_float64_oracle(self):
        """THE ONE THAT MATTERS: the checkpoint-native two-level scale, end to end through ctypes.

        The tolerance is 1e-6 and not 1e-2 because the oracle quantizes the activations the same
        way the core does — so this bounds the ARITHMETIC (tile layout, +16 bias correction, both
        scale foldings), not the int8 activation cost, which is measured on real weights in
        tools/cpu_moe/validate_so.py.
        """
        b, rel = _run(True)
        try:
            assert b.policy.name == "vnni_nvfp4_e4m3_g16"
            assert rel < 1e-6, f"rel_rms {rel:.3e}"
        finally:
            b.close()

    def test_fp16_policy_still_works(self):
        """The A/B comparand. Adding the e4m3 policy must not have moved the folded-scale path."""
        b, rel = _run(False)
        try:
            assert b.policy.name == "vnni_nvfp4_fp16_g16"
            assert rel < 1e-6, f"rel_rms {rel:.3e}"
        finally:
            b.close()

    def test_the_per_channel_global_is_load_bearing(self):
        """Perturb ONE output channel's global and the answer must move.

        Without this, a core that ignored `_global_op` entirely, or applied channel 0's value to
        all N, would pass every other test in this file as long as the oracle made the same
        mistake — and the real-world cost of that is a uniform ~4.8e3x weight error that produces
        fluent, wrong text rather than a crash.
        """
        w13 = _FakeExperts(2 * I, H, e4m3=True, seed=11)
        w2 = _FakeExperts(H, I, e4m3=True, seed=22)
        b = _backend(torch.float8_e4m3fn)
        try:
            g = w2._global_op.view(torch.float32)
            g[0, 0] *= 4.0                      # one expert, one output channel
            b.pack_layer(0, w13, w2, E)
            rng = np.random.default_rng(7)
            x = (rng.standard_normal(H) * 0.02).astype(np.float32)
            ids = np.array([0, 1], dtype=np.int32)
            rw = np.array([0.5, 0.5], dtype=np.float32)
            args = (torch.from_numpy(x).reshape(1, H), torch.from_numpy(ids).reshape(1, TOPK),
                    torch.from_numpy(rw).reshape(1, TOPK))
            before = b.compute(*args, layer=0).numpy().copy()
            g[0, 0] /= 4.0
            after = b.compute(*args, layer=0).numpy()
            # Channel 0 of the down projection moved; nothing else did.
            assert abs(before[0, 0] - after[0, 0]) > 1e-9, "the global was never read"
            assert np.allclose(before[0, 1:], after[0, 1:], atol=0, rtol=0), \
                "a per-CHANNEL global changed other channels: it is being applied per matrix"
        finally:
            b.close()

    def test_a_scale_dtype_with_no_policy_is_refused(self):
        _so_or_skip()
        with pytest.raises(CpuTierError, match="no CPU-core WLoad policy"):
            _backend(torch.bfloat16)

    def test_omitting_the_scale_dtype_is_refused_rather_than_defaulted(self):
        """There is no default. The old default WAS fp16, which is the silent-wrong-answer."""
        _so_or_skip()
        with pytest.raises(CpuTierError, match="needs `scales_dtype`"):
            _cn.NativeVnniBackend(H, I, TOPK, 1, [1], so_path=_so_or_skip(), scales_dtype=None)

    def test_a_layer_whose_scales_disagree_with_the_worker_is_refused(self):
        """One worker serves ONE scale encoding, because the core is instantiated per policy."""
        b = _backend(torch.float16)
        try:
            w13 = _FakeExperts(2 * I, H, e4m3=True, seed=11)
            w2 = _FakeExperts(H, I, e4m3=True, seed=22)
            with pytest.raises(CpuTierError, match="opened for policy"):
                b.pack_layer(0, w13, w2, E)
        finally:
            b.close()

    def test_an_e4m3_layer_with_no_global_is_refused(self):
        b = _backend(torch.float8_e4m3fn)
        try:
            w13 = _FakeExperts(2 * I, H, e4m3=True, seed=11)
            w2 = _FakeExperts(H, I, e4m3=True, seed=22)
            del w13._global_op
            with pytest.raises(CpuTierError, match="no `_global_op`"):
                b.pack_layer(0, w13, w2, E)
        finally:
            b.close()

    def test_counters_report_both_sides_of_the_seam(self):
        """`engaged()` saturates at one; these are what distinguish 1 layer from 12."""
        b, _ = _run(True)
        try:
            c = b.counters()
            assert c["layers_registered"] == 1
            assert c["python_layer_calls"] == 1 == c["native_layer_calls"]
            assert c["python_tokens"] == 1 == c["native_tokens"]
            assert c["policy"] == "vnni_nvfp4_e4m3_g16"
        finally:
            b.close()
