"""Stage B — the chunked load driver, on a toy model.

The GPU proof is `tests/qwen4exp_stage_b_test.py` (22 of 48 real layers vs 4 unchunked). This file
covers the parts that are cheap to get wrong and expensive to notice: the totality ledger, the
one-and-only-once finalize, and the `missing_ok` partial fill that makes chunking possible at all.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minisgl.layers.base import BaseOP, OPList  # noqa: E402
from minisgl.weights.stage_b import (  # noqa: E402
    ChunkedLoadError,
    ChunkedWeightLoader,
    DeviceLayerSink,
    LoadChunk,
)


class _Leaf(BaseOP):
    """Stands in for a quantized expert container: `post_load` DELETES what it read."""

    def __init__(self, n: int) -> None:
        self.weight = torch.zeros(n)
        self.scale = torch.zeros(n)
        self.post_loads = 0

    def forward(self, *a, **kw):  # pragma: no cover - storage only
        raise RuntimeError

    def post_load(self) -> None:
        self.packed = self.weight + self.scale
        del self.weight, self.scale
        self.post_loads += 1


class _Block(BaseOP):
    def __init__(self, n: int) -> None:
        self.experts = _Leaf(n)
        self.norm = torch.zeros(n)

    def forward(self, *a, **kw):  # pragma: no cover
        raise RuntimeError


class _Toy(BaseOP):
    def __init__(self, n_layers: int, n: int = 4) -> None:
        self.layers = OPList([_Block(n) for _ in range(n_layers)])
        self.embed = torch.zeros(n)

    def forward(self, *a, **kw):  # pragma: no cover
        raise RuntimeError


def _chunks(n_layers: int):
    """Body first, then one chunk per layer — the shape `qwen4_exp_chunked_source` emits."""
    yield LoadChunk(name="body", files=())
    for i in range(n_layers):
        yield LoadChunk(name=f"layer-{i}", files=(), finalize_paths=(f"layers.{i}",))


def _stream(model, chunk):
    """One chunk's tensors, filled with a value derived from the key so a mix-up is visible."""
    keys = (
        [k for k in model.state_dict() if ".experts." not in k]
        if chunk.name == "body"
        else [k for k in model.state_dict() if k.startswith(chunk.finalize_paths[0] + ".experts.")]
    )
    for k in keys:
        yield k, torch.full_like(model.state_dict()[k], float(len(k)))


@pytest.fixture(autouse=True)
def _discoverable(monkeypatch):
    """`finalize_paths` are resolved through `discover_moe_layers`, which keys on `MoELayer`.

    The toy model has no `MoELayer`, so point the driver's resolver at the toy's own blocks. That
    keeps this file free of the whole quant/kernel stack while exercising the SAME code path the
    real model takes — the driver never learns what a MoE layer is, it asks.
    """
    from minisgl.weights import stage_b

    monkeypatch.setattr(
        ChunkedWeightLoader,
        "_ops_by_path",
        lambda self: {f"layers.{i}": b for i, b in enumerate(self.model.layers.op_list)},
    )
    return stage_b


class TestChunkedFill:
    def test_fills_everything_exactly_once_and_finalizes_each_layer(self):
        m = _Toy(3)
        loader = ChunkedWeightLoader(m, sink=DeviceLayerSink())
        expected = set(m.state_dict())
        led = loader.run(list(_chunks(3)), lambda c: _stream(m, c))
        assert led.chunks == 4
        assert led.keys_filled == len(expected)
        assert led.placed_device_layers == 3
        for b in m.layers.op_list:
            assert b.experts.post_loads == 1
            assert hasattr(b.experts, "packed")

    def test_the_trailing_whole_model_post_load_does_not_re_finalize(self):
        """A second `post_load` on a quantized container is an AttributeError, not a no-op — the
        buffers it reads are gone. `_post_load_done` is what makes the ordinary
        `model.post_load()` safe to call after a chunked load."""
        m = _Toy(2)
        ChunkedWeightLoader(m, sink=DeviceLayerSink()).run(
            list(_chunks(2)), lambda c: _stream(m, c)
        )
        m.post_load()  # would raise AttributeError without the guard
        assert [b.experts.post_loads for b in m.layers.op_list] == [1, 1]

    def test_a_missing_shard_is_named_at_the_end_not_left_as_garbage(self):
        m = _Toy(2)
        loader = ChunkedWeightLoader(m, sink=DeviceLayerSink())
        chunks = [c for c in _chunks(2) if c.name != "layer-1"]
        with pytest.raises(ChunkedLoadError) as exc:
            loader.run(chunks, lambda c: _stream(m, c))
        assert "unfilled" in str(exc.value)
        assert "layers.1.experts.weight" in str(exc.value)

    def test_overlapping_chunks_are_refused(self):
        """Two chunks delivering the same key means one of the two reads is dead weight and which
        one wins depends on chunk order — silent, and order-dependent."""
        m = _Toy(2)
        loader = ChunkedWeightLoader(m, sink=DeviceLayerSink())
        chunks = list(_chunks(2))
        chunks.insert(1, chunks[0])  # the body, twice
        with pytest.raises(ChunkedLoadError) as exc:
            loader.run(chunks, lambda c: _stream(m, c))
        assert "re-delivers" in str(exc.value)

    def test_a_layer_finalized_twice_is_refused(self):
        """The other half of the same hazard: re-finalizing a container whose `post_load` already
        consumed its buffers is an AttributeError deep in a quant path, not a diagnosis."""
        m = _Toy(2)
        loader = ChunkedWeightLoader(m, sink=DeviceLayerSink())
        chunks = list(_chunks(2))
        chunks.append(LoadChunk(name="layer-0-again", files=(), finalize_paths=("layers.0",)))
        with pytest.raises(ChunkedLoadError) as exc:
            loader.run(chunks, lambda c: _stream(m, c))
        assert "already finalized" in str(exc.value)

    def test_a_key_the_model_has_no_home_for_is_refused(self):
        m = _Toy(1)
        loader = ChunkedWeightLoader(m, sink=DeviceLayerSink())

        def stream(c):
            yield from _stream(m, c)
            if c.name == "body":
                yield "not.a.real.key", torch.zeros(4)

        with pytest.raises(ChunkedLoadError) as exc:
            loader.run(list(_chunks(1)), stream)
        assert "no home for" in str(exc.value)

    def test_finalize_path_that_is_not_in_the_model_is_refused(self):
        m = _Toy(1)
        loader = ChunkedWeightLoader(m, sink=DeviceLayerSink())
        chunks = [LoadChunk(name="body", files=()), LoadChunk("x", (), ("layers.9",))]
        with pytest.raises(ChunkedLoadError) as exc:
            loader.run(chunks, lambda c: _stream(m, c) if c.name == "body" else iter(()))
        assert "layers.9" in str(exc.value)

    def test_cast_is_applied_to_every_tensor(self):
        m = _Toy(1)
        seen = []
        loader = ChunkedWeightLoader(
            m, cast=lambda k, v: (seen.append(k), v)[1], sink=DeviceLayerSink()
        )
        loader.run(list(_chunks(1)), lambda c: _stream(m, c))
        assert set(seen) == {"embed", "layers.0.norm", "layers.0.experts.weight",
                             "layers.0.experts.scale"}


class TestPartialLoadStateDict:
    """`missing_ok` is the primitive chunking rests on; without it the first chunk KeyErrors."""

    def test_absent_keys_leave_the_buffer_alone(self):
        m = _Toy(1)
        before = m.layers.op_list[0].experts.weight.clone()
        m.load_state_dict({"embed": torch.ones(4)}, missing_ok=True)
        assert torch.equal(m.embed, torch.ones(4))
        assert torch.equal(m.layers.op_list[0].experts.weight, before)

    def test_strict_by_default(self):
        m = _Toy(1)
        with pytest.raises(KeyError):
            m.load_state_dict({"embed": torch.ones(4)})
