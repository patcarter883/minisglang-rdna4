"""The new IPC fields must survive the ZMQ hop.

`message/utils.py` packs the hot message types from an ALLOWLIST (`_FAST_SCALAR_FIELDS`), not from
the dataclass. A field added to the dataclass but not to that tuple is silently dropped on the wire
and simply reads as its default at the far end — no error, no warning. Two of the three fixes here
ride on new fields (`DetokenizeMsg.error`, `StatsMsg.prefill_seconds`), so pin the round-trip.
"""

from __future__ import annotations

from minisgl.message.frontend import StatsFrontendMsg, UserReply
from minisgl.message.tokenizer import DetokenizeMsg, StatsMsg
from minisgl.message.utils import deserialize_type, serialize_type

CLS = {c.__name__: c for c in (DetokenizeMsg, UserReply, StatsMsg, StatsFrontendMsg)}


def _round_trip(msg):
    return deserialize_type(CLS, serialize_type(msg))


def test_detokenize_error_survives_the_wire():
    m = DetokenizeMsg(uid=7, next_token=0, finished=True, finish_reason="length",
                      error="input sequence length 93296 exceeds 73872")
    got = _round_trip(m)
    assert got.error == m.error, "rejection reason dropped -> client hangs again"
    assert got.finished is True


def test_detokenize_error_defaults_to_none():
    got = _round_trip(DetokenizeMsg(uid=1, next_token=5, finished=False))
    assert got.error is None
    assert got.next_token == 5


def test_user_reply_error_survives_the_wire():
    got = _round_trip(UserReply(uid=3, incremental_output="", finished=True,
                                finish_reason="length", error="rejected"))
    assert got.error == "rejected", "HTTP layer would return an empty 200 instead of a 4xx"


def _stats(cls, **kw):
    import dataclasses
    req = {f.name: (0 if f.type in ("int", int) else 0.0)
           for f in dataclasses.fields(cls) if f.default is dataclasses.MISSING}
    return cls(**{**req, **kw})


def test_stats_prefill_seconds_survives_the_wire():
    got = _round_trip(_stats(StatsMsg, prefill_seconds=12.5))
    assert got.prefill_seconds == 12.5, "prefill throughput panel stays blank"
    got2 = _round_trip(_stats(StatsFrontendMsg, prefill_seconds=3.25))
    assert got2.prefill_seconds == 3.25


def test_allowlist_covers_every_field_of_the_fast_types():
    """The real guard: any FUTURE field added to a fast-path type without updating the allowlist."""
    import dataclasses

    from minisgl.message.utils import _FAST_SCALAR_FIELDS

    for name, cls in CLS.items():
        allow = _FAST_SCALAR_FIELDS.get(name)
        if allow is None:
            continue
        declared = [f.name for f in dataclasses.fields(cls)]
        missing = [d for d in declared if d not in allow]
        assert not missing, f"{name}: fields {missing} are NOT in _FAST_SCALAR_FIELDS -> dropped on the wire"
