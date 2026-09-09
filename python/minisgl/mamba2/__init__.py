"""Mamba-2 (SSD) support for Nemotron-H. See reference.py for the recurrence itself."""

from .reference import (
    discretize_dt,
    mamba2_chunked,
    mamba2_decode,
    mamba2_sequential,
    segment_sum,
)

__all__ = [
    "discretize_dt",
    "mamba2_chunked",
    "mamba2_decode",
    "mamba2_sequential",
    "segment_sum",
]
