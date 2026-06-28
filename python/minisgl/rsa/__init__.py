"""Markovian Recursive Self-Aggregation (RSA) shim for minisglang.

RSA (Venkatraman et al., arXiv:2509.26626) is a test-time-compute method:
expand a query into N rollouts, then repeatedly aggregate K-subsets of the
population into improved candidates over T rounds, and select a final answer.

This package implements the *Markovian* variant used by the ZAYA1-8B report:
each aggregation round conditions ONLY on the immediately preceding round's
population (a Markov chain over rounds), never on the full round history.
It runs as an OpenAI-compatible shim proxy in front of a minisglang server
rather than inside the engine process — see ``docs/zaya-port/RSA_SHIM.md``.
"""

from .config import RSAParams, ShimConfig
from .core import RSAResult, run_markovian_rsa

__all__ = ["RSAParams", "ShimConfig", "RSAResult", "run_markovian_rsa"]
