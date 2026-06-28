"""RSA shim parameters and proxy server configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

Selection = Literal["auto", "majority", "final_agg", "sample"]


class RSAParams(BaseModel):
    """Tunable Markovian-RSA parameters.

    Defaults follow the ZAYA1-8B report's Markovian RSA configuration
    (N=16, K=4, T=2, tail=4096). Set ``tail_tokens=0`` to carry the full
    previous-round traces into aggregation (generalized RSA, still Markovian
    over rounds).
    """

    enabled: bool = True
    n: int = Field(default=16, ge=1, description="population size N")
    k: int = Field(default=4, ge=1, description="aggregation set size K")
    t: int = Field(default=2, ge=1, description="total rounds T (round 0 = expand)")
    tail_tokens: int = Field(
        default=4096,
        ge=0,
        description="carry only the final tail_tokens of each previous-round trace "
        "into aggregation; 0 = full trace",
    )
    max_tokens: int = Field(
        default=8192, ge=1, description="per-rollout completion budget (round 0)"
    )
    agg_max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="completion budget for aggregation rounds (t>=1) and the final "
        "selection call; None = use max_tokens",
    )
    temperature: float = Field(default=0.8, ge=0.0)
    selection: Selection = Field(
        default="auto",
        description="'auto': majority vote when >=2 boxed answers extract, else a "
        "final aggregation call; 'majority'/'final_agg'/'sample' force one",
    )
    max_concurrency: int = Field(default=16, ge=1)
    request_timeout: float = Field(default=1800.0, gt=0)
    max_retries: int = Field(default=1, ge=0)


def merge_params(defaults: RSAParams, rsa_value) -> RSAParams | None:
    """Merge a request's ``rsa`` extra-body value over server defaults.

    Returns None when the request opts out (``"rsa": false`` or
    ``{"enabled": false}``). ``"rsa": true`` or absent -> server defaults.
    A dict patches the defaults field-by-field.
    """
    if rsa_value is None or rsa_value is True:
        merged = defaults
    elif rsa_value is False:
        return None
    elif isinstance(rsa_value, dict):
        merged = RSAParams(**{**defaults.model_dump(), **rsa_value})
    else:
        raise ValueError(f"invalid 'rsa' value: {rsa_value!r}")
    return merged if merged.enabled else None


@dataclass
class ShimConfig:
    """Configuration for the RSA shim proxy process."""

    backend_base_url: str = "http://127.0.0.1:1919/v1"
    host: str = "127.0.0.1"
    port: int = 2929
    api_key: str = "EMPTY"
    log_level: str = "info"
    tokenizer: str | None = None  # HF name/path for local token-exact tails
    defaults: RSAParams = field(default_factory=RSAParams)

    @property
    def backend_root(self) -> str:
        """Backend root URL (without /v1)."""
        return self.backend_base_url.rstrip("/").removesuffix("/v1")


def add_rsa_args(parser: argparse.ArgumentParser) -> None:
    d = RSAParams()
    g = parser.add_argument_group("Markovian RSA parameters")
    g.add_argument("--rsa-n", type=int, default=d.n, help="population size N")
    g.add_argument("--rsa-k", type=int, default=d.k, help="aggregation set size K")
    g.add_argument(
        "--rsa-t", type=int, default=d.t, help="total rounds T (round 0 = expand)"
    )
    g.add_argument(
        "--rsa-tail-tokens",
        type=int,
        default=d.tail_tokens,
        help="tail tokens of each previous-round trace fed to aggregation (0 = full)",
    )
    g.add_argument(
        "--rsa-max-tokens",
        type=int,
        default=d.max_tokens,
        help="per-rollout completion budget (round 0)",
    )
    g.add_argument(
        "--rsa-agg-max-tokens",
        type=int,
        default=d.agg_max_tokens,
        help="completion budget for aggregation rounds + final selection "
        "(default: --rsa-max-tokens)",
    )
    g.add_argument("--rsa-temperature", type=float, default=d.temperature)
    g.add_argument(
        "--rsa-selection",
        choices=["auto", "majority", "final_agg", "sample"],
        default=d.selection,
    )
    g.add_argument("--rsa-max-concurrency", type=int, default=d.max_concurrency)
    g.add_argument("--rsa-request-timeout", type=float, default=d.request_timeout)
    g.add_argument("--rsa-max-retries", type=int, default=d.max_retries)


def params_from_args(args: argparse.Namespace) -> RSAParams:
    return RSAParams(
        n=args.rsa_n,
        k=args.rsa_k,
        t=args.rsa_t,
        tail_tokens=args.rsa_tail_tokens,
        max_tokens=args.rsa_max_tokens,
        agg_max_tokens=args.rsa_agg_max_tokens,
        temperature=args.rsa_temperature,
        selection=args.rsa_selection,
        max_concurrency=args.rsa_max_concurrency,
        request_timeout=args.rsa_request_timeout,
        max_retries=args.rsa_max_retries,
    )
