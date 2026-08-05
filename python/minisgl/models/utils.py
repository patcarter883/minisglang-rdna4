from __future__ import annotations

from typing import TYPE_CHECKING

from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    RMSNorm,
    gelu_and_mul,
    silu_and_mul,
)
from minisgl.models import ModelConfig
from minisgl.quant import create_linear_method
from minisgl.utils import init_logger, nvtx_annotate

if TYPE_CHECKING:
    import torch


_ACTQ_LOGGED = False
_actq_logger = init_logger(__name__)


def norm_then_mlp(norm, mlp, x, residual, *, fuse_actquant: bool, time_mlp=None):
    """post-attention RMSNorm -> MLP, with the MLP's activation quant FUSED INTO THE NORM when the
    MLP can take it. Returns (mlp_out, residual).

    `fuse_actquant` is decided ONCE at layer construction (`mlp_accepts_producer_actquant`), not per
    step: whether the MLP's GEMM1 can consume a pre-quantized activation depends on the quant scheme
    and the EP topology, both fixed at load. Re-deciding it every decode step would put a python
    branch in the hot loop for an answer that cannot change.

    Why this exists at all: at decode the step is ~68% idle on inter-kernel GAPS, so what costs is
    the number of dispatches, not their shapes. The norm is already a one-block-per-row kernel
    holding the whole row in registers, so emitting its e4m3 form is nearly free — and it deletes the
    MoE's own per-token act-quant launch. It is a LAUNCH-COUNT win and it does not scale with M.

    When the producer kernel is unavailable (`forward_quant` returns a None pair — non-contiguous
    input, unsupported dtype), this degrades to exactly the old two calls. That is not a silent
    numerics fallback: `out`/`residual` are bit-identical either way, and the MLP re-quantizes as it
    always did. The only observable difference is one extra dispatch, and the engage ledger shows it
    (`+prequant` present or absent).

    `time_mlp` (if given) wraps ONLY the MLP call, so the layer-profiler's "ffn" bucket keeps
    meaning exactly what it meant before this seam existed — the norm stays outside it. Folding the
    norm into that bucket would silently re-baseline every recorded ffn number.
    """
    if not fuse_actquant:
        x, residual = norm.forward(x, residual)
        out = time_mlp(mlp.forward, x) if time_mlp is not None else mlp.forward(x)
        return out, residual
    x, residual, x_fp8, act_scales = norm.forward_quant(x, residual)

    def _run(h):
        return mlp.forward(h, x_fp8, act_scales)

    out = time_mlp(_run, x) if time_mlp is not None else _run(x)
    return out, residual


def mlp_accepts_producer_actquant(mlp) -> bool:
    """Does `mlp` have a GEMM1 that can consume the producer's (x_fp8, act_scales) pair?

    Opt-IN by the block, never inferred: a block that quietly accepted and dropped the pair would
    make an A/B read as though producer fusion applied while the pre-kernel still ran. Dense MLPs
    (GatedMLP) and unquantized/EP sparse blocks answer False and pay nothing — the norm then does not
    even compute the quant epilogue.
    """
    probe = getattr(mlp, "accepts_producer_actquant", None)
    consumer = bool(probe()) if callable(probe) else False
    # BOTH halves, not just the consumer. The producer op is optional (an older baked tail_hip .so
    # has neither quant twin), and answering True with no producer would make the norm take the
    # forward_quant branch, get a None pair back, and hand the MLP nothing — a wasted branch and,
    # worse, a boot line claiming a fusion that is not running. Ask the producer as well.
    from minisgl.layers import _tail_hip as _th
    producer = hasattr(_th, "rms_norm_quant") and hasattr(_th, "rms_norm_add_quant")
    ans = consumer and producer
    # Say so ONCE per process. Whether this seam engaged is exactly the thing an A/B has to assert,
    # and inferring it from a kernel-name ledger is indirect; a boot line that names the MLP type and
    # the answer makes "the fusion did not engage" a 2-second check instead of a re-run.
    global _ACTQ_LOGGED
    if not _ACTQ_LOGGED:
        _ACTQ_LOGGED = True
        _actq_logger.info(
            "producer-side MoE act-quant: %s (mlp=%s consumer=%s producer=%s)",
            "ENGAGED" if ans else "OFF", type(mlp).__name__, consumer, producer,
        )
    return ans


class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        qm = create_linear_method(config.quant)
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
            quant_method=qm,
        )

        FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        act_fn = FN_MAP.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
            quant_method=qm,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


class MoEMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.gate = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        return final_hidden_states


class RopeAttn(BaseOP):
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
    ):
        head_dim = config.head_dim
        qm = create_linear_method(config.quant)
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
            quant_method=qm,
        )
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
            quant_method=qm,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj.forward(x)
        del x
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)


__all__ = ["GatedMLP", "RopeAttn", "MoEMLP"]
