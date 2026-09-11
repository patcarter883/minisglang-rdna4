"""NemotronHForCausalLM — NVIDIA Nemotron-3.5-Lightning (Mamba-2 + MoE + sparse global attention).

Architecture (verified against the real checkpoint's tensor inventory, not just its config —
see docs/measurements/NEMOTRON_INVENTORY/ and tools/nemotron/probe_checkpoint.py):

    52 layers, ONE sublayer each, dispatched off `block_types`:
        23 x mamba      Mamba-2 (SSD) mixer
        23 x moe        the MoE *is* the mixer, not something that follows one
         6 x attention  plain GQA, GLOBAL (sliding_window: null), ids [5,12,19,26,33,42]

THE SHAPE OF A LAYER IS THE THING TO GET RIGHT FIRST. Every other model in this repo is
`x = x + mixer(norm1(x)); x = x + mlp(norm2(x))` — two norms, two sublayers. Nemotron-H is
`x = x + mixer(norm(x))` with ONE norm and ONE sublayer, and "moe" is a peer of "attention" in
that schedule rather than a thing that follows it. The checkpoint says so unambiguously: an
attention layer ships exactly one `norm.weight`, and so does a moe layer. Building this as
mixer+MLP would need a second norm the checkpoint does not contain.

Weight names map STRUCTURALLY: `BaseOP.state_dict` walks attribute names, so the module tree is
spelled to match the checkpoint (`backbone.embeddings`, `backbone.layers.N.mixer.*`,
`backbone.norm_f`, `lm_head`) rather than through a rename table. That is why the inner model
attribute is `backbone` and not `model`.

Mixed precision in ONE checkpoint, by module (46 FP8 targets, 5,935 NVFP4 — an explicit per-module
assignment, not a global scheme with exceptions):
    mamba in_proj/out_proj   FP8 e4m3, per-tensor, STATIC activation scale (`input_scale` ships)
    routed + shared experts  NVFP4 (e2m1 pack + e4m3 group-16 block scale + f32 global)
    attention q/k/v/o        bf16, with fp8 KV `k_scale`/`v_scale`
    gate, norms, conv1d, A_log/D/dt_bias   bf16 (gate weight is f32)

TWO THINGS THIS FAMILY DOES THAT NOTHING ELSE HERE DOES, both load-bearing and both silent if
missed:

  1. **The experts are relu², and are NOT gated.** One `up_proj`, one `down_proj`, no gate half —
     confirmed from the inventory (a routed expert ships exactly two weights) and from HF's own
     `NemotronHExperts` ("Unlike Mixtral or DeepSeek which use gated MLPs..."). Every MoE in this
     repo before it was gate+up fused with a SiLU epilogue, and `layers/moe.py` gates its allowed
     activations to `("silu", "gelu")` — both GATED. Handing Nemotron's experts to that path would
     read a gate half that does not exist: right shapes, wrong numbers, no error. Per
     KERNEL_CORE_POLICY the fused form of this is an EPILOGUE POLICY on the existing MoE core, not
     a new kernel — see `NemotronHExperts` below for the seam and why it is torch today.

  2. **Mamba-2 is not GDN.** Both are chunked linear recurrences behind a causal conv1d with a
     gated RMSNorm epilogue, and they share a state layout — but GDN is a delta-rule update and
     Mamba-2 is a selective SSM (`S <- S*exp(dt*A) + dt*B x^T`, `y = C*S + D*x`). `config.py` keeps
     `layer_types` None for this family precisely so `is_gdn_hybrid` stays False and no GDN kernel,
     slot manager or state cache can be reached from here.

STATUS: the recurrence runs through `minisgl.mamba2.reference` (torch, validated to float64 against
the sequential definition by tests/mamba2_reference_test.py). That is correct and slow, and it is
the seam the HIP chunk-scan drops into — see `NemotronHMamba2Mixer.forward`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn.functional as F
from minisgl.distributed import get_tp_info
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    RMSNormFused,
    VocabParallelEmbedding,
)
from minisgl.mamba2 import discretize_dt, mamba2_chunked
from minisgl.quant import create_linear_method
from minisgl.utils import div_even, init_logger, nvtx_annotate

from .base import BaseLLMModel

if TYPE_CHECKING:
    from .config import ModelConfig

logger = init_logger(__name__)


def _quant_method(config: "ModelConfig", name: str) -> object:
    """Per-projection quant, config-driven — the same generic dispatcher every other model uses.

    A projection is quantized iff the checkpoint's quant config declares its module quantized. This
    family needs it to be per-module rather than global: FP8 lands on the mamba projections and
    NVFP4 on the experts, in ONE checkpoint, and the attention stays bf16. No model-name branch —
    the precision falls out of the config."""
    q = config.quant
    return create_linear_method(q.for_module(name) if q is not None else None)


# ---------------------------------------------------------------------------------------------
# attention — the easy one
# ---------------------------------------------------------------------------------------------
class NemotronHAttention(BaseOP):
    """Plain GQA: 32 q / 2 kv heads, head_dim 128, FULL rotary (`partial_rotary_factor: 1.0`),
    theta 1e4, no bias, no q/k norm, and no output gate. GLOBAL attention — `sliding_window` is
    null in the config, so all six of these layers see the whole sequence.

    Two kv heads is why `min_tp=2` is a ceiling as well as a floor for this model: 2 kv heads
    cannot shard four ways."""

    def __init__(self, config: "ModelConfig", layer_id: int, *, attn_kv_id: int):
        head_dim = config.head_dim
        nqo, nkv = config.num_qo_heads, config.num_kv_heads
        prefix = f"backbone.layers.{layer_id}.mixer"

        self.q_proj = LinearColParallelMerged(
            config.hidden_size, [nqo * head_dim], has_bias=False,
            quant_method=_quant_method(config, f"{prefix}.q_proj"),
        )
        self.k_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False,
            quant_method=_quant_method(config, f"{prefix}.k_proj"),
        )
        self.v_proj = LinearColParallelMerged(
            config.hidden_size, [nkv * head_dim], has_bias=False,
            quant_method=_quant_method(config, f"{prefix}.v_proj"),
        )
        self.attn = AttentionLayer(
            # Index the paged KV pool by the COMPACT attention position (0..5 over the six
            # attention layers), never the global layer id (5,12,...,42): the pool has one slot per
            # ATTENTION layer. Getting this wrong is the 8.7x KV stranding `config.py`'s
            # `full_attn_layer_ids` fix already had to deal with once for this family.
            layer_id=attn_kv_id,
            head_dim=head_dim,
            num_qo_heads=nqo,
            num_kv_heads=nkv,
            rotary_config=config.rotary_config,
        )
        self.o_proj = LinearOProj(
            head_dim * nqo, config.hidden_size, has_bias=False,
            quant_method=_quant_method(config, f"{prefix}.o_proj"),
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)
        o = self.attn.forward(torch.cat([q, k, v], dim=-1))
        return self.o_proj.forward(o)


# ---------------------------------------------------------------------------------------------
# Mamba-2 (SSD)
# ---------------------------------------------------------------------------------------------
class NemotronHMamba2Mixer(BaseOP):
    """Mamba-2 selective SSM: in_proj -> causal conv1d -> SSD scan -> gated RMSNorm -> out_proj.

    `in_proj` emits ONE fused tensor and the split order is load-bearing (verified against HF's
    `NemotronHMamba2Mixer`, which splits `[d_mlp, d_mlp, intermediate, conv_dim, num_heads]` with
    d_mlp == 0 here):

        10304 = z(4096) | x·B·C(6144) | dt(64)
                 gate     the conv'd part   per-head timestep

    and the conv1d runs over x, B and C TOGETHER (conv_dim 6144 = 4096 + 2*8*128), which is what
    `conv1d.weight [6144, 1, 4]` is. Splitting x/B/C before the convolution instead of after gives
    the right shapes and the wrong numbers.

    TP: head-parallel over the 64 mamba heads, mirroring the GDN mixer — local heads on each rank,
    `out_proj` all-reduces. `n_groups` 8 divides evenly by 2, so B/C shard with the heads.
    """

    def __init__(self, config: "ModelConfig", layer_id: int):
        prefix = f"backbone.layers.{layer_id}.mixer"
        tp = get_tp_info().size

        self.num_heads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.ssm_state = config.mamba_ssm_state
        self.n_groups = config.mamba_n_groups
        self.conv_kernel = config.mamba_conv_kernel
        self.chunk_size = config.mamba_chunk_size

        inner = self.num_heads * self.head_dim                      # 4096
        groups_state = self.n_groups * self.ssm_state               # 1024
        self.conv_dim = inner + 2 * groups_state                    # 6144
        in_proj_out = inner + self.conv_dim + self.num_heads        # 10304

        # LOCAL (post-TP) geometry — the forward reshapes with these, never the global counts.
        self.local_heads = div_even(self.num_heads, tp)
        self.local_groups = div_even(self.n_groups, tp)
        self.local_inner = self.local_heads * self.head_dim
        self.local_groups_state = self.local_groups * self.ssm_state
        self.local_conv_dim = self.local_inner + 2 * self.local_groups_state

        # FP8 e4m3 with a STATIC per-tensor activation scale — `in_proj.input_scale` really does
        # ship (confirmed in the inventory), so this is the calibrated-static path, not the
        # dynamic-per-token one the other arms use. If the method rejects a static scale that is a
        # loud failure at load, which is what we want; it must never silently fall back to dynamic.
        self.in_proj = LinearColParallelMerged(
            config.hidden_size, [in_proj_out], has_bias=config.mamba_proj_bias,
            quant_method=_quant_method(config, f"{prefix}.in_proj"),
        )
        self.out_proj = LinearOProj(
            inner, config.hidden_size, has_bias=config.mamba_proj_bias,
            quant_method=_quant_method(config, f"{prefix}.out_proj"),
        )
        # conv1d is a depthwise convolution stored as [conv_dim, 1, kernel]; it is bf16 and stays
        # bf16. Held as raw tensors (not an nn.Conv1d) so the names match the checkpoint exactly.
        self.conv1d = _Conv1dWeights(self.local_conv_dim, self.conv_kernel, bias=config.mamba_conv_bias)

        # Per-head SSM parameters. A_log/D/dt_bias are [num_heads] and shard with the heads.
        self.A_log = torch.empty(self.local_heads)
        self.D = torch.empty(self.local_heads)
        self.dt_bias = torch.empty(self.local_heads)
        # Gated RMSNorm over the INNER dim (4096), plain-weight convention (HF's NemotronHRMSNorm
        # initialises to ones and multiplies — no `1 + weight`).
        self.norm = _GatedRMSNorm(self.local_inner, eps=config.rms_norm_eps)

        self._time_step_limit = (config.mamba_dt_min, config.mamba_dt_max)

    @nvtx_annotate("Mamba2")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prefill/verify path. `x` is [num_tokens, hidden] — the engine's flat token layout.

        SEAM: the SSD scan below is `minisgl.mamba2.reference.mamba2_chunked`, i.e. torch. It is
        the validated definition (tests/mamba2_reference_test.py pins it to the sequential form in
        float64), and it is what the HIP `mamba2_prefill_chunked` kernel must reproduce. Swapping
        the kernel in is a one-line change HERE; everything around it — the split, the conv, the
        gate, the norm — is already the shipped arrangement.
        """
        T = x.shape[0]
        projected = self.in_proj.forward(x)                                     # [T, 10304/tp]
        z, xBC, dt = torch.split(
            projected, [self.local_inner, self.local_conv_dim, self.local_heads], dim=-1
        )

        # Depthwise causal conv1d over (x, B, C) jointly, then SiLU — `mamba_hidden_act`.
        xBC = self.conv1d.causal_forward(xBC)
        xBC = F.silu(xBC)

        xs, B, C = torch.split(
            xBC, [self.local_inner, self.local_groups_state, self.local_groups_state], dim=-1
        )

        # `mamba2_chunked` takes A_log (it applies -exp itself) and an ALREADY-DISCRETIZED dt, so
        # the softplus+bias+clamp happens here. Splitting it this way is the reference's contract,
        # not an accident: the kernel will discretize on the host side too, because dt_bias and the
        # clamp are per-head constants that do not belong in the scan's inner loop.
        dt = discretize_dt(
            dt.float(), self.dt_bias.float(),
            time_step_min=self._time_step_limit[0], time_step_max=self._time_step_limit[1],
        )
        y, _final_state = mamba2_chunked(
            x=xs.view(T, self.local_heads, self.head_dim).float(),
            dt=dt,
            A_log=self.A_log.float(),
            B=B.view(T, self.local_groups, self.ssm_state).float(),
            C=C.view(T, self.local_groups, self.ssm_state).float(),
            D=self.D.float(),
            chunk_size=self.chunk_size,
        )
        # `_final_state` is the carry a decode step resumes from. It is DROPPED here because there
        # is nowhere to put it yet: the per-slot Mamba-2 state cache is Phase 3 (plan N5). Until
        # that exists this mixer is prefill-only, and a decode step would silently restart the
        # recurrence from zero — which is why no decode path is wired rather than one that looks
        # like it works.
        y = y.reshape(T, self.local_inner).to(x.dtype)
        y = self.norm.forward(y, z)                                             # gated RMSNorm
        return self.out_proj.forward(y)


class _Conv1dWeights(BaseOP):
    """Depthwise causal conv1d held as `weight [dim, 1, k]` + `bias [dim]`, named to match the
    checkpoint. Not an `nn.Conv1d` because the state_dict walk keys off attribute names and a
    module would introduce a level the checkpoint does not have."""

    def __init__(self, dim: int, kernel: int, *, bias: bool):
        self.weight = torch.empty(dim, 1, kernel)
        if bias:
            self.bias = torch.empty(dim)
        self._dim, self._kernel = dim, kernel

    def causal_forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` [T, dim] -> [T, dim], left-padded so position t sees only t-k+1..t.

        SEAM: this is the torch form. `gdn_hip.causal_conv1d_fwd` computes exactly this op and is
        reusable as-is (the plan's N1 note) — only `conv_dim` differs. Wiring it is Phase 3 work,
        and it needs the per-slot conv state the state cache has not been written for yet."""
        xt = x.t().unsqueeze(0)                                     # [1, dim, T]
        xt = F.pad(xt, (self._kernel - 1, 0))
        out = F.conv1d(xt, self.weight, getattr(self, "bias", None), groups=self._dim)
        return out.squeeze(0).t()


class _GatedRMSNorm(BaseOP):
    """RMSNorm(x * silu(gate)) — Mamba-2's epilogue. Plain-weight convention.

    Note the gate is applied BEFORE the normalisation, which is the Mamba-2 convention and the
    opposite of a post-norm gate; HF's mixer does `hidden_states * self.act(gate)` inside the norm
    call. `gdn_hip.rmsnorm_gated` implements this exact op and is reusable (plan N1)."""

    def __init__(self, size: int, eps: float):
        self.weight = torch.empty(size)
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        h = (x * F.silu(gate.to(x.dtype))).float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        # Cast to the activation dtype BEFORE the weight multiply, matching HF's
        # `self.weight * hidden_states.to(input_dtype)`. Doing the multiply in fp32 and casting
        # afterwards is more accurate and measurably DIFFERENT (~1e-1 max on bf16 activations) —
        # this epilogue follows the reference the checkpoint was published against, and a
        # deliberate accuracy change here belongs in its own commit with its own measurement.
        return self.weight * h.to(x.dtype)


# ---------------------------------------------------------------------------------------------
# MoE — the mixer, not an MLP
# ---------------------------------------------------------------------------------------------
class NemotronHTopkGate(BaseOP):
    """Sigmoid router with a per-expert correction bias used ONLY for top-k selection; the routing
    weights are the UN-biased sigmoid scores. `n_group` is 1 on this checkpoint, so HF's group-topk
    block is an identity and is not reproduced here.

    The gate weight ships as F32 and the router runs in F32 (HF casts hidden states up before the
    matmul). The correction bias is kept F32 DELIBERATELY: the GLM fast path
    (`kernels.moe_route_sigmoid_bias`) consumes an already-downcast bias because that model's
    checkpoint is served that way, and feeding a truncated bias here would change expert SELECTION
    on near-ties versus HF. Different, invisible to a tok/s A/B, and a parity failure."""

    def __init__(self, hidden_size: int, num_experts: int):
        self.weight = torch.empty(num_experts, hidden_size)
        self.e_score_correction_bias = torch.empty(num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(x.float(), self.weight.float())
        scores = logits.sigmoid()
        choice = scores + self.e_score_correction_bias.float()
        topk_ids = torch.topk(choice, k=self._top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_ids)
        if self._norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self._routed_scaling_factor
        return topk_weights, topk_ids

    def configure(self, top_k: int, norm_topk_prob: bool, routed_scaling_factor: float) -> None:
        self._top_k = top_k
        self._norm_topk_prob = norm_topk_prob
        self._routed_scaling_factor = routed_scaling_factor


class NemotronHExperts(BaseOP):
    """128 routed relu² experts: `down_proj(relu(up_proj(x))**2)`. NO gate half.

    SEAM / WHY THIS IS TORCH TODAY. `layers/moe.py`'s fused path allows only `("silu", "gelu")` and
    both are GATED — its gemm1 reads a `2*intermediate` weight and splits it. Nemotron's expert
    ships a single `[intermediate, hidden]` up_proj, so that path would consume a gate half that
    does not exist. Per KERNEL_CORE_POLICY the fix is an EPILOGUE POLICY on the shared MoE core
    (an ungated relu² tail alongside the silu/gelu ones), NOT a forked kernel — copying
    `w4a8_moe` and editing the epilogue is exactly the debt that document forbids. That is kernel
    work and needs a GPU to validate, so it is deliberately not attempted here; this torch path is
    correct and slow, and it is what the kernel must reproduce."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        # Stacked [E, out, in], matching HF's 3D parameter layout and the loader's gather.
        self.up_proj = torch.empty(num_experts, intermediate_size, hidden_size)
        self.down_proj = torch.empty(num_experts, hidden_size, intermediate_size)
        self.num_experts = num_experts

    def forward(
        self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        # Iterate only the experts that actually received a token, like HF does — with top-6 of 128
        # and a short batch most experts are idle and a dense loop would be 20x the work.
        #
        # `torch.where` on a [T, top_k] mask returns (dim0, dim1) = (token, k-position) IN THAT
        # ORDER. Naming them the other way round still indexes and still runs, right up until
        # top_k != T — which is why this is written out rather than unpacked by feel.
        hit = torch.unique(topk_ids)
        for e in hit.tolist():
            tok, pos = torch.where(topk_ids == e)
            if tok.numel() == 0:
                continue
            h = F.linear(x[tok], self.up_proj[e])
            h = F.relu(h).pow(2)                              # relu2 — squared ReLU, ungated
            h = F.linear(h, self.down_proj[e])
            # Accumulate in fp32: a token can land on top_k experts and bf16 would round each
            # partial sum before the next one arrives.
            out.index_add_(0, tok, h.float() * topk_weights[tok, pos, None].float())
        return out.to(x.dtype)


class NemotronHSharedExpert(BaseOP):
    """The always-on shared expert — also relu², also ungated, intermediate 3712 (2x a routed
    expert's 1856). Added to the routed output, not gated against it.

    REPLICATED across TP ranks rather than sharded, following the GLM shared expert for the same
    reason: each rank computes the whole thing and adds it to the already-all-reduced routed
    output, so there is no double-count and no extra collective."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        prefix = f"backbone.layers.{layer_id}.mixer.shared_experts"
        inter = config.moe_shared_intermediate
        self.up_proj = LinearReplicated(
            config.hidden_size, inter, has_bias=False,
            quant_method=_quant_method(config, f"{prefix}.up_proj"),
        )
        self.down_proj = LinearReplicated(
            inter, config.hidden_size, has_bias=False,
            quant_method=_quant_method(config, f"{prefix}.down_proj"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(F.relu(self.up_proj.forward(x)).pow(2))


class NemotronHMoE(BaseOP):
    """The MoE block, which on this model IS the layer's mixer."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        self.gate = NemotronHTopkGate(config.hidden_size, config.num_experts)
        self.gate.configure(
            top_k=config.num_experts_per_tok,
            norm_topk_prob=config.norm_topk_prob,
            routed_scaling_factor=config.routed_scaling_factor,
        )
        self.experts = NemotronHExperts(
            config.num_experts, config.hidden_size, config.moe_intermediate_size
        )
        self.shared_experts = NemotronHSharedExpert(config, layer_id)

    @nvtx_annotate("MoE")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk_weights, topk_ids = self.gate.forward(x)
        routed = self.experts.forward(x, topk_weights, topk_ids)
        return routed + self.shared_experts.forward(x)


# ---------------------------------------------------------------------------------------------
# the stack
# ---------------------------------------------------------------------------------------------
class NemotronHDecoderLayer(BaseOP):
    """ONE norm, ONE mixer. See the module docstring — this is the structural difference from
    every other model here, and it is what the checkpoint contains."""

    def __init__(self, config: "ModelConfig", layer_id: int, *, attn_kv_id: int | None):
        kind = config.block_types[layer_id]
        if kind == "attention":
            assert attn_kv_id is not None, "an attention layer needs a compact KV-pool slot id"
            self.mixer: BaseOP = NemotronHAttention(config, layer_id, attn_kv_id=attn_kv_id)
        elif kind == "mamba":
            self.mixer = NemotronHMamba2Mixer(config, layer_id)
        elif kind == "moe":
            self.mixer = NemotronHMoE(config, layer_id)
        else:
            raise ValueError(f"unknown Nemotron-H block type {kind!r} at layer {layer_id}")
        self.norm = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self._layer_id = layer_id
        self._kind = kind

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.norm.forward(x, residual)
        return self.mixer.forward(x), residual


class NemotronHModel(BaseOP):
    """Named `backbone` by its parent so state_dict keys line up with the checkpoint."""

    def __init__(self, config: "ModelConfig"):
        self.embeddings = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        # Compact attention position -> paged-KV slot. Six slots, not fifty-two.
        attn_pos = {lid: pos for pos, lid in enumerate(config.full_attn_layer_ids)}
        self.layers = OPList(
            [
                NemotronHDecoderLayer(config, i, attn_kv_id=attn_pos.get(i))
                for i in range(config.num_layers)
            ]
        )
        self.norm_f = RMSNormFused(size=config.hidden_size, eps=config.rms_norm_eps)
        self._capture_layer_ids: List[int] | None = None

    def set_capture_layers(self, ids: List[int] | None) -> None:
        self._capture_layer_ids = list(ids) if ids else None

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False):
        x = self.embeddings.forward(input_ids)
        residual = None
        aux: List[torch.Tensor] = []
        capture = self._capture_layer_ids
        for i, layer in enumerate(self.layers):
            x, residual = layer.forward(x, residual)
            if capture is not None and i in capture:
                aux.append(x)
        x, _ = self.norm_f.forward(x, residual)
        if not return_hidden:
            return x
        return x, (torch.stack(aux) if aux else None)


class NemotronHForCausalLM(BaseLLMModel):
    def __init__(self, config: "ModelConfig"):
        self.backbone = NemotronHModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.config = config

    def forward(self, return_hidden: bool = False):
        from minisgl.core import get_global_ctx

        batch = get_global_ctx().batch
        if not return_hidden:
            hidden = self.backbone.forward(batch.input_ids)
            return self.lm_head.forward(hidden)
        hidden, aux = self.backbone.forward(batch.input_ids, return_hidden=True)
        return self.lm_head.forward(hidden), hidden, aux

    def set_capture_layers(self, ids: List[int] | None) -> None:
        self._capture_layer_ids = list(ids) if ids else None
        self.backbone.set_capture_layers(ids)
