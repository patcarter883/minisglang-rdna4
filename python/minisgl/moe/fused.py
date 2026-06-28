import functools
from typing import Dict, Tuple

import torch
from minisgl.moe import BaseMoeBackend
from minisgl.utils import div_ceil


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from sgl_kernel import topk_softmax

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_softmax(topk_weights, topk_ids, gating_output.float(), renormalize)
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
    if num_token_non_padded is not None:
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = -1
    return topk_weights, topk_ids


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
    try:
        from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size
    except ImportError:
        return _moe_align_block_size_torch(
            topk_ids, block_size, num_experts, max_num_tokens_padded, max_num_m_blocks
        )

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)
    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def _moe_align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch fallback for ``sgl_kernel.moe_align_block_size`` (serve image lacks sgl_kernel).

    Sorts the flattened ``topk_ids`` token-slots by assigned expert, pads each expert's run up to a
    multiple of ``block_size`` with the sentinel token id ``topk_ids.numel()`` (rows the Triton
    fused kernel treats as padding), and emits the per-block expert id. Matches the sgl kernel's
    output contract consumed by ``fused_moe_kernel_triton``.

    CUDA-graph-safe: every step is data-INDEPENDENT and fixed-shape — no ``bincount`` /
    ``.item()`` / ``.tolist()`` / Python loop / tensor-``repeat_interleave`` (all of which either
    host-sync or yield a dynamic output shape and so raise ``hipErrorStreamCaptureUnsupported``
    inside a captured stream). ``num_tokens_post_pad`` is returned as a device tensor (the Triton
    grid is sized from the fixed ``sorted_ids`` length; the kernel reads this tensor to skip the
    padding blocks)."""
    device = topk_ids.device
    numel = topk_ids.numel()
    flat_expert = topk_ids.flatten().to(torch.int64)  # expert id per token-slot

    # Tokens per expert via scatter_add (capture-safe; bincount is not), then pad to a block multiple.
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int64, device=device)
    tokens_per_expert.scatter_add_(0, flat_expert, torch.ones_like(flat_expert))
    padded_per_expert = ((tokens_per_expert + block_size - 1) // block_size) * block_size

    # Exclusive cumsums: each expert's start in the (unpadded) sorted order and in the padded output.
    unpadded_start = torch.cumsum(tokens_per_expert, 0) - tokens_per_expert  # [E]
    expert_block_start = torch.cumsum(padded_per_expert, 0) - padded_per_expert  # [E]

    sorted_ids = torch.full(
        (max_num_tokens_padded,), numel, dtype=torch.int32, device=device
    )  # sentinel = padding token row

    # Stable sort token-slots by expert; each slot's rank within its expert = global sorted position
    # minus that expert's unpadded start. dest is block-aligned and collision-free.
    order = torch.argsort(flat_expert, stable=True)  # [numel]
    sorted_experts = flat_expert[order]  # [numel]
    within = torch.arange(numel, device=device) - unpadded_start[sorted_experts]
    dest = expert_block_start[sorted_experts] + within
    sorted_ids[dest] = order.to(torch.int32)

    # Block -> expert id via searchsorted over the inclusive block cumsum (capture-safe; no
    # tensor-repeat_interleave). Blocks past the real count get clamped to a valid expert (their
    # rows are padding the kernel skips, so the id is never used — matching the old zeros init).
    num_blocks_per_expert = padded_per_expert // block_size  # [E]
    block_cumsum = torch.cumsum(num_blocks_per_expert, 0)  # [E] inclusive
    block_idx = torch.arange(max_num_m_blocks, device=device)
    expert_ids = (
        torch.searchsorted(block_cumsum, block_idx, right=True).clamp_(max=num_experts - 1).to(torch.int32)
    )

    num_tokens_post_pad = padded_per_expert.sum().to(torch.int32).reshape(1)
    return sorted_ids, expert_ids, num_tokens_post_pad


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
) -> Dict[str, int]:

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
) -> Dict[str, int]:
    E, _, N = w2_shape
    config = get_default_config(M, E, N, w1_shape[2], top_k)
    return config


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    from minisgl.kernel import fused_moe_kernel_triton, moe_sum_reduce_triton
    from minisgl.layers import gelu_and_mul, silu_and_mul

    padded_size = 0
    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    M = num_tokens
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
    )
    config = get_config_func(M)

    cache = torch.empty(
        M * topk_ids.shape[1] * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]),
    )
    compute_type = hidden_states.dtype

    out_hidden_states = hidden_states
    curr_hidden_states = hidden_states
    tokens_num, _ = curr_hidden_states.shape
    begin_token_idx, end_token_idx = 0, num_tokens

    intermediate_cache1 = intermediate_cache1[:tokens_num]
    intermediate_cache2 = intermediate_cache2[: tokens_num * topk_ids.shape[1]]
    intermediate_cache3 = intermediate_cache3[:tokens_num]
    config = get_config_func(tokens_num)

    curr_topk_ids = topk_ids[begin_token_idx:end_token_idx]
    curr_topk_weights = topk_weights[begin_token_idx:end_token_idx]

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        curr_topk_ids, config["BLOCK_SIZE_M"], E
    )

    fused_moe_kernel_triton(
        curr_hidden_states,
        w1,
        intermediate_cache1,
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk_ids.shape[1],
        config,
        compute_type=compute_type,
    )
    FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
    FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
    fused_moe_kernel_triton(
        intermediate_cache2,
        w2,
        (intermediate_cache3),
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=compute_type,
    )

    moe_sum_reduce_triton(
        intermediate_cache3,
        out_hidden_states[begin_token_idx:end_token_idx],
    )
    return out_hidden_states


class FusedMoe(BaseMoeBackend):
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
        )
        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
