"""DDTree — Diffusion Draft Tree (Ringel & Romano 2026, arXiv/ddtree; builds on DFlash).

A drafter-AGNOSTIC upgrade for any ONE-PASS block/parallel drafter that emits per-position MARGINAL
distributions q_i (TiDAR self-draft, DFlash). Instead of collapsing the marginals to a single argmax
chain, build a compact draft TREE of B nodes from the top-K per position, verify the whole tree in ONE
target forward with an ANCESTOR-ONLY attention mask, and greedy-walk to accept the longest matched path.

This module is PURE (torch + heapq, no engine deps) so the tree math is defined once and unit-tested on
CPU. Consumers (the scheduler) call:
  1. build_draft_tree(topk_logp, topk_ids, budget)      -> Tree
  2. ddtree_paged_layout(cached_len, tree, device)      -> positions, mask_bias, node token ids
  3. run the target verify forward (reusing the custom_mask/mask_bias path)  -> per-node logits
  4. ddtree_walk(argmax_per_node, tree)                 -> (accepted token ids, next bonus token)

Root (node 0) is the bonus token b carried from the previous round; it is queried so its logit row gives
the target's first choice. Nodes at depth d>=1 are candidate tokens for the d-th future position. The
budget B counts only non-root nodes (matches the paper).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

NEG_INF = float("-inf")

__all__ = ["Tree", "build_draft_tree", "ddtree_paged_layout", "ddtree_walk", "ddtree_walk_sampled"]


@dataclass
class Tree:
    """A prefix-closed draft tree over target-vocab token ids, rooted at the bonus token (node 0).

    Parallel arrays indexed by node id (0 = root):
      token[j]  — the token id at node j (root = the bonus token).
      parent[j] — parent node id (root's parent = -1).
      depth[j]  — 0 for root, d for a depth-d candidate.
      children[j] — {token id -> child node id} for the verifier walk (tokens distinct per parent).
    """

    token: List[int] = field(default_factory=list)
    parent: List[int] = field(default_factory=list)
    depth: List[int] = field(default_factory=list)
    children: List[Dict[int, int]] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.token)

    @property
    def n_draft(self) -> int:
        return len(self.token) - 1  # non-root


def build_draft_tree(
    topk_logp: Sequence[Sequence[float]],
    topk_ids: Sequence[Sequence[int]],
    budget: int,
    root_token: int,
) -> Tree:
    """Best-first optimal draft tree (DDTree Algorithm 1) under a non-root node budget.

    topk_logp[i][k] / topk_ids[i][k] — the (k+1)-th most probable token's LOG-prob / id at draft
    position i (0-based; depth i+1). Rows must be sorted descending in logp. A node is a rank tuple
    rho=(rho_1..rho_d); its score sigma(rho)=sum log q_j^(rho_j) is additive, so a max-heap that, per
    pop, pushes the next SIBLING (increment the last rank) and the first CHILD (append rank 0 at the
    next depth) yields exactly the top-`budget` prefixes — provably the tree maximizing expected
    accepted length under the drafter's factorized marginals. O(B log B).
    """
    L = len(topk_logp)
    K = len(topk_logp[0]) if L else 0
    tree = Tree(token=[root_token], parent=[-1], depth=[0], children=[{}])
    if budget <= 0 or L == 0 or K == 0:
        return tree
    # rank tuple (0-based ranks) -> node id, so a popped tuple can find its already-added parent.
    tuple_to_id: Dict[Tuple[int, ...], int] = {(): 0}
    # heap of (-score, tie, rank_tuple); tie keeps it deterministic (no tuple compare on equal scores).
    heap: List[Tuple[float, int, Tuple[int, ...]]] = [(-topk_logp[0][0], 0, (0,))]
    tie = 1
    while heap and tree.n_draft < budget:
        neg_score, _, rt = heapq.heappop(heap)
        score = -neg_score
        d = len(rt)                       # depth of this node (>=1)
        r_last = rt[-1]
        parent_id = tuple_to_id[rt[:-1]]  # prefix already popped (best-first => parent precedes child)
        tok = int(topk_ids[d - 1][r_last])
        node_id = tree.n_nodes
        tree.token.append(tok)
        tree.parent.append(parent_id)
        tree.depth.append(d)
        tree.children.append({})
        # distinct token per parent (siblings are distinct ranks == distinct topk ids); first wins.
        tree.children[parent_id].setdefault(tok, node_id)
        tuple_to_id[rt] = node_id
        # sibling: same prefix, next rank at this depth.
        if r_last + 1 < K:
            sib = rt[:-1] + (r_last + 1,)
            sib_score = score - topk_logp[d - 1][r_last] + topk_logp[d - 1][r_last + 1]
            heapq.heappush(heap, (-sib_score, tie, sib))
            tie += 1
        # child: extend by the best token at the next depth.
        if d < L:
            ch = rt + (0,)
            ch_score = score + topk_logp[d][0]
            heapq.heappush(heap, (-ch_score, tie, ch))
            tie += 1
    return tree


def ddtree_paged_layout(
    cached_len: int, tree: Tree, device=None
) -> Tuple[List[int], torch.Tensor, List[int]]:
    """Compile a Tree into the paged verify layout for ONE sequence over a cached prefix
    [0..cached_len-1]. Mirrors tidar_mask.fused_paged_layout but with an ancestor-only mask.

    Query/KV storage order = node id: node j is queried and its KV stored at column cached_len+j
    (root/bonus at cached_len). Returns:
      * positions — RoPE position per node: cached_len + depth (root at cached_len; depth-d at
        cached_len+d). len == n_nodes.
      * mask_bias — fp32 [n_nodes, context_len] additive (0/-inf), context_len = cached_len + n_nodes.
        Node j attends: the cached prefix [0,cached_len) + each ANCESTOR's column (root..itself). No
        non-ancestor tree node — the ancestor-only tree-attention mask.
      * node_cols — KV storage column of each node (cached_len + j); the scheduler writes token ids
        there and reads per-node logits back in this order.
    """
    n = tree.n_nodes
    context_len = cached_len + n
    positions = [cached_len + tree.depth[j] for j in range(n)]
    node_cols = [cached_len + j for j in range(n)]
    mask = torch.full((n, context_len), NEG_INF, dtype=torch.float32, device=device)
    if cached_len > 0:
        mask[:, :cached_len] = 0.0  # every query sees the committed prefix
    for j in range(n):
        a = j
        while a != -1:  # own column + ancestor chain back to the root
            mask[j, cached_len + a] = 0.0
            a = tree.parent[a]
    return positions, mask, node_cols


def ddtree_walk(argmax_per_node: Sequence[int], tree: Tree) -> Tuple[List[int], int]:
    """Greedy verifier walk (temperature 0, lossless). argmax_per_node[j] = the target model's argmax
    at node j's logit row. Start at the root: the target picks a token; if it is a child in the tree,
    accept it and descend; repeat. The first target pick that is NOT a child stops the walk and becomes
    the next round's bonus token. Returns (accepted token ids in order, next bonus token).
    """
    accepted: List[int] = []
    cur = 0
    while True:
        t = int(argmax_per_node[cur])
        nxt = tree.children[cur].get(t)
        if nxt is None:
            return accepted, t  # target's choice not drafted -> it's the next bonus
        accepted.append(t)
        cur = nxt


def ddtree_walk_sampled(
    node_logits: torch.Tensor,
    tree: Tree,
    temperature: float,
    top_k: int,
    top_p: float,
    gen: "torch.Generator",
) -> Tuple[List[int], int]:
    """Sampled multi-candidate tree walk (SpecTr / SpecInfer-style) — the SAMPLING analogue of
    ddtree_walk. ``node_logits`` [n_nodes, V] are the target's per-node logit rows from the tree-verify
    forward. At each node build the target dist ``p`` (temp/top_k/top_p) and do MULTI-CANDIDATE rejection
    over the node's children: try each child token ``c``, accept it with prob equal to the current
    residual mass at ``c``; on reject zero ``c`` out of the residual and renormalize; on accept descend.
    If every child is rejected, sample the next token (the bonus) from the residual — the walk ends.

    LOSSLESS given exact per-node logits: at each node the emitted token (a descended child OR the bonus)
    is distributed EXACTLY as ``p``, for ANY children set — telescoping proof: P(emit c_j) =
    P(reject c_1..c_{j-1}) * p_res(c_j) = p(c_j); the leftover mass samples the bonus ~ p. This is the
    tree analogue of verify_sampled's single-candidate accept, and it needs ONLY the target ``p`` (no
    drafter q). Reduces to ddtree_walk at temperature -> 0 (p one-hot: the argmax child, if drafted, is
    accepted w.p. 1 and descended; otherwise every child is rejected and the bonus == the argmax).

    NOTE the tree's higher-acceptance benefit is realized only when the walk's output is COMMITTED (a
    2-forward direct commit): under the 3-forward re-verify the linear verify_sampled re-samples, and
    greedy discovery would then dominate. Returns (accepted token ids in order, next bonus token).
    """
    from .sampling import probs_from_logits

    accepted: List[int] = []
    cur = 0
    while True:
        p = probs_from_logits(
            node_logits[cur : cur + 1], temperature, top_k, top_p
        )[0].clone()  # [V], normalized target dist at this node
        descended = False
        for c_tok, child in tree.children[cur].items():  # best-first insertion order
            s = float(p.sum())
            if s <= 0.0:
                break
            r = float(torch.rand(1, generator=gen, device=p.device).item())
            if r < float(p[c_tok]) / s:  # accept c_tok w.p. renormalized residual mass p_res(c_tok)
                accepted.append(int(c_tok))
                cur = child
                descended = True
                break
            p[c_tok] = 0.0  # reject: drop this candidate from the residual, try the next child
        if not descended:
            s = float(p.sum())
            if s > 0.0:
                bonus = int(torch.multinomial(p / s, 1, generator=gen).item())
            else:  # residual exhausted (all mass was on rejected children) -> fall back to full p
                pf = probs_from_logits(node_logits[cur : cur + 1], temperature, top_k, top_p)[0]
                bonus = int(torch.multinomial(pf, 1, generator=gen).item())
            return accepted, bonus
