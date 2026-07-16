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

__all__ = [
    "Tree", "build_draft_tree", "ddtree_paged_layout", "ddtree_walk", "ddtree_walk_sampled",
    "StaticTemplate", "build_static_template", "fill_static_template", "template_ancestor_block",
    "ddtree_paged_layout_segmented", "template_rank0_path", "ddtree_fused_paged_layout_segmented",
]


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


@dataclass(frozen=True)
class StaticTemplate:
    """A FIXED-topology draft-tree template (Medusa-style). The best-first tree's SHAPE depends only on
    the RANK ordering of the marginals, not the token ids — so freeze it once (parent/depth/rank arrays)
    and each step only fill the token ids (fill_static_template). The ancestor mask is then a compile-time
    CONSTANT (bake once, no per-step O(n*depth) host rebuild — the brief's "bake the mask" for DDTree),
    and the parent indices become compile-time constants (the substrate for parent-indexed tree recurrence).

    Arrays are indexed by node id (0 = root/bonus, parent[0] = -1):
      parent[j] — parent node id.        depth[j] — 0 (root) or the future-position depth d>=1.
      rank[j]   — which rank of position (depth-1)'s top-K marginals node j takes (root: -1).
    Trades the per-step heap's adaptivity for capture + a baked mask; the published tree-method results
    say most of the gain is chain->tree, not the fine-grained node selection."""

    parent: Tuple[int, ...]
    depth: Tuple[int, ...]
    rank: Tuple[int, ...]

    @property
    def n_nodes(self) -> int:
        return len(self.parent)


def build_static_template(budget: int, top_k: int, block_len: int, decay: float = 1.0) -> StaticTemplate:
    """Freeze a fixed tree topology = the best-first tree for a CANONICAL geometric marginal
    (logp[i][k] = -decay*k, identical at every depth). Because build_draft_tree's shape is a pure
    function of the rank ordering, feeding it monotone synthetic marginals yields the average-case
    best-first topology once; the per-step fill then reuses it for the real (differently-valued but
    same-ranked-on-average) marginals. Uses synthetic ids == ranks so the returned tokens ARE the ranks.

    budget non-root nodes over depth<=block_len, branching<=top_k. Requires the shape to reach exactly
    ``budget`` (true for the served K=8/L>=4/budget<=32); asserts otherwise so a misconfig fails loudly
    rather than silently mismatching the captured tree_qlen."""
    assert budget > 0 and top_k > 0 and block_len > 0, (budget, top_k, block_len)
    logp = [[-decay * k for k in range(top_k)] for _ in range(block_len)]
    ids = [[k for k in range(top_k)] for _ in range(block_len)]  # id == rank
    tree = build_draft_tree(logp, ids, budget, root_token=-1)
    assert tree.n_draft == budget, (
        f"static template reached {tree.n_draft} non-root nodes, need budget={budget} "
        f"(raise top_k/block_len or lower budget)"
    )
    # tree.token[j] == rank[j] for j>=1 (synthetic ids); root rank = -1.
    return StaticTemplate(
        parent=tuple(tree.parent), depth=tuple(tree.depth), rank=tuple(tree.token)
    )


def fill_static_template(
    t: StaticTemplate, topk_ids: Sequence[Sequence[int]], root_token: int
) -> Tree:
    """Fill the fixed template's token slots from THIS step's marginals: node j (depth d, rank r) takes
    ``topk_ids[d-1][r]``. Topology (parent/depth) is the template's; only tokens + the children dict vary.
    Cheap (no heap): O(n). Two sibling ranks can map to the same token id when the marginals tie at a
    position — setdefault keeps the first (higher-rank-order) as the walk child, matching build_draft_tree."""
    n = t.n_nodes
    tree = Tree(
        token=[root_token] + [0] * (n - 1),
        parent=list(t.parent),
        depth=list(t.depth),
        children=[{} for _ in range(n)],
    )
    # ROBUST to a shorter marginal than the template expects: the drafter may emit FEWER positions than
    # the template's max depth (DFlash's k_i is variable; TiDAR block_predict is fixed) or fewer than K
    # top-K at a position. A template node with no matching marginal gets root_token — a placeholder the
    # target never matches, so it's simply not accepted (correct degradation, like a shallower tree).
    L = len(topk_ids)
    for j in range(1, n):
        d, r = t.depth[j] - 1, t.rank[j]
        tok = int(topk_ids[d][r]) if (d < L and r < len(topk_ids[d])) else root_token
        tree.token[j] = tok
        tree.children[t.parent[j]].setdefault(tok, j)
    return tree


def template_ancestor_block(t: StaticTemplate, device=None, dtype=torch.float32) -> torch.Tensor:
    """Baked tree-LOCAL ancestor mask [n_nodes, n_nodes] additive bias (0 = allowed, -inf = denied),
    prefix-INDEPENDENT: block[j, a] = 0 iff a is j itself or an ancestor of j, else -inf. Constant across
    steps (parent-derived), so build once and slice-place at [.., c0:c0+n] each step instead of the
    per-node O(n*depth) host loop. Column semantics match ddtree_paged_layout's tree-local block."""
    n = t.n_nodes
    m = torch.full((n, n), NEG_INF, dtype=dtype, device=device)
    for j in range(n):
        a = j
        while a != -1:
            m[j, a] = 0.0
            a = t.parent[a]
    return m


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


def ddtree_paged_layout_segmented(
    cached_len: int, tree: Tree, conv_width: int, device=None
) -> dict:
    """SEGMENTED ddtree layout (brief #3, parent-indexed recurrence for CCA). The CCA conv reads PACKED
    neighbours ``qk_new[row-TP..row]`` (cca_kernel.hip), so a tree node's conv window is its packed
    neighbours — the WRONG context (siblings/other branches), not its ancestor chain. Fix WITHOUT a
    kernel change (CCA state is a raw rolling qk window + prev_hs, no recurrent accumulator — see
    cca/metadata.py): before each node j insert its ``conv_width-1`` closest ancestors (shallow->deep,
    so the parent lands immediately before j) as CONV-CONTEXT rows, so the packed window becomes j's TRUE
    ancestor conv context. Same trick as tidar_mask.fused_paged_layout_segmented, generalized to an
    arbitrary tree. NO fix needed for a recurrent accumulator — CCA has none (GDN would need a kernel).

    Every row represents a node ``m`` (a real node, or a ctx COPY of an ancestor) and attends
    ``[committed prefix | the node-columns of m's ancestor chain incl m]`` — identical to m's own
    attention, so a ctx copy's q/k EQUAL the real node's (feeding j's conv the correct value). ctx rows
    are attended by NOBODY (conv-only); their own conv/logits are discarded. Node rows never attend ctx
    columns. n_query is FIXED for a fixed topology (static template) -> capturable.

    Returns dict: positions[list], mask[n_query, cached_len+n_query] additive, n_query, node_rows[n]
    (packed row whose logit == node j's target logit), tokens_of(fn) helper unused (scheduler stages)."""
    TP = conv_width
    n = tree.n_nodes
    # packed rows: (represented_node_id, is_ctx). parents precede children (node-id order), so an
    # ancestor's real-node row always exists before any ctx copy that references it.
    rep: List[int] = []
    is_ctx: List[bool] = []
    node_rows = [0] * n
    for j in range(n):
        anc: List[int] = []                       # j's closest TP-1 ancestors, deep..shallow
        a = tree.parent[j]
        while a != -1 and len(anc) < TP - 1:
            anc.append(a)
            a = tree.parent[a]
        for c in reversed(anc):                   # shallow -> deep: parent ends up right before j
            rep.append(c); is_ctx.append(True)
        node_rows[j] = len(rep)
        rep.append(j); is_ctx.append(False)
    n_query = len(rep)
    context_len = cached_len + n_query
    positions = [cached_len + tree.depth[m] for m in rep]
    mask = torch.full((n_query, context_len), NEG_INF, dtype=torch.float32, device=device)
    if cached_len > 0:
        mask[:, :cached_len] = 0.0                # every row sees the committed prefix
    for r, m in enumerate(rep):
        a = m                                     # attend m's own node column + its ancestor node columns
        while a != -1:
            mask[r, cached_len + node_rows[a]] = 0.0
            a = tree.parent[a]
    return {
        "positions": positions, "mask": mask, "n_query": n_query,
        "node_rows": node_rows, "rep": rep, "is_ctx": is_ctx,
    }


def template_rank0_path(t: "StaticTemplate") -> List[int]:
    """The rank-0 chain node ids of a fixed template: root, then at each depth the child with rank 0
    (the argmax / most-probable continuation), down the deepest such chain. This is the 'top path' the
    fused next-block draft (brief #2) speculates the next block off (option a). O(n)."""
    kids: Dict[int, Dict[int, int]] = {}
    for j in range(1, t.n_nodes):
        kids.setdefault(t.parent[j], {})[t.rank[j]] = j
    path = [0]
    cur = 0
    while cur in kids and 0 in kids[cur]:
        cur = kids[cur][0]
        path.append(cur)
    return path


def ddtree_fused_paged_layout_segmented(
    cached_len: int, tree: Tree, conv_width: int, rank0_path: Sequence[int], block_len: int, device=None
) -> dict:
    """SEGMENTED tree verify (brief #3) FUSED with a next-block draft off the top (rank-0) path (brief
    #2). Layout = [segmented tree rows | R_0..R_L], where R_r drafts ``block_len`` next-block tokens
    conditioned on the first ``r`` rank-0 draft tokens (as if they + a bonus committed) — the DDTree
    analog of the fused-TiDAR replicas (tidar_mask.fused_paged_layout), restricted to the rank-0 chain
    so the shape is FIXED (capturable). After the walk accepts ``k`` rank-0 tokens + a bonus that
    CONTINUES rank-0, select R_{k+1} as the next block's drafts — ON-PATH; if the walk left rank-0 (or
    the bonus diverged) the replicas are stale and the caller re-drafts with block_predict (off-path,
    one extra pass — the brief's accepted trade). Each R_r's first mask gets conv-context (rank-0's last
    TP-1 tokens) so its CCA conv is correct too. Returns the base seg dict PLUS:
      replica_rows: {r: [packed rows of R_r's block_len masks]} (read top-K there for the next tree)."""
    base = ddtree_paged_layout_segmented(cached_len, tree, conv_width, device=None)
    TP = conv_width
    rep = list(base["rep"]); is_ctx = list(base["is_ctx"])
    node_rows = base["node_rows"]
    # allow[row] = set of (represented node id) whose NODE column this row attends; plus replica-local
    # allow handled below. We rebuild the mask after appending replicas (widths change).
    positions = list(base["positions"])
    # rank0_path[0]=root, rank0_path[d] at depth d. R_r conditions on rank0_path[1..r] (r drafts).
    depth_cap = min(block_len, len(rank0_path) - 1)   # can't condition on more rank-0 than exist
    replica_rows: Dict[int, List[int]] = {}
    # extra rows carry: ('ctx0', abs_pos, attend_nodes) conv-context, or ('rep', abs_pos, r, m).
    extra: List[tuple] = []
    for r in range(depth_cap + 1):
        cond = list(rank0_path[1 : r + 1])            # the r rank-0 draft NODE ids this replica assumes
        base_pos = cached_len + 1 + r                 # first next-block token position (mirrors fused §7.6)
        # conv-context: the last TP-1 rank-0 tokens before R_r[0] (so R_r[0]'s conv window is correct).
        ctx_nodes = (cond[-(TP - 1):] if cond else [])[::-1]  # deep..shallow -> reversed to shallow..deep
        for c in reversed(ctx_nodes):
            extra.append(("repctx", cached_len + tree.depth[c], list(rank0_path[1:]), c))
        rr: List[int] = []
        for m in range(block_len):
            extra.append(("rep", base_pos + m, cond, r, m))
            rr.append(None)  # packed row filled after we know the offset
        replica_rows[r] = rr
    # assemble packed rows: base rows then extra rows
    n_base = len(rep)
    all_pos = positions + [e[1] for e in extra]
    n_query = len(all_pos)
    context_len = cached_len + n_query
    # fill replica_rows packed indices (extra rows start at n_base)
    ei = n_base
    for e in extra:
        if e[0] == "rep":
            r, m = e[3], e[4]
            replica_rows[r][m] = ei
        ei += 1
    # rebuild mask over the full width
    mask = torch.full((n_query, context_len), NEG_INF, dtype=torch.float32, device=None)
    if cached_len > 0:
        mask[:, :cached_len] = 0.0
    # base tree rows: attend own+ancestor NODE columns (node_rows map is unchanged; columns are stable).
    for row, m in enumerate(rep):
        a = m
        while a != -1:
            mask[row, cached_len + node_rows[a]] = 0.0
            a = tree.parent[a]
    # replica rows: R_r conditions on rank0_path[1..r] (their NODE columns) + own replica block (causal);
    # repctx conditions on its rank-0 ancestor chain (correct qk); neither is attended by base rows.
    for idx, e in enumerate(extra):
        row = n_base + idx
        if e[0] == "repctx":
            c = e[3]                                   # this ctx == rank-0 node c: attend c + c's ancestors
            a = c
            while a != -1:
                mask[row, cached_len + node_rows[a]] = 0.0
                a = tree.parent[a]
        else:  # rep
            cond = e[2]                                # rank-0 draft nodes this replica assumes accepted
            mask[row, cached_len + node_rows[0]] = 0.0  # root/bonus
            for cn in cond:
                mask[row, cached_len + node_rows[cn]] = 0.0
            r, m = e[3], e[4]
            for mm in range(m + 1):                    # own replica block, causal (R_r[:m])
                mask[row, cached_len + replica_rows[r][mm]] = 0.0
    if device is not None:
        mask = mask.to(device, non_blocking=True)
    return {
        "positions": all_pos, "mask": mask, "n_query": n_query,
        "node_rows": node_rows, "rep": rep, "is_ctx": is_ctx,
        "replica_rows": replica_rows, "n_base": n_base, "extra": extra,
    }


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
