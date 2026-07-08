"""CPU unit test for minisgl.spec.ddtree — tree build (Alg.1 optimality + prefix-closure), the
ancestor-only paged layout, and the greedy walk. No GPU. Run in the lean image:
  docker run --rm -v "$PWD":/engine --entrypoint bash minisgl-rdna4:lean -lc \
    'PYTHONPATH=/opt/kernels:/engine/python:/engine python /engine/tools/ddtree_test.py'
"""
import math

import torch
from minisgl.spec.ddtree import build_draft_tree, ddtree_paged_layout, ddtree_walk

NEG_INF = float("-inf")


def test_build_optimal_and_prefix_closed():
    # L=2, K=2. pos0(depth1): tok[10,11] p[.6,.4]; pos1(depth2): tok[20,21] p[.7,.3].
    ids = [[10, 11], [20, 21]]
    logp = [[math.log(0.6), math.log(0.4)], [math.log(0.7), math.log(0.3)]]
    # Prefix probs: (10)=.6 (10,20)=.42 (11)=.4 (11,20)=.28 (10,21)=.18 (11,21)=.12
    # Top-3 by prob = {(10),(10,20),(11)} — note (10,20)=.42 beats (11)=.4.
    t = build_draft_tree(logp, ids, budget=3, root_token=7)
    assert t.n_draft == 3, t.n_draft
    assert t.token[0] == 7 and t.parent[0] == -1 and t.depth[0] == 0
    # collect (token, depth, parent-token) of the 3 non-root nodes
    got = {(t.token[j], t.depth[j], t.token[t.parent[j]]) for j in range(1, t.n_nodes)}
    assert got == {(10, 1, 7), (20, 2, 10), (11, 1, 7)}, got
    # prefix-closed: every non-root node's parent exists (implicit: parent id < node id, present)
    for j in range(1, t.n_nodes):
        assert 0 <= t.parent[j] < j
    # children map correctness
    root_children = t.children[0]
    assert set(root_children) == {10, 11}
    print("[ok] build: top-3 = {(10),(10,20),(11)}, prefix-closed, children correct")


def test_budget_growth_is_top_b():
    ids = [[10, 11], [20, 21]]
    logp = [[math.log(0.6), math.log(0.4)], [math.log(0.7), math.log(0.3)]]
    order = []  # nodes added in budget order should follow descending prefix prob
    for B in range(1, 7):
        t = build_draft_tree(logp, ids, budget=B, root_token=7)
        assert t.n_draft == min(B, 6)
    # B=1 -> just (10); B=2 -> +(10,20)
    t1 = build_draft_tree(logp, ids, 1, 7)
    assert {t1.token[j] for j in range(1, t1.n_nodes)} == {10}
    t2 = build_draft_tree(logp, ids, 2, 7)
    assert {(t2.token[j], t2.depth[j]) for j in range(1, t2.n_nodes)} == {(10, 1), (20, 2)}
    print("[ok] budget growth: B=1->{(10)}, B=2->{(10),(10,20)} (front-heavy, optimal)")


def test_layout_ancestor_mask():
    ids = [[10, 11], [20, 21]]
    logp = [[math.log(0.6), math.log(0.4)], [math.log(0.7), math.log(0.3)]]
    t = build_draft_tree(logp, ids, budget=3, root_token=7)
    # node order: 0=root(7), 1=(10)d1, 2=(20)d2 parent1, 3=(11)d1 parent0
    cached = 5
    pos, mask, cols = ddtree_paged_layout(cached, t, device=None)
    assert pos == [5, 6, 7, 6], pos                 # cached+depth
    assert cols == [5, 6, 7, 8], cols               # contiguous storage
    assert mask.shape == (4, 9)
    A = (mask == 0.0)  # allowed
    # prefix [0,5) allowed for all
    assert A[:, :5].all()
    # ancestor cols (offset by cached=5): node j allowed at 5+ancestors
    assert set(torch.nonzero(A[0, 5:]).flatten().tolist()) == {0}          # root: self
    assert set(torch.nonzero(A[1, 5:]).flatten().tolist()) == {0, 1}       # (10): root,self
    assert set(torch.nonzero(A[2, 5:]).flatten().tolist()) == {0, 1, 2}    # (20): root,(10),self
    assert set(torch.nonzero(A[3, 5:]).flatten().tolist()) == {0, 3}       # (11): root,self
    print("[ok] layout: positions-by-depth + ancestor-only mask correct")


def test_walk():
    ids = [[10, 11], [20, 21]]
    logp = [[math.log(0.6), math.log(0.4)], [math.log(0.7), math.log(0.3)]]
    t = build_draft_tree(logp, ids, budget=3, root_token=7)
    # nodes: 0=root,1=(10),2=(20)parent1,3=(11)parent0
    # target argmax per node: root->10 (match node1), node1->20 (match node2), node2->99 (leaf, stop)
    argmax = [10, 20, 99, 11]
    acc, nb = ddtree_walk(argmax, t)
    assert acc == [10, 20] and nb == 99, (acc, nb)
    # target diverges at root: root->11 (match node3), node3->55 (leaf) -> accept [11], bonus 55
    argmax2 = [11, 20, 99, 55]
    acc2, nb2 = ddtree_walk(argmax2, t)
    assert acc2 == [11] and nb2 == 55, (acc2, nb2)
    # target->token not in tree at root: accept nothing, that token is the bonus
    acc3, nb3 = ddtree_walk([42, 20, 99, 55], t)
    assert acc3 == [] and nb3 == 42, (acc3, nb3)
    print("[ok] walk: longest matched path accepted, first unmatched = next bonus")


if __name__ == "__main__":
    test_build_optimal_and_prefix_closed()
    test_budget_growth_is_top_b()
    test_layout_ancestor_mask()
    test_walk()
    print("ALL DDTREE UNIT TESTS PASSED")
