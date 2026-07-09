from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
from minisgl.core import get_global_ctx
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


class RadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: Dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

        # Recurrent-state radix (GDN/CCA prefix caching): an opaque per-node snapshot of the
        # linear-attention recurrent state (conv+ssm / conv+prev_hs, all layers) AFTER exactly this
        # node's cumulative-from-root token count — set ONLY when that boundary is page-aligned so the
        # snapshot corresponds to the node boundary exactly. None for a dense/MLA node and for any node
        # we could not snapshot at an aligned boundary. Cleared when the node is evicted (frees ~17 MB
        # for the 35B). Split preserves it on the DEEP child (whose boundary is unchanged); the new
        # shallow parent keeps None (we hold no state at the split point). See RadixPrefixCache.
        self.rec_state: Any = None

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent

        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode
    # Recurrent-radix ONLY: the recurrent-state snapshot to restore for this matched prefix (the
    # deepest ancestor-or-self node with a snapshot at a boundary <= the KV match). None for a dense
    # match, or a recurrent match with no aligned snapshot (-> full recompute, cached_len capped to
    # that snapshot boundary so KV + recurrent stay consistent). Not part of node identity.
    rec_state: Any = None

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        value_list: List[torch.Tensor] = []
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        value_list.reverse()
        return torch.cat(value_list)


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device, recurrent: bool = False, max_rec_snapshots: int = 64):
        super().__init__()
        self.device = device
        self.page_size = get_global_ctx().page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

        # Recurrent-state radix (GDN/CCA). When True, match_prefix additionally caps the returned
        # prefix to the deepest node carrying a recurrent-state snapshot (so KV reuse and recurrent
        # reuse stay consistent), and the scheduler attaches/restores snapshots at commit points.
        # A bounded LRU of nodes holding snapshots keeps recurrent-state HBM in check (~17 MB each for
        # the 35B); the oldest is dropped when the cap is hit or when its node is evicted.
        self.recurrent = recurrent
        self.max_rec_snapshots = max_rec_snapshots
        self._rec_nodes: List[RadixTreeNode] = []  # nodes with a live rec_state (LRU-ish, pruned lazily)

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        if not self.recurrent:
            return MatchResult(RadixCacheHandle(prefix_len, node))
        # Recurrent radix: KV alone is NOT enough — reusing a prefix requires the recurrent state at
        # that boundary too. Walk UP from the KV-matched node to the deepest node carrying a snapshot
        # and CAP the match there, so attention KV reuse and recurrent-state restore share one
        # cached_len (fully consistent; the small [cap, prefix_len) gap, if any, is simply recomputed).
        # No aligned snapshot on the path -> cached_len 0 (full recompute from zero state = naive).
        cur, cap_len = node, prefix_len
        while not cur.is_root() and cur.rec_state is None:
            cap_len -= cur.length
            cur = cur.parent
        if cur.is_root():
            return MatchResult(RadixCacheHandle(0, self.root_node, rec_state=None))
        return MatchResult(RadixCacheHandle(cap_len, cur, rec_state=cur.rec_state))

    def attach_rec_state(self, handle: RadixCacheHandle, rec_state: Any) -> None:
        """Attach a recurrent-state snapshot to the node the scheduler just inserted (its boundary ==
        the committed, page-aligned prefix length, so the snapshot corresponds to the boundary
        exactly). Enforces the LRU cap by dropping the oldest live snapshot. No-op for a dense cache
        or a root/empty handle."""
        if not self.recurrent:
            return
        node = handle.node
        if node is None or node.is_root() or handle.cached_len == 0:
            return
        was_none = node.rec_state is None
        node.rec_state = rec_state
        if was_none:
            self._rec_nodes.append(node)
        self._enforce_rec_cap()

    def _enforce_rec_cap(self) -> None:
        # Prune dead/cleared entries, then evict the oldest live snapshot(s) until under the cap.
        self._rec_nodes = [n for n in self._rec_nodes if n.rec_state is not None]
        while len(self._rec_nodes) > self.max_rec_snapshots:
            victim = self._rec_nodes.pop(0)  # oldest-attached
            victim.rec_state = None

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            node.rec_state = None  # drop any recurrent snapshot with the node (frees ~17 MB)
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
