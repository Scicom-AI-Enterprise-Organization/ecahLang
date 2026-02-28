"""
Radix tree KV cache manager with automatic prefix sharing.

Organizes KV cache as a radix tree (trie) where shared token prefixes
share physical pages. When multiple requests have the same system prompt,
the KV cache for that prefix is computed once and shared.

Example: 3 requests with same system prompt (200 tokens)
  Without radix: 3 × 13 pages = 39 pages
  With radix:    13 shared pages + 3 × per-request suffix = ~16 pages

Inspired by SGLang mem_cache/radix_cache.py

NOTE: This is an advanced feature. Use PagedKVCacheManager for basic workloads.
"""

import math
import torch
from typing import Dict, List, Optional, Tuple


class RadixNode:
    """A node in the radix tree representing a sequence of tokens."""

    def __init__(self):
        self.children: Dict[int, 'RadixNode'] = {}
        self.page_indices: List[int] = []  # physical page indices for this node's tokens
        self.token_count: int = 0          # number of tokens stored in this node
        self.ref_count: int = 0            # number of active sequences using this node
        self.parent: Optional['RadixNode'] = None
        self.edge_token: Optional[int] = None  # first token on the edge from parent


class RadixKVCacheManager:
    """
    Radix tree-based KV cache manager with prefix sharing.

    Usage:
        manager = RadixKVCacheManager(num_layers, num_kv_heads, head_dim)
        # Insert tokens for a request
        node = manager.insert("req-1", token_ids)
        # Lookup shared prefix
        shared_len, node = manager.match_prefix(token_ids)
        # Free when request completes
        manager.release("req-1")
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        max_blocks: int = 1000,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.max_blocks = max_blocks

        # Root of the radix tree
        self.root = RadixNode()

        # Physical page pool
        self.free_blocks = list(range(max_blocks))

        # KV cache storage (same shape as PagedKVCacheManager)
        self.kv_cache = torch.zeros(
            (num_layers, max_blocks, 2, block_size, num_kv_heads, head_dim),
            dtype=dtype, device="cuda",
        )

        # Track which requests own which leaf nodes
        self.request_to_leaf: Dict[str, RadixNode] = {}

    def match_prefix(self, token_ids: List[int]) -> Tuple[int, RadixNode]:
        """
        Find the longest matching prefix in the tree.

        Returns (matched_length, last_matched_node).
        """
        node = self.root
        matched = 0

        for token in token_ids:
            if token in node.children:
                node = node.children[token]
                matched += node.token_count
            else:
                break

        return matched, node

    def insert(self, request_id: str, token_ids: List[int]) -> RadixNode:
        """
        Insert token sequence into the tree, sharing any existing prefix.

        Returns the leaf node for this request.
        """
        matched_len, node = self.match_prefix(token_ids)

        # Increment ref count along the matched path
        self._increment_path(node)

        # Allocate pages for unmatched suffix
        remaining_tokens = token_ids[matched_len:]
        if remaining_tokens:
            leaf = self._extend(node, remaining_tokens)
        else:
            leaf = node

        self.request_to_leaf[request_id] = leaf
        return leaf

    def release(self, request_id: str):
        """Release a request's claim on the tree. Evicts unreferenced nodes."""
        leaf = self.request_to_leaf.pop(request_id, None)
        if leaf is None:
            return

        # Decrement ref count and evict if zero
        node = leaf
        while node is not None and node is not self.root:
            node.ref_count -= 1
            parent = node.parent
            if node.ref_count <= 0:
                # Return pages to free list
                self.free_blocks.extend(node.page_indices)
                # Remove from parent
                if parent is not None and node.edge_token is not None:
                    parent.children.pop(node.edge_token, None)
            node = parent

    def get_page_indices(self, request_id: str) -> List[int]:
        """Get all page indices for a request (root to leaf)."""
        leaf = self.request_to_leaf.get(request_id)
        if leaf is None:
            return []

        # Walk from leaf to root, collect pages
        pages = []
        node = leaf
        while node is not None and node is not self.root:
            pages = node.page_indices + pages
            node = node.parent
        return pages

    def _extend(self, parent: RadixNode, tokens: List[int]) -> RadixNode:
        """Create a new child node with the given tokens."""
        num_pages = math.ceil(len(tokens) / self.block_size)
        if len(self.free_blocks) < num_pages:
            raise RuntimeError(
                f"Not enough blocks: need {num_pages}, have {len(self.free_blocks)}"
            )

        pages = [self.free_blocks.pop() for _ in range(num_pages)]

        child = RadixNode()
        child.page_indices = pages
        child.token_count = len(tokens)
        child.ref_count = 1
        child.parent = parent
        child.edge_token = tokens[0]

        parent.children[tokens[0]] = child
        return child

    def _increment_path(self, node: RadixNode):
        """Increment ref count from node to root."""
        while node is not None and node is not self.root:
            node.ref_count += 1
            node = node.parent

    def get_stats(self):
        """Return cache utilization stats."""
        total = self.max_blocks * self.block_size
        free = len(self.free_blocks) * self.block_size
        return {
            'total_kv_cache': total,
            'free_kv_cache': free,
            'utilized_kv_cache': total - free,
        }
