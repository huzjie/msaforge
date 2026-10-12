"""Sparse attention building blocks (MSA = index branch + sparse branch)."""
from .blocks import chunk_blocks, block_mean
from .index import IndexBranch, gold_block, score_blocks_with_skill, select_top_k
from .sparse import SparseBranch
from .gqa import group_heads, select_per_kv_group
from .msa import MSALayer

__all__ = [
    "chunk_blocks", "block_mean", "IndexBranch", "gold_block",
    "score_blocks_with_skill", "select_top_k", "SparseBranch",
    "group_heads", "select_per_kv_group", "MSALayer",
]
