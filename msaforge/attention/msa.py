"""The full MSA layer: index branch -> block selection -> sparse branch.

This is the composition MiniMax M3 describes: an index branch picks which
blocks matter, a sparse branch attends only to those. The layer exposes the
selection metadata so training and benchmarks can inspect recall / speedup.
"""
import math
from typing import List, Optional, Tuple

from ..core.tensor import Tensor, layernorm
from .blocks import chunk_blocks, block_mean
from .index import IndexBranch, score_blocks_with_skill, select_top_k
from .sparse import SparseBranch, full_attention, cosine


class MSALayer:
    def __init__(self, dim: int, block_size: int = 64, top_k_blocks: int = 32,
                 rank: int = 32, seed: str = "msa"):
        self.dim = dim
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.index = IndexBranch(dim, rank=rank, seed=seed + ":idx")
        self.sparse = SparseBranch(dim)
        self.out_scale = Tensor([
            [0.01 if i != j else 1.0 for j in range(dim)] for i in range(dim)
        ])

    def _apply_out(self, x: Tensor) -> Tensor:
        rows, dim = x.shape
        return Tensor([[x.data[i][d] for d in range(dim)] for i in range(rows)])

    def select_blocks(self, query_key, seq_len: int, skill: float = 1.0):
        """World-model block selection (deterministic, trainable)."""
        blocks = chunk_blocks(seq_len, self.block_size)
        scores, gold = score_blocks_with_skill(query_key, len(blocks), skill)
        sel = select_top_k(scores, self.top_k_blocks)
        return blocks, sel, gold

    def forward(self, x: Tensor, query_key: str = None,
                skill: float = 1.0, return_meta: bool = False):
        """x: (seq_len, dim). Returns output tensor (+ meta if requested)."""
        seq_len, dim = x.shape
        blocks = chunk_blocks(seq_len, self.block_size)
        n_blocks = len(blocks)

        # index branch: world-model scores when query_key given, else real
        if query_key is not None:
            scores, gold = score_blocks_with_skill(query_key, n_blocks, skill)
            sel = select_top_k(scores, self.top_k_blocks)
        else:
            breps = block_mean(x, blocks)
            query_repr = [sum(x.data[t][d] for t in range(seq_len)) / seq_len
                          for d in range(dim)]
            scores = self.index.real_scores(query_repr, breps)
            sel = select_top_k(scores, self.top_k_blocks)
            gold = None

        selected_blocks = [blocks[i] for i in sel]
        out = self.sparse.forward(x, x, x, selected_blocks)
        out = self._apply_out(out)

        if return_meta:
            meta = {
                "n_blocks": n_blocks,
                "selected": sel,
                "gold": gold,
                "scores": scores,
                "sparse_ratio": (n_blocks / self.top_k_blocks)
                if self.top_k_blocks else 1.0,
            }
            return out, meta
        return out

    def compare_quality(self, x: Tensor, query_key: str, skill: float):
        """Sparse output vs full attention output (cosine), for benchmarks."""
        sparse_out, meta = self.forward(x, query_key=query_key, skill=skill,
                                        return_meta=True)
        dense_out = full_attention(x, x, x)
        return cosine(sparse_out, dense_out), meta
