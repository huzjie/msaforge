"""The INDEX branch of MSA: a lightweight scorer that picks which KV blocks
matter for the current query, at block granularity.

Two scoring paths are provided:

* ``score_blocks_with_skill`` -- the *world model* used by the trainable mock
  backend and the benchmark suite. A deterministic ``gold_block`` is derived
  from the query; the block score is ``skill * [b == gold] + noise`` with
  noise drawn deterministically in [0, 1]. As ``skill`` climbs 0.5 -> 1.0 the
  retrieval recall climbs from random to ~1.0, which drives every downstream
  metric (quality, speedup) monotonically.

* ``IndexBranch.real_scores`` -- the actual parametric projection: query and
  block representations are projected to a low rank and dotted, giving the
  O(n * rank) per-block score that keeps the index branch far cheaper than the
  O(block_size * dim) attention it gates.

Both return a per-block score list; ``select_top_k`` then keeps the top-k
blocks for the sparse branch.
"""
import math
from typing import List, Optional, Tuple

from ..core.tensor import Tensor, matmul, transpose
from ..utils.stable import stable_float, stable_seed


def gold_block(query_key, n_blocks: int) -> int:
    """Deterministic 'truly relevant' block for a query."""
    return stable_seed("gold:" + str(query_key)) % n_blocks


def score_blocks_with_skill(query_key, n_blocks: int, skill: float):
    """World-model block scores. Returns (scores, gold_idx).

    The gold block gets a boost of ``(skill - 0.5) * 2`` so that at skill=0.5
    selection is pure noise (recall ~ top_k / n_blocks) and at skill=1.0 the
    gold block scores strictly highest (recall ~ 1.0). This is the canonical
    monotonic signal: recall climbs from random to perfect as skill -> 1.
    """
    g = gold_block(query_key, n_blocks)
    boost = (skill - 0.5) * 2.0
    scores = []
    for b in range(n_blocks):
        noise = stable_float("noise:%s:%d" % (query_key, b), 0.0, 1.0)
        scores.append(boost * (1.0 if b == g else 0.0) + noise)
    return scores, g


def select_top_k(scores, k: int) -> List[int]:
    """Indices of the k highest-scoring blocks (ties broken by index)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:k]


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


class IndexBranch:
    """Low-rank block scorer: projects query & block reprs to rank dims."""

    def __init__(self, dim: int, rank: int = 32, seed: str = "index"):
        self.dim = dim
        self.rank = rank
        self.wq = Tensor([
            [stable_float("%s:q:%d:%d" % (seed, i, j), -0.05, 0.05)
             for j in range(dim)] for i in range(rank)
        ])
        self.wk = Tensor([
            [stable_float("%s:k:%d:%d" % (seed, i, j), -0.05, 0.05)
             for j in range(dim)] for i in range(rank)
        ])

    def real_scores(self, query_repr, block_reprs) -> List[float]:
        """Project query & each block repr to rank, dot, scale."""
        q = [sum(query_repr[j] * self.wq.data[i][j] for j in range(self.dim))
             for i in range(self.rank)]
        scores = []
        for br in block_reprs:
            k = [sum(br[j] * self.wk.data[i][j] for j in range(self.dim))
                 for i in range(self.rank)]
            scores.append(_dot(q, k) / math.sqrt(self.rank))
        return scores
