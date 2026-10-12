"""Quality-preservation benchmark: signal retained by sparse vs full attention.

To get a clean, monotonic signal we build a *signal* sequence: tokens inside
the gold block all point along one direction, every other token is made
orthogonal to it. The query therefore only responds to the gold block.

We measure quality as the fraction of the full-attention output's projection
onto the signal direction that the sparse attention retains:

  * gold block selected     -> sparse ~ full -> quality ~ 1.0
  * gold block not selected -> sparse ~ noise -> quality ~ 0.0

Quality therefore tracks retrieval recall and climbs monotonically with skill.
"""
import math

from ..core.tensor import Tensor
from ..attention.sparse import full_attention
from ..attention.index import score_blocks_with_skill, select_top_k
from ..attention.blocks import chunk_blocks
from ..utils.stable import stable_vector

SIGNAL_KEY = "quality:signal"


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _orthogonalize(v, signal):
    dot = sum(a * b for a, b in zip(v, signal))
    norm = sum(a * a for a in signal) or 1.0
    return [a - (dot / norm) * b for a, b in zip(v, signal)]


def make_signal_sequence(seq_len: int, dim: int, gold_block: int,
                         block_size: int) -> Tensor:
    signal = stable_vector(SIGNAL_KEY, dim)
    rows = []
    for i in range(seq_len):
        if gold_block * block_size <= i < (gold_block + 1) * block_size:
            rows.append(list(signal))
        else:
            rows.append(_orthogonalize(stable_vector("quality:noise:%d" % i, dim),
                                       signal))
    return Tensor(rows)


def _proj(t: Tensor, unit) -> float:
    """Mean projection of every row onto a unit vector."""
    return sum(sum(row[d] * unit[d] for d in range(len(unit)))
               for row in t.data) / len(t.data)


def run_quality(cfg, seq_len: int = 64, block_size: int = 16,
                top_k: int = 2, n_queries: int = 8):
    dim = cfg.attention.dim
    blocks = chunk_blocks(seq_len, block_size)
    n_blocks = len(blocks)
    signal_unit = _unit(stable_vector(SIGNAL_KEY, dim))
    points = []
    for skill in [0.5, 0.7, 0.9, 1.0]:
        total = 0.0
        for q in range(n_queries):
            scores, gold = score_blocks_with_skill("q%d" % q, n_blocks, skill)
            sel = select_top_k(scores, top_k)
            x = make_signal_sequence(seq_len, dim, gold, block_size)
            selected = [blocks[i] for i in sel]
            sparse = _sparse_out(x, selected)
            dense = full_attention(x, x, x)
            sp = _proj(sparse, signal_unit)
            dp = _proj(dense, signal_unit)
            total += (sp / dp) if abs(dp) > 1e-6 else 0.0
        points.append((skill, total / n_queries))
    return points


def _sparse_out(x: Tensor, selected_blocks):
    from ..attention.sparse import SparseBranch
    return SparseBranch(x.shape[1]).forward(x, x, x, selected_blocks)
