"""The SPARSE branch of MSA: real scaled-dot-product attention computed
ONLY over the selected top-k blocks (everything else is masked to -inf).

This is what actually saves FLOPs. With ``selected`` covering only ``top_k``
of ``n_blocks`` blocks, attention cost drops from O(seq^2 * dim) to
O(seq * selected_tokens * dim), i.e. roughly a factor of ``n_blocks / top_k``.
"""
import math
from typing import List, Tuple

from ..core.tensor import Tensor, softmax


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


class SparseBranch:
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                selected_blocks: List[Tuple[int, int]],
                scale: float = None) -> Tensor:
        """q/k/v: (seq_len, dim). selected_blocks: inclusive/exclusive ranges.

        Returns an output tensor of shape (seq_len, dim).
        """
        seq_len, dim = q.shape
        scale = scale if scale is not None else (1.0 / math.sqrt(dim))

        # gather selected token indices (deduplicated, ordered)
        idxs = []
        for (s, e) in selected_blocks:
            idxs.extend(range(s, e))
        idxs = sorted(set(idxs))

        out = []
        for i in range(seq_len):
            qi = q.data[i]
            scores = []
            for t in idxs:
                scores.append(_dot(qi, k.data[t]) * scale)
            # numerically stable softmax inline
            mx = max(scores) if scores else 0.0
            exps = [math.exp(s - mx) for s in scores]
            denom = sum(exps)
            weights = [e / denom for e in exps]
            row = [0.0] * dim
            for w, t in zip(weights, idxs):
                vt = v.data[t]
                for d in range(dim):
                    row[d] += w * vt[d]
            out.append(row)
        return Tensor(out)


def full_attention(q: Tensor, k: Tensor, v: Tensor, scale: float = None) -> Tensor:
    """Dense O(seq^2) attention, used as the quality-preservation baseline."""
    seq_len, dim = q.shape
    scale = scale if scale is not None else (1.0 / math.sqrt(dim))
    out = []
    for i in range(seq_len):
        qi = q.data[i]
        scores = [_dot(qi, k.data[t]) * scale for t in range(seq_len)]
        mx = max(scores)
        exps = [math.exp(s - mx) for s in scores]
        denom = sum(exps)
        weights = [e / denom for e in exps]
        row = [0.0] * dim
        for w, t in zip(weights, range(seq_len)):
            vt = v.data[t]
            for d in range(dim):
                row[d] += w * vt[d]
        out.append(row)
    return Tensor(out)


def cosine(a: Tensor, b: Tensor) -> float:
    """Mean cosine similarity between two (seq_len, dim) tensors."""
    seq_len, dim = a.shape
    total = 0.0
    for i in range(seq_len):
        ai, bi = a.data[i], b.data[i]
        na = math.sqrt(sum(x * x for x in ai)) or 1.0
        nb = math.sqrt(sum(x * x for x in bi)) or 1.0
        dot = sum(x * y for x, y in zip(ai, bi))
        total += dot / (na * nb)
    return total / seq_len
