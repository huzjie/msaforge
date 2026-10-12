"""Block-level chunking of a token sequence.

MSA operates at BLOCK granularity rather than token granularity: the KV cache
is split into fixed-size blocks and the index branch ranks whole blocks. This
is the key design change that makes sparse attention both easier to parallelize
and friendlier to GPU memory access (one contiguous read per block instead of
scattered single-token reads).
"""
from typing import List, Tuple


def chunk_blocks(seq_len: int, block_size: int) -> List[Tuple[int, int]]:
    """Return inclusive/exclusive block boundaries [(s0,e0), (s1,e1), ...]."""
    blocks = []
    for s in range(0, seq_len, block_size):
        e = min(s + block_size, seq_len)
        blocks.append((s, e))
    return blocks


def block_mean(x, blocks: List[Tuple[int, int]]):
    """Mean-pool each block of a (seq_len, dim) tensor -> list of (dim,) reprs."""
    seq_len, dim = x.shape
    out = []
    for (s, e) in blocks:
        n = e - s
        col = [0.0] * dim
        for t in range(s, e):
            row = x.data[t]
            for d in range(dim):
                col[d] += row[d]
        out.append([v / n for v in col])
    return out
