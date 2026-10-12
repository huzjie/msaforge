"""Show index-branch real (low-rank) scores vs world-model scores."""
import _bootstrap  # noqa: F401
from msaforge.attention.index import IndexBranch, score_blocks_with_skill
from msaforge.attention.blocks import chunk_blocks, block_mean
from msaforge.core.tensor import Tensor

dim = 64
seq_len = 128
x = Tensor([[0.1 * ((i * 5 + j * 7) % 13) for j in range(dim)]
            for i in range(seq_len)])
blocks = chunk_blocks(seq_len, 32)
breps = block_mean(x, blocks)
query_repr = [sum(x.data[t][d] for t in range(seq_len)) / seq_len for d in range(dim)]

idx = IndexBranch(dim, rank=16)
print("real scores :", [round(s, 3) for s in idx.real_scores(query_repr, breps)])
scores, gold = score_blocks_with_skill("q", len(blocks), 1.0)
print("world scores:", [round(s, 3) for s in scores], "gold=%d" % gold)
