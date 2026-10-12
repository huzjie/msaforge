"""Compare sparse vs full attention on a small sequence."""
import _bootstrap  # noqa: F401
from msaforge.attention.msa import MSALayer
from msaforge.attention.sparse import full_attention, cosine
from msaforge.core.tensor import Tensor

dim = 64
seq_len = 128
x = Tensor([[0.1 * ((i * 7 + j * 13) % 11) for j in range(dim)]
            for i in range(seq_len)])
layer = MSALayer(dim, block_size=32, top_k_blocks=4)
sparse, meta = layer.forward(x, query_key="demo", skill=1.0, return_meta=True)
dense = full_attention(x, x, x)
print("n_blocks=%d selected=%d sparse_ratio=%.1fx"
      % (meta["n_blocks"], len(meta["selected"]), meta["sparse_ratio"]))
print("cosine(sparse, full)=%.4f" % cosine(sparse, dense))
