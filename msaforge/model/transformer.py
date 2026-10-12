"""A transformer whose attention layers are all MSA sparse layers."""
from ..core.tensor import Tensor, matmul, layernorm
from ..core.nn import Embedding, Linear
from ..attention.msa import MSALayer


class SparseTransformer:
    """Stack of MSA layers over token embeddings (reference forward only).

    The block-selection ``skill`` is injected so a single model class serves
    both the reference path (skill=1.0) and the trainable mock world model.
    """

    def __init__(self, dim: int = 256, n_layers: int = 6, vocab_size: int = 4096,
                 block_size: int = 64, top_k_blocks: int = 32):
        self.dim = dim
        self.n_layers = n_layers
        self.embed = Embedding(vocab_size, dim, seed="tok")
        self.layers = [MSALayer(dim, block_size=block_size,
                                top_k_blocks=top_k_blocks,
                                seed="msa:%d" % i) for i in range(n_layers)]
        self.head = Linear(dim, vocab_size, seed="lmhead")

    def forward(self, ids, skill: float = 1.0, query_key: str = None):
        x = self.embed.forward(ids)
        for layer in self.layers:
            x = layer.forward(x, query_key=query_key, skill=skill)
            x = layernorm(x)
        return x

    def logits(self, ids, skill: float = 1.0, query_key: str = None) -> Tensor:
        h = self.forward(ids, skill=skill, query_key=query_key)
        return self.head.forward(h)
