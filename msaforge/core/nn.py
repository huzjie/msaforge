"""Tiny neural-network modules built on the core Tensor."""
from .tensor import Tensor, matmul, layernorm, transpose
from ..utils.stable import stable_vector, stable_seed


class Linear:
    """A dense layer with deterministic weights derived from a seed key."""

    def __init__(self, in_dim: int, out_dim: int, seed: str = "w"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Tensor(
            [stable_vector("%s:%d:%d" % (seed, i, out_dim), in_dim)
             for i in range(out_dim)]
        )

    def forward(self, x: Tensor) -> Tensor:
        return matmul(x, transpose(self.weight))


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return layernorm(x, self.eps)


class Embedding:
    """A token/vocab embedding table with deterministic weights."""

    def __init__(self, vocab_size: int, dim: int, seed: str = "emb"):
        self.vocab_size = vocab_size
        self.dim = dim
        self.table = Tensor(
            [stable_vector("%s:%d" % (seed, i), dim) for i in range(vocab_size)]
        )

    def forward(self, ids) -> Tensor:
        if isinstance(ids, int):
            ids = [ids]
        return Tensor([self.table[i] for i in ids])
