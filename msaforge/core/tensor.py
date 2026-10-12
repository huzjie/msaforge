"""A tiny pure-Python tensor (list-of-lists) with the ops msaforge needs.

This is a reference implementation: no numpy, no torch. Only the operations
the sparse-attention core requires are implemented, with explicit dimension
checks so mismatches surface loudly instead of silently misbehaving.
"""
import math
from typing import List, Optional, Sequence, Union

Number = Union[int, float]


class Tensor:
    __slots__ = ("data", "shape")

    def __init__(self, data):
        if isinstance(data, Tensor):
            self.data = data.data
        elif isinstance(data, (list, tuple)):
            if len(data) > 0 and isinstance(data[0], (list, tuple)):
                self.data = [[float(x) for x in row] for row in data]
            else:
                self.data = [float(x) for x in data]
        else:
            raise TypeError("Tensor expects a list")
        if self.data and isinstance(self.data[0], list):
            self.shape = (len(self.data), len(self.data[0]))
        else:
            self.shape = (len(self.data),)

    def __repr__(self):
        return "Tensor(shape=%s)" % (self.shape,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __add__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        if self.shape[0] != len(other):
            raise ValueError("add shape mismatch")
        if len(self.shape) == 2:
            return Tensor([[self.data[i][j] + other[i][j] for j in range(self.shape[1])]
                           for i in range(self.shape[0])])
        return Tensor([self.data[i] + other[i] for i in range(self.shape[0])])

    def __sub__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        if len(self.shape) == 2:
            return Tensor([[self.data[i][j] - other[i][j] for j in range(self.shape[1])]
                           for i in range(self.shape[0])])
        return Tensor([self.data[i] - other[i] for i in range(self.shape[0])])

    def __mul__(self, scalar):
        if len(self.shape) == 2:
            return Tensor([[self.data[i][j] * scalar for j in range(self.shape[1])]
                           for i in range(self.shape[0])])
        return Tensor([x * scalar for x in self.data])

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        if len(self.shape) == 2:
            return Tensor([[self.data[i][j] / scalar for j in range(self.shape[1])]
                           for i in range(self.shape[0])])
        return Tensor([x / scalar for x in self.data])

    def tolist(self):
        return self.data

    def sum(self):
        if len(self.shape) == 2:
            return sum(sum(row) for row in self.data)
        return sum(self.data)


def zeros(rows: int, cols: Optional[int] = None) -> Tensor:
    if cols is None:
        return Tensor([0.0] * rows)
    return Tensor([[0.0] * cols for _ in range(rows)])


def ones(rows: int, cols: Optional[int] = None) -> Tensor:
    if cols is None:
        return Tensor([1.0] * rows)
    return Tensor([[1.0] * cols for _ in range(rows)])


def transpose(m: Tensor) -> Tensor:
    if len(m.shape) != 2:
        raise ValueError("transpose expects a 2D tensor")
    rows, cols = m.shape
    return Tensor([[m.data[i][j] for i in range(rows)] for j in range(cols)])


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication (a @ b)."""
    if len(a.shape) == 1:
        a = Tensor([a.data])
    if len(b.shape) == 1:
        b = Tensor([[x] for x in b.data])
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul shape mismatch: %s @ %s" % (a.shape, b.shape))
    m, k = a.shape
    k2, n = b.shape
    out = [[0.0] * n for _ in range(m)]
    bT = transpose(b).data
    for i in range(m):
        ai = a.data[i]
        for j in range(n):
            bj = bT[j]
            s = 0.0
            for t in range(k):
                s += ai[t] * bj[t]
            out[i][j] = s
    return Tensor(out)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax along the last axis (a 2D tensor of scores)."""
    if len(x.shape) == 1:
        x = Tensor([x.data])
    rows, cols = x.shape
    out = []
    for i in range(rows):
        row = x.data[i]
        mx = max(row)
        exps = [math.exp(v - mx) for v in row]
        s = sum(exps)
        out.append([e / s for e in exps])
    return Tensor(out)


def layernorm(x: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm along the last axis, handling both 1D and 2D (token seq)."""
    if len(x.shape) == 1:
        mean = sum(x.data) / len(x.data)
        var = sum((v - mean) ** 2 for v in x.data) / len(x.data)
        inv = 1.0 / math.sqrt(var + eps)
        return Tensor([(v - mean) * inv for v in x.data])
    rows, cols = x.shape
    out = []
    for i in range(rows):
        row = x.data[i]
        mean = sum(row) / cols
        var = sum((v - mean) ** 2 for v in row) / cols
        inv = 1.0 / math.sqrt(var + eps)
        out.append([(v - mean) * inv for v in row])
    return Tensor(out)
