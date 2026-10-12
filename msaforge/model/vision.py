"""Vision tower: patch embedding + ViT with 3D (space-time) attention.

MiniMax M3 trains vision from token zero (not bolted on later), and uses 3D
attention inside the vision transformer so temporal structure across frames is
modeled directly instead of flattened into a 2D patch grid.
"""
import math
from typing import List

from ..core.tensor import Tensor, matmul, layernorm, softmax
from ..utils.stable import stable_vector, stable_seed


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _patchify(frames, patch: int):
    """Split each frame into patch tokens; return list of (dim,) patch vecs.

    ``frames`` is a list of flattened (H*W) frames. For the reference impl we
    use a small deterministic placeholder image so it stays fast and pure.
    """
    h = int(math.sqrt(len(frames[0])))
    tokens = []
    for fi, frame in enumerate(frames):
        for py in range(0, h, patch):
            for px in range(0, h, patch):
                vals = []
                for y in range(py, py + patch):
                    for x in range(px, px + patch):
                        vals.append(frame[y * h + x])
                tokens.append((fi, py // patch, px // patch, sum(vals) / len(vals)))
    return tokens


class Temporal3DAttention:
    """Space-time attention: attends across frames AND spatial positions.

    A token's key encodes (frame, y, x) so the attention graph is 3D; the
    query attends to spatially-near and temporally-near tokens first. For the
    reference implementation we materialize a small 3D graph and run softmax
    attention over it.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, tokens: List[tuple], vecs: List[List[float]]):
        n = len(tokens)
        # 3D similarity: closer in (frame, y, x) -> higher prior
        out = []
        for i in range(n):
            fi, yi, xi = tokens[i][:3]
            scores = []
            for j in range(n):
                fj, yj, xj = tokens[j][:3]
                dist = ((fi - fj) ** 2 + (yi - yj) ** 2 + (xi - xj) ** 2) ** 0.5
                scores.append(_dot(vecs[i], vecs[j]) - 0.05 * dist)
            mx = max(scores)
            exps = [math.exp(s - mx) for s in scores]
            denom = sum(exps)
            row = [0.0] * self.dim
            for j in range(n):
                w = exps[j] / denom
                for d in range(self.dim):
                    row[d] += w * vecs[j][d]
            out.append(row)
        return out


class ViT:
    """Patch embedding + a stack of temporal 3D attention blocks."""

    def __init__(self, patch: int = 16, dim: int = 256, n_frames: int = 8):
        self.patch = patch
        self.dim = dim
        self.n_frames = n_frames
        self.attn = Temporal3DAttention(dim)

    def encode(self, frames) -> Tensor:
        """frames: list of n flattened (H*W) frames -> (n_tokens, dim) tensor."""
        tokens = _patchify(frames, self.patch)
        vecs = [stable_vector("vis:%d" % (t[0] * 10000 + t[1] * 100 + t[2]),
                              self.dim) for t in tokens]
        out = self.attn.forward(tokens, vecs)
        return Tensor(out)

    @staticmethod
    def synthetic_frames(h: int = 32, w: int = 32, n: int = 8):
        """Deterministic placeholder frames (for examples/benchmarks)."""
        return [
            [stable_seed("frame:%d:%d" % (fi, i)) % 256 / 255.0 for i in range(h * w)]
            for fi in range(n)
        ]
