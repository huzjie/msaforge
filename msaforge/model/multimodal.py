"""Native multimodal model: text + vision fused from the first token.

Instead of a text model with a vision adapter bolted on afterwards, the two
modalities are projected into the SAME token space and interleaved into one
sequence fed to the same sparse transformer -- the "from token zero" strategy
MiniMax reports as strictly more stable than late vision fusion.
"""
from ..core.tensor import Tensor, matmul, layernorm
from ..core.nn import Embedding, Linear
from ..attention.msa import MSALayer
from .vision import ViT


class NativeMultimodalModel:
    def __init__(self, dim: int = 256, n_layers: int = 6, vocab_size: int = 4096,
                 patch: int = 16, n_frames: int = 8,
                 block_size: int = 64, top_k_blocks: int = 32):
        self.dim = dim
        self.text_embed = Embedding(vocab_size, dim, seed="txt")
        self.vision = ViT(patch=patch, dim=dim, n_frames=n_frames)
        # project vision tokens into the shared text-token space
        self.vis_proj = Linear(dim, dim, seed="visproj")
        self.layers = [MSALayer(dim, block_size=block_size,
                                top_k_blocks=top_k_blocks,
                                seed="mm:%d" % i) for i in range(n_layers)]

    def fuse(self, text_ids, frames):
        """Interleave text tokens and vision tokens into one sequence."""
        txt = self.text_embed.forward(text_ids)          # (T, dim)
        vis = self.vision.encode(frames)                  # (V, dim)
        vis = self.vis_proj.forward(vis)
        seq = txt.data + vis.data                         # concatenated rows
        return Tensor(seq)

    def forward(self, text_ids, frames, skill: float = 1.0, query_key: str = None):
        x = self.fuse(text_ids, frames)
        for layer in self.layers:
            x = layer.forward(x, query_key=query_key, skill=skill)
            x = layernorm(x)
        return x
