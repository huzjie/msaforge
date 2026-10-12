"""CPU reference backend: runs the pure-Python MSA layer directly."""
from .base import BaseBackend, register_backend
from ..attention.index import score_blocks_with_skill, select_top_k
from ..model.transformer import SparseTransformer
from ..utils.stable import stable_ints


@register_backend("cpu")
class CpuBackend(BaseBackend):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = SparseTransformer(
            dim=cfg.attention.dim, n_layers=cfg.model.n_layers,
            vocab_size=cfg.model.vocab_size,
            block_size=cfg.attention.block_size,
            top_k_blocks=cfg.attention.top_k_blocks,
        )

    def select_blocks(self, query_key, n_blocks, n_kv_heads=1, top_k=None):
        top_k = top_k if top_k is not None else self.cfg.attention.top_k_blocks
        scores, gold = score_blocks_with_skill(query_key, n_blocks, 1.0)
        return [select_top_k(scores, top_k)], [gold]

    def generate(self, prompt, max_tokens=128, **kwargs):
        # deterministic reference output (tiny reference forward)
        n = max(1, min(max_tokens, 24))
        ids = stable_ints("cpu:" + str(prompt), n, 0, self.cfg.model.vocab_size - 1)
        logits = self.model.logits(ids)
        top = [max(range(logits.shape[1]), key=lambda j: logits.data[i][j])
               for i in range(logits.shape[0])]
        return "cpu-reference: " + " ".join(str(t) for t in top)
