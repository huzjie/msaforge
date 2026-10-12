"""LangChain-compatible wrapper.

Works without LangChain installed (the class is self-contained); if LangChain
is present it can be used as a drop-in LLM. Delegates generation to a backend
and exposes sparse block selection as a tool.
"""
from ..backends import get_backend


class MSAChain:
    """A minimal LLM-compatible wrapper around an msaforge backend."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = get_backend(cfg.backend.name, cfg)

    def invoke(self, prompt, max_tokens=128):
        return self.backend.generate(prompt, max_tokens=max_tokens)

    __call__ = invoke

    def select(self, query, n_blocks=32, top_k=None):
        top_k = top_k or self.cfg.attention.top_k_blocks
        return self.backend.select_blocks(query, n_blocks, top_k=top_k)
