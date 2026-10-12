"""Hugging Face transformers backend stub (optional heavy dependency).

Only used when ``transformers`` is importable; otherwise it delegates to a
pure-Python fallback so the backend stays loadable in dependency-free envs.
"""
from .base import BaseBackend, register_backend
from ..attention.index import score_blocks_with_skill, select_top_k


@register_backend("transformers")
class TransformersBackend(BaseBackend):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._pipe = None
        try:
            from transformers import pipeline  # type: ignore
            self._pipe = pipeline("text-generation", model=self.cfg.backend.model_id)
        except Exception:
            self._pipe = None

    def select_blocks(self, query_key, n_blocks, n_kv_heads=1, top_k=None):
        top_k = top_k if top_k is not None else self.cfg.attention.top_k_blocks
        scores, gold = score_blocks_with_skill(query_key, n_blocks, 1.0)
        return [select_top_k(scores, top_k)], [gold]

    def generate(self, prompt, max_tokens=128, **kwargs):
        if self._pipe is None:
            return "[transformers backend: library not installed] " + str(prompt)[:40]
        out = self._pipe(prompt, max_new_tokens=max_tokens, do_sample=False)
        return out[0]["generated_text"]
