"""OpenAI-compatible backend (delegates to a remote endpoint).

Requests go through the stdlib http.client so no third-party HTTP library is
required. Set ``backend.endpoint`` and ``backend.model_id`` in config.
"""
import json
from urllib import request as urlreq

from .base import BaseBackend, register_backend
from ..attention.index import score_blocks_with_skill, select_top_k


@register_backend("openai")
class OpenAIBackend(BaseBackend):
    def select_blocks(self, query_key, n_blocks, n_kv_heads=1, top_k=None):
        top_k = top_k if top_k is not None else self.cfg.attention.top_k_blocks
        scores, gold = score_blocks_with_skill(query_key, n_blocks, 1.0)
        return [select_top_k(scores, top_k)], [gold]

    def generate(self, prompt, max_tokens=128, **kwargs):
        endpoint = (self.cfg.backend.endpoint or "").rstrip("/")
        if not endpoint:
            return self._fallback(prompt)
        body = json.dumps({
            "model": self.cfg.backend.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }).encode("utf-8")
        req = urlreq.Request(endpoint + "/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
        if self.cfg.backend.api_key:
            req.add_header("Authorization", "Bearer " + self.cfg.backend.api_key)
        with urlreq.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def _fallback(self, prompt):
        return "[openai backend: no endpoint configured] " + str(prompt)[:40]
