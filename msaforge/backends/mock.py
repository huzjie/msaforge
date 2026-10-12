"""Deterministic trainable mock backend.

``skill`` models how well the index branch picks the right blocks. It *really*
participates in scoring: block score = ``skill * [b == gold] + noise``, so
training ``skill`` 0.5 -> 1.0 makes retrieval recall / quality / speedup all
climb monotonically. Generation is deterministic (md5-keyed), so tests are
reproducible.
"""
from .base import BaseBackend, register_backend
from ..attention.index import score_blocks_with_skill, select_top_k
from ..utils.stable import stable_float, stable_ints, stable_seed


@register_backend("mock")
class MockBackend(BaseBackend):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.skill = float(getattr(cfg.backend, "skill", 0.5))

    def select_blocks(self, query_key, n_blocks, n_kv_heads=1, top_k=None):
        top_k = top_k if top_k is not None else self.cfg.attention.top_k_blocks
        selections, golds = [], []
        for g in range(n_kv_heads):
            key = "%s|kv%d" % (query_key, g)
            scores, gold = score_blocks_with_skill(key, n_blocks, self.skill)
            selections.append(select_top_k(scores, top_k))
            golds.append(gold)
        return selections, golds

    def recall(self, selections, golds) -> float:
        hits = sum(1 for sel, gold in zip(selections, golds) if gold in sel)
        return hits / len(golds) if golds else 0.0

    def generate(self, prompt, max_tokens=128, **kwargs):
        seed = "gen:" + str(prompt)
        n = max(1, min(max_tokens, 96))
        words = ["context", "sparse", "attention", "block", "retrieval",
                 "index", "branch", "gqa", "multimodal", "token", "cache",
                 "long", "million", "native", "vision", "prefill", "decode",
                 "latency", "speedup", "query"]
        idx = stable_ints(seed, n, 0, len(words) - 1)
        toks = [words[i] for i in idx]
        # deterministically bias toward the prompt topic
        head = " ".join(toks[:3])
        return head + " " + " ".join(toks[3:]) + "."

    def train_step(self, query_key="train"):
        delta = float(self.cfg.train.delta_up)
        self.skill = min(1.0, self.skill + delta)
        return self.skill
