"""Block-retrieval recall benchmark across the skill range.

With ``skill`` low the index branch picks blocks essentially at random, so the
gold block lands in the top-k with probability ~k/n_blocks. As skill -> 1 the
gold block scores highest and recall -> 1.0. This is the canonical, monotonic
signal for the trainable mock.
"""
from ..backends.mock import MockBackend


def run_recall(cfg, n_blocks: int = 32, top_k: int = 8, n_queries: int = 20,
               n_kv_heads: int = 1):
    backend = MockBackend(cfg)
    points = []
    for skill in [0.5, 0.7, 0.9, 1.0]:
        backend.skill = skill
        hits = 0
        for q in range(n_queries):
            sel, golds = backend.select_blocks("q%d" % q, n_blocks,
                                               n_kv_heads=n_kv_heads,
                                               top_k=top_k)
            hits += backend.recall(sel, golds)
        points.append((skill, hits / n_queries))
    return points
