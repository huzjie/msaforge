"""Train the index branch's block-selection skill.

The mock backend's ``skill`` models how well the index branch picks relevant
blocks. Training nudges ``skill`` upward by a fixed positive delta every step
(the always-positive rule that keeps the signal monotonic), and we record
retrieval recall alongside to show it climbing 0.5 -> 1.0.
"""
from typing import List, Tuple


class Trainer:
    def __init__(self, cfg, backend):
        self.cfg = cfg
        self.backend = backend

    def train(self, n_steps: int = None, queries: List[str] = None,
              n_blocks: int = None, n_kv_heads: int = 1,
              top_k: int = None) -> List[Tuple[float, float]]:
        n_steps = n_steps or self.cfg.train.epochs
        queries = queries or ["q%d" % i for i in range(4)]
        n_blocks = n_blocks or 32
        top_k = top_k or max(1, n_blocks // 4)
        hist = []
        for step in range(n_steps):
            q = queries[step % len(queries)]
            sel, golds = self.backend.select_blocks(q, n_blocks,
                                                    n_kv_heads=n_kv_heads,
                                                    top_k=top_k)
            recall = self.backend.recall(sel, golds)
            hist.append((self.backend.skill, recall))
            self.backend.train_step(q)
        return hist
