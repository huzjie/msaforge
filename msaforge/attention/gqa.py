"""Grouped-Query Attention compatibility for sparse block selection.

The whole reason MSA picks blocks per KV-head GROUP is that GQA exists to let
different head groups attend to different parts of the input. If a single
token/block selection were reused across all KV heads, every group would stare
at the same selected tokens and GQA's diversity would be quietly defeated --
exactly what happens when DeepSeek's DSA (built for MQA) is naively ported to
GQA. Here each group runs its own top-k selection.
"""
from typing import List

from .index import score_blocks_with_skill, select_top_k


def group_heads(n_heads: int, n_kv_heads: int) -> List[List[int]]:
    """Map each KV head to the list of query heads it serves."""
    assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
    per = n_heads // n_kv_heads
    return [[g * per + i for i in range(per)] for g in range(n_kv_heads)]


def select_per_kv_group(query_key, n_blocks: int, n_kv_heads: int,
                        top_k: int, skill: float):
    """Return (selections, golds) where selections[g] = top-k blocks for group g.

    Each group derives its own gold block from a per-group query key, so the
    groups attend to (mostly) different blocks -- preserving GQA diversity.
    """
    selections = []
    golds = []
    for g in range(n_kv_heads):
        key = "%s|kv%d" % (query_key, g)
        scores, gold = score_blocks_with_skill(key, n_blocks, skill)
        selections.append(select_top_k(scores, top_k))
        golds.append(gold)
    return selections, golds
