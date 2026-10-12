"""Show per-KV-group block selection diversity (GQA compatibility)."""
import _bootstrap  # noqa: F401
from msaforge.attention.gqa import select_per_kv_group

for skill in (0.5, 0.75, 1.0):
    sel, golds = select_per_kv_group("q", n_blocks=32, n_kv_heads=4,
                                     top_k=4, skill=skill)
    uniq = len(set(map(tuple, sel)))
    print("skill=%.2f  groups=%d  distinct_selections=%d  (4=full diversity)"
          % (skill, len(sel), uniq))
