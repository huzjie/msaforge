"""Show index-branch block selection for a query."""
import _bootstrap  # noqa: F401
from msaforge.config import load_config
from msaforge.backends import get_backend

cfg = load_config()
backend = get_backend(cfg.backend.name, cfg)
for skill in (0.5, 0.75, 1.0):
    backend.skill = skill
    sel, golds = backend.select_blocks("demo-query", n_blocks=32, top_k=8)
    print("skill=%.2f  selected=%s  gold=%s  recall=%.2f"
          % (skill, sel, golds, backend.recall(sel, golds)))
