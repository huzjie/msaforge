"""Run all benchmarks and print a summary table."""
from ..backends.mock import MockBackend
from .speed import run_speed
from .recall import run_recall
from .quality import run_quality


def _print_table(title, rows, col1, col2):
    print("\n== %s ==" % title)
    for a, b in rows:
        print("  %-14s %s" % (a, b))


def run_all(cfg):
    print("\n=== msaforge benchmark suite ===")

    # speed (1M context extrapolation)
    sp = run_speed()
    print("\n== Speed @ 1M context (FLOPs model) ==")
    print("  prefill speedup : %.2fx" % sp["prefill_speedup"])
    print("  decode speedup  : %.2fx" % sp["decode_speedup"])
    print("  n_blocks        : %d (block_size=%d)" % (sp["n_blocks"], sp["block_size"]))
    print("  top_k prefill   : %d  -> sparse ratio %.2fx"
          % (sp["top_k_prefill"], sp["sparse_ratio_prefill"]))
    print("  top_k decode    : %d  -> sparse ratio %.2fx"
          % (sp["top_k_decode"], sp["sparse_ratio_decode"]))

    # recall
    rec = run_recall(cfg)
    print("\n== Block-retrieval recall (skill -> recall) ==")
    for skill, r in rec:
        print("  skill %.2f -> recall %.3f" % (skill, r))

    # quality
    ql = run_quality(cfg)
    print("\n== Quality preservation (signal retention ratio) ==")
    for skill, c in ql:
        print("  skill %.2f -> retention %.3f" % (skill, c))

    return {"speed": sp, "recall": rec, "quality": ql}


def summarize(cfg) -> dict:
    """Return the benchmark results as a dict (for programmatic use)."""
    return run_all(cfg)
