"""Benchmark suite: speed, recall, quality."""
from .speed import flops_speedup, theoretical_speedup, run_speed
from .recall import run_recall
from .quality import make_signal_sequence, run_quality
from .summary import summarize, run_all

__all__ = [
    "flops_speedup", "theoretical_speedup", "run_speed", "run_recall",
    "make_signal_sequence", "run_quality", "summarize", "run_all",
]
