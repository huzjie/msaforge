"""Run the full benchmark suite."""
import _bootstrap  # noqa: F401
from msaforge.config import load_config
from msaforge.bench.summary import run_all

run_all(load_config())
