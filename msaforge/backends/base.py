"""Backend registry and the common interface."""
from typing import Dict, List, Optional, Type

_BACKENDS: Dict[str, Type["BaseBackend"]] = {}


def register_backend(name: str):
    def deco(cls):
        _BACKENDS[name] = cls
        return cls
    return deco


def list_backends() -> List[str]:
    return sorted(_BACKENDS.keys())


def get_backend(name: str, cfg):
    if name not in _BACKENDS:
        raise KeyError("unknown backend '%s' (available: %s)"
                       % (name, ", ".join(sorted(_BACKENDS)) or "none"))
    return _BACKENDS[name](cfg)


class BaseBackend:
    """Common sparse-attention backend interface.

    A backend answers two questions: (1) which blocks does the index branch
    select for a query, and (2) what text does the model generate. The mock
    backend models block-selection *skill*; real backends delegate generation
    to a remote/CPU model and report selection via the reference index branch.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def select_blocks(self, query_key: str, n_blocks: int, n_kv_heads: int = 1,
                      top_k: Optional[int] = None):
        """Return (selections, golds) per KV-head group."""
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 128, **kwargs) -> str:
        raise NotImplementedError

    def train_step(self, query_key: str = "train") -> float:
        """One training step; returns the (updated) skill."""
        return 1.0
