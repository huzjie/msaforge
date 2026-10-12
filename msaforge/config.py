"""Configuration for msaforge.

Reads YAML config files. Falls back to a tiny built-in YAML-subset parser when
PyYAML is not installed (the managed Python runtime is dependency-free), and
to dataclass defaults when no file is present.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore

    _HAS_PYYAML = True
except Exception:  # pragma: no cover
    _HAS_PYYAML = False


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_yaml(text: str) -> Dict[str, Any]:
    if _HAS_PYYAML:
        return yaml.safe_load(text) or {}
    from .utils.yamlish import parse

    return parse(text)


@dataclass
class AttentionConfig:
    dim: int = 256
    n_heads: int = 8
    head_dim: int = 32
    n_kv_heads: int = 2          # GQA: query heads share these KV heads
    block_size: int = 64         # tokens per KV block
    top_k_blocks: int = 32       # how many blocks the index branch keeps
    softmax_scale: Optional[float] = None


@dataclass
class ModelConfig:
    n_layers: int = 6
    vocab_size: int = 4096
    max_seq_len: int = 8192
    vision_patch: int = 16
    vision_dim: int = 256
    n_frames: int = 8


@dataclass
class BackendConfig:
    name: str = "mock"
    endpoint: str = ""
    model_id: str = "MiniMax-M3"
    api_key: str = ""
    skill: float = 0.5           # mock block-selection skill (trainable)
    temperature: float = 1.0


@dataclass
class TrainConfig:
    lr: float = 0.1
    epochs: int = 20
    delta_up: float = 0.05       # always-positive skill increment per step


@dataclass
class Config:
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        d = d or {}
        return cls(
            attention=AttentionConfig(**{k: v for k, v in d.get("attention", {}).items()
                                          if k in AttentionConfig.__dataclass_fields__}),
            model=ModelConfig(**{k: v for k, v in d.get("model", {}).items()
                                  if k in ModelConfig.__dataclass_fields__}),
            backend=BackendConfig(**{k: v for k, v in d.get("backend", {}).items()
                                      if k in BackendConfig.__dataclass_fields__}),
            train=TrainConfig(**{k: v for k, v in d.get("train", {}).items()
                                  if k in TrainConfig.__dataclass_fields__}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Optional[str] = None) -> Config:
    """Load config from a YAML file, falling back to defaults."""
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        here = Path(__file__).parent
        candidates.append(here / "configs" / "default.yaml")
        candidates.append(Path.cwd() / "configs" / "default.yaml")
    for p in candidates:
        if p.exists():
            data = _parse_yaml(p.read_text(encoding="utf-8"))
            return Config.from_dict(data)
    return Config()
