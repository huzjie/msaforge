"""msaforge: MiniMax Sparse Attention (MSA) reference implementation.

A zero-dependency, fully runnable reference implementation of the sparse
attention architecture behind MiniMax M3 -- the open-weight frontier model that
combines a 1-million-token context window with native text+vision multimodality
by replacing full quadratic attention with a two-branch sparse mechanism.

MiniMax M3's MSA (MiniMax Sparse Attention) speeds up prefill ~9x and decoding
~15x versus full attention at long context, at negligible quality loss. The
design has two branches:

    index branch  -- a lightweight scorer that decides, at BLOCK granularity,
                     which KV blocks matter for the current query
    sparse branch -- real scaled-dot-product attention computed ONLY over the
                     selected top-k blocks (everything else is masked)

This repo distills the ideas into reusable, runnable code:

1. Block-level retrieval -- chunk the KV cache into fixed-size blocks and rank
   whole blocks instead of individual tokens (simpler to parallelize, friendlier
   to GPU memory access, and a negligible quality sacrifice).
2. GQA-compatible selection -- each KV-head group runs its OWN block selection,
   so grouped-query attention keeps its diversity instead of every group
   staring at the same tokens (the exact failure mode when DeepSeek's DSA is
   naively ported to GQA).
3. Native multimodality -- text and vision are fused from the very first token,
   including 3D attention inside the vision transformer for temporal structure.

It ships with a deterministic trainable mock backend (block-selection skill
0.5 -> 1.0, retrieval recall + quality + speedup all monotonically climb),
five backends, an OpenAI-compatible stdlib HTTP server, a speed/recall/quality
benchmark suite, and CLI / Docker / K8s / Helm / CI scaffolding.

Highlights
----------
- Zero-dependency pure-Python core (index branch + sparse branch + GQA + ViT)
- Deterministic trainable mock backend (skill 0.5 -> 1.0)
- Sparse-attention speedup benchmark (prefill ~9x / decode ~15x)
- Block-retrieval recall benchmark (k/n_blocks -> 1.0)
- Quality-preservation benchmark (sparse vs full-attention cosine -> 1.0)
- OpenAI-compatible stdlib HTTP server
- CLI / Docker / K8s / Helm / CI + prompts + recipes + docs
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
