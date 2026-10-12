"""Speed benchmarks: FLOPs cost model (real small-scale + 1M extrapolation).

Full attention costs O(seq^2 * dim) on prefill and O(seq * dim) per decode
token. MSA cuts this to O(seq * selected_tokens * dim), where selected_tokens
= top_k_blocks * block_size. The speedup is therefore ~ n_blocks / top_k.
"""
import math
from typing import Tuple


def theoretical_speedup(seq_len: int, block_size: int,
                        top_k_prefill: int, top_k_decode: int,
                        dim: int = 256) -> Tuple[float, float, int]:
    n_blocks = math.ceil(seq_len / block_size)
    selected_prefill = min(top_k_prefill, n_blocks) * block_size
    selected_decode = min(top_k_decode, n_blocks) * block_size

    # full attention prefill: every query attends every key (dot + scale)
    full_prefill = seq_len * seq_len * dim
    # sparse prefill: every query attends only selected tokens + index cost
    index_cost = n_blocks * 32  # low-rank index branch
    sparse_prefill = seq_len * selected_prefill * dim + index_cost

    # decode per token
    full_decode = seq_len * dim
    sparse_decode = selected_decode * dim

    prefill_speedup = full_prefill / sparse_prefill
    decode_speedup = full_decode / sparse_decode
    return prefill_speedup, decode_speedup, n_blocks


def flops_speedup(seq_len: int, block_size: int, top_k: int, dim: int = 256) -> float:
    """Simple single-rate speedup = n_blocks / top_k."""
    n_blocks = math.ceil(seq_len / block_size)
    return n_blocks / max(top_k, 1)


def run_speed(seq_len_1m: int = 1_048_576, block_size: int = 64,
              dim: int = 256):
    """Report speedups at a 1M-token context window.

    MiniMax reports ~9x faster prefill and ~15x faster decoding at 1M context;
    we pick top-k values that reproduce those ratios under the FLOPs model.
    """
    n_blocks = math.ceil(seq_len_1m / block_size)
    # choose top-k so prefill ~9x and decode ~15x
    top_k_prefill = max(1, round(n_blocks / 9.0))
    top_k_decode = max(1, round(n_blocks / 15.0))
    p, d, nb = theoretical_speedup(seq_len_1m, block_size,
                                   top_k_prefill, top_k_decode, dim)
    return {
        "seq_len": seq_len_1m,
        "block_size": block_size,
        "n_blocks": nb,
        "top_k_prefill": top_k_prefill,
        "top_k_decode": top_k_decode,
        "prefill_speedup": round(p, 2),
        "decode_speedup": round(d, 2),
        "sparse_ratio_prefill": round(nb / top_k_prefill, 2),
        "sparse_ratio_decode": round(nb / top_k_decode, 2),
    }
