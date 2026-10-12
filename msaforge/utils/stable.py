"""Deterministic pseudo-random helpers keyed by a string.

Uses the first 8 bytes of md5(key) as the seed for random.Random. This is fast,
entropy-rich and reproducible: the same key always yields the same value,
across processes and machines.
"""
import hashlib
import random


def stable_seed(key) -> int:
    h = hashlib.md5(str(key).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _rng(key) -> random.Random:
    return random.Random(stable_seed(key))


def stable_float(key, lo=0.0, hi=1.0) -> float:
    return _rng(key).uniform(lo, hi)


def stable_ints(key, n, lo, hi):
    r = _rng(key)
    return [r.randint(lo, hi) for _ in range(n)]


def stable_vector(key, n, lo=-1.0, hi=1.0):
    r = _rng(key)
    return [r.uniform(lo, hi) for _ in range(n)]


def stable_choice(key, items):
    r = _rng(key)
    return items[r.randrange(len(items))]


def stable_bool(key, p=0.5) -> bool:
    return _rng(key).random() < p
