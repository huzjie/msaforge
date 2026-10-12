"""Backend registry.

Each backend is a class decorated with ``@register_backend("name")``; the
explicit imports below force the decorators to run so ``list_backends()`` and
``get_backend()`` see every backend.
"""
from .base import BaseBackend, register_backend, get_backend, list_backends
from . import mock, cpu, openai, vllm, transformers  # noqa: F401

__all__ = ["BaseBackend", "register_backend", "get_backend", "list_backends"]
