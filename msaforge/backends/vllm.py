"""vLLM backend stub: local OpenAI-compatible server on the vLLM port."""
from .openai import OpenAIBackend
from .base import register_backend


@register_backend("vllm")
class VllmBackend(OpenAIBackend):
    def generate(self, prompt, max_tokens=128, **kwargs):
        if not self.cfg.backend.endpoint:
            return "[vllm backend: no endpoint configured] " + str(prompt)[:40]
        return super().generate(prompt, max_tokens, **kwargs)
