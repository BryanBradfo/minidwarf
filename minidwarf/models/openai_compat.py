# SPDX-License-Identifier: Apache-2.0
from openai import OpenAI
from .base import Model, GenResult
from ..evalconfig import EvalConfig


class OpenAICompatModel(Model):
    """Model backed by any OpenAI-compatible /v1/chat/completions endpoint
    (local Ollama/vLLM or a hosted OpenAI-compatible API like Gemini's)."""
    def __init__(self, config: EvalConfig):
        self._cfg = config
        # free-tier APIs return bursts of 429s: retry with the SDK's exponential backoff rather than
        # recording an empty kernel, which would be scored as a model failure
        self._client = OpenAI(base_url=config.base_url, api_key=config.resolve_api_key(),
                              max_retries=10, timeout=600)

    def generate(self, prompt: str) -> GenResult:
        try:
            resp = self._client.chat.completions.create(
                model=self._cfg.provider_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._cfg.temperature,
                max_tokens=self._cfg.max_output_tokens,
            )
            text = resp.choices[0].message.content or ""
            return GenResult(text=text, status="success" if text.strip() else "empty")
        except Exception as e:  # network/API failures must not crash a run
            return GenResult(text="", status="error", error=str(e))
