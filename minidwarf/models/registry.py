# SPDX-License-Identifier: Apache-2.0
from .base import Model
from .dummy import DummyModel
from ..evalconfig import EvalConfig


def create_model(config: EvalConfig, *, canned_text: str | None = None) -> Model:
    """Build a Model from a config's provider field."""
    if config.provider == "dummy":
        return DummyModel(canned_text or "")
    if config.provider == "openai_compat":
        from .openai_compat import OpenAICompatModel
        return OpenAICompatModel(config)
    raise ValueError(f"unknown provider: {config.provider!r}")
