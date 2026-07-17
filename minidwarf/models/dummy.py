# SPDX-License-Identifier: Apache-2.0
from .base import Model, GenResult


class DummyModel(Model):
    """Network-free model that always returns a fixed canned response (for tests)."""
    def __init__(self, canned_text: str):
        self._canned = canned_text

    def generate(self, prompt: str) -> GenResult:
        return GenResult(text=self._canned, status="success")
