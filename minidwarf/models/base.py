# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenResult:
    """Result of one model generation: the raw text plus a coarse status."""
    text: str
    status: str = "success"
    error: str | None = None


class Model(ABC):
    """A model that maps a prompt string to a GenResult."""
    @abstractmethod
    def generate(self, prompt: str) -> GenResult: ...
