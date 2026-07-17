# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import dataclass
from pathlib import Path
import yaml

_REQUIRED = ["model_name", "base_url", "provider_model_name"]

@dataclass(frozen=True)
class EvalConfig:
    model_name: str
    base_url: str
    provider_model_name: str
    provider: str = "openai_compat"
    api_key_env_var: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 4096
    n_samples: int = 1

    def resolve_api_key(self) -> str:
        """Return the API key, or the dummy "EMPTY" for keyless local servers."""
        if self.api_key_env_var:
            return os.getenv(self.api_key_env_var) or "EMPTY"
        return "EMPTY"

def load_eval_config(path: Path) -> EvalConfig:
    """Load an eval run config from YAML; raises ValueError on missing required fields."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ValueError(f"{path} missing required fields: {missing}")
    known = {"model_name", "base_url", "provider_model_name", "provider",
             "api_key_env_var", "temperature", "max_output_tokens", "n_samples"}
    return EvalConfig(**{k: v for k, v in data.items() if k in known})
