import os
from pathlib import Path
import pytest
from minidwarf.evalconfig import load_eval_config, EvalConfig

FIX = Path(__file__).parent / "fixtures/eval"

def test_loads_defaults():
    c = load_eval_config(FIX / "local.yaml")
    assert c.provider == "openai_compat" and c.temperature == 0.2
    assert c.n_samples == 1 and c.max_output_tokens == 4096

def test_keyless_uses_dummy_key():
    c = load_eval_config(FIX / "local.yaml")
    assert c.resolve_api_key() == "EMPTY"

def test_resolves_env_key(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-123")
    c = EvalConfig(model_name="m", base_url="u", provider_model_name="p",
                   api_key_env_var="MY_KEY")
    assert c.resolve_api_key() == "sk-123"

def test_missing_required_raises(tmp_path):
    (tmp_path / "bad.yaml").write_text("model_name: x\n")
    with pytest.raises(ValueError):
        load_eval_config(tmp_path / "bad.yaml")
