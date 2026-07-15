# SPDX-License-Identifier: Apache-2.0
from minidwarf.models.base import Model, GenResult
from minidwarf.models.dummy import DummyModel
from minidwarf.models.registry import create_model
from minidwarf.evalconfig import EvalConfig


def test_dummy_returns_canned():
    m = DummyModel("KERNEL_SRC")
    r = m.generate("any prompt")
    assert isinstance(r, GenResult) and r.text == "KERNEL_SRC" and r.status == "success"


def test_registry_builds_dummy():
    cfg = EvalConfig(model_name="d", base_url="u", provider_model_name="p", provider="dummy")
    m = create_model(cfg, canned_text="X")
    assert isinstance(m, DummyModel) and m.generate("p").text == "X"


def test_registry_unknown_provider_raises():
    cfg = EvalConfig(model_name="d", base_url="u", provider_model_name="p", provider="nope")
    import pytest
    with pytest.raises(ValueError):
        create_model(cfg)
