# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch
from minidwarf.evalconfig import EvalConfig
from minidwarf.models.openai_compat import OpenAICompatModel

CFG = EvalConfig(model_name="m", base_url="http://localhost:11434/v1",
                 provider_model_name="qwen2.5-coder:7b")  # keyless

def _fake_completion(text):
    msg = MagicMock(); msg.content = text
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp

def test_generate_success():
    with patch("minidwarf.models.openai_compat.OpenAI") as OC:
        client = OC.return_value
        client.chat.completions.create.return_value = _fake_completion("```cuda\nX\n```")
        m = OpenAICompatModel(CFG)
        r = m.generate("prompt")
        assert r.status == "success" and "X" in r.text
        # keyless -> dummy key injected
        _, kw = OC.call_args
        assert kw["api_key"] == "EMPTY" and kw["base_url"] == CFG.base_url

def test_generate_error_is_captured():
    with patch("minidwarf.models.openai_compat.OpenAI") as OC:
        OC.return_value.chat.completions.create.side_effect = RuntimeError("boom")
        r = OpenAICompatModel(CFG).generate("p")
        assert r.status == "error" and "boom" in (r.error or "")
