# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from minidwarf.evalconfig import EvalConfig
from minidwarf.models.dummy import DummyModel
from minidwarf.generate import run_generation, SYSTEM_PROMPT

SMOKE = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_run_generation_writes_artifacts(tmp_path):
    cfg = EvalConfig(model_name="dummy", base_url="u", provider_model_name="p",
                     provider="dummy", n_samples=2)
    kernel = (SMOKE / "solutions/expert_v1.cu").read_text()
    run_dir = run_generation(cfg, [SMOKE], tmp_path, run_id="r1",
                             model=DummyModel(f"```cuda\n{kernel}\n```"))
    assert (run_dir / "results.jsonl").exists()
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2  # n_samples
    assert all(r["problem"] == "vector_add" for r in rows)
    assert (run_dir / "kernels/vector_add__s0.cu").exists()
    assert "minidwarf_solve" in (run_dir / "kernels/vector_add__s0.cu").read_text()

def test_system_prompt_mentions_abi():
    assert "minidwarf_solve" in SYSTEM_PROMPT
