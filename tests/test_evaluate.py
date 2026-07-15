# SPDX-License-Identifier: Apache-2.0
from minidwarf.grade import ProblemResult
from minidwarf.evaluate import best_result

def mk(status, correct, sp): return ProblemResult("n", "d", status, correct, sp)

def test_best_prefers_highest_speedup_among_correct():
    r = best_result([mk("ok", True, 1.5), mk("ok", True, 3.0), mk("wrong_output", False, None)])
    assert r.correct and r.speedup == 3.0

def test_best_falls_back_to_least_bad_status():
    r = best_result([mk("compile_error", False, None), mk("wrong_output", False, None)])
    assert r.status == "wrong_output"


import shutil, pytest
from pathlib import Path
from minidwarf.evalconfig import EvalConfig
from minidwarf.models.dummy import DummyModel
from minidwarf.generate import run_generation
from minidwarf.evaluate import score_run

@pytest.mark.skipif(shutil.which("nvcc") is None, reason="needs CUDA toolchain")
def test_score_run_e2e_smoke(tmp_path):
    SMOKE = Path(__file__).parent / "fixtures/smoke/vector_add"
    # generate against the smoke problem using its own expert as the "model output"
    cfg = EvalConfig(model_name="dummy", base_url="u", provider_model_name="p",
                     provider="dummy", n_samples=1)
    kernel = (SMOKE / "solutions/expert_v1.cu").read_text()
    run_dir = run_generation(cfg, [SMOKE], tmp_path, "r1",
                             model=DummyModel(f"```cuda\n{kernel}\n```"))
    scores = score_run(run_dir, problems_root=SMOKE.parent.parent)  # tests/fixtures/smoke
    import json
    data = json.loads(scores.read_text())
    assert data["per_problem"][0]["correct"] is True
