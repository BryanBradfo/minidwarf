# SPDX-License-Identifier: Apache-2.0
import shutil, pytest
from pathlib import Path
from minidwarf.grade import ProblemResult
from minidwarf.evaluate import best_result, score_run

def mk(status, correct, sp): return ProblemResult("n", "d", status, correct, sp)

def test_best_prefers_highest_speedup_among_correct():
    r = best_result([mk("ok", True, 1.5), mk("ok", True, 3.0), mk("wrong_output", False, None)])
    assert r.correct and r.speedup == 3.0

def test_best_falls_back_to_least_bad_status():
    r = best_result([mk("compile_error", False, None), mk("wrong_output", False, None)])
    assert r.status == "wrong_output"


def test_forbidden_api_ranks_below_wrong_output():
    r = best_result([mk("forbidden_api", False, None), mk("wrong_output", False, None)])
    assert r.status == "wrong_output"

def test_score_run_refuses_busy_gpu(tmp_path, monkeypatch):
    import json as _json
    import minidwarf.preflight as pf
    from minidwarf.preflight import GpuBusyError
    monkeypatch.setattr(pf, "_smi", lambda args: "42, python\n")
    run = tmp_path / "r"; run.mkdir()
    (run / "results.jsonl").write_text(_json.dumps(
        {"problem": "vector_add", "dwarf": "smoke", "model": "m", "kernel_path": "kernels/x.cu"}) + "\n")
    with pytest.raises(GpuBusyError):
        score_run(run, problems_root=Path(__file__).parent / "fixtures")

from minidwarf.evalconfig import EvalConfig
from minidwarf.models.dummy import DummyModel
from minidwarf.generate import run_generation

@pytest.mark.skipif(shutil.which("nvcc") is None, reason="needs CUDA toolchain")
def test_score_run_e2e_smoke(tmp_path):
    SMOKE = Path(__file__).parent / "fixtures/smoke/vector_add"
    # generate against the smoke problem using its own expert as the "model output"
    cfg = EvalConfig(model_name="dummy", base_url="u", provider_model_name="p",
                     provider="dummy", n_samples=1)
    kernel = (SMOKE / "solutions/expert_v1.cu").read_text()
    run_dir = run_generation(cfg, [SMOKE], tmp_path, "r1",
                             model=DummyModel(f"```cuda\n{kernel}\n```"))
    scores = score_run(run_dir, problems_root=SMOKE.parent.parent, allow_busy_gpu=True)  # tests/fixtures/smoke
    import json
    data = json.loads(scores.read_text())
    assert data["per_problem"][0]["correct"] is True
    assert data["harness_version"] == 3 and "env" in data
    assert data["per_problem"][0]["checks"] and data["per_problem"][0]["timings"]


def _run_dir(tmp_path):
    import json as _json
    run = tmp_path / "r"; run.mkdir()
    (run / "results.jsonl").write_text(_json.dumps(
        {"problem": "vector_add", "dwarf": "smoke", "model": "m", "kernel_path": "kernels/x.cu"}) + "\n")
    return run

def test_busy_seen_recorded(tmp_path, monkeypatch):
    import json as _json
    import minidwarf.preflight as pf, minidwarf.evaluate as ev
    monkeypatch.setattr(pf, "_smi", lambda args: "42, python\n")
    monkeypatch.setattr(ev, "grade_problem", lambda *a, **k: ProblemResult("vector_add", "smoke", "ok", True, 1.0))
    data = _json.loads(score_run(_run_dir(tmp_path), allow_busy_gpu=True).read_text())
    assert data["busy_seen"] == {"vector_add": ["42, python"]}

def test_stale_scores_removed_on_busy_abort(tmp_path, monkeypatch):
    import minidwarf.preflight as pf
    from minidwarf.preflight import GpuBusyError
    monkeypatch.setattr(pf, "_smi", lambda args: "42, python\n")
    run = _run_dir(tmp_path); (run / "scores.json").write_text("{}")
    with pytest.raises(GpuBusyError):
        score_run(run)
    assert not (run / "scores.json").exists()

def test_scores_json_is_strict(tmp_path, monkeypatch):
    import json as _json
    import minidwarf.preflight as pf, minidwarf.evaluate as ev
    monkeypatch.setattr(pf, "_smi", lambda args: None)
    monkeypatch.setattr(pf.shutil, "which", lambda n: None)
    r = ProblemResult("vector_add", "smoke", "ok", True, 1.0, checks=[{"ratio": float("inf")}])
    monkeypatch.setattr(ev, "grade_problem", lambda *a, **k: r)
    text = score_run(_run_dir(tmp_path)).read_text()
    def bad(c): raise ValueError(c)
    data = _json.loads(text, parse_constant=bad)
    assert data["per_problem"][0]["checks"] == [{"ratio": None}]
