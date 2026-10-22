# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location("mf", Path(__file__).resolve().parents[1] / "scripts/make_figures.py")
mf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mf)

def _p(name, dwarf, status, speedup, sample_statuses):
    return {"name": name, "dwarf": dwarf, "status": status, "correct": status == "ok", "speedup": speedup,
            "samples": [{"sample": i, "status": s, "correct": s == "ok"} for i, s in enumerate(sample_statuses)]}

SCORES = {"model": "m", "harness_version": 3, "per_problem": [
    _p("a", "dense", "ok", 3.0, ["compile_error", "ok"]),
    _p("b", "dense", "wrong_output", None, ["wrong_output", "wrong_output"]),
    _p("c", "nbody", "ok", 1.0, ["ok", "ok"]),
]}

def test_summarize_run_metrics():
    s = mf.summarize_run(SCORES, eps_global=0.0, per={})
    assert s["model"] == "m" and s["n_problems"] == 3
    assert s["correct"] == pytest.approx(2 / 3)
    assert s["pass1"] == pytest.approx((0.5 + 0 + 1) / 3)  # mean per-problem sample success
    assert s["fast_p"][0.0] == pytest.approx(2 / 3) and s["fast_p"][2.0] == pytest.approx(1 / 3)
    assert s["by_dwarf"] == {"dense": pytest.approx(0.5), "nbody": pytest.approx(1.0)}
    assert s["outcomes"] == {"ok": 3, "wrong_output": 2, "compile_error": 1}

def test_noise_floor_raises_the_speedup_bar():
    s = mf.summarize_run(SCORES, eps_global=0.0, per={"c": 0.5})
    assert s["fast_p"][1.0] == pytest.approx(1 / 3)  # c at 1.0x is no longer >= 1 * (1 + 0.5)

def test_collect_keeps_most_complete_run_per_model(tmp_path):
    for run, probs in (("big", SCORES["per_problem"]), ("smoke", SCORES["per_problem"][:1])):
        (tmp_path / run).mkdir()
        (tmp_path / run / "scores.json").write_text(json.dumps({**SCORES, "per_problem": probs}))
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "scores.json").write_text(json.dumps({**SCORES, "model": "v2", "harness_version": 2}))
    runs = mf.collect(tmp_path)
    assert [r["model"] for r in runs] == ["m"] and len(runs[0]["per_problem"]) == 3
