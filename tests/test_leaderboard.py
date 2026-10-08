# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from minidwarf.leaderboard import build_leaderboard, load_noise_floor
from minidwarf.score import HARNESS_VERSION


def _write(run, model, rows, version=HARNESS_VERSION):
    run.mkdir(parents=True)
    data = {"model": model, "per_problem": rows}
    if version is not None:
        data["harness_version"] = version
    (run / "scores.json").write_text(json.dumps(data))

def _row(sp, dwarf="dense"):
    return {"name": "p1", "dwarf": dwarf, "status": "ok", "correct": True, "speedup": sp}


def test_leaderboard_has_rows_and_dwarf_breakdown(tmp_path):
    _write(tmp_path / "a", "modelA", [
        _row(3.0),
        {"name": "p2", "dwarf": "sparse", "status": "wrong_output", "correct": False, "speedup": None},
    ])
    md = build_leaderboard(tmp_path, eps=0.0)
    assert "modelA" in md
    assert "fast_p@1" in md and "compile" in md
    assert "dense" in md and "sparse" in md

def test_old_harness_runs_are_skipped(tmp_path):
    _write(tmp_path / "old", "oldModel", [_row(3.0)], version=None)
    _write(tmp_path / "new", "newModel", [_row(3.0)])
    md = build_leaderboard(tmp_path, eps=0.0)
    assert "newModel" in md and "oldModel" not in md and "Skipped 1 run" in md

def test_noise_floor_raises_the_bar(tmp_path):
    _write(tmp_path / "a", "m", [_row(1.03)])
    assert "| m | 100.0% | 100.0% | 100.0% |" in build_leaderboard(tmp_path, eps=0.0)
    assert "| m | 100.0% | 100.0% | 0.0% |" in build_leaderboard(tmp_path, eps=0.05)

def test_load_noise_floor(tmp_path):
    assert load_noise_floor(tmp_path / "missing.json") == (0.0, {}, set())
    (tmp_path / "nf.json").write_text(json.dumps({"eps": 0.04}))  # old format: global eps only
    assert load_noise_floor(tmp_path / "nf.json") == (0.04, {}, set())
    (tmp_path / "nf2.json").write_text(json.dumps({"eps": 0.01, "eps_global": 0.01, "eps_max": 0.2,
                                                   "per_problem": {"p1": {"eps": 0.2, "speedups": [1.2], "unstable": True},
                                                                   "p2": {"eps": 0.01, "unstable": False}}}))
    assert load_noise_floor(tmp_path / "nf2.json") == (0.01, {"p1": 0.2, "p2": 0.01}, {"p1"})

def test_per_problem_eps_raises_the_bar_for_that_problem_only(tmp_path):
    _write(tmp_path / "a", "m", [_row(1.1), {**_row(1.1), "name": "p2"}])
    nf = tmp_path / "nf.json"
    nf.write_text(json.dumps({"eps": 0.01, "eps_global": 0.01, "per_problem": {"p1": {"eps": 0.2}}}))
    md = build_leaderboard(tmp_path, noise_floor=nf)
    assert "| m | 100.0% | 100.0% | 50.0% |" in md  # p1 needs 1.2x, p2 (unmeasured) needs 1.01x
    assert "eps_global = 0.010" in md and "eps_max = 0.200" in md
    assert "| m | 100.0% | 100.0% | 100.0% |" in build_leaderboard(tmp_path, eps=0.05, noise_floor=nf)  # override

def test_unstable_problems_are_flagged(tmp_path):
    _write(tmp_path / "a", "m", [_row(1.1, dwarf="sparse"), {**_row(1.1), "name": "p2"}])
    nf = tmp_path / "nf.json"
    nf.write_text(json.dumps({"eps": 0.01, "eps_global": 0.01,
                              "per_problem": {"p1": {"eps": 0.25, "unstable": True}, "p2": {"eps": 0.01}}}))
    md = build_leaderboard(tmp_path, noise_floor=nf)
    assert "Unstable timing: p1 (eps 0.25)" in md and "low-confidence" in md
    assert "| m | sparse (unstable timing: p1) |" in md and "| m | dense |" in md
    assert "Unstable timing" not in build_leaderboard(tmp_path, eps=0.0)

def test_load_noise_floor_tolerates_null_and_missing_eps(tmp_path):
    import pytest
    nf = tmp_path / "nf.json"
    nf.write_text(json.dumps({"eps_global": 0.03}))  # no "eps" key: must not KeyError
    assert load_noise_floor(nf)[0] == 0.03
    nf.write_text(json.dumps({"eps": 0.04, "eps_global": None}))
    assert load_noise_floor(nf)[0] == 0.04
    nf.write_text(json.dumps({"eps": None, "eps_global": None, "per_problem": {"p1": {"eps": None}, "p2": {"eps": 0.1}}}))
    with pytest.warns(UserWarning):
        assert load_noise_floor(nf) == (0.0, {"p2": 0.1}, set())
    nf.write_text(json.dumps({"per_problem": {}}))
    with pytest.warns(UserWarning):
        assert load_noise_floor(nf)[0] == 0.0
