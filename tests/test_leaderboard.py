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
    assert load_noise_floor(tmp_path / "missing.json") == 0.0
    (tmp_path / "nf.json").write_text(json.dumps({"eps": 0.04}))
    assert load_noise_floor(tmp_path / "nf.json") == 0.04
