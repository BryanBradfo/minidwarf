# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from minidwarf.leaderboard import build_leaderboard


def _write(run, model, rows):
    run.mkdir(parents=True)
    (run / "scores.json").write_text(json.dumps({"model": model, "per_problem": rows}))


def test_leaderboard_has_rows_and_dwarf_breakdown(tmp_path):
    _write(tmp_path / "a", "modelA", [
        {"name": "p1", "dwarf": "dense", "status": "ok", "correct": True, "speedup": 3.0},
        {"name": "p2", "dwarf": "sparse", "status": "wrong_output", "correct": False, "speedup": None},
    ])
    md = build_leaderboard(tmp_path)
    assert "modelA" in md
    assert "fast_p@1" in md and "compile" in md
    assert "dense" in md and "sparse" in md  # per-dwarf breakdown present
