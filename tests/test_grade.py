# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from minidwarf.grade import grade_problem

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_expert_grades_ok(tmp_path):
    r = grade_problem(FIX, FIX / "solutions/expert_v1.cu", tmp_path)
    assert r.status == "ok" and r.correct is True and r.speedup > 0

def test_cheater_is_caught(tmp_path):
    r = grade_problem(FIX, FIX / "solutions/cheater.cu", tmp_path)
    assert r.correct is False

def test_broken_is_compile_error(tmp_path):
    bad = tmp_path / "bad.cu"; bad.write_text("nope")
    r = grade_problem(FIX, bad, tmp_path)
    assert r.status == "compile_error"
