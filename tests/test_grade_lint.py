# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from minidwarf.grade import grade_problem

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_lint_runs_before_compile_on_non_utf8_source(tmp_path):
    bad = tmp_path / "bad.cu"; bad.write_bytes(b"\xff\xfe#include <cublas_v2.h>\n")
    r = grade_problem(FIX, bad, tmp_path)
    assert r.status == "forbidden_api" and r.lint == ["cublas"] and r.correct is False and r.speedup is None
