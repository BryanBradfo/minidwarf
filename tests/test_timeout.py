# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from minidwarf.grade import grade_problem

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_hanging_kernel_scores_timeout_not_crash(tmp_path):
    r = grade_problem(FIX, FIX / "solutions/hang.cu", tmp_path, timeout_s=8)
    assert r.status == "timeout"
    assert r.correct is False
    assert r.speedup is None
