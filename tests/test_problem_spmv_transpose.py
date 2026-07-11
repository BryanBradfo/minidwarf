# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from minidwarf.grade import grade_problem

P = Path(__file__).resolve().parents[1] / "problems/sparse/spmv_transpose"

def test_expert_ok(tmp_path):
    r = grade_problem(P, P / "solutions/expert_v1.cu", tmp_path)
    assert r.status == "ok" and r.correct
