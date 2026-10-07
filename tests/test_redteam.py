# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path
import pytest
from minidwarf.grade import grade_problem

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"
RT = Path(__file__).parent / "fixtures/redteam"
pytestmark = pytest.mark.skipif(shutil.which("nvcc") is None, reason="needs CUDA toolchain")

@pytest.mark.parametrize("name,status", [
    ("memo_static", "wrong_output"), ("memo_copy", "wrong_output"), ("noop", "wrong_output"),
    ("skip_timed", "wrong_output"), ("index_cache", "wrong_output"),
    ("assumes_mult32", "wrong_output"), ("calls_cublas", "forbidden_api"), ("fopen_grader", "forbidden_api"),
])
def test_redteam_fixture_is_caught(tmp_path, name, status):
    r = grade_problem(FIX, RT / f"{name}.cu", tmp_path)
    assert r.status == status and r.correct is False
    if name in ("memo_static", "memo_copy", "noop", "skip_timed", "index_cache"): assert r.bad_calls > 0

def test_side_stream_work_is_timed(tmp_path):
    honest = grade_problem(FIX, FIX / "solutions/expert_v1.cu", tmp_path / "h")
    side = grade_problem(FIX, RT / "side_stream_nosync.cu", tmp_path / "s")
    assert side.status == "ok"
    for h, s in zip(honest.timings, side.timings):
        assert s["cand_median_ms"] >= 0.7 * h["cand_median_ms"]

def test_check_shapes_catch_what_eval_shapes_miss(tmp_path):
    r = grade_problem(FIX, RT / "assumes_mult32.cu", tmp_path)
    assert all(c["passed"] for c in r.checks if c["kind"] == "eval")  # both eval sizes are multiples of 32
    assert not all(c["passed"] for c in r.checks if c["kind"] == "check")
