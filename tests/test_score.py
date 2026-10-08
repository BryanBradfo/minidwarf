# SPDX-License-Identifier: Apache-2.0
from minidwarf.grade import ProblemResult
from minidwarf.score import fast_p, compile_rate, correctness_rate, summarize

def mk(status, correct, sp): return ProblemResult("n","d",status,correct,sp)

R = [mk("ok",True,3.0), mk("ok",True,1.2), mk("wrong_output",False,None),
     mk("compile_error",False,None)]

def test_fast_p():
    assert fast_p(R, 0, eps=0.0) == 0.5            # 2 of 4 correct
    assert fast_p(R, 2, eps=0.0) == 0.25           # only speedup>=2
def test_rates():
    assert compile_rate(R) == 0.75
    assert correctness_rate(R) == 0.5
def test_compile_rate_excludes_forbidden_api():
    assert compile_rate(R + [mk("forbidden_api", False, None)]) == 0.6
def test_summarize_shape():
    s = summarize(R, eps=0.0)
    assert s["compile_rate"] == 0.75 and s["fast_p"][2] == 0.25

def test_fast_p_applies_noise_floor(tmp_path):
    import json
    nf = tmp_path / "nf.json"
    nf.write_text(json.dumps({"eps_global": 0.01, "per_problem": {"slow": {"eps": 0.5}}}))
    rs = [ProblemResult("slow", "d", "ok", True, 1.2), ProblemResult("other", "d", "ok", True, 1.2)]
    assert fast_p(rs, 1, noise_floor=nf) == 0.5        # slow needs 1.5x, other needs 1.01x
    assert fast_p(rs, 1, eps=0.0, noise_floor=nf) == 1.0  # uniform override
    assert summarize(rs, noise_floor=nf)["fast_p"][1] == 0.5
    assert fast_p([ProblemResult("other", "d", "ok", True, 1.005)], 1, noise_floor=nf) == 0.0

import pytest
from minidwarf.score import geomean_speedup, HARNESS_VERSION

def test_geomean_speedup():
    assert geomean_speedup([2.0, 8.0], [1.0, 1.0]) == pytest.approx(4.0)
    assert geomean_speedup([1.0], [4.0]) == pytest.approx(0.25)

def test_geomean_speedup_invalid_is_none():
    assert geomean_speedup([], []) is None
    assert geomean_speedup([1.0], [0.0]) is None
    assert geomean_speedup([1.0, 2.0], [1.0]) is None

def test_harness_version():
    assert HARNESS_VERSION == 3
