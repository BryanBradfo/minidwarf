# SPDX-License-Identifier: Apache-2.0
from minidwarf.grade import ProblemResult
from minidwarf.score import fast_p, compile_rate, correctness_rate, summarize

def mk(status, correct, sp): return ProblemResult("n","d",status,correct,sp)

R = [mk("ok",True,3.0), mk("ok",True,1.2), mk("wrong_output",False,None),
     mk("compile_error",False,None)]

def test_fast_p():
    assert fast_p(R, 0) == 0.5            # 2 of 4 correct
    assert fast_p(R, 2) == 0.25           # only speedup>=2
def test_rates():
    assert compile_rate(R) == 0.75
    assert correctness_rate(R) == 0.5
def test_summarize_shape():
    s = summarize(R)
    assert s["compile_rate"] == 0.75 and s["fast_p"][2] == 0.25

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
