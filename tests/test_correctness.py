# SPDX-License-Identifier: Apache-2.0
import numpy as np
from minidwarf.correctness import check_correct


def test_close_passes():
    a = np.ones(4, np.float32)
    b = a + 1e-7
    assert check_correct([a], [b], rtol=1e-5, atol=1e-6)


def test_far_fails():
    a = np.ones(4, np.float32)
    b = a + 1.0
    assert not check_correct([a], [b], rtol=1e-5, atol=1e-6)

import pytest
from minidwarf.correctness import compare

def test_compare_reports_errors():
    r = compare([np.array([1.0, 2.0], np.float32)], [np.array([1.0, 2.5], np.float32)], 1e-3, 1e-4)
    assert r["passed"] is False and r["max_abs_err"] == pytest.approx(0.5)
    assert r["nan_count"] == 0 and r["tol_ratio"] > 1

def test_compare_nan_and_shape_mismatch():
    r = compare([np.array([np.nan], np.float32)], [np.array([1.0], np.float32)], 1e-3, 1e-4)
    assert r["passed"] is False and r["nan_count"] == 1 and r["max_abs_err"] == float("inf")
    assert compare([np.zeros(2)], [np.zeros(3)], 1e-3, 1e-4)["passed"] is False

def test_compare_exact_has_zero_ratio():
    r = compare([np.ones(3, np.float32)], [np.ones(3, np.float32)], 1e-3, 1e-4)
    assert r["passed"] is True and r["tol_ratio"] == 0.0
