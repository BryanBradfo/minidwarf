# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_row_ok():
    m = _load("size_shapes")
    assert m.row_ok(1.5, 800) and not m.row_ok(0.4, 800) and not m.row_ok(3.0, 2500)

def test_eps_from_speedups():
    m = _load("noise_floor")
    assert m.eps_from_speedups([1.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert m.eps_from_speedups([1.0, 1.1, 1 / 1.1]) == pytest.approx(0.1, rel=1e-6)

def test_verdict():
    m = _load("calibrate_tol")
    assert m.verdict(0.0) == "exact" and m.verdict(0.5) == "ok"
    assert m.verdict(2.0) == "fail" and m.verdict(0.001) == "too_loose"
