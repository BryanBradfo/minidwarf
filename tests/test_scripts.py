# SPDX-License-Identifier: Apache-2.0
import importlib.util, json
from pathlib import Path
from types import SimpleNamespace
import pytest
from minidwarf.report import sanitize, write_report

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def _noop(*a, **k): return []

def _fake_problems(root, n):
    for i in range(n):
        d = root / f"c{i}" / f"p{i}"; d.mkdir(parents=True)
        (d / "spec.yaml").write_text(""); (d / "baseline.cu").write_text("")
    return root

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

def test_calibrate_fails_on_grade_status(monkeypatch):
    m = _load("calibrate_tol")
    pdir = next((ROOT / "problems").glob("*/*"))
    def fake(status, bad=0):
        return lambda *a, **k: SimpleNamespace(status=status, checks=[{"tol_ratio": 0.5}], bad_calls=bad)
    monkeypatch.setattr(m, "grade_problem", fake("ok"))
    assert m.calibrate(pdir)["verdict"] == "ok"
    monkeypatch.setattr(m, "grade_problem", fake("runtime_error"))
    row = m.calibrate(pdir)
    assert row["verdict"] == "fail" and row["status"] == "runtime_error"
    monkeypatch.setattr(m, "grade_problem", fake("wrong_output", bad=2))
    row = m.calibrate(pdir)
    assert row["verdict"] == "fail" and row["bad_calls"] == 2

def test_sanitize_non_finite_to_none():
    assert sanitize({"a": float("inf"), "b": [1.0, float("nan")], "c": 2.5}) == {"a": None, "b": [1.0, None], "c": 2.5}

def test_write_report_is_strict_json(tmp_path):
    out = tmp_path / "r.json"
    write_report(out, {"x": float("inf"), "y": 1.5})
    def reject(c): raise ValueError(c)
    assert json.loads(out.read_text(), parse_constant=reject) == {"x": None, "y": 1.5}
    assert not (tmp_path / "r.json.tmp").exists()

def test_empty_problems_root_exits_2(tmp_path, monkeypatch):
    for name in ("size_shapes", "noise_floor", "calibrate_tol"):
        m = _load(name); monkeypatch.setattr(m, "ensure_gpu_idle", _noop)
        args = ["--problems-root", str(tmp_path)] + (["--check-all"] if name == "size_shapes" else [])
        assert m.main(args) == 2, name

def test_noise_floor_none_speedup_is_failure(tmp_path, monkeypatch):
    m = _load("noise_floor")
    root = _fake_problems(tmp_path / "probs", 3)
    seq = iter([1.0, None, 1.1])
    monkeypatch.setattr(m, "grade_problem", lambda *a, **k: SimpleNamespace(status="ok", speedup=next(seq)))
    monkeypatch.setattr(m, "ensure_gpu_idle", _noop); monkeypatch.setattr(m, "env_record", lambda: {})
    out = tmp_path / "nf.json"
    assert m.main(["--problems-root", str(root), "--out", str(out), "--repeats", "1"]) == 1
    rep = json.loads(out.read_text())
    assert "p1" in rep["failed"] and rep["eps"] is not None and "p1" not in rep["per_problem"]

def test_noise_floor_refuses_eps_below_half(tmp_path, monkeypatch):
    m = _load("noise_floor")
    root = _fake_problems(tmp_path / "probs", 3)
    seq = iter([None, None, 1.0])
    monkeypatch.setattr(m, "grade_problem", lambda *a, **k: SimpleNamespace(status="ok", speedup=next(seq)))
    monkeypatch.setattr(m, "ensure_gpu_idle", _noop); monkeypatch.setattr(m, "env_record", lambda: {})
    out = tmp_path / "nf.json"
    assert m.main(["--problems-root", str(root), "--out", str(out), "--repeats", "1"]) == 1
    assert json.loads(out.read_text())["eps"] is None

def test_noise_floor_pools_repeats(tmp_path, monkeypatch):
    m = _load("noise_floor")
    root = _fake_problems(tmp_path / "probs", 2)
    seq = iter([1.0, 1.1, 1 / 1.1, 1.0, 1.0, 1.0])
    monkeypatch.setattr(m, "grade_problem", lambda *a, **k: SimpleNamespace(status="ok", speedup=next(seq)))
    monkeypatch.setattr(m, "ensure_gpu_idle", _noop); monkeypatch.setattr(m, "env_record", lambda: {"t": 1})
    out = tmp_path / "nf.json"
    assert m.main(["--problems-root", str(root), "--out", str(out), "--repeats", "3", "--unstable-repeats", "3"]) == 0
    rep = json.loads(out.read_text())
    p0 = rep["per_problem"]["p0"]
    assert p0["speedups"] == pytest.approx([1.0, 1.1, 1 / 1.1]) and p0["max"] == pytest.approx(1.1)
    assert rep["eps"] == pytest.approx(m.eps_from_speedups([1.0, 1.1, 1 / 1.1, 1.0, 1.0, 1.0]))
    assert rep["repeats"] == 3 and rep["env_end"] == {"t": 1}
    g = rep["eps_global"]
    assert g == rep["eps"] and p0["eps"] == pytest.approx(max(0.1, g)) and rep["eps_max"] == pytest.approx(0.1)
    assert rep["per_problem"]["p1"]["eps"] == pytest.approx(g)  # floored at eps_global
    assert p0["unstable"] and p0["n_repeats"] == 3 and not rep["per_problem"]["p1"]["unstable"]

def test_noise_floor_extends_unstable_problems(tmp_path, monkeypatch):
    m = _load("noise_floor")
    root = _fake_problems(tmp_path / "probs", 2)
    seq = iter([1.0, 1.2, 1.0, 1.0, 1.0, 0.8] + [1.0] * 20)  # p0 bimodal after 2, extended to 6; p1 stable
    monkeypatch.setattr(m, "grade_problem", lambda *a, **k: SimpleNamespace(status="ok", speedup=next(seq)))
    monkeypatch.setattr(m, "ensure_gpu_idle", _noop); monkeypatch.setattr(m, "env_record", lambda: {})
    out = tmp_path / "nf.json"
    assert m.main(["--problems-root", str(root), "--out", str(out), "--repeats", "2", "--unstable-repeats", "6"]) == 0
    rep = json.loads(out.read_text()); p0, p1 = rep["per_problem"]["p0"], rep["per_problem"]["p1"]
    assert p0["unstable"] and p0["n_repeats"] == 6 and p0["eps"] == pytest.approx(0.25)  # 1/0.8 over all samples
    assert not p1["unstable"] and p1["n_repeats"] == 2
    assert rep["eps_global"] == pytest.approx(m.eps_from_speedups([1.0, 1.2, 1.0, 1.0]))  # first 2 per problem only
