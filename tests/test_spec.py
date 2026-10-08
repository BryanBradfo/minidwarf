# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import pytest
from minidwarf.spec import load_problem, Problem

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_load_problem_parses_fields():
    p = load_problem(FIX)
    assert isinstance(p, Problem)
    assert p.name == "vector_add" and p.dwarf == "structured_grids"
    assert p.n_inputs == 2 and p.n_outputs == 1
    assert p.eval_shapes == [[1048576], [1500000]]
    assert p.rtol == pytest.approx(1e-5)

def test_load_problem_missing_field_raises(tmp_path):
    (tmp_path / "spec.yaml").write_text("name: x\n")
    with pytest.raises(ValueError):
        load_problem(tmp_path)

def test_load_problem_missing_file_raises(tmp_path):
    with pytest.raises(ValueError):
        load_problem(tmp_path)

def test_load_problem_non_dict_raises(tmp_path):
    (tmp_path / "spec.yaml").write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_problem(tmp_path)

BASE = ("name: x\ndwarf: d\ndifficulty: easy\nrtol: 1.0e-5\natol: 1.0e-6\n"
        "n_inputs: 1\nn_outputs: 1\neval_shapes: [[4]]\nbaseline: author_kernel\n")

def test_optional_fields_default_empty(tmp_path):
    (tmp_path / "spec.yaml").write_text(BASE)
    p = load_problem(tmp_path)
    assert p.check_shapes == [] and p.allowed_libs == []

def test_optional_fields_parsed(tmp_path):
    (tmp_path / "spec.yaml").write_text(BASE + "check_shapes: [[1], [33]]\nallowed_libs: [curand]\n")
    p = load_problem(tmp_path)
    assert p.check_shapes == [[1], [33]] and p.allowed_libs == ["curand"]

def test_unknown_allowed_lib_raises(tmp_path):
    (tmp_path / "spec.yaml").write_text(BASE + "allowed_libs: [mkl]\n")
    with pytest.raises(ValueError):
        load_problem(tmp_path)

def test_scalar_allowed_libs_rejected_clearly(tmp_path):
    (tmp_path / "spec.yaml").write_text(BASE + "allowed_libs: curand\n")
    with pytest.raises(ValueError, match="list"):
        load_problem(tmp_path)

def test_malformed_check_shapes_rejected(tmp_path):
    (tmp_path / "spec.yaml").write_text(BASE + "check_shapes: 5\n")
    with pytest.raises(ValueError, match="check_shapes"):
        load_problem(tmp_path)
