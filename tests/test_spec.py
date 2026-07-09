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
