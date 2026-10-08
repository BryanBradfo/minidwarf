# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import numpy as np
import pytest
from minidwarf.spec import load_problem
from minidwarf.problem_io import load_module_fn

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = sorted(p.parent for p in (ROOT / "problems").glob("*/*/spec.yaml"))

@pytest.mark.parametrize("pdir", PROBLEMS, ids=lambda p: p.name)
def test_check_shapes_are_valid(pdir):
    p = load_problem(pdir)
    assert len(p.check_shapes) >= 3
    assert all(len(s) == len(p.eval_shapes[0]) for s in p.check_shapes)
    assert any(s[0] % 32 for s in p.check_shapes)  # at least one non-multiple-of-32 leading dim
    gen = load_module_fn(pdir, "inputs.py", "generate"); ref = load_module_fn(pdir, "reference.py", "run")
    for i, shape in enumerate(p.check_shapes):
        ins = gen(shape, 1 + i); outs = ref(ins, shape)
        assert len(ins) == p.n_inputs and len(outs) == p.n_outputs
        assert all(np.isfinite(np.asarray(o)).all() for o in outs)
