# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import numpy as np
import pytest
from minidwarf.spec import load_problem
from minidwarf.refcache import cached_case

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = sorted(p.parent for p in (ROOT / "problems").glob("*/*/spec.yaml"))

@pytest.mark.parametrize("pdir", PROBLEMS, ids=lambda p: p.name)
def test_reference_outputs_are_not_near_constant(pdir):
    p = load_problem(pdir)
    _, outs = cached_case(pdir, p.eval_shapes[0], 12345)  # same case grading uses (eval shape 0, set 0)
    for o in outs:
        o = np.asarray(o, np.float64)
        assert o.std() / (abs(o.mean()) + 1e-12) > 1e-3, f"{p.name}: near-constant reference output"
