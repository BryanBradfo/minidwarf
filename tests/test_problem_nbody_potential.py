# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from minidwarf.grade import grade_problem

P = Path(__file__).resolve().parents[1] / "problems/nbody/nbody_potential"

def test_expert_ok(tmp_path):
    r = grade_problem(P, P / "solutions/expert_v1.cu", tmp_path)
    assert r.status == "ok" and r.correct

_EPS = 1e-2

def _full(inputs, shape):  # pre-chunk reference (full N x N temporaries), kept to pin the chunked math
    import numpy as np
    pos = inputs[0].astype(np.float64)
    d = pos[None, :] - pos[:, None]  # d[i, j] = pos[j] - pos[i]
    r2 = d * d
    val = 1.0 / np.sqrt(r2 + _EPS)
    np.fill_diagonal(val, 0.0)
    out = val.sum(axis=1)
    return [out.astype(np.float32)]

def test_chunked_reference_matches_full():
    import numpy as np
    from minidwarf.problem_io import load_module_fn
    gen = load_module_fn(P, "inputs.py", "generate"); ref = load_module_fn(P, "reference.py", "run")
    ins = gen([3000], 5)
    np.testing.assert_allclose(ref(ins, [3000])[0], _full(ins, [3000])[0], rtol=1e-12, atol=0)
