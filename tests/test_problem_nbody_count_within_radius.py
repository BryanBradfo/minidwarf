# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from minidwarf.grade import grade_problem

P = Path(__file__).resolve().parents[1] / "problems/nbody/nbody_count_within_radius"

def test_expert_ok(tmp_path):
    r = grade_problem(P, P / "solutions/expert_v1.cu", tmp_path)
    assert r.status == "ok" and r.correct

_RADIUS = 3.0

def _full(inputs, shape):  # pre-chunk reference (full N x N temporaries), kept to pin the chunked math
    import numpy as np
    pos = inputs[0].astype(np.float64)
    d = np.abs(pos[:, None] - pos[None, :])
    within = d < _RADIUS
    np.fill_diagonal(within, False)
    out = within.sum(axis=1).astype(np.float64)
    return [out.astype(np.float32)]

def test_chunked_reference_matches_full():
    import numpy as np
    from minidwarf.problem_io import load_module_fn
    gen = load_module_fn(P, "inputs.py", "generate"); ref = load_module_fn(P, "reference.py", "run")
    ins = gen([3000], 5)
    np.testing.assert_allclose(ref(ins, [3000])[0], _full(ins, [3000])[0], rtol=1e-12, atol=0)

def test_float32_counts_match_float64_on_eval_data():
    # positions sit on a 2^-16 grid, so float32 |pj - pi| < 3 decides every pair like the float64 reference
    import numpy as np
    from minidwarf.refcache import cached_case
    from minidwarf.spec import load_problem
    for i, shape in enumerate(load_problem(P).eval_shapes):
        (pos,), (expected,) = cached_case(P, shape, 12345 + 1000 * i)  # data set 0 of each eval shape
        assert pos.dtype == np.float32
        counts = np.empty(pos.size, np.float32)
        for s in range(0, pos.size, 2048):
            within = np.abs(pos[None, :] - pos[s:s + 2048, None]) < np.float32(3.0)  # float32 arithmetic
            within[np.arange(within.shape[0]), np.arange(s, s + within.shape[0])] = False
            counts[s:s + 2048] = within.sum(axis=1)
        np.testing.assert_array_equal(counts, expected)
