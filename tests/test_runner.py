# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import numpy as np
from minidwarf.compile import compile_binary
from minidwarf.runner import run_binary

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"

def test_run_expert_produces_correct_sum(tmp_path):
    exe = compile_binary(FIX / "solutions/expert_v1.cu", tmp_path)
    a = np.arange(1024, dtype=np.float32); b = np.ones(1024, dtype=np.float32)
    res = run_binary(exe, [a, b], dims=[1024], output_shapes=[(1024,)], reps=5)
    assert res.median_ms > 0
    np.testing.assert_allclose(res.outputs[0], a + b, rtol=1e-5, atol=1e-6)
