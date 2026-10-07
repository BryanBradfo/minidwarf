# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import numpy as np
import pytest
from minidwarf.compile import compile_binary
from minidwarf.runner import run_binary

FIX = Path(__file__).parent / "fixtures/smoke/vector_add"
NOOP = '#include <cuda_runtime.h>\nextern "C" void minidwarf_solve(const void* const*, void* const*, const long*, int){}\n'

def _sets(n, k):
    rng = np.random.default_rng(0)
    return [[rng.standard_normal(n).astype(np.float32), rng.standard_normal(n).astype(np.float32)] for _ in range(k)]

def test_run_returns_one_output_per_set(tmp_path):
    exe = compile_binary(FIX / "solutions/expert_v1.cu", tmp_path)
    sets = _sets(1024, 4)
    res = run_binary(exe, sets, dims=[1024], output_shapes=[(1024,)], reps=8, warmup=1)
    assert res.median_ms > 0 and len(res.times_ms) == 8
    assert res.p25_ms <= res.median_ms <= res.p75_ms
    for (a, b), outs in zip(sets, res.outputs):
        np.testing.assert_allclose(outs[0], a + b, rtol=1e-5, atol=1e-6)

def test_noop_kernel_outputs_are_poisoned(tmp_path):
    src = tmp_path / "noop.cu"; src.write_text(NOOP)
    exe = compile_binary(src, tmp_path / "b")
    res = run_binary(exe, _sets(256, 4), dims=[256], output_shapes=[(256,)], reps=4, warmup=0)
    assert all(np.isnan(o[0]).all() for o in res.outputs)

def test_reps_must_cover_all_sets(tmp_path):
    exe = compile_binary(FIX / "solutions/expert_v1.cu", tmp_path)
    with pytest.raises(ValueError):
        run_binary(exe, _sets(16, 4), [16], [(16,)], reps=3)
