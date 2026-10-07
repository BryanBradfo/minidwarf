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

SKIP = '''#include <cuda_runtime.h>
__global__ void vadd(const float* a,const float* b,float* c,long n){ long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* d, int nd){
  static int calls; if(++calls>=5 && calls<=20) return;
  vadd<<<(int)((d[0]+255)/256),256>>>((const float*)in[0],(const float*)in[1],(float*)out[0],d[0]); }
'''
# computes each slot once (first use), afterwards replays the slot keyed by call index
CACHE = '''#include <cuda_runtime.h>
__global__ void vadd(const float* a,const float* b,float* c,long n){ long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* d, int nd){
  static int calls; static float* slot[4]; static bool have[4];
  int k=calls++ % 4; size_t nb=d[0]*4;
  if(!slot[k]) cudaMalloc(&slot[k], nb);
  if(!have[k]){ vadd<<<(int)((d[0]+255)/256),256>>>((const float*)in[0],(const float*)in[1],slot[k],d[0]); have[k]=true; }
  cudaMemcpy(out[0], slot[k], nb, cudaMemcpyDeviceToDevice); }
'''

def _exp(sets): return [[a + b] for a, b in sets]

def _build(tmp_path, src):
    f = tmp_path / "k.cu"; f.write_text(src); return compile_binary(f, tmp_path / "b")

def test_every_call_is_verified(tmp_path):
    exe = _build(tmp_path, SKIP); sets = _sets(1024, 4)
    res = run_binary(exe, sets, [1024], [(1024,)], reps=20, warmup=3, expected_sets=_exp(sets), rtol=1e-5, atol=1e-6)
    assert res.n_bad_calls > 0

def test_honest_kernel_has_no_bad_calls(tmp_path):
    exe = compile_binary(FIX / "solutions/expert_v1.cu", tmp_path); sets = _sets(1024, 4)
    res = run_binary(exe, sets, [1024], [(1024,)], reps=20, warmup=3, expected_sets=_exp(sets), rtol=1e-5, atol=1e-6)
    assert res.n_bad_calls == 0

def test_set_order_is_randomized(tmp_path):
    exe = compile_binary(FIX / "solutions/expert_v1.cu", tmp_path); sets = _sets(256, 4)
    seqs = [run_binary(exe, sets, [256], [(256,)], reps=20, warmup=1).timed_sets for _ in range(3)]
    assert all(sorted(s) == sorted(i % 4 for i in range(20)) for s in seqs)
    assert len({tuple(s) for s in seqs}) > 1

def test_index_keyed_cache_is_caught(tmp_path):
    exe = _build(tmp_path, CACHE); sets = _sets(1024, 4)
    res = run_binary(exe, sets, [1024], [(1024,)], reps=20, warmup=3, expected_sets=_exp(sets), rtol=1e-5, atol=1e-6)
    assert res.n_bad_calls > 0

def test_mismatched_set_shapes_rejected(tmp_path):
    exe = compile_binary(FIX / "solutions/expert_v1.cu", tmp_path)
    with pytest.raises(ValueError):
        run_binary(exe, _sets(16, 1) + _sets(32, 1), [16], [(16,)], reps=4)

def test_expected_sets_mismatch_rejected(tmp_path):
    exe = compile_binary(FIX / "solutions/expert_v1.cu", tmp_path); sets = _sets(16, 4)
    exp = _exp(sets)
    with pytest.raises(ValueError): run_binary(exe, sets, [16], [(16,)], reps=4, expected_sets=exp[:3])
    with pytest.raises(ValueError): run_binary(exe, sets, [16], [(16,)], reps=4, expected_sets=[[np.zeros(8, np.float32)] for _ in sets])
