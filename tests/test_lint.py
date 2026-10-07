# SPDX-License-Identifier: Apache-2.0
from minidwarf.lint import lint_source

CLEAN = '''#include <cuda_runtime.h>
// we do not use cublas or fopen here
/* nor cusparse */
__global__ void k(float* x){ float cube = x[0]*x[0]*x[0]; long system_size = 3; x[0] = cube + system_size; }
extern "C" void minidwarf_solve(const void* const* i, void* const* o, const long* d, int n){
  const char* s = "cublas fopen( system("; (void)s; }
'''

def test_clean_source_passes():
    assert lint_source(CLEAN) == []

def test_cublas_include_and_call_flagged():
    assert lint_source("#include <cublas_v2.h>\nvoid f(){ cublasHandle_t h; cublasCreate(&h); }") == ["cublas"]

def test_quoted_include_flagged():
    assert lint_source('#include "cusparse.h"\n') == ["cusparse"]

def test_thrust_and_cub_flagged():
    assert lint_source("void f(){ thrust::sort(a, b); cub::DeviceReduce::Sum(); }") == ["cub", "thrust"]

def test_file_process_dlopen_getenv_flagged():
    src = 'void f(){ FILE* f = fopen("x", "r"); system("ls"); void* h = dlopen("a", 0); getenv("HOME"); }'
    assert lint_source(src) == ["dynamic_loading", "environment", "file_io", "process"]

def test_allowed_lib_not_flagged():
    src = "#include <curand_kernel.h>\n"
    assert lint_source(src) == ["curand"]
    assert lint_source(src, allowed_libs=["curand"]) == []
