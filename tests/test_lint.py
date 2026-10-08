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

def test_double_slash_in_string_does_not_hide_call():
    assert lint_source('const char* s="//"; FILE*f=fopen("x","r");') == ["file_io"]

def test_block_comment_markers_in_strings_do_not_hide_call():
    assert lint_source('const char* a="/*"; fopen("x","r"); const char* b="*/";') == ["file_io"]

def test_char_literal_quote_does_not_hide_call():
    assert lint_source("char c='\"'; fopen(\"x\",\"r\"); char d='\"';") == ["file_io"]

def test_execve_flagged():
    assert lint_source('void f(){ execve("/bin/sh", 0, 0); }') == ["process"]

def test_quoted_thrust_include_flagged():
    assert lint_source('#include "thrust/sort.h"\n') == ["thrust"]

def test_digit_separator_does_not_hide_call():
    assert lint_source('int n = 1\'000; fopen("x","r"); int m = 2\'0;') == ["file_io"]

def test_prefixed_char_literal_does_not_hide_call():
    assert lint_source('char c = u8\'"\'; fopen("x","r"); char d = L\'"\';') == ["file_io"]

def test_raw_string_flagged():
    assert lint_source('const char* s = R"(x)";') == ["raw_string"]

from minidwarf.lint import lint_symbols

def test_constructor_attribute_flagged():
    assert lint_source("__attribute__((constructor)) void f(){}") == ["constructor"]
    assert lint_source("[[gnu::constructor]] void f(){}") == ["constructor"]
    assert lint_source("// __attribute__((constructor))\nvoid f(){}") == []

def test_lint_symbols_policy():
    mangled = "_ZNSt6thread15_M_start_threadESt10unique_ptrINS_6_StateESt14default_deleteIS1_EEPFvvE"
    got = lint_symbols(["cudaMalloc", "atexit", "fopen", "cublasCreate_v2", mangled])
    assert got == sorted(["symbol:fopen", "symbol:cublasCreate_v2", "symbol:" + mangled])
    assert lint_symbols(["curand_init"], ["curand"]) == []
    assert lint_symbols(["curand_init"]) == ["symbol:curand_init"]

def test_extra_banned_symbols():
    assert lint_symbols(["environ", "__open_2", "fdopen", "clone", "cudaMalloc"]) == sorted(
        ["symbol:environ", "symbol:__open_2", "symbol:fdopen", "symbol:clone"])

def test_lint_defined():
    from minidwarf.lint import lint_defined
    got = lint_defined([(n, "T") for n in ["minidwarf_solve", "cudaEventElapsedTime", "_Z4vaddPf", "fwrite", "__cudaFoo", "cuInit", "cube"]], {"fwrite"})
    assert got == sorted(["defines:cudaEventElapsedTime", "defines:fwrite", "defines:__cudaFoo", "defines:cuInit"])

def test_mprotect_and_reserved_prefixes():
    from minidwarf.lint import lint_defined
    assert lint_symbols(["mprotect", "pkey_mprotect"]) == ["symbol:mprotect", "symbol:pkey_mprotect"]
    assert lint_defined([("libcudart_static_abc", "T"), ("__cudart1", "T")], set()) == ["defines:__cudart1", "defines:libcudart_static_abc"]

def test_driver_names_forbidden_only_when_strong():
    from minidwarf.lint import lint_defined
    syms = [("_ZSt7shuffle", "T"), ("_ZNSt6vectorIfED2Ev", "W"), ("_ZNSt6vectorIlED2Ev", "V"), ("_ZSt4sort", "D")]
    drv = {"_ZSt7shuffle", "_ZNSt6vectorIfED2Ev", "_ZNSt6vectorIlED2Ev"}
    assert lint_defined(syms, set(), drv) == ["defines:_ZSt7shuffle"]
    assert lint_defined([("fwrite", "W")], {"fwrite"}) == ["defines:fwrite"]  # always-forbidden applies to weak too

def test_link_puts_driver_before_candidate_object():
    from minidwarf.compile import _link_cmd, DRIVER
    cmd = _link_cmd("c.o", "x.bin", "sm_120", ["-lm"])
    assert cmd.index(str(DRIVER)) < cmd.index("c.o") and cmd[-1] == "-lm"

def test_lint_defined_type_allowlist():
    from minidwarf.lint import lint_defined
    drv = {"_ZdrvW", "_ZdrvV", "_ZdrvT", "_ZdrvI"}
    syms = [("_ZdrvW", "W"), ("_ZdrvV", "V"), ("_ZdrvT", "T"), ("_ZdrvI", "i"), ("other", "i"), ("_Z6kernelPf", "T"),
            ("abs_sym", "A"), ("uniq", "u")]
    assert lint_defined(syms, set(), drv) == sorted(["defines:_ZdrvT", "defines:_ZdrvI", "defines:other",
                                                    "defines:abs_sym", "defines:uniq"])

def test_exec_prefixed_helpers_not_flagged():
    assert lint_source("__device__ void execute_tile(float* x){}\nvoid g(float* x){ execute_tile(x); }") == []
    assert lint_source("__device__ float executed(float x){ return x; }") == []
    assert lint_source('void f(){ execvp("sh", 0); fork(); popen("x","r"); }') == ["process"]
    assert lint_source('void f(){ execveat(0, "sh", 0, 0, 0); }') == ["process"]
    assert lint_symbols(["execveat"]) == ["symbol:execveat"]

def test_exit_and_set_device_symbols_allowed():
    assert lint_symbols(["exit", "_exit", "_Exit", "quick_exit", "cudaSetDevice"]) == []
    assert lint_symbols(["cudaDeviceReset"]) == ["symbol:cudaDeviceReset"]
