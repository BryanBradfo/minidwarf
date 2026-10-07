# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import shutil
import pytest
from minidwarf.compile import compile_object, object_hits
from minidwarf.grade import grade_problem, N_SETS

pytestmark = pytest.mark.skipif(shutil.which("nvcc") is None, reason="needs nvcc")
FIX = Path(__file__).parent / "fixtures/smoke/vector_add"
ROOT = Path(__file__).resolve().parents[1]
EXPERTS = sorted(ROOT.glob("problems/*/*/solutions/expert_v1.cu")) + [FIX / "solutions/expert_v1.cu"]

VADD = ('#include <cuda_runtime.h>\n__global__ void vadd(const float* a,const float* b,float* c,long n){'
        'long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }\n')
def _entry(pre, body=""):
    return VADD + 'extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){\n' + pre + (
        'long n=1; for(int i=0;i<nd;i++) n*=dims[i]; vadd<<<(int)((n+255)/256),256>>>((const float*)in[0],(const float*)in[1],(float*)out[0],n);' + body + '}\n')

def test_expert_grades_ok_with_detail(tmp_path):
    r = grade_problem(FIX, FIX / "solutions/expert_v1.cu", tmp_path)
    assert r.status == "ok" and r.correct is True and r.speedup > 0 and r.bad_calls == 0
    evals = [c for c in r.checks if c["kind"] == "eval"]
    assert len(evals) == 2 * 2 * N_SETS  # 2 eval shapes x 2 candidate runs (ABBA) x D sets
    assert len(r.timings) == 2 and all(t["cand_median_ms"] > 0 for t in r.timings)

def test_cheater_is_caught(tmp_path):
    r = grade_problem(FIX, FIX / "solutions/cheater.cu", tmp_path)
    assert r.correct is False and r.status == "wrong_output"

def test_broken_is_compile_error(tmp_path):
    bad = tmp_path / "bad.cu"; bad.write_text("nope")
    r = grade_problem(FIX, bad, tmp_path)
    assert r.status == "compile_error"

def test_baseline_trusted_against_itself(tmp_path):
    r = grade_problem(FIX, FIX / "baseline.cu", tmp_path, trusted=True)
    assert r.status == "ok" and 0.5 < r.speedup < 2.0

def test_macro_pasted_fopen_caught_by_symbol_check(tmp_path):
    k = tmp_path / "k.cu"
    k.write_text('#include <cstdio>\n#define CAT(a,b) a##b\n' + _entry(
        'FILE* f = CAT(fo,pen)("/proc/self/cmdline","rb"); if(f) fclose(f);\n'))
    r = grade_problem(FIX, k, tmp_path / "w")
    assert r.status == "forbidden_api" and "symbol:fopen" in r.lint

def test_call_counting_kernel_is_caught(tmp_path):
    k = tmp_path / "k.cu"
    k.write_text(_entry('static int calls; if(++calls>=5 && calls<=20) return;\n'))
    r = grade_problem(FIX, k, tmp_path / "w")
    assert r.status == "wrong_output" and r.bad_calls > 0 and r.correct is False

@pytest.mark.parametrize("cu", EXPERTS, ids=lambda p: p.parents[1].name)
def test_expert_passes_symbol_check(cu, tmp_path):
    assert object_hits(compile_object(cu, tmp_path)) == []

def _hits(tmp_path, src, **kw):
    k = tmp_path / "k.cu"; k.write_bytes(src if isinstance(src, bytes) else src.encode())
    return grade_problem(FIX, k, tmp_path / "w", **kw)

def test_optimize_guarded_fopen_caught(tmp_path):
    r = _hits(tmp_path, '#include <cstdio>\n#define CAT(a,b) a##b\n#ifdef __OPTIMIZE__\n'
              'static int s = []{ FILE* f = CAT(fo,pen)("/etc/hostname","rb"); if(f) fclose(f); return 1; }();\n#endif\n' + _entry(""))
    assert r.status == "forbidden_api" and "symbol:fopen" in r.lint

def test_timer_interposer_caught(tmp_path):
    r = _hits(tmp_path, '#include <cuda_runtime.h>\nextern "C" cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b)'
              '{*ms=0.001f; return cudaSuccess;}\n' + _entry(""))
    assert r.status == "forbidden_api" and "defines:cudaEventElapsedTime" in r.lint

def test_inline_syscall_caught(tmp_path):
    r = _hits(tmp_path, _entry('__asm__ volatile("syscall");\n'))
    assert r.status == "forbidden_api" and "asm:syscall" in r.lint

def test_non_utf8_syntax_error_is_compile_error(tmp_path):
    r = _hits(tmp_path, b'// \xff\xfe\nthis is not c++ \xff\n')
    assert r.status == "compile_error"

def test_non_utf8_stderr_does_not_raise(tmp_path):
    r = _hits(tmp_path, '#include <cstdio>\nstatic int s = (fprintf(stderr, "\\xff\\xfe"), 1);\n' + _entry(""))
    assert r.status == "ok"
