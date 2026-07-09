# SPDX-License-Identifier: Apache-2.0
"""Verify the CUDA toolchain: nvcc present and a trivial kernel compiles & runs."""
import shutil, subprocess, sys, tempfile
from pathlib import Path

TRIVIAL = r"""
#include <cstdio>
__global__ void k(int* x){ *x = 42; }
int main(){ int *d,h=0; cudaMalloc(&d,sizeof(int)); k<<<1,1>>>(d);
  cudaMemcpy(&h,d,sizeof(int),cudaMemcpyDeviceToHost); printf("%d\n",h);
  return h==42?0:1; }
"""

def main() -> int:
    if not shutil.which("nvcc"):
        print("ERROR: nvcc not found. Install CUDA Toolkit >= 12.8 "
              "(Blackwell/sm_120) and put nvcc on PATH.", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as d:
        src, exe = Path(d) / "t.cu", Path(d) / "t"
        src.write_text(TRIVIAL)
        c = subprocess.run(["nvcc", "-arch=sm_120", str(src), "-o", str(exe)],
                           capture_output=True, text=True)
        if c.returncode != 0:
            print("ERROR: nvcc failed to compile trivial kernel:\n" + c.stderr,
                  file=sys.stderr)
            return 1
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        if r.returncode != 0 or r.stdout.strip() != "42":
            print("ERROR: trivial kernel ran incorrectly.", file=sys.stderr)
            return 1
    print("OK: nvcc present, sm_120 kernel compiles and runs.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
