// SPDX-License-Identifier: Apache-2.0
// Fixture for test_timeout.py: a candidate that launches a kernel which
// never returns, so the driver's device sync after the kernel launch never
// completes. Used to prove grade_problem scores this as a timeout instead
// of letting subprocess.TimeoutExpired crash the whole suite run.
#include <cuda_runtime.h>
__global__ void hang(){
  // A plain `while (true) {}` with no side effects is UB per the C++
  // standard (an infinite loop that does nothing observable), so nvcc is
  // free to optimize it away entirely. The volatile read forces a real,
  // observable memory access on every iteration so the compiler can't
  // eliminate the loop, and the device sync in the driver truly never
  // returns.
  volatile int spin = 1;
  while (spin) {}
}
extern "C" void minidwarf_solve(const void* const* inputs, void* const* outputs, const long* dims, int n_dims){
  hang<<<1,1>>>();
}
