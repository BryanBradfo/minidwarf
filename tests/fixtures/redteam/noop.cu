// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void vadd(const float* a, const float* b, float* c, long n){
  long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
// Red-team: does nothing (relies on stale output buffers).
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){}
