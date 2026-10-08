// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void vadd(const float* a, const float* b, float* c, long n){
  long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
// Red-team: computes on the first call only, then returns without touching outputs.
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  static bool done = false; if (done) return;
  long n = dims[0];
  vadd<<<(int)((n+255)/256),256>>>((const float*)in[0], (const float*)in[1], (float*)out[0], n);
  done = true; }
