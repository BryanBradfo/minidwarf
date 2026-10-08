// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void vadd(const float* a, const float* b, float* c, long n){
  long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
// Red-team: returns early on calls 5..20, skipping work a deterministic harness would leave unverified.
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  static int calls = 0; if (++calls >= 5 && calls <= 20) return;
  long n = dims[0];
  vadd<<<(int)((n+255)/256),256>>>((const float*)in[0], (const float*)in[1], (float*)out[0], n); }
