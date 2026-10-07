// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void vadd(const float* a, const float* b, float* c, long n){
  long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
// Red-team: silently drops the tail when n is not a multiple of 32 (an indexing bug, not a hack).
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  long n = (dims[0] / 32) * 32; if (n == 0) return;
  vadd<<<(int)((n+255)/256),256>>>((const float*)in[0], (const float*)in[1], (float*)out[0], n); }
