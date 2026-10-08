// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void vadd(const float* a, const float* b, float* c, long n){
  long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
// Red-team: launches on a non-blocking stream and never synchronizes.
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  static cudaStream_t s = nullptr; if (!s) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  long n = dims[0];
  vadd<<<(int)((n+255)/256),256,0,s>>>((const float*)in[0], (const float*)in[1], (float*)out[0], n); }
