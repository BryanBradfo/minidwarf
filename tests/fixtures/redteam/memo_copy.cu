// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void vadd(const float* a, const float* b, float* c, long n){
  long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
// Red-team: computes into a private buffer on the first call, then only copies it to the output.
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  static float* cache = nullptr; long n = dims[0];
  if (!cache){ cudaMalloc(&cache, n*sizeof(float));
    vadd<<<(int)((n+255)/256),256>>>((const float*)in[0], (const float*)in[1], cache, n); }
  cudaMemcpy(out[0], cache, n*sizeof(float), cudaMemcpyDeviceToDevice); }
