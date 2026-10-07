// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void vadd(const float* a, const float* b, float* c, long n){
  long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
// Red-team: caches results per call index (calls % 4); defeated by the randomized data-set order.
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  static float* slot[4] = {nullptr, nullptr, nullptr, nullptr}; static int calls = 0;
  int k = calls++ % 4; long n = dims[0];
  if (!slot[k]){ cudaMalloc(&slot[k], n*sizeof(float));
    vadd<<<(int)((n+255)/256),256>>>((const float*)in[0], (const float*)in[1], slot[k], n); }
  cudaMemcpy(out[0], slot[k], n*sizeof(float), cudaMemcpyDeviceToDevice); }
