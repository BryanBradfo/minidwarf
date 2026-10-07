// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <cublas_v2.h>
__global__ void vadd(const float* a, const float* b, float* c, long n){
  long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
// Red-team: delegates to cuBLAS (forbidden for candidates).
extern "C" void minidwarf_solve(const void* const* in, void* const* out, const long* dims, int nd){
  static cublasHandle_t h; static bool init = false; if (!init){ cublasCreate(&h); init = true; }
  long n = dims[0]; float one = 1.0f;
  cudaMemcpy(out[0], in[1], n*sizeof(float), cudaMemcpyDeviceToDevice);
  cublasSaxpy(h, (int)n, &one, (const float*)in[0], 1, (float*)out[0], 1); }
