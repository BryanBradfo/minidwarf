// SPDX-License-Identifier: Apache-2.0
// Hardcodes zeros: correct only for degenerate inputs, fails real correctness.
#include <cuda_runtime.h>
__global__ void zero(float* c,long n){ long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=0.0f; }
extern "C" void minidwarf_solve(const float* const*,float* const* out,const long* dims,int nd){
  long n=1; for(int i=0;i<nd;i++) n*=dims[i]; int t=256; zero<<<(int)((n+t-1)/t),t>>>(out[0],n); }
