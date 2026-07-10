// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void vadd(const float* a,const float* b,float* c,long n){
  long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i]; }
extern "C" void minidwarf_solve(const float* const* in,float* const* out,
                                const long* dims,int nd){
  long n=1; for(int i=0;i<nd;i++) n*=dims[i];
  int t=256; vadd<<<(int)((n+t-1)/t),t>>>(in[0],in[1],out[0],n); }
