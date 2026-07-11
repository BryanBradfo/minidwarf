// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void jac(const float* u,float* o,int nx,int ny,int nz){
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int j=blockIdx.y*blockDim.y+threadIdx.y;
  int i=blockIdx.z*blockDim.z+threadIdx.z;
  if(i>=nx||j>=ny||k>=nz) return;
  long idx=((long)i*ny+j)*nz+k;
  if(i==0||j==0||k==0||i==nx-1||j==ny-1||k==nz-1){ o[idx]=u[idx]; return; }
  o[idx]=(u[idx+ny*nz]+u[idx-ny*nz]+u[idx+nz]+u[idx-nz]+u[idx+1]+u[idx-1]+u[idx])/7.0f;
}
extern "C" void minidwarf_solve(const void* const* in_,void* const* out_,const long* d,int){
  const float* in0=(const float*)in_[0]; float* out0=(float*)out_[0];
  int nx=d[0],ny=d[1],nz=d[2]; dim3 t(32,4,2);
  dim3 g((nz+t.x-1)/t.x,(ny+t.y-1)/t.y,(nx+t.z-1)/t.z);
  jac<<<g,t>>>(in0,out0,nx,ny,nz);
}
