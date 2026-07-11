// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void sobel(const float* u, float* o, long ny, long nx) {
  long j = blockIdx.x * blockDim.x + threadIdx.x;
  long i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= ny || j >= nx) return;
  long idx = i * nx + j;
  if (i == 0 || j == 0 || i == ny - 1 || j == nx - 1) { o[idx] = 0.0f; return; }
  float tl = u[idx - nx - 1], tc = u[idx - nx], tr = u[idx - nx + 1];
  float ml = u[idx - 1],                        mr = u[idx + 1];
  float bl = u[idx + nx - 1], bc = u[idx + nx], br = u[idx + nx + 1];
  float gx = -tl + tr - 2.0f * ml + 2.0f * mr - bl + br;
  float gy = -tl - 2.0f * tc - tr + bl + 2.0f * bc + br;
  o[idx] = sqrtf(gx * gx + gy * gy);
}
extern "C" void minidwarf_solve(const void* const* in_, void* const* out_, const long* d, int) {
  const float* in0 = (const float*)in_[0];
  float* out0 = (float*)out_[0];
  long ny = d[0], nx = d[1];
  dim3 t(32, 8);
  dim3 g((nx + t.x - 1) / t.x, (ny + t.y - 1) / t.y);
  sobel<<<g, t>>>(in0, out0, ny, nx);
}
