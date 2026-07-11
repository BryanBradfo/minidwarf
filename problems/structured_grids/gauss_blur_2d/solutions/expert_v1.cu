// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void gauss_blur(const float* u, float* o, long ny, long nx) {
  long j = blockIdx.x * blockDim.x + threadIdx.x;
  long i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= ny || j >= nx) return;
  long idx = i * nx + j;
  if (i == 0 || j == 0 || i == ny - 1 || j == nx - 1) { o[idx] = u[idx]; return; }
  float sum =
      1.0f * u[idx - nx - 1] + 2.0f * u[idx - nx] + 1.0f * u[idx - nx + 1] +
      2.0f * u[idx - 1]      + 4.0f * u[idx]       + 2.0f * u[idx + 1] +
      1.0f * u[idx + nx - 1] + 2.0f * u[idx + nx]  + 1.0f * u[idx + nx + 1];
  o[idx] = sum / 16.0f;
}
extern "C" void minidwarf_solve(const void* const* in_, void* const* out_, const long* d, int) {
  const float* in0 = (const float*)in_[0];
  float* out0 = (float*)out_[0];
  long ny = d[0], nx = d[1];
  dim3 t(32, 8);
  dim3 g((nx + t.x - 1) / t.x, (ny + t.y - 1) / t.y);
  gauss_blur<<<g, t>>>(in0, out0, ny, nx);
}
