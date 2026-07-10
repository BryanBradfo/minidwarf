// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
__global__ void laplacian(const float* u, float* o, long ny, long nx) {
  long j = blockIdx.x * blockDim.x + threadIdx.x;
  long i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= ny || j >= nx) return;
  long idx = i * nx + j;
  if (i == 0 || j == 0 || i == ny - 1 || j == nx - 1) { o[idx] = u[idx]; return; }
  o[idx] = u[idx + nx] + u[idx - nx] + u[idx + 1] + u[idx - 1] - 4.0f * u[idx];
}
extern "C" void minidwarf_solve(const float* const* in, float* const* out, const long* d, int) {
  long ny = d[0], nx = d[1];
  dim3 t(32, 8);
  dim3 g((nx + t.x - 1) / t.x, (ny + t.y - 1) / t.y);
  laplacian<<<g, t>>>(in[0], out[0], ny, nx);
}
