// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// Row-major y(M) = A(M x N) * x(N). One thread per output row, dot product
// over N.
__global__ void gemv_kernel(const float* __restrict__ A, const float* __restrict__ x,
                             float* __restrict__ y, int M, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;
  const float* a_row = A + (long)row * N;
  float acc = 0.0f;
  for (int j = 0; j < N; ++j) {
    acc += a_row[j] * x[j];
  }
  y[row] = acc;
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* A = (const float*)inputs_[0];
  const float* x = (const float*)inputs_[1];
  float* y = (float*)outputs_[0];
  int m = (int)dims[0];
  int n = (int)dims[1];

  int threads = 256;
  int blocks = (m + threads - 1) / threads;
  gemv_kernel<<<blocks, threads>>>(A, x, y, m, n);
}
