// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// Row-major C(M x M) = A(M x K) * A(M x K)^T, brute force: one thread per
// output element, full matrix written directly (no triangle trick).
__global__ void syrk_kernel(const float* __restrict__ A, float* __restrict__ C,
                             int M, int K) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= M) return;

  const float* a_row = A + (long)i * K;
  const float* b_row = A + (long)j * K;
  float acc = 0.0f;
  for (int p = 0; p < K; ++p) {
    acc += a_row[p] * b_row[p];
  }
  C[(long)i * M + j] = acc;
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* A = (const float*)inputs_[0];
  float* C = (float*)outputs_[0];
  int m = (int)dims[0];
  int k = (int)dims[1];

  dim3 threads(16, 16);
  dim3 blocks((m + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
  syrk_kernel<<<blocks, threads>>>(A, C, m, k);
}
