// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// One Jacobi iteration: x_new[i] = (b[i] - sum_{j != i} A[i,j]*x[j]) / A[i,i]
// over a square CSR matrix that has a stored diagonal entry in every row.
// No clean cuSPARSE op covers this fused gather-and-divide, so this
// baseline is a hand-written kernel (baseline: author_kernel), not a
// vendor library call. One thread per row: each thread linearly scans its
// row to find the diagonal entry (by matching col_idx[k] == i) while
// accumulating the full row dot product, then subtracts the diagonal
// contribution and divides.
__global__ void jacobi_sparse_kernel(const float* __restrict__ values, const int* __restrict__ row_ptr,
                                      const int* __restrict__ col_idx, const float* __restrict__ b,
                                      const float* __restrict__ x, float* __restrict__ x_new, int R) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= R) return;

  int start = row_ptr[i];
  int end = row_ptr[i + 1];

  float full_sum = 0.0f;
  float diag = 0.0f;
  for (int k = start; k < end; ++k) {
    int j = col_idx[k];
    float v = values[k];
    float contrib = v * x[j];
    full_sum += contrib;
    if (j == i) {
      diag = v;
    }
  }

  float off_sum = full_sum - diag * x[i];
  x_new[i] = (b[i] - off_sum) / diag;
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* values = (const float*)inputs_[0];
  const int* row_ptr = (const int*)inputs_[1];
  const int* col_idx = (const int*)inputs_[2];
  const float* b = (const float*)inputs_[3];
  const float* x = (const float*)inputs_[4];
  float* x_new = (float*)outputs_[0];

  int R = (int)dims[0];

  int threads = 256;
  int blocks = (R + threads - 1) / threads;
  jacobi_sparse_kernel<<<blocks, threads>>>(values, row_ptr, col_idx, b, x, x_new, R);
}
