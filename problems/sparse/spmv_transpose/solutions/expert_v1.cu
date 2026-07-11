// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// y = A^T * x, A is a sparse R x C matrix in CSR. One thread per row:
// each thread reads x[i] once and scatters values[k] * x[i] into
// y[col_idx[k]] via atomicAdd for every nonzero k in its row. y must be
// zeroed first (via cudaMemset) since the output buffer is not pre-zeroed
// by the caller.
__global__ void spmv_transpose_kernel(const float* __restrict__ values, const int* __restrict__ row_ptr,
                                       const int* __restrict__ col_idx, const float* __restrict__ x,
                                       float* __restrict__ y, int R) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= R) return;

  int start = row_ptr[i];
  int end = row_ptr[i + 1];
  float xi = x[i];
  for (int k = start; k < end; ++k) {
    atomicAdd(&y[col_idx[k]], values[k] * xi);
  }
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* values = (const float*)inputs_[0];
  const int* row_ptr = (const int*)inputs_[1];
  const int* col_idx = (const int*)inputs_[2];
  const float* x = (const float*)inputs_[3];
  float* y = (float*)outputs_[0];

  int R = (int)dims[0];
  int C = (int)dims[1];

  cudaMemset(y, 0, (size_t)C * sizeof(float));

  int threads = 256;
  int blocks = (R + threads - 1) / threads;
  spmv_transpose_kernel<<<blocks, threads>>>(values, row_ptr, col_idx, x, y, R);
}
