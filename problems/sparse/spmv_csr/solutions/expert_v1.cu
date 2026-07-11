// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// y = A * x, A is a sparse R x C matrix in CSR. One thread per row: each
// thread walks its row's nonzeros (row_ptr[i] .. row_ptr[i+1)) and
// accumulates values[k] * x[col_idx[k]].
__global__ void spmv_csr_kernel(const float* __restrict__ values, const int* __restrict__ row_ptr,
                                 const int* __restrict__ col_idx, const float* __restrict__ x,
                                 float* __restrict__ y, int R) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= R) return;

  int start = row_ptr[i];
  int end = row_ptr[i + 1];
  float acc = 0.0f;
  for (int k = start; k < end; ++k) {
    acc += values[k] * x[col_idx[k]];
  }
  y[i] = acc;
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* values = (const float*)inputs_[0];
  const int* row_ptr = (const int*)inputs_[1];
  const int* col_idx = (const int*)inputs_[2];
  const float* x = (const float*)inputs_[3];
  float* y = (float*)outputs_[0];

  int R = (int)dims[0];

  int threads = 256;
  int blocks = (R + threads - 1) / threads;
  spmv_csr_kernel<<<blocks, threads>>>(values, row_ptr, col_idx, x, y, R);
}
