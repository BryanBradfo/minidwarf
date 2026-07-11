// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// Y = A * B, A is a sparse R x C matrix in CSR, B/Y are dense row-major
// matrices of shape (C, D) / (R, D). One thread per (row, d): the thread
// walks its row's nonzeros (row_ptr[i] .. row_ptr[i+1)) and accumulates
// values[k] * B[col_idx[k] * D + d].
__global__ void spmm_csr_kernel(const float* __restrict__ values, const int* __restrict__ row_ptr,
                                 const int* __restrict__ col_idx, const float* __restrict__ B,
                                 float* __restrict__ Y, int R, int D) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = R * D;
  if (idx >= total) return;

  int i = idx / D;
  int d = idx % D;

  int start = row_ptr[i];
  int end = row_ptr[i + 1];
  float acc = 0.0f;
  for (int k = start; k < end; ++k) {
    acc += values[k] * B[col_idx[k] * D + d];
  }
  Y[i * D + d] = acc;
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* values = (const float*)inputs_[0];
  const int* row_ptr = (const int*)inputs_[1];
  const int* col_idx = (const int*)inputs_[2];
  const float* B = (const float*)inputs_[3];
  float* Y = (float*)outputs_[0];

  int R = (int)dims[0];
  int D = (int)dims[3];

  int total = R * D;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  spmm_csr_kernel<<<blocks, threads>>>(values, row_ptr, col_idx, B, Y, R, D);
}
