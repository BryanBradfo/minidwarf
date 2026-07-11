// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// out_values[e] = dot(A[row(e), :], B[:, col_idx[e]]) for each nonzero e of
// a CSR sparsity pattern over an R x C matrix, A dense R x K row-major, B
// dense K x C row-major. One thread per nonzero: binary-search row_ptr to
// find row(e), then walk the K-length dot product.
__global__ void sddmm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                              const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
                              float* __restrict__ out_values, int R, int C, int K, int NNZ) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= NNZ) return;

  // Binary search for the row i such that row_ptr[i] <= e < row_ptr[i+1].
  int lo = 0, hi = R;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (row_ptr[mid + 1] <= e) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  int row = lo;
  int col = col_idx[e];

  const float* a_row = A + (size_t)row * K;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += a_row[k] * B[(size_t)k * C + col];
  }
  out_values[e] = acc;
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* A = (const float*)inputs_[0];
  const float* B = (const float*)inputs_[1];
  const int* row_ptr = (const int*)inputs_[2];
  const int* col_idx = (const int*)inputs_[3];
  float* out_values = (float*)outputs_[0];

  int R = (int)dims[0];
  int C = (int)dims[1];
  int K = (int)dims[2];
  int NNZ = (int)dims[3];

  int threads = 256;
  int blocks = (NNZ + threads - 1) / threads;
  sddmm_kernel<<<blocks, threads>>>(A, B, row_ptr, col_idx, out_values, R, C, K, NNZ);
}
