// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// One Jacobi iteration: x_new[i] = (b[i] - sum_{j != i} A[i,j]*x[j]) / A[i,i].
// One thread per row: since col_idx is sorted within each row and every row
// is guaranteed to contain the diagonal column i, the diagonal's position
// is located with a binary search instead of a linear scan, then the row's
// dot product is accumulated skipping that position directly.
__global__ void jacobi_sparse_kernel(const float* __restrict__ values, const int* __restrict__ row_ptr,
                                      const int* __restrict__ col_idx, const float* __restrict__ b,
                                      const float* __restrict__ x, float* __restrict__ x_new, int R) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= R) return;

  int start = row_ptr[i];
  int end = row_ptr[i + 1];

  // Binary search within [start, end) for the position holding column i.
  int lo = start, hi = end;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (col_idx[mid] < i) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  int diag_pos = lo;
  float diag = values[diag_pos];

  float off_sum = 0.0f;
  for (int k = start; k < end; ++k) {
    if (k == diag_pos) continue;
    off_sum += values[k] * x[col_idx[k]];
  }

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
