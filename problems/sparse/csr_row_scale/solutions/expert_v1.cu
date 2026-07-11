// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// out[k] = values[k] * scale[row(k)]. One thread per nonzero: each thread
// binary-searches row_ptr to find its row, then applies the scale factor.
// This differs from the row-per-thread baseline: it exposes NNZ-way
// parallelism instead of R-way, which is friendlier when rows are uneven or
// R is small relative to NNZ.
__global__ void csr_row_scale_kernel(const float* __restrict__ values, const int* __restrict__ row_ptr,
                                      const float* __restrict__ scale, float* __restrict__ out, int R, int NNZ) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= NNZ) return;

  // Binary search for the row i such that row_ptr[i] <= k < row_ptr[i+1].
  int lo = 0, hi = R;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (row_ptr[mid + 1] <= k) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  out[k] = values[k] * scale[lo];
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* values = (const float*)inputs_[0];
  const int* row_ptr = (const int*)inputs_[1];
  const float* scale = (const float*)inputs_[2];
  float* out = (float*)outputs_[0];

  int R = (int)dims[0];
  int NNZ = (int)dims[2];

  int threads = 256;
  int blocks = (NNZ + threads - 1) / threads;
  csr_row_scale_kernel<<<blocks, threads>>>(values, row_ptr, scale, out, R, NNZ);
}
