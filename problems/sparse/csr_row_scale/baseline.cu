// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// out[k] = values[k] * scale[row(k)], where row(k) is the row containing
// nonzero k of a CSR matrix (row_ptr length R+1). No clean cuSPARSE op
// covers this per-row scaling, so this baseline is a hand-written kernel
// (baseline: author_kernel), not a vendor library call. One thread per row:
// each thread walks its row's slice [row_ptr[i], row_ptr[i+1)) and scales it
// by scale[i].
__global__ void csr_row_scale_kernel(const float* __restrict__ values, const int* __restrict__ row_ptr,
                                      const float* __restrict__ scale, float* __restrict__ out, int R) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= R) return;

  int start = row_ptr[i];
  int end = row_ptr[i + 1];
  float s = scale[i];
  for (int k = start; k < end; ++k) {
    out[k] = values[k] * s;
  }
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* values = (const float*)inputs_[0];
  const int* row_ptr = (const int*)inputs_[1];
  const float* scale = (const float*)inputs_[2];
  float* out = (float*)outputs_[0];

  int R = (int)dims[0];

  int threads = 256;
  int blocks = (R + threads - 1) / threads;
  csr_row_scale_kernel<<<blocks, threads>>>(values, row_ptr, scale, out, R);
}
