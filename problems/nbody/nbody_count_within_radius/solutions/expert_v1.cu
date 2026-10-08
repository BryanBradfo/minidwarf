// SPDX-License-Identifier: Apache-2.0
// Canonical float32 expert: shared-memory tiling of positions, float32 distances.
#include <cuda_runtime.h>

#define TILE 256

__global__ void count_within_radius(const float* __restrict__ pos, float* __restrict__ out, long n) {
  __shared__ float tile[TILE];
  long i = (long)blockIdx.x * TILE + threadIdx.x;
  float pi = i < n ? pos[i] : 0.0f;
  int acc = 0;
  for (long base = 0; base < n; base += TILE) {
    long j = base + threadIdx.x;
    tile[threadIdx.x] = j < n ? pos[j] : 0.0f;
    __syncthreads();
    int m = (int)min((long)TILE, n - base);
    for (int t = 0; t < m; ++t) {
      if (base + t == i) continue;
      acc += fabsf(tile[t] - pi) < 3.0f;  // exact: inputs are on a 2^-16 grid
    }
    __syncthreads();
  }
  if (i < n) out[i] = (float)acc;
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* pos = (const float*)inputs_[0];
  float* out = (float*)outputs_[0];
  long n = dims[0];
  long blocks = (n + TILE - 1) / TILE;
  count_within_radius<<<(unsigned int)blocks, TILE>>>(pos, out, n);
}
