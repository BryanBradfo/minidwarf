// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#define TILE 16

// Row-major C(M x N) = A(M x K) * B(K x N), classic shared-memory tiled SGEMM.
__global__ void sgemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;  // index into M
  int col = blockIdx.x * TILE + threadIdx.x;  // index into N

  float acc = 0.0f;
  int n_tiles = (K + TILE - 1) / TILE;
  for (int t = 0; t < n_tiles; ++t) {
    int a_col = t * TILE + threadIdx.x;  // column in A (K dim)
    int b_row = t * TILE + threadIdx.y;  // row in B (K dim)

    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[(long)row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[(long)b_row * N + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int i = 0; i < TILE; ++i) {
      acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[(long)row * N + col] = acc;
  }
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* A = (const float*)inputs_[0];
  const float* B = (const float*)inputs_[1];
  float* C = (float*)outputs_[0];
  int m = (int)dims[0];
  int n = (int)dims[1];
  int k = (int)dims[2];

  dim3 threads(TILE, TILE);
  dim3 blocks((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  sgemm_tiled<<<blocks, threads>>>(A, B, C, m, n, k);
}
