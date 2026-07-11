// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#define TILE 32

// Row-major B(N x M) = A(M x N)^T, classic shared-memory tiled transpose
// with a padded tile to avoid shared-memory bank conflicts.
__global__ void transpose_tiled(const float* __restrict__ A, float* __restrict__ B,
                                 int M, int N) {
  __shared__ float tile[TILE][TILE + 1];

  int a_col = blockIdx.x * TILE + threadIdx.x;  // index into N
  int a_row = blockIdx.y * TILE + threadIdx.y;  // index into M

  if (a_row < M && a_col < N) {
    tile[threadIdx.y][threadIdx.x] = A[(long)a_row * N + a_col];
  }
  __syncthreads();

  int b_col = blockIdx.y * TILE + threadIdx.x;  // index into M
  int b_row = blockIdx.x * TILE + threadIdx.y;  // index into N

  if (b_row < N && b_col < M) {
    B[(long)b_row * M + b_col] = tile[threadIdx.x][threadIdx.y];
  }
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* A = (const float*)inputs_[0];
  float* B = (float*)outputs_[0];
  int m = (int)dims[0];
  int n = (int)dims[1];

  dim3 threads(TILE, TILE);
  dim3 blocks((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  transpose_tiled<<<blocks, threads>>>(A, B, m, n);
}
