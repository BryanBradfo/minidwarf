// SPDX-License-Identifier: Apache-2.0
// Naive float32 baseline: one thread per body, plain loop over all others from global memory.
#include <cuda_runtime.h>

__global__ void coulomb(const float* pos, float* out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float pi = pos[i];
  float acc = 0.0f;
  for (long j = 0; j < n; ++j) {
    if (j == i) continue;
    float d = pi - pos[j];
    float r2 = d * d + 1e-2f;
    acc += d / (r2 * sqrtf(r2));
  }
  out[i] = acc;
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* in0 = (const float*)inputs_[0];
  float* out0 = (float*)outputs_[0];
  long n = dims[0];
  int threads = 256;
  long blocks = (n + threads - 1) / threads;
  coulomb<<<(unsigned int)blocks, threads>>>(in0, out0, n);
}
