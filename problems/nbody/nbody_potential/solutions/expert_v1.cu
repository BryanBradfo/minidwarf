// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

__global__ void potential(const float* pos, float* out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double pi = (double)pos[i];
  double acc = 0.0;
  for (long j = 0; j < n; ++j) {
    if (j == i) continue;
    double d = (double)pos[j] - pi;
    double r2 = d * d;
    acc += 1.0 / sqrt(r2 + 1e-2);
  }
  out[i] = (float)acc;
}

extern "C" void minidwarf_solve(const float* const* inputs, float* const* outputs,
                                 const long* dims, int n_dims) {
  long n = dims[0];
  int threads = 256;
  long blocks = (n + threads - 1) / threads;
  potential<<<(unsigned int)blocks, threads>>>(inputs[0], outputs[0], n);
}
