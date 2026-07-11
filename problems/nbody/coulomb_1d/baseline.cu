// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

__global__ void coulomb(const float* pos, float* out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double pi = (double)pos[i];
  double acc = 0.0;
  for (long j = 0; j < n; ++j) {
    if (j == i) continue;
    double d = pi - (double)pos[j];
    double r2 = d * d;
    double denom = pow(r2 + 1e-2, 1.5);
    acc += d / denom;
  }
  out[i] = (float)acc;
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
