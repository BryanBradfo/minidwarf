// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUBLAS_CHECK(call) do { cublasStatus_t _s=(call); if(_s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error at %s:%d: status %d\n",__FILE__,__LINE__,(int)_s); exit(1);} } while(0)
#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

// Row-major C(M x N) = A(M x K) * B(K x N) + bias(M x N).
//
// Same row-major/col-major aliasing trick as sgemm: computing
// C^T = B^T * A^T via cublasSgemm(N, N, n=N, m=M, k=K, ...) yields exactly
// the row-major product A*B. We first copy bias into C, then call sgemm
// with beta=1 so that C = alpha*A*B + beta*C = A*B + bias.
extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* A = (const float*)inputs_[0];
  const float* B = (const float*)inputs_[1];
  const float* bias = (const float*)inputs_[2];
  float* C = (float*)outputs_[0];
  int m = (int)dims[0];
  int n = (int)dims[1];
  int k = (int)dims[2];

  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    CUBLAS_CHECK(cublasCreate(&handle));
  }

  CUDA_CHECK(cudaMemcpy(C, bias, (size_t)m * n * sizeof(float), cudaMemcpyDeviceToDevice));

  const float alpha = 1.0f;
  const float beta = 1.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            n, m, k,
                            &alpha,
                            B, n,
                            A, k,
                            &beta,
                            C, n));
}
