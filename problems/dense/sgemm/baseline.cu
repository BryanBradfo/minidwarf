// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUBLAS_CHECK(call) do { cublasStatus_t _s=(call); if(_s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error at %s:%d: status %d\n",__FILE__,__LINE__,(int)_s); exit(1);} } while(0)

// Row-major C(M x N) = A(M x K) * B(K x N).
//
// cuBLAS expects column-major matrices. A row-major M x N matrix, read as-is,
// is exactly a column-major N x M matrix (same underlying memory layout).
// So rather than transposing anything explicitly, we compute
//   C^T (col-major N x M) = B^T (col-major N x K) * A^T (col-major K x M)
// which is precisely what cublasSgemm(N, N, m=N, n=M, k=K, B, ldb=N, A, lda=K,
// C, ldc=N) computes when B/A/C are interpreted as the column-major matrices
// that alias our row-major B(K x N)/A(M x K)/C(M x N) buffers.
extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* A = (const float*)inputs_[0];
  const float* B = (const float*)inputs_[1];
  float* C = (float*)outputs_[0];
  int m = (int)dims[0];
  int n = (int)dims[1];
  int k = (int)dims[2];

  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    CUBLAS_CHECK(cublasCreate(&handle));
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            n, m, k,
                            &alpha,
                            B, n,
                            A, k,
                            &beta,
                            C, n));
}
