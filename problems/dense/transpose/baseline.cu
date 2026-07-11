// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUBLAS_CHECK(call) do { cublasStatus_t _s=(call); if(_s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error at %s:%d: status %d\n",__FILE__,__LINE__,(int)_s); exit(1);} } while(0)

// Row-major B(N x M) = A(M x N)^T.
//
// A row-major M x N buffer, reinterpreted col-major with lda=N, is exactly
// an N x M matrix Aview == A^T (standard row-major/col-major aliasing).
// B row-major N x M buffer, reinterpreted col-major with ldb=M, is exactly
// an M x N matrix Bview == B^T == A.
//
// So we want Bview (M x N, col-major, ldc=M) = Aview^T (M x N). That is a
// single cublasSgeam transpose call: C = alpha*op(A) + beta*op(B) with
// m=M, n=N, transa=T, A=buffer_A (N x M view, lda=N), beta=0 (B unused,
// but pass the destination itself for a valid in-place-safe geam call).
extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* A = (const float*)inputs_[0];
  float* B = (float*)outputs_[0];
  int m = (int)dims[0];
  int n = (int)dims[1];

  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    CUBLAS_CHECK(cublasCreate(&handle));
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            m, n,
                            &alpha,
                            A, n,
                            &beta,
                            B, m,
                            B, m));
}
