// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUBLAS_CHECK(call) do { cublasStatus_t _s=(call); if(_s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error at %s:%d: status %d\n",__FILE__,__LINE__,(int)_s); exit(1);} } while(0)

// Row-major y(M) = A(M x N) * x(N).
//
// cuBLAS expects column-major matrices. A row-major M x N matrix, read as-is,
// is exactly a column-major N x M matrix (same underlying memory layout), so
// our buffer aliases a col-major matrix Acm (N rows, M cols, lda=N) with
// Acm == A^T. Calling cublasSgemv with trans=T on Acm computes
// op(Acm) = Acm^T = A, so y = op(Acm) * x = A * x, exactly what we need.
extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* A = (const float*)inputs_[0];
  const float* x = (const float*)inputs_[1];
  float* y = (float*)outputs_[0];
  int m = (int)dims[0];
  int n = (int)dims[1];

  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    CUBLAS_CHECK(cublasCreate(&handle));
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T,
                            n, m,
                            &alpha,
                            A, n,
                            x, 1,
                            &beta,
                            y, 1));
}
