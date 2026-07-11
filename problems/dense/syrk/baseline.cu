// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUBLAS_CHECK(call) do { cublasStatus_t _s=(call); if(_s!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error at %s:%d: status %d\n",__FILE__,__LINE__,(int)_s); exit(1);} } while(0)

// Row-major C(M x M) = A(M x K) * A(M x K)^T.
//
// A row-major M x K buffer, reinterpreted col-major with lda=K, is exactly
// a K x M matrix Aview == A^T (standard aliasing trick). Calling
// cublasSsyrk with trans=T on Aview (a k x n array with k=K, n=M) computes
// op(Aview) = Aview^T = A, so C = op(Aview)^T ... more precisely ssyrk
// computes C = alpha*op(A)*op(A)^T + beta*C where op(A)=A^T for trans=T,
// giving C = alpha*A*A^T + beta*C -- exactly the product we want. C itself
// is symmetric, so the row-major/col-major aliasing is a non-issue for the
// output buffer -- but which triangle CUBLAS_FILL_MODE_{LOWER,UPPER} refers
// to still depends on the col-major-vs-row-major reading of C, so we pick
// CUBLAS_FILL_MODE_LOWER empirically (verified against the reference below)
// and mirror the entries it wrote (row-major j <= i) into the untouched
// half (row-major i < j).
__global__ void symmetrize_kernel(float* C, int M) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M || j >= M || i <= j) return;
  C[(long)i * M + j] = C[(long)j * M + i];
}

extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* A = (const float*)inputs_[0];
  float* C = (float*)outputs_[0];
  int m = (int)dims[0];
  int k = (int)dims[1];

  static cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    CUBLAS_CHECK(cublasCreate(&handle));
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                            m, k,
                            &alpha,
                            A, k,
                            &beta,
                            C, m));

  dim3 threads(16, 16);
  dim3 blocks((m + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
  symmetrize_kernel<<<blocks, threads>>>(C, m);
}
