// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusparse.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)

#define CUSPARSE_CHECK(call) do { cusparseStatus_t _s=(call); if(_s!=CUSPARSE_STATUS_SUCCESS){ \
  fprintf(stderr,"cuSPARSE error at %s:%d: status %d\n",__FILE__,__LINE__,(int)_s); exit(1);} } while(0)

// y = A * x, A is a sparse R x C matrix in CSR (row_ptr length R+1, col_idx/
// values length NNZ), x is a dense vector of length C, y is dense of length
// R. Uses the cuSPARSE generic SpMV API (cusparseCreateCsr / cusparseSpMV).
extern "C" void minidwarf_solve(const void* const* inputs_, void* const* outputs_,
                                 const long* dims, int n_dims) {
  const float* values = (const float*)inputs_[0];
  const int* row_ptr = (const int*)inputs_[1];
  const int* col_idx = (const int*)inputs_[2];
  const float* x = (const float*)inputs_[3];
  float* y = (float*)outputs_[0];

  int64_t R = dims[0];
  int64_t C = dims[1];
  int64_t NNZ = dims[2];

  static cusparseHandle_t handle = nullptr;
  if (handle == nullptr) {
    CUSPARSE_CHECK(cusparseCreate(&handle));
  }

  cusparseConstSpMatDescr_t matA;
  CUSPARSE_CHECK(cusparseCreateConstCsr(&matA, R, C, NNZ,
                                        (const void*)row_ptr, (const void*)col_idx, (const void*)values,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  cusparseConstDnVecDescr_t vecX;
  CUSPARSE_CHECK(cusparseCreateConstDnVec(&vecX, C, (const void*)x, CUDA_R_32F));

  cusparseDnVecDescr_t vecY;
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, R, (void*)y, CUDA_R_32F));

  const float alpha = 1.0f;
  const float beta = 0.0f;

  size_t buffer_size = 0;
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha, matA, vecX, &beta, vecY,
                                         CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size));

  void* d_buffer = nullptr;
  if (buffer_size > 0) {
    CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
  }

  CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha, matA, vecX, &beta, vecY,
                              CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));

  if (d_buffer != nullptr) {
    CUDA_CHECK(cudaFree(d_buffer));
  }
  CUSPARSE_CHECK(cusparseDestroySpMat(matA));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
}
