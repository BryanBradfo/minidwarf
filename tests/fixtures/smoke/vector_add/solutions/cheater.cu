// SPDX-License-Identifier: Apache-2.0
// Anti-gaming fixture: intentionally computes the CORRECT result only for
// n == 1048576 (this fixture's spec.yaml eval_shapes[0] == [1048576]) and
// writes wrong output (zeros) for any other size, in particular
// eval_shapes[1] == [1500000]. This is coupled to spec.yaml's eval_shapes:
// if those shapes change, this fixture must be updated to match.
//
// The purpose is to prove that grade_problem truly requires the candidate
// to pass ALL eval shapes, not just the first one. A grader bug that only
// checked eval_shapes[0] would incorrectly mark this cheater as correct;
// the real grader must catch it via eval_shapes[1].
#include <cuda_runtime.h>
__global__ void add(const float* a, const float* b, float* c, long n){
  long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+b[i];
}
__global__ void zero(float* c, long n){
  long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) c[i]=0.0f;
}
extern "C" void minidwarf_solve(const float* const* inputs, float* const* outputs, const long* dims, int n_dims){
  long n=1; for(int i=0;i<n_dims;i++) n*=dims[i];
  int t=256; int b=(int)((n+t-1)/t);
  if (n == 1048576) {
    add<<<b,t>>>(inputs[0], inputs[1], outputs[0], n);
  } else {
    zero<<<b,t>>>(outputs[0], n);
  }
}
